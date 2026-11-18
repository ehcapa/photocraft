//! `portrait.create`: the fork's smart-portrait orchestrator (crates/portrait,
//! docs/fork/2026-11-18-portrait-pipeline.md).
//!
//! Analyzes the person through the Select Subject line (the `photocraft-ml` seed when
//! available, the classical saliency otherwise), then materializes `photocraft_portrait`'s
//! plan as ONE history step of standard layers: a named group holding a person-masked
//! Hue/Saturation adjustment layer. No new layer kind — what lands in the document is
//! ordinary Photoshop-shaped layers the user keeps tuning, so a PSD round trip and an
//! upstream sync see nothing new.

use std::sync::Arc;

use photocraft_algo::matting;
use photocraft_algo::segment::subject;
use photocraft_doc::{Document, Layer, LayerContent, LayerMask};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::smartselect_cmds::with_doc_sampler;
use crate::{EngineError, Result, Session};

const CMD: &str = "portrait.create";

fn enabled(s: &Session) -> std::result::Result<(), String> {
    s.active().ok_or("no document open")?;
    Ok(())
}

/// The recipe from the command's params, field by field: anything absent keeps the
/// `photocraft-portrait` default, wrong types included (the adjustment layer's own values are
/// range-checked where they land, in `adjust_params`). A `hairHue`/`hairSaturation`/
/// `hairLightness` value opts into the hair-recolor layer.
fn recipe(p: &Value) -> photocraft_portrait::Recipe {
    let mut r = photocraft_portrait::Recipe::default();
    if let Some(name) = p.get("groupName").and_then(Value::as_str) {
        r.group_name = name.into();
    }
    let h = &mut r.hue_saturation;
    if let Some(v) = p.get("hue").and_then(Value::as_f64) {
        h.hue = v as f32;
    }
    if let Some(v) = p.get("saturation").and_then(Value::as_f64) {
        h.saturation = v as f32;
    }
    if let Some(v) = p.get("lightness").and_then(Value::as_f64) {
        h.lightness = v as f32;
    }
    let asked = |key: &str| p.get(key).and_then(Value::as_f64).map(|v| v as f32);
    if [p.get("hairHue"), p.get("hairSaturation"), p.get("hairLightness")].iter().any(|v| v.is_some()) {
        r.hair = Some(photocraft_portrait::HueSat {
            hue: asked("hairHue").unwrap_or(0.0),
            saturation: asked("hairSaturation").unwrap_or(0.0),
            lightness: asked("hairLightness").unwrap_or(0.0),
        });
    }
    r
}

/// The parsing regions of `doc`, upsampled to document resolution and intersected with the
/// person: the model's per-class probabilities grouped into regions
/// (`photocraft_portrait::parsing`), each resampled like the subject mask rides back and
/// clipped to the person the pipeline is about (hair, skin and clothes belong to the
/// person; on off-distribution input the gate also suppresses stray predictions). Empty
/// masks stay out — a region with no support anywhere must not produce a layer.
fn parsing_regions(
    doc: &Document,
    target: Option<photocraft_doc::LayerId>,
    sample_all: bool,
    person: &photocraft_raster::Surface,
) -> Vec<(photocraft_portrait::Region, photocraft_raster::Surface)> {
    let Some(found) = with_doc_sampler(doc, target, sample_all, |smp, d| {
        let step = photocraft_algo::segment::scale_for(d.bounds(), photocraft_ml::PARSING_SIDE * photocraft_ml::PARSING_SIDE);
        let img = smp.rgb_scaled(d.bounds(), step);
        photocraft_ml::parsing_probs(&img.px, img.w, img.h)
    }) else {
        return Vec::new();
    };
    let (dw, dh) = (doc.size.width as usize, doc.size.height as usize);
    // Read the person gate per pixel through a LayerMask (that is where `value` lives).
    let person = photocraft_doc::LayerMask { surface: person.clone(), ..photocraft_doc::LayerMask::reveal_all() };
    photocraft_portrait::parsing::region_planes(&found.planes, found.w, found.h)
        .into_iter()
        .filter_map(|(region, mask)| {
            let support = mask.iter().fold(0.0f32, |m, v| m.max(*v));
            if support < 0.5 {
                return None;
            }
            let mut up = photocraft_ml::resize_mask(&mask, found.w, found.h, dw, dh)?;
            for y in 0..dh {
                for x in 0..dw {
                    up[y * dw + x] *= person.value(x as i32, y as i32);
                }
            }
            let mut surface = photocraft_raster::Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
            surface.write_region(photocraft_geom::Rect::new(0, 0, doc.size.width as i32, doc.size.height as i32), &up);
            Some((region, surface))
        })
        .collect()
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let recipe = recipe(p);
    let sample_all = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let cached = s.portrait_cache.get(&st.doc, crate::layer_multi_cmds::same_pixels);
    let doc = Arc::clone(&st.doc);
    let target = st.active_layer;
    crate::jobs::run(
        s,
        "Smart Portrait",
        true,
        move |ctx| {
            if let Some(analysis) = cached {
                return Ok(analysis);
            }
            // Cancel reaches the min cuts inside Select Subject, so Esc doesn't wait for them.
            let found = ctx
                .stage(0.05, 0.4, "Finding person", |ctl| {
                    with_doc_sampler(&doc, target, sample_all, |smp, d| {
                        let seed = crate::smartselect_cmds::subject_seed(smp, d.bounds());
                        subject::select_subject_seeded_with(smp, d.bounds(), seed.as_ref(), ctl)
                    })
                })
                .map_err(|_| EngineError::Cancelled)?;
            let region = found.ok_or_else(|| EngineError::Other("no person found".into()))?;
            let mut analysis = photocraft_portrait::Analysis::new(doc.size.width, doc.size.height);
            let person = matting::region_surface(&region);
            analysis.set(photocraft_portrait::Region::Person, person.clone());
            // The parsing stage when a model is handed in (`PHOTOCRAFT_PORTRAIT_PARSING`):
            // skin/hair/clothes masks join the person mask; without one the slice is the
            // person alone and every layer but the first stays out.
            let regions = ctx.stage(0.4, 0.7, "Parsing regions", |_| parsing_regions(&doc, target, sample_all, &person));
            for (region, mask) in regions {
                analysis.set(region, mask);
            }
            Ok(Arc::new(analysis))
        },
        move |s, analysis| {
            let plan = photocraft_portrait::plan(&recipe, &analysis);
            let group = materialize(s, &plan, &analysis)?;
            // Keyed to the post-edit document, past this edit's own cache reset (the same
            // re-point Select Subject does); an intervening edit still invalidates by pixels.
            if let Some(doc) = s.active().map(|st| Arc::clone(&st.doc)) {
                s.portrait_cache.store(&doc, (*analysis).clone());
            }
            Ok(json!({ "group": group.0 }))
        },
    )
}

/// One history step named "Smart Portrait": the plan's standard layer subtree, built directly
/// (`Layer::group` takes children; masks are assigned, no selection roundtrip). Unimplemented
/// stages plan nothing, so the match's other arms are the generative stages' landing spots.
fn materialize(s: &mut Session, plan: &photocraft_portrait::LayerPlan, analysis: &photocraft_portrait::Analysis) -> Result<photocraft_doc::LayerId> {
    s.edit("Smart Portrait", |doc, active| {
        let mut children = Vec::new();
        for step in &plan.layers {
            match step {
                photocraft_portrait::PlannedLayer::Adjustment { name, kind, params, mask } => {
                    let adjustment = crate::adjust_params::from_params(kind, params, None, doc.mode)?;
                    let mut layer = Layer::new(name.clone(), LayerContent::Adjustment(adjustment));
                    if let Some(region) = mask
                        && let Some(surface) = analysis.mask(*region)
                    {
                        layer.mask = Some(crate::commands::mask_at_depth(LayerMask { surface: surface.clone(), ..LayerMask::reveal_all() }, doc.depth)?);
                    }
                    children.push(layer);
                }
                photocraft_portrait::PlannedLayer::Group { .. } | photocraft_portrait::PlannedLayer::Raster { .. } => {}
            }
        }
        let group = Layer::group(plan.group_name.clone(), children);
        let bg = crate::background_cmds::background_id(doc);
        let id = crate::layer_place::insert_new(doc, *active, group, crate::layer_place::Place::Auto);
        crate::background_cmds::keep_background_at_bottom(bg, doc)?;
        *active = Some(id);
        Ok(id)
    })
}

/// Smart Portrait command spec (Layer › New Adjustment Layer, though what it adds is a group
/// of standard layers — the fork's entry point for the portrait pipeline).
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Smart Portrait…",
        menu: &["Layer", "New Adjustment Layer"],
        shortcut: None,
        params: r##"{"sampleAllLayers":bool=false,"groupName":text="Portrait","hue":-180..180=0,"saturation":-100..100=0,"lightness":-100..100=0,"hairHue":-180..180?,"hairSaturation":-100..100?,"hairLightness":-100..100?} → {group} (analyzes the person — and, with PHOTOCRAFT_PORTRAIT_PARSING, the skin/hair/clothes regions — then adds a named group of standard adjustment layers masked to them; any hair value opts into a Hair Color layer; one history step)"##,
        enabled,
        run,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_algo::segment::Rng;
    use photocraft_geom::Rect;

    fn session_with(w: u32, h: u32, px: impl Fn(u32, u32) -> [f32; 3]) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h})).unwrap();
        s.edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let mut data = Vec::with_capacity((w * h) as usize * n);
            for y in 0..h {
                for x in 0..w {
                    let c = px(x, y);
                    data.extend_from_slice(&c[..n.min(3)]);
                    if n == 4 {
                        data.push(1.0);
                    }
                }
            }
            bg.write_region(Rect::new(0, 0, w as i32, h as i32), &data);
            Ok(())
        })
        .unwrap();
        s
    }

    /// Textured disc (centre (100, 80), radius 45) on a noisy background — the cutout tests'
    /// stand-in for a person.
    fn disc() -> (Session, impl Fn(i32, i32) -> bool) {
        let mut rng = Rng::new(2);
        let nz: Vec<f32> = (0..200 * 160).map(|_| rng.normal()).collect();
        let inside = |x: i32, y: i32| {
            let (dx, dy) = (x as f32 + 0.5 - 100.0, y as f32 + 0.5 - 80.0);
            dx * dx + dy * dy <= 45.0 * 45.0
        };
        let s = session_with(200, 160, move |x, y| {
            let n = 0.05 * nz[(y * 200 + x) as usize];
            if inside(x as i32, y as i32) {
                let t = ((x + y) as f32 * 0.35).sin() * 0.5 + 0.5;
                [0.85 + 0.05 * t + n, 0.2 + 0.3 * t + n, 0.15 + n]
            } else {
                [0.25 + n, 0.45 + n, 0.7 + n]
            }
            .map(|v| v.clamp(0.0, 1.0))
        });
        (s, inside)
    }

    /// The one layer of the created group's only child, with its mask coverage probe.
    fn child_of_portrait_group(s: &Session) -> (&Layer, impl Fn(i32, i32) -> f32 + '_) {
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        assert_eq!(layer.name, "Portrait");
        let child = group.children.first().expect("one child");
        let cov = move |x: i32, y: i32| child.mask.as_ref().map_or(0.0, |m| m.value(x, y));
        (child, cov)
    }

    #[test]
    fn creates_a_masked_hue_group_as_one_history_step_and_undoes() {
        let (mut s, inside) = disc();
        assert!(s.is_enabled(CMD));
        let steps = s.active().unwrap().history.past_len();
        let r = s.execute(CMD, json!({})).unwrap();
        let group = s.active().unwrap().active_layer.unwrap();
        assert_eq!(r["group"], group.0);
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "one history step");
        // All reads before the undo: the child probe borrows the document.
        {
            let (child, cov) = child_of_portrait_group(&s);
            let LayerContent::Adjustment(photocraft_doc::Adjustment::HueSaturation { hue, saturation, lightness, .. }) = &child.content else {
                panic!("not a Hue/Saturation adjustment");
            };
            assert_eq!((*hue, *saturation, *lightness), (0.0, 0.0, 0.0));
            // The mask follows the person, like Remove Background's does.
            let (mut hits, mut union) = (0, 0);
            for y in 0..160 {
                for x in 0..200 {
                    let (a, b) = (cov(x, y) >= 0.5, inside(x, y));
                    hits += (a && b) as i32;
                    union += (a || b) as i32;
                }
            }
            let iou = hits as f32 / union.max(1) as f32;
            assert!(iou >= 0.75, "IoU {iou}");
        }
        s.execute("edit.undo", json!({})).unwrap();
        assert!(s.active().unwrap().doc.layer(group).is_none(), "the group survived undo");
        s.execute("edit.redo", json!({})).unwrap();
        assert!(s.active().unwrap().doc.layer(group).is_some(), "the group did not come back");
    }

    #[test]
    fn hue_params_pass_through_to_the_adjustment_layer() {
        let (mut s, _) = disc();
        s.execute(CMD, json!({"hue": 30, "saturation": -20, "lightness": 5})).unwrap();
        let (child, _) = child_of_portrait_group(&s);
        let LayerContent::Adjustment(photocraft_doc::Adjustment::HueSaturation { hue, saturation, lightness, .. }) = &child.content else {
            panic!("not a Hue/Saturation adjustment");
        };
        assert_eq!((*hue, *saturation, *lightness), (30.0, -20.0, 5.0));
    }

    #[test]
    fn a_second_run_hits_the_analysis_cache() {
        let (mut s, _) = disc();
        s.execute(CMD, json!({})).unwrap();
        let first = s.active().unwrap().active_layer.unwrap();
        s.execute(CMD, json!({})).unwrap();
        let second = s.active().unwrap().active_layer.unwrap();
        assert_ne!(first, second, "the second run adds its own group");
        // The store survived its own edit's cache reset (it runs after the edit), so the
        // session holds an entry for this document.
        assert!(s.active().is_some() && s.portrait_cache.get(&s.active().unwrap().doc, |_, _| true).is_some());
    }

    #[test]
    fn cancelled_while_finding_the_person_changes_nothing() {
        let (mut s, _) = disc();
        let before = s.active().unwrap().revision;
        let ctx = crate::jobs::JobCtx::new();
        ctx.cancel();
        s.set_inline_job_ctx(Some(ctx.clone()));
        assert!(matches!(s.execute(CMD, json!({})), Err(EngineError::Cancelled)));
        assert_eq!(ctx.message(), "Finding person");
        assert_eq!(s.active().unwrap().revision, before);
        s.set_inline_job_ctx(None);
        s.execute(CMD, json!({})).unwrap();
        assert!(s.active().unwrap().active_layer.is_some());
    }

    #[test]
    fn without_a_person_the_document_is_left_alone() {
        let mut s = session_with(32, 32, |_, _| [0.5, 0.5, 0.5]);
        let before = s.active().unwrap().revision;
        assert!(s.execute(CMD, json!({})).is_err());
        assert_eq!(s.active().unwrap().revision, before);
    }

    /// The parsing stage only exists when a model is handed in; this test runs with one
    /// and asserts the hair region flows through to a second, hair-masked layer. Without
    /// the env var it reports the skip (CI has no weights in the tree).
    #[test]
    fn parsing_regions_feed_a_hair_layer_when_a_model_is_provided() {
        if std::env::var_os("PHOTOCRAFT_PORTRAIT_PARSING").is_none() {
            eprintln!("skipping: PHOTOCRAFT_PORTRAIT_PARSING unset");
            return;
        }
        let (mut s, inside) = disc();
        s.execute(CMD, json!({"hairHue": -15.0, "hairSaturation": 40.0})).unwrap();
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        assert_eq!(group.children.len(), 2, "person + hair layers");
        let hair = &group.children[1];
        assert_eq!(hair.name, "Hair Color");
        // The hair mask sits inside the person: any hair pixel is also a disc pixel.
        let Some(mask) = &hair.mask else { panic!("hair layer unmasked") };
        let mut hair_px = 0;
        for y in 0..160 {
            for x in 0..200 {
                if mask.value(x, y) >= 0.5 {
                    hair_px += 1;
                    assert!(inside(x, y), "hair mask outside the person at ({x}, {y})");
                }
            }
        }
        // A synthetic disc has no hair: the wiring proof is that the layer exists, is
        // masked, and its mask sits inside the person (the gate above). Real hair volume
        // is the real-photo validation the spike archive records, not this test.
        assert!(hair_px > 0, "hair mask has no support at all");
    }

    #[test]
    fn no_document_is_disabled() {
        assert!(!Session::new().is_enabled(CMD));
        assert!(Session::new().execute(CMD, json!({})).is_err());
    }
}
