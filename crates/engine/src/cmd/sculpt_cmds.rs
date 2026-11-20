//! Filter › Body Sculpt: whole-body shaping from the body-pose foundation — sixteen
//! bidirectional sliders (PixCake's 全身美型 panel in this fork's terms) whose warp
//! [`photocraft_algo::sculpt`] writes analytically over Liquify's own field, so the pixel
//! work, the selection limit and the smart-filter replay are all the distortion family's.
//!
//! The panel (and `portrait.analyze`) hands the command the anchors it analysed once; with
//! them Apply builds the field and warps — no portrait pipeline on the button. A script that
//! sends none falls back to one lightweight analysis: the person mask and the pose, not the
//! full Smart Portrait stages. The anchors it used embed into the params it stores: a smart
//! filter re-renders at composite time, when no session and no analysis exist, so the stored
//! params alone must rebuild the field. A linked layer mask rides the same field — slimming
//! pixels under a stale mask is a broken silhouette (#3083's rule, Puppet Warp's precedent).

use photocraft_algo::contour;
use photocraft_algo::liquify::{LiquifyField, apply_liquify, auto_cell};
use photocraft_algo::sculpt::{SYMMETRY_PAIRS, SculptAnchors, SculptHead, SculptReport, SculptSliders, SculptSpec, sculpt_field};
use photocraft_doc::LayerContent;
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::photo_cmds::Stopwatch;
use crate::{EngineError, Result, Session};

pub const SCULPT: &str = "filter.bodySculpt";

/// The refusal a run without usable anchors answers (`portrait.analyze` reports it too).
pub(crate) const NO_PERSON: &str = "no person found — Body Sculpt needs the body-pose landmarks";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: SCULPT.into(), msg: msg.into() }
}

/// One slider out of params: absent is zero (the dialog sends all sixteen, a script the few
/// it means), a non-number is refused, the range clamps like the dialog's own slider stops.
fn slider(p: &Value, key: &str) -> Result<f32> {
    let raw = match p.get(key) {
        None => 0.0,
        Some(v) => v.as_f64().ok_or_else(|| bad(format!("`{key}` must be a number in -100..100")))?,
    };
    Ok(raw.clamp(-100.0, 100.0) as f32)
}

fn sliders(p: &Value) -> Result<SculptSliders> {
    Ok(SculptSliders {
        slim_body: slider(p, "slimBody")?,
        waist: slider(p, "waist")?,
        belly: slider(p, "belly")?,
        arms: slider(p, "arms")?,
        thighs: slider(p, "thighs")?,
        calves: slider(p, "calves")?,
        shoulders: slider(p, "shoulders")?,
        hips: slider(p, "hips")?,
        bust: slider(p, "bust")?,
        head_size: slider(p, "headSize")?,
        neck: slider(p, "neck")?,
        height: slider(p, "height")?,
        leg_length: slider(p, "legLength")?,
        waist_length: slider(p, "waistLength")?,
        torso_length: slider(p, "torsoLength")?,
        neck_length: slider(p, "neckLength")?,
        ai_sculpt: slider(p, "aiSculpt")?,
        straight_shoulders: slider(p, "straightShoulders")?,
        swan_neck: slider(p, "swanNeck")?,
        bust_lift: slider(p, "bustLift")?,
        double_chin: slider(p, "doubleChin")?,
    })
}

/// The field resolution, Liquify's own `meshSize` convention.
fn cell(p: &Value, canvas: Rect) -> Result<f64> {
    match p.get("meshSize") {
        None | Some(Value::Null) => Ok(auto_cell(canvas)),
        Some(v) => {
            let c = v.as_f64().ok_or_else(|| bad("`meshSize` is px per field node (1..64)"))?;
            if !(1.0..=64.0).contains(&c) {
                return Err(bad("`meshSize` must be 1..64"));
            }
            Ok(c)
        }
    }
}

/// The anchors as the stored params carry them (the smart-filter re-render path): sliders,
/// `anchors` and `meshSize`. `None` when the anchors are absent or malformed — a filter this
/// command never stored.
fn stored_spec(p: &Value, canvas: Rect) -> Option<(SculptSpec, f64)> {
    let anchors: SculptAnchors = serde_json::from_value(p.get("anchors")?.clone()).ok()?;
    Some((SculptSpec { sliders: sliders(p).ok()?, anchors }, cell(p, canvas).ok()?))
}

/// The params' own anchors — the panel's one analysis (`portrait.analyze`), or a script's.
/// `None` when absent: the lightweight analysis answers instead. Present but malformed is
/// refused by its shape rather than warping from garbage — the field builders index every
/// point, so nothing downstream may see a short or non-finite pose.
fn anchors_from_params(p: &Value) -> Result<Option<SculptAnchors>> {
    let Some(raw) = p.get("anchors") else { return Ok(None) };
    let anchors: SculptAnchors =
        serde_json::from_value(raw.clone()).map_err(|e| bad(format!("`anchors` must be the anchors `portrait.analyze` returns: {e}")))?;
    let finite = anchors.doc.iter().chain(&anchors.chin).chain(anchors.head.centre.iter()).all(|v| v.is_finite())
        && anchors.head.radius.is_finite()
        && anchors.pose.iter().flatten().all(|v| v.is_finite());
    if anchors.pose.len() != photocraft_algo::sculpt::POSE_COUNT || !finite {
        return Err(bad("`anchors` must carry 33 finite pose points [x, y, visibility], a finite head and chin, and the document size"));
    }
    Ok(Some(anchors))
}

/// The head and chin anchors: the face oval's own centroid and radius (grown to cover the
/// hair) and its lowest point; without a mesh run, proportions of the nose and shoulders
/// stand in — the head sliders are coarse enough that the estimate carries them.
fn head_anchors(a: &photocraft_portrait::Analysis) -> (SculptHead, [f32; 2]) {
    if a.landmarks.len() > 460 {
        let ring: Vec<[f64; 2]> = photocraft_portrait::face::FACE_OVAL.iter().map(|&i| [a.landmarks[i][0] as f64, a.landmarks[i][1] as f64]).collect();
        let n = ring.len() as f64;
        let (cx, cy) = ring.iter().fold((0.0, 0.0), |(x, y), p| (x + p[0] / n, y + p[1] / n));
        let radius = ring.iter().map(|p| (p[0] - cx).hypot(p[1] - cy)).sum::<f64>() / n * 1.15;
        let chin = a.landmarks[152];
        return (SculptHead { centre: [cx as f32, cy as f32], radius: radius as f32 }, [chin[0], chin[1]]);
    }
    let nose = a.pose.first().copied().unwrap_or_default();
    let (l, r) = (a.pose.get(11).copied().unwrap_or_default(), a.pose.get(12).copied().unwrap_or_default());
    let shoulder_mid = [(l[0] + r[0]) * 0.5, (l[1] + r[1]) * 0.5];
    let lerp = |t: f32| [nose[0] + (shoulder_mid[0] - nose[0]) * t, nose[1] + (shoulder_mid[1] - nose[1]) * t];
    let d = (shoulder_mid[0] - nose[0]).hypot(shoulder_mid[1] - nose[1]);
    (SculptHead { centre: lerp(-0.28), radius: 0.55 * d }, lerp(0.38))
}

/// The anchors of an analysis: the pose as `[x, y, visibility]`, the head and the chin.
/// `None` when the pose stage never answered — the caller's "no person" wording.
pub(crate) fn anchors_from_analysis(a: &photocraft_portrait::Analysis) -> Option<SculptAnchors> {
    if a.pose.len() != photocraft_algo::sculpt::POSE_COUNT {
        return None;
    }
    let pose = a.pose.iter().map(|q| [q[0], q[1], q[3]]).collect();
    let (head, chin) = head_anchors(a);
    Some(SculptAnchors { doc: [a.width as f32, a.height as f32], pose, head, chin })
}

/// The person mask's silhouette as a closed polyline: the left and right rims of every
/// sampled row, walked down one side and back up the other. Rows that read empty — or
/// edge-to-edge, a subject tier that cannot separate this photo's background — contribute
/// nothing, so a full-width mask yields no contour and the smooth slider says so instead of
/// smoothing the frame's box.
pub(crate) fn person_contour(mask: &Surface) -> Option<Vec<[f64; 2]>> {
    let b = mask.content_bounds();
    let w = b.x1.max(0);
    let rows: usize = b.height() as usize;
    let step = (rows / 120).max(1);
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut y = b.y0;
    while y < b.y1 {
        let (mut l, mut r) = (None, None);
        for x in 0..w {
            if mask.sample_channel(x, y, 0) > 0.5 {
                l.get_or_insert(x);
                r = Some(x);
            }
        }
        if let (Some(l), Some(r)) = (l, r)
            && r - l < w * 9 / 10
            && r - l > 4
        {
            left.push([l as f64 + 0.5, y as f64 + 0.5]);
            right.push([r as f64 + 0.5, y as f64 + 0.5]);
        }
        y += step as i32;
    }
    if left.len() < 8 {
        return None;
    }
    right.reverse();
    left.extend(right);
    Some(left)
}

/// The symmetry amount (对称液化·上半身, one-directional 0..100).
fn symmetry_of(p: &Value) -> f32 {
    p.get("symmetry").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, 100.0) as f32
}

/// The smoothing amount (身型流畅, one-directional 0..100).
fn smooth_of(p: &Value) -> f32 {
    p.get("smooth").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, 100.0) as f32
}

/// The stored contour as `[[x, y], …]` document pixels, when the params carry one.
fn contour_of(p: &Value) -> Option<Vec<[f64; 2]>> {
    serde_json::from_value::<Vec<[f64; 2]>>(p.get("contour")?.clone()).ok().filter(|c| c.len() >= 8)
}

/// Stamps the contour relaxation into a built field (the run path builds it from the
/// analysis, the replay path reads it back from the stored params — same stamp either way).
fn stamp_contour(field: &mut photocraft_algo::liquify::LiquifyField, pts: &[[f64; 2]], smooth: f32) {
    let len: f64 = pts.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum::<f64>()
        + (pts[0][0] - pts[pts.len() - 1][0]).hypot(pts[0][1] - pts[pts.len() - 1][1]);
    contour::stamp(field, pts, len / pts.len() as f64, f64::from(smooth));
}

/// The field every path builds: the slider warp, the symmetry pair alignment and the
/// contour relaxation. Apply, the smart-filter replay and the panel's live preview all call
/// it with the same values, so the field they show cannot drift from the one they commit.
fn field_with(
    values: SculptSliders,
    anchors: &SculptAnchors,
    contour: Option<&[[f64; 2]]>,
    symmetry: f32,
    smooth: f32,
    canvas: Rect,
    cell: f64,
) -> (LiquifyField, SculptReport) {
    let spec = SculptSpec { sliders: values, anchors: anchors.clone() };
    let (mut field, mut report) = sculpt_field(&spec, canvas, cell);
    // 对称液化·上半身: the paired joints align toward each other's mirror about the
    // torso axis — pure geometry of the stored joints, replayed for free.
    if symmetry > 0.0 {
        let pts: Vec<[f64; 2]> = spec.anchors.pose.iter().map(|q| [q[0] as f64, q[1] as f64]).collect();
        let shoulder_mid = [(pts[11][0] + pts[12][0]) * 0.5, (pts[11][1] + pts[12][1]) * 0.5];
        let hip_mid = [(pts[23][0] + pts[24][0]) * 0.5, (pts[23][1] + pts[24][1]) * 0.5];
        let reach = ((pts[11][0] - pts[12][0]).hypot(pts[11][1] - pts[12][1]) * 0.5).max(8.0);
        contour::align_pairs(
            &mut field,
            &pts,
            SYMMETRY_PAIRS,
            shoulder_mid,
            [hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]],
            f64::from(symmetry),
            reach,
        );
        report.applied.push("symmetry");
    }
    // 身型流畅: the silhouette's own contour, low-passed and stamped over the parts. The
    // contour rides the stored params like the anchors, so the re-render replays it with
    // no mask and no analysis.
    if smooth > 0.0 {
        match contour {
            Some(pts) => {
                stamp_contour(&mut field, pts, smooth);
                report.applied.push("smooth");
            }
            None => report.skipped.push(("smooth", "the subject mask has no contour to relax")),
        }
    }
    (field, report)
}

/// The field `params` alone describe — exactly the one Apply builds, from the anchors
/// `portrait.analyze` found and the contour it embedded. The panel's live preview calls it
/// with the params Apply will send, so what is shown is what is committed.
pub fn field_for(p: &Value, canvas: Rect) -> Result<(LiquifyField, SculptReport)> {
    let values = sliders(p)?;
    let anchors = anchors_from_params(p)?.ok_or_else(|| bad("`anchors` are required here — `portrait.analyze` supplies them"))?;
    let cell = cell(p, canvas)?;
    Ok(field_with(values, &anchors, contour_of(p).as_deref(), symmetry_of(p), smooth_of(p), canvas, cell))
}

/// The one lightweight analysis a run without anchors needs: the person mask and the
/// body-pose landmarks — not the Smart Portrait regions. Select Subject's precedent of a
/// synchronous, progress-less ML analysis (a detached job context whose progress goes
/// nowhere).
fn lightweight(s: &Session, all_layers: bool) -> Result<std::sync::Arc<photocraft_portrait::Analysis>> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let target = d.active_layer.filter(|l| !all_layers && d.doc.layer(*l).and_then(photocraft_doc::Layer::surface).is_some());
    let ctx = crate::jobs::JobCtx::new();
    crate::portrait_cmds::analyze_sculpt(&d.doc, target, all_layers, crate::portrait_cmds::SculptKind::Body, &ctx)
}

fn sculpt(s: &mut Session, p: &Value) -> Result<Value> {
    let values = sliders(p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let id = crate::commands::layer_param(s, p)?;
    let smooth = smooth_of(p);
    let symmetry = symmetry_of(p);
    if values == SculptSliders::default() && smooth <= 0.0 && symmetry <= 0.0 {
        return Ok(json!({"layer": id.0, "changed": false}));
    }
    let cell = cell(p, canvas)?;
    let all_layers = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let t0 = Stopwatch::start();
    // The anchors the panel analysed are the fast path: Apply is a field build and a warp.
    // Without them (a script, an action) one lightweight analysis answers the pose — never
    // the full Smart Portrait pipeline.
    let explicit = anchors_from_params(p)?;
    let (anchors, mut analysis) = match explicit {
        Some(anchors) => (anchors, None),
        None => {
            let analysis = lightweight(s, all_layers)?;
            let anchors = anchors_from_analysis(&analysis).ok_or_else(|| EngineError::Other(NO_PERSON.into()))?;
            (anchors, Some(analysis))
        }
    };
    // 身型流畅: the silhouette contour needs the person mask. The params' own contour (what
    // `portrait.analyze` and the smart filter store) spares that ask; without one, the
    // lightweight analysis answers it, and a photo it cannot segment only skips the pass.
    let contour = if smooth > 0.0 {
        match contour_of(p) {
            Some(pts) => Some(pts),
            None => {
                if analysis.is_none() {
                    analysis = lightweight(s, all_layers).ok();
                }
                analysis.as_ref().and_then(|a| a.mask(photocraft_portrait::Region::Person)).and_then(person_contour)
            }
        }
    } else {
        None
    };
    let (field, report) = field_with(values, &anchors, contour.as_deref(), symmetry, smooth, canvas, cell);
    let smart = matches!(s.active().and_then(|d| d.doc.layer(id)).map(|l| &l.content), Some(LayerContent::Smart(_)));
    if field.is_identity() && !smart {
        return Ok(json!({"layer": id.0, "changed": false, "parts": parts(&report)}));
    }
    // What a smart filter stores: the sliders the dialog sent plus the anchors they were
    // resolved against, so the re-render rebuilds this exact field with no analysis.
    let mut stored = p.clone();
    if let Value::Object(m) = &mut stored {
        m.insert("anchors".into(), serde_json::to_value(&anchors).map_err(|e| bad(format!("the anchors would not serialise: {e}")))?);
        if let Some(pts) = &contour {
            m.insert("contour".into(), serde_json::to_value(pts).map_err(|e| bad(format!("the contour would not serialise: {e}")))?);
        }
    }
    let r = crate::distort_cmds::run_on_layer(
        s,
        SCULPT,
        "Body Sculpt",
        &stored,
        false,
        &|surf, sel, _| {
            let out = apply_liquify(surf, &field);
            Ok(match sel {
                Some(sel) => crate::distort_cmds::mix_by_selection(surf, &out, sel)?,
                None => out,
            })
        },
        // A linked mask deforms on the same field (#3083): a slimmed body under an unslimmed
        // mask is a broken silhouette.
        Some(&|mask, _| Ok(apply_liquify(mask, &field))),
    )?;
    let mut r = r;
    r["ms"] = json!(t0.ms());
    r["maxDisplacement"] = json!(field.max_displacement());
    r["parts"] = parts(&report);
    Ok(r)
}

fn parts(report: &photocraft_algo::sculpt::SculptReport) -> Value {
    json!({
        "applied": report.applied,
        "skipped": report.skipped.iter().map(|(name, why)| json!({"part": name, "why": why})).collect::<Vec<_>>(),
    })
}

/// Smart-filter hook: re-applies a stored Body Sculpt to a rendered smart object.
pub(crate) fn apply_to_surface(params: &Value, surf: &Surface, canvas: Rect) -> Option<Surface> {
    let (spec, cell) = stored_spec(params, canvas)?;
    let (field, _) = field_with(spec.sliders, &spec.anchors, contour_of(params).as_deref(), symmetry_of(params), smooth_of(params), canvas, cell);
    Some(apply_liquify(surf, &field))
}

pub fn specs() -> Vec<CommandSpec> {
    const P: &str = r##"{"slimBody":-100..100=0,"waist":-100..100=0,"belly":-100..100=0,"arms":-100..100=0,"thighs":-100..100=0,"calves":-100..100=0,"shoulders":-100..100=0,"hips":-100..100=0,"bust":-100..100=0,"headSize":-100..100=0,"neck":-100..100=0,"height":-100..100=0,"legLength":-100..100=0,"waistLength":-100..100=0,"torsoLength":-100..100=0,"neckLength":-100..100=0,"aiSculpt":-100..100=0 (composition: a gentle whole-body slim layered over the shaping sliders),"swanNeck":-100..100=0 (composition: slim + lengthen + square the shoulders),"straightShoulders":-100..100=0,"bustLift":-100..100=0,"doubleChin":-100..100=0,"symmetry":0..100=0 (对称液化·上半身: the paired joints align toward each other's mirror about the torso axis),"smooth":0..100=0 (身型流畅: the silhouette contour, low-passed and relaxed — bumps like a trapezius slope or an arm bulge flatten to their base; the contour is embedded from the person mask at apply time, like the anchors) (each slider's positive direction is its label's action — Waist narrows, Hips rounds, Leg Length lengthens; with `anchors` the analysis never runs on Apply),"anchors":{…}? (`portrait.analyze`'s document-pixel anchors — the panel supplies them so Apply is a field build and a warp; without them one lightweight pose analysis runs. Stored with the smart filter so re-renders replay exactly; never typed),"meshSize":px? (field resolution, px per node),"sampleAllLayers":bool=false,"layer":id?} — a selection limits the effect; on a smart object it becomes a smart filter"##;
    vec![CommandSpec {
        id: SCULPT,
        label: "Body Sculpt…",
        menu: &["Filter"],
        shortcut: None,
        params: P,
        enabled: crate::filters::has_filterable_layer,
        journal: true,
        run: sculpt,
    }]
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::Session;

    /// A checkerboard the warp's resampling visibly changes (a flat fill would not).
    fn patterned(w: u32, h: u32) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h})).unwrap();
        s.edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let mut data = Vec::with_capacity((w * h) as usize * n);
            for y in 0..h {
                for x in 0..w {
                    let t = ((x + y) % 2) as f32;
                    data.extend_from_slice(&[t, 1.0 - t, 0.5][..n.min(3)]);
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

    /// A full 33-joint pose on a `w × h` document: shoulders, hips and ankles bracketing the
    /// torso the shaping sliders move.
    fn anchors_on(w: f32, h: f32) -> SculptAnchors {
        let mut pose = vec![[0.0f32; 3]; photocraft_algo::sculpt::POSE_COUNT];
        let (sx, sy) = (w / 512.0, h / 768.0);
        for &(i, x, y) in
            &[(0usize, 256.0, 92.0), (11, 204.0, 180.0), (12, 308.0, 180.0), (23, 222.0, 400.0), (24, 292.0, 400.0), (27, 230.0, 680.0), (28, 284.0, 680.0)]
        {
            pose[i] = [x * sx, y * sy, 1.0];
        }
        SculptAnchors { doc: [w, h], pose, head: SculptHead { centre: [w * 0.5, 88.0 * sy], radius: 52.0 * sx }, chin: [w * 0.5, 138.0 * sy] }
    }

    fn pixels(s: &Session, w: i32, h: i32) -> Vec<f32> {
        s.active().unwrap().doc.layers[0].surface().unwrap().read_region(Rect::new(0, 0, w, h))
    }

    /// The panel's fast path: the params carry `portrait.analyze`'s anchors, so Apply builds
    /// the field and warps — the fallback analysis, which would fail on this document, never
    /// runs.
    #[test]
    fn explicit_anchors_apply_without_a_person_analysis() {
        let mut s = patterned(256, 384);
        let before = pixels(&s, 256, 384);
        let steps = s.active().unwrap().history.past_len();
        let anchors = anchors_on(256.0, 384.0);
        let r = s.execute(SCULPT, json!({"waist": -60.0, "anchors": serde_json::to_value(&anchors).unwrap()})).unwrap();
        assert_eq!(r["changed"], json!(true));
        assert!(r["maxDisplacement"].as_f64().unwrap_or_default() > 0.0);
        assert_ne!(before, pixels(&s, 256, 384), "the warp must move pixels");
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "one history step");
    }

    /// Malformed anchors are refused by their shape, not warped from garbage: the field
    /// builders index every joint, so a short pose must never reach them.
    #[test]
    fn malformed_anchors_are_refused_not_analysed() {
        let mut s = patterned(64, 64);
        let steps = s.active().unwrap().history.past_len();
        let mut anchors = serde_json::to_value(anchors_on(64.0, 64.0)).unwrap();
        anchors["pose"] = serde_json::Value::Array(vec![json!([0.0, 0.0, 1.0]); 3]);
        let err = s.execute(SCULPT, json!({"waist": 40.0, "anchors": anchors})).unwrap_err().to_string();
        assert!(err.contains("33"), "{err}");
        assert_eq!(s.active().unwrap().history.past_len(), steps, "a refused run edits nothing");
    }

    #[test]
    fn zero_sliders_change_nothing_without_an_analysis() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 96, "height": 128, "background": "white"})).unwrap();
        let r = s.execute(SCULPT, json!({})).unwrap();
        assert_eq!(r, json!({"layer": s.active().unwrap().active_layer.unwrap().0, "changed": false}));
    }

    #[test]
    fn a_personless_photo_reports_no_person() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 96, "height": 128, "background": "white"})).unwrap();
        let err = s.execute(SCULPT, json!({"waist": 40})).unwrap_err().to_string();
        assert!(err.contains("no person"), "{err}");
        // A failed run leaves no history step and no edit.
        assert!(s.execute("edit.undo", json!({})).is_err());
    }

    #[test]
    fn sliders_clamp_and_bad_ones_are_refused() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32, "background": "white"})).unwrap();
        // 250 clamps to 100 (a number in range is accepted), which then asks for a person
        // the plain image cannot give — the clamp passed, the analysis answered.
        let err = s.execute(SCULPT, json!({"waist": 250.0})).unwrap_err().to_string();
        assert!(err.contains("no person"), "{err}");
        assert!(s.execute(SCULPT, json!({"waist": "slim"})).is_err());
    }

    #[test]
    fn stored_params_replay_the_same_field() {
        // A stored filter's params, anchors included, build one field — the smart filter's
        // re-render is a pure function of them, with no session or analysis to ask.
        let mut pose = vec![[0.0f32; 3]; photocraft_algo::sculpt::POSE_COUNT];
        for &(i, x, y) in
            &[(0usize, 256.0, 92.0), (11, 204.0, 180.0), (12, 308.0, 180.0), (23, 222.0, 400.0), (24, 292.0, 400.0), (27, 230.0, 680.0), (28, 284.0, 680.0)]
        {
            pose[i] = [x as f32, y as f32, 1.0];
        }
        let anchors = SculptAnchors { doc: [512.0, 768.0], pose, head: SculptHead { centre: [256.0, 88.0], radius: 52.0 }, chin: [256.0, 138.0] };
        let params = json!({"waist": -40.0, "legLength": 50.0, "anchors": serde_json::to_value(&anchors).unwrap(), "meshSize": 4});
        let bounds = Rect::new(0, 0, 512, 768);
        let (spec, _) = stored_spec(&params, bounds).unwrap();
        assert_eq!(spec.sliders.waist, -40.0);
        assert_eq!(spec.anchors, anchors, "the anchors survive the params round-trip");
        let f1 = sculpt_field(&spec, bounds, 4.0).0;
        let f2 = sculpt_field(&stored_spec(&params, bounds).unwrap().0, bounds, 4.0).0;
        assert!(!f1.is_identity());
        assert_eq!(f1, f2, "the same params must rebuild the same field");
        // The live preview and Apply share this builder, so all three must agree.
        assert_eq!(f1, field_for(&params, bounds).unwrap().0, "field_for and the stored replay must build the same field");
        // Without anchors it is not this command's stored form.
        assert!(stored_spec(&json!({"waist": -40.0}), bounds).is_none());
    }

    /// The mask row's covered extent, `(first x, last x)` above half coverage — `None` when
    /// the row reads full-width or nearly empty: a subject matte that cannot separate this
    /// photo's background (the u2netp tier's known wall-on-white failure; the ort tier
    /// exists for it) has no rim to measure, and the gauge says so instead of lying.
    fn row_rim(mask: &Surface, y: i32, w: i32) -> Option<(i32, i32)> {
        let mut first: Option<i32> = None;
        let mut last = 0i32;
        for x in 0..w {
            if mask.sample_channel(x, y, 0) > 0.5 {
                first.get_or_insert(x);
                last = x;
            }
        }
        let l = first?;
        ((last - l) >= w / 10 && (last - l) < w * 9 / 10).then_some((l, last))
    }

    /// The narrowing between two rims, when both rows have one.
    fn narrowing(before: Option<(i32, i32)>, after: Option<(i32, i32)>) -> Option<i32> {
        Some(after?.1 - after?.0 - (before?.1 - before?.0))
    }

    /// The fraction of a horizontal band whose pixels moved by more than a visible step
    /// between two surfaces — the gauge that needs no mask: the warp's own footprint.
    fn band_changed(before: &Surface, after: &Surface, y0: i32, y1: i32, w: i32) -> f64 {
        let region = Rect::new(0, y0, w, y1.max(y0 + 1));
        let (a, b) = (before.read_region(region), after.read_region(region));
        let n = before.format().channels().min(3);
        let step = 4.0 / 255.0;
        let ch = a.chunks_exact(n).zip(b.chunks_exact(n)).filter(|(x, y)| (0..n).any(|c| (x[c] - y[c]).abs() > step)).count();
        ch as f64 / a.len() as f64 * n as f64
    }

    /// The real-photo acceptance and calibration instrument: `PHOTOCRAFT_SCULPT_PHOTO=<image>`
    /// imports it, analyses the person exactly as the command does and prints the per-part
    /// table for every slider alone at +100 (applied or skipped and why, the field's reach) —
    /// then measures what a warp does to the Person mask's own silhouette: the waist row
    /// narrows with the waist slider and roughly in proportion, the head row narrows with
    /// head size, and a smart-object run stores the anchors and replays bit-identically
    /// through the real re-render path. The flattened result exports for eyeballing
    /// (`PHOTOCRAFT_SCULPT_OUT`, default `/tmp/sculpt_out.png`). Skipped without the env
    /// var, like the portrait acceptance: CI has neither the weights nor the photos.
    #[test]
    fn real_photo_sculpt_diagnostic() {
        let Some(photo) = std::env::var_os("PHOTOCRAFT_SCULPT_PHOTO") else {
            eprintln!("skipping: PHOTOCRAFT_SCULPT_PHOTO unset");
            return;
        };
        let run = || -> Result<()> {
            let mut s = Session::new();
            let name = std::path::Path::new(&photo).file_name().and_then(|n| n.to_str()).unwrap_or("photo").to_string();
            let imported =
                photocraft_io::import_file(&name, std::path::Path::new(&photo), &photocraft_io::ImportOptions::default(), &photocraft_raster::Interrupt::NONE)
                    .map_err(|e| EngineError::Other(format!("import: {e}")))?;
            s.open_document(imported.document, None);
            let (canvas, analysis, photo) = {
                let d = s.active().ok_or(EngineError::NoDocument)?;
                let target = d.active_layer.filter(|l| d.doc.layer(*l).and_then(photocraft_doc::Layer::surface).is_some());
                let ctx = crate::jobs::JobCtx::new();
                let analysis = crate::portrait_cmds::analyze_person(&d.doc, target, false, None, &ctx)?;
                let photo =
                    target.and_then(|l| d.doc.layer(l)).and_then(|l| l.surface()).cloned().ok_or_else(|| EngineError::Other("no pixel layer".into()))?;
                (d.doc.bounds(), analysis, photo)
            };
            let anchors = anchors_from_analysis(&analysis).ok_or_else(|| EngineError::Other("no pose landmarks in the photo".into()))?;
            let person = analysis.mask(photocraft_portrait::Region::Person).ok_or_else(|| EngineError::Other("no person mask".into()))?.clone();
            let cell = auto_cell(canvas);

            // The calibration table: every slider alone at +100.
            type Single = (&'static str, fn(f32) -> SculptSliders);
            let singles: Vec<Single> = vec![
                ("slimBody", |v| SculptSliders { slim_body: v, ..SculptSliders::default() }),
                ("waist", |v| SculptSliders { waist: v, ..SculptSliders::default() }),
                ("belly", |v| SculptSliders { belly: v, ..SculptSliders::default() }),
                ("arms", |v| SculptSliders { arms: v, ..SculptSliders::default() }),
                ("thighs", |v| SculptSliders { thighs: v, ..SculptSliders::default() }),
                ("calves", |v| SculptSliders { calves: v, ..SculptSliders::default() }),
                ("shoulders", |v| SculptSliders { shoulders: v, ..SculptSliders::default() }),
                ("hips", |v| SculptSliders { hips: v, ..SculptSliders::default() }),
                ("bust", |v| SculptSliders { bust: v, ..SculptSliders::default() }),
                ("headSize", |v| SculptSliders { head_size: v, ..SculptSliders::default() }),
                ("neck", |v| SculptSliders { neck: v, ..SculptSliders::default() }),
                ("height", |v| SculptSliders { height: v, ..SculptSliders::default() }),
                ("legLength", |v| SculptSliders { leg_length: v, ..SculptSliders::default() }),
                ("waistLength", |v| SculptSliders { waist_length: v, ..SculptSliders::default() }),
                ("torsoLength", |v| SculptSliders { torso_length: v, ..SculptSliders::default() }),
                ("neckLength", |v| SculptSliders { neck_length: v, ..SculptSliders::default() }),
                ("aiSculpt", |v| SculptSliders { ai_sculpt: v, ..SculptSliders::default() }),
                ("straightShoulders", |v| SculptSliders { straight_shoulders: v, ..SculptSliders::default() }),
                ("swanNeck", |v| SculptSliders { swan_neck: v, ..SculptSliders::default() }),
                ("bustLift", |v| SculptSliders { bust_lift: v, ..SculptSliders::default() }),
                ("doubleChin", |v| SculptSliders { double_chin: v, ..SculptSliders::default() }),
            ];
            eprintln!("slider        rows applied/skipped, max displacement (px)");
            for (name, make) in &singles {
                let (field, report) = sculpt_field(&SculptSpec { sliders: make(100.0), anchors: anchors.clone() }, canvas, cell);
                let skipped = report.skipped.iter().map(|(n, w)| format!("{n}: {w}")).collect::<Vec<_>>().join("; ");
                eprintln!("{name:12}  {:>2} applied, {skipped}, max {:.1}px", report.applied.len(), field.max_displacement());
            }

            // The silhouette gauge: the waist row narrows with the slider, in proportion;
            // the head row answers head size. Both only where the mask has a rim to read.
            let mid_y = |a: usize, b: usize| ((anchors.pose[a][1] + anchors.pose[b][1]) * 0.5) as i32;
            let waist_y = (mid_y(11, 12) + mid_y(23, 24)) / 2;
            let head_y = anchors.head.centre[1] as i32;
            let w = canvas.width() as i32;
            let rim = |m: &Surface, y: i32| row_rim(m, y, w);
            let waist0 = rim(&person, waist_y);
            let mut rims = vec![waist0];
            for v in [25.0f32, 50.0, 100.0] {
                let (field, _) =
                    sculpt_field(&SculptSpec { sliders: SculptSliders { waist: v, ..SculptSliders::default() }, anchors: anchors.clone() }, canvas, cell);
                rims.push(rim(&apply_liquify(&person, &field), waist_y));
            }
            eprintln!("waist row {waist_y}: rims at 0/25/50/100 = {rims:?}");
            if let Some(step100) = narrowing(rims[0], rims[3]).filter(|_| rims[0].is_some()) {
                assert!(step100 < 0, "waist at 100 must narrow the rim: {rims:?}");
                assert!(step100 <= -2, "waist at 100 must narrow measurably: {rims:?}");
                assert!(narrowing(rims[0], rims[1]).is_none_or(|s| s >= step100), "narrowing must grow with the slider: {rims:?}");
            } else {
                eprintln!("waist row reads without a rim — this photo's subject mask cannot gauge the waist; the field itself is covered by the algo tests");
            }
            let (head_field, _) =
                sculpt_field(&SculptSpec { sliders: SculptSliders { head_size: 100.0, ..SculptSliders::default() }, anchors: anchors.clone() }, canvas, cell);
            let head1 = rim(&apply_liquify(&person, &head_field), head_y);
            let head0 = rim(&person, head_y);
            eprintln!("head row {head_y}: {head0:?} → {head1:?}");
            if let Some(step) = narrowing(head0, head1).filter(|_| head0.is_some()) {
                assert!(step < 0, "head size must shrink the head row: {step}");
            } else {
                eprintln!("head row reads without a rim; skipping the head gauge");
            }

            // The footprint gauge, mask-free: waist at 100 moves pixels in the waist band
            // and only there — the head and legs bands must stay put.
            let (sh_y, hip_y) = (mid_y(11, 12) as f32, mid_y(23, 24) as f32);
            let at = |t: f32| (sh_y + (hip_y - sh_y) * t) as i32;
            let (waist_field, _) =
                sculpt_field(&SculptSpec { sliders: SculptSliders { waist: 100.0, ..SculptSliders::default() }, anchors: anchors.clone() }, canvas, cell);
            let warped = apply_liquify(&photo, &waist_field);
            let (head_band, waist_band, legs_band) = (
                band_changed(&photo, &warped, head_y - anchors.head.radius as i32, head_y + anchors.head.radius as i32, w),
                band_changed(&photo, &warped, at(0.30), at(0.78), w),
                band_changed(&photo, &warped, (hip_y + 40.0) as i32, canvas.height() as i32, w),
            );
            eprintln!("waist-100 footprint: head {head_band:.3}, waist {waist_band:.3}, legs {legs_band:.3}");
            assert!(waist_band > 0.01, "waist at 100 must move the waist band: {waist_band:.3}");
            assert!(
                head_band < waist_band / 4.0 && legs_band < waist_band / 4.0,
                "the warp must stay in its band: head {head_band:.3} legs {legs_band:.3} vs waist {waist_band:.3}"
            );

            // A smart-object run stores the anchors and replays through the real path.
            s.execute("select.all", json!({}))?;
            s.execute("layer.smartObjects.convertToSmartObject", json!({}))?;
            let r = s.execute(SCULPT, json!({"waist": -40.0}))?;
            assert_eq!(r["changed"], json!(true), "{r}");
            let (params, source) = {
                let d = s.active().ok_or(EngineError::NoDocument)?;
                let id = d.active_layer.ok_or(EngineError::Other("no active layer".into()))?;
                let photocraft_doc::LayerContent::Smart(sm) = &d.doc.layer(id).ok_or(EngineError::NoLayer(id))?.content else { panic!("just converted") };
                assert_eq!(sm.smart_filters.len(), 1);
                (sm.smart_filters[0].params.clone(), sm.cache.clone().ok_or_else(|| EngineError::Other("no rendered source".into()))?)
            };
            assert!(params.get("anchors").is_some(), "the anchors must ride the stored params");
            let a = crate::filters::apply_filter_to_surface(SCULPT, &params, &source, source.content_bounds(), None, canvas)?
                .ok_or_else(|| EngineError::Other("the re-render path refused the stored filter".into()))?;
            let b = crate::filters::apply_filter_to_surface(SCULPT, &params, &source, source.content_bounds(), None, canvas)?
                .ok_or_else(|| EngineError::Other("the re-render path refused the stored filter".into()))?;
            assert_eq!(a.read_region(a.content_bounds()), b.read_region(b.content_bounds()), "re-render must be bit-identical");

            // Flatten and export for the eyeball pass.
            s.execute("layer.flattenImage", json!({}))?;
            let out = std::env::var_os("PHOTOCRAFT_SCULPT_OUT").map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "/tmp/sculpt_out.png".into());
            s.execute("layer.quickExportAsPng", json!({"path": out}))?;
            eprintln!("exported: {out}");
            Ok(())
        };
        if let Err(e) = run() {
            panic!("sculpt diagnostic failed: {e}");
        }
    }
}
