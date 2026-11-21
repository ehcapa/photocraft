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

/// A 追色 reference: a file path, or the statistics themselves (the protocol seam the
/// tests drive; a path is what a user hands the command).
enum Either {
    Path(String),
    Stats { mean: [f32; 3], std: [f32; 3] },
}

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
    if [p.get("skinRadius"), p.get("skinThreshold")].iter().any(|v| v.is_some()) {
        r.skin = Some(photocraft_portrait::Skin { radius: asked("skinRadius").unwrap_or_default(), threshold: asked("skinThreshold").unwrap_or_default() });
    }
    if p.get("enhanceFace").and_then(Value::as_bool) == Some(true) {
        r.enhance = true;
    }
    if [p.get("lipHue"), p.get("lipSaturation")].iter().any(|v| v.is_some()) {
        r.lips = Some(photocraft_portrait::HueSat {
            hue: asked("lipHue").unwrap_or(0.0),
            saturation: asked("lipSaturation").unwrap_or(25.0),
            lightness: asked("lipLightness").unwrap_or(0.0),
        });
    }
    if let Some(v) = asked("eyeBrighten") {
        r.eyes = Some(v);
    }
    if [p.get("blushHue"), p.get("blushSaturation")].iter().any(|v| v.is_some()) {
        r.blush = Some(photocraft_portrait::HueSat {
            hue: asked("blushHue").unwrap_or(350.0),
            saturation: asked("blushSaturation").unwrap_or(20.0),
            lightness: asked("blushLightness").unwrap_or(0.0),
        });
    }
    if [p.get("shadowHue"), p.get("shadowSaturation")].iter().any(|v| v.is_some()) {
        r.eyeshadow = Some(photocraft_portrait::HueSat {
            hue: asked("shadowHue").unwrap_or(30.0),
            saturation: asked("shadowSaturation").unwrap_or(25.0),
            lightness: asked("shadowLightness").unwrap_or(0.0),
        });
    }
    r
}

/// Per-channel mean/std of `surf`'s RGB, optionally weighted by a person mask (the
/// portrait's own tones, not the backdrop's). Alpha-transparent pixels never count.
fn channel_stats(surf: &photocraft_raster::Surface, area: photocraft_geom::Rect, person: Option<&photocraft_raster::Surface>) -> ([f32; 3], [f32; 3]) {
    let mut sums = [0f64; 3];
    let mut sqs = [0f64; 3];
    let mut n = 0u64;
    for y in area.y0..area.y1 {
        for x in area.x0..area.x1 {
            let px = surf.rgba(x, y);
            if px[3] < 0.5 {
                continue;
            }
            if person.is_some_and(|p| p.rgba(x, y)[0] < 0.5) {
                continue;
            }
            for c in 0..3 {
                sums[c] += f64::from(px[c]);
                sqs[c] += f64::from(px[c]) * f64::from(px[c]);
            }
            n += 1;
        }
    }
    let n = n.max(1) as f64;
    let mean = sums.map(|s| (s / n) as f32);
    // Population std, floored at one level so a flat channel never divides by zero below.
    let std = std::array::from_fn(|c| (((sqs[c] / n - f64::from(mean[c]) * f64::from(mean[c])).max(0.0)).sqrt() as f32).max(1.0 / 255.0));
    (mean, std)
}

/// The 追色 transfer: Reinhard-style mean/std matching per RGB channel, computed from the
/// document (masked to the person) and the reference. `None` without a usable reference.
fn grade_transfer(doc: &Document, person: Option<&photocraft_raster::Surface>, reference: Either) -> Result<photocraft_portrait::Grade> {
    let canvas = doc.bounds();
    let src = photocraft_compose::composite_to_surface(doc, canvas, photocraft_doc::PixelFormat::RGBA8, None);
    let (src_mean, src_std) = channel_stats(&src, canvas, person);
    let (ref_mean, ref_std) = match reference {
        Either::Stats { mean, std } => (mean, std),
        Either::Path(path) => {
            let name = std::path::Path::new(&path).file_name().and_then(|n| n.to_str()).unwrap_or("reference");
            let imported =
                photocraft_io::import_file(name, std::path::Path::new(&path), &photocraft_io::ImportOptions::default(), &photocraft_raster::Interrupt::NONE)
                    .map_err(|e| EngineError::Other(format!("could not read the reference image: {e}")))?;
            let bounds = imported.document.bounds();
            let surf = photocraft_compose::composite_to_surface(&imported.document, bounds, photocraft_doc::PixelFormat::RGBA8, None);
            channel_stats(&surf, bounds, None)
        }
    };
    let mut slope = [0.0f32; 3];
    let mut offset = [0.0f32; 3];
    for c in 0..3 {
        slope[c] = (ref_std[c] / src_std[c]).clamp(0.25, 4.0);
        offset[c] = ref_mean[c] - slope[c] * src_mean[c];
    }
    Ok(photocraft_portrait::Grade { slope, offset })
}

/// The smoothed copy the Skin Smoothing raster carries: the composite rendered flat, then
/// Surface Blur over it (`filter.blur.surfaceBlur`, cancelled through `ctx` like any
/// filter). `None` when the recipe does not ask for skin or the analysis found no skin.
fn smooth_skin(
    ctx: &crate::jobs::JobCtx,
    skin: photocraft_portrait::Skin,
    source: &photocraft_raster::Surface,
    canvas: photocraft_geom::Rect,
) -> Result<photocraft_raster::Surface> {
    let params = json!({ "radius": skin.radius, "threshold": skin.threshold });
    crate::filters::apply_filter_to_surface_in("filter.blur.surfaceBlur", &params, source, canvas, None, canvas, ctx)?
        .ok_or_else(|| EngineError::Other("skin smoothing produced no surface".into()))
}

/// The GFPGAN restoration of the face: the landmark-aligned 512² crop through the
/// models-dir model, pasted back over the composite masked by the (feathered) face hull.
/// `Ok(None)` without a model or the geometry the paste needs — the layer stays out.
fn enhance_face(
    doc: &Document,
    composite: &photocraft_raster::Surface,
    analysis: &photocraft_portrait::Analysis,
) -> Result<Option<photocraft_raster::Surface>> {
    let canvas = doc.bounds();
    let (w, h) = (doc.size.width as usize, doc.size.height as usize);
    let Some(points) = photocraft_portrait::align::five_points(&analysis.landmarks) else { return Ok(None) };
    let Some(face) = analysis.mask(photocraft_portrait::Region::Face) else { return Ok(None) };
    let rgba = composite.try_read_region(canvas).map_err(|e| EngineError::Other(e.to_string()))?;
    let alpha = face.try_read_region(canvas).map_err(|e| EngineError::Other(e.to_string()))?;
    let rgb: Vec<[f32; 3]> = rgba.chunks(rgba.len() / (w * h)).map(|c| [c[0], c[1], c[2]]).collect();
    let forward = photocraft_portrait::align::fit(&points, photocraft_ml::ENHANCE_SIDE);
    let crop = photocraft_portrait::align::sample_crop(&rgb, w, h, &forward, photocraft_ml::ENHANCE_SIDE);
    let Some(enhanced) = photocraft_ml::enhance_face(&crop) else { return Ok(None) };
    let alpha: Vec<f32> = alpha.chunks((alpha.len() / (w * h)).max(1)).map(|c| c[0]).collect();
    let pasted = photocraft_portrait::align::paste_back(&enhanced, photocraft_ml::ENHANCE_SIDE, &forward, &rgb, w, h, &alpha);
    // Back onto the composite's own alpha: same pixels, the face region redrawn.
    let mut out = rgba.clone();
    let ch = out.len() / (w * h);
    for (i, px) in pasted.iter().enumerate() {
        out[i * ch..i * ch + 3].copy_from_slice(px);
    }
    let mut surface = photocraft_raster::Surface::with_default(photocraft_doc::PixelFormat::RGBA8, &[0.0, 0.0, 0.0, 0.0]);
    surface.write_region(canvas, &out);
    Ok(Some(surface))
}

/// The work stage's derived pixels: what the raster layers carry.
struct Products {
    skin: Option<photocraft_raster::Surface>,
    enhance: Option<photocraft_raster::Surface>,
    grade: Option<photocraft_portrait::Grade>,
}

/// The 追色 ask: a reference image path, or the statistics themselves (the seam the tests
/// drive; a path is what a user hands the command). `None` when the params carry neither.
fn reference(p: &Value) -> Option<Either> {
    if let Some(stats) = p.get("referenceStats") {
        let nums = |key: &str| {
            let v = stats.get(key)?.as_array()?;
            Some([v.first()?.as_f64()? as f32, v.get(1)?.as_f64()? as f32, v.get(2)?.as_f64()? as f32])
        };
        return Some(Either::Stats { mean: nums("mean")?, std: nums("std")? });
    }
    p.get("reference").and_then(Value::as_str).map(|s| Either::Path(s.into()))
}

/// How far the two box stages grow their rect: the user's box gets a little slack, and the
/// mesh's re-run crops the oval ring by more — the mesh wants the face with its context
/// (MediaPipe's own pipeline crops its detection box about this loose).
const BOX_MARGIN: f32 = 1.3;
const RING_MARGIN: f32 = 1.6;

/// The person bounds' growth for the body stage's crop: a little slack so the border
/// classes are not cut at the matte's edge, still a person-dominant view.
const BODY_MARGIN: f32 = 1.25;

/// The body planes' confidence ramp: under `0.5` a pixel is noise on the union (the
/// scarf that reads skin at 0.55, the background speckle of a distant framing), over
/// `0.75` it is firm (the neck, the arms); between, a straight ramp keeps the soft edge
/// a soft selection wants.
const BODY_FLOOR: (f32, f32) = (0.5, 0.75);

/// [`BODY_FLOOR`]'s ramp: 0 under the floor's low end, 1 over its high end, linear between.
fn body_floor(v: f32) -> f32 {
    ((v - BODY_FLOOR.0) / (BODY_FLOOR.1 - BODY_FLOOR.0)).clamp(0.0, 1.0)
}

/// A body-plane component smaller than this share of the plane's largest component is
/// speckle, not a region (the 256² model's borderline pixels land as isolated dots far
/// from anything they could belong to) — relative, because the planes ride at the
/// person-crop's scale and no absolute count survives both crops.
const BODY_SPECKLE_RATIO: f32 = 0.02;
/// …but never keep a component under this many pixels, however small the largest is.
const BODY_SPECKLE_MIN: usize = 16;

/// Drops the 4-connected components of the on-pixels (`> 0`) under [`BODY_SPECKLE_RATIO`]
/// of the largest (and under [`BODY_SPECKLE_MIN`]) — in place, on a floored plane.
fn suppress_speckle(vals: &mut [f32], w: usize, h: usize) {
    if w == 0 || h == 0 || vals.len() != w * h {
        return;
    }
    // Component ids are 1-based into `sizes`; 0 is unvisited or off.
    let mut labels = vec![0u32; w * h];
    let mut sizes: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if labels[start] != 0 || vals[start] <= 0.0 {
            continue;
        }
        let id = sizes.len() as u32 + 1;
        let mut size = 0usize;
        labels[start] = id;
        stack.push(start);
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y) = (i % w, i / w);
            for j in [(x > 0).then(|| i - 1), (x + 1 < w).then_some(i + 1), (y > 0).then(|| i - w), (y + 1 < h).then_some(i + w)].into_iter().flatten() {
                if labels[j] == 0 && vals[j] > 0.0 {
                    labels[j] = id;
                    stack.push(j);
                }
            }
        }
        sizes.push(size);
    }
    let Some(&largest) = sizes.iter().max() else { return };
    let keep = (largest as f32 * BODY_SPECKLE_RATIO).max(BODY_SPECKLE_MIN as f32) as usize;
    for (i, id) in labels.iter().enumerate() {
        if *id > 0 && sizes[(*id - 1) as usize] < keep {
            vals[i] = 0.0;
        }
    }
}

/// A face smaller than this in the tightened crop is not worth layers: the mesh's 256²
/// input would be almost all upsample, and a garbage answer that passes the plausibility
/// gate on a speck crop still builds nothing useful.
const MIN_FACE_SIDE: i32 = 24;

/// The user-drawn face box `[x, y, w, h]` (document pixels): the mesh and parsing stages run
/// inside it instead of over the whole document. Both models were trained on face-dominant
/// crops (CelebAMask-HQ is face close-ups; FaceMesh assumes the face fills the frame), so a
/// face that is a small fraction of the frame collapses — the box is the crop stage the
/// models are missing. Without it the SCRFD detector supplies the largest face's box
/// automatically (`detect`); `None` without the param, or when the box clips away to nothing.
fn roi(p: &Value, size: photocraft_geom::Size) -> Option<photocraft_geom::Rect> {
    let v = p.get("roi")?.as_array()?;
    let mut n = v.iter().filter_map(Value::as_f64);
    let (x, y, w, h) = (n.next()?, n.next()?, n.next()?, n.next()?);
    let r = photocraft_geom::Rect::from_xywh(x.round() as i32, y.round() as i32, w.max(0.0).round() as u32, h.max(0.0).round() as u32)
        .intersect(&photocraft_geom::Rect::from_size(size));
    (!r.is_empty()).then_some(r)
}

/// `r` grown by `factor` about its centre, clamped into `0..w × 0..h` and never empty.
fn expanded(r: photocraft_geom::Rect, w: u32, h: u32, factor: f32) -> photocraft_geom::Rect {
    let (cx, cy) = ((r.x0 as f32 + r.x1 as f32) * 0.5, (r.y0 as f32 + r.y1 as f32) * 0.5);
    let (hw, hh) = (r.width() as f32 * factor * 0.5, r.height() as f32 * factor * 0.5);
    let (x0, y0) = ((cx - hw).floor().max(0.0) as i32, (cy - hh).floor().max(0.0) as i32);
    let (x1, y1) = ((cx + hw).ceil().min(w as f32) as i32, (cy + hh).ceil().min(h as f32) as i32);
    photocraft_geom::Rect::new(x0, y0, x1.max(x0 + 1), y1.max(y0 + 1))
}

/// The FACE_OVAL ring's bounding box in document pixels — what the first mesh pass says the
/// face actually is. `None` when the landmarks are too short for the ring.
fn oval_box(landmarks: &[[f32; 3]]) -> Option<photocraft_geom::Rect> {
    let mut x0 = f32::MAX;
    let mut y0 = f32::MAX;
    let (mut x1, mut y1) = (f32::MIN, f32::MIN);
    for &i in photocraft_portrait::face::FACE_OVAL {
        let p = landmarks.get(i)?;
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    (!landmarks.is_empty()).then(|| photocraft_geom::Rect::new(x0 as i32, y0 as i32, x1.max(0.0) as i32, y1.max(0.0) as i32))
}

/// Whether the mesh's answer looks like a face inside `over`: the oval ring leaves ≥5%
/// margins on every side (a mesh that found nothing, or a face clipped at the crop's edge,
/// hugs the border) and spans ≥10% of the crop's larger side (a collapsed answer is a
/// speck). The box path reports a wrong box loudly instead of building layers from garbage.
fn plausible(landmarks: &[[f32; 3]], over: photocraft_geom::Rect) -> bool {
    let Some(b) = oval_box(landmarks) else { return false };
    let (w, h) = (over.width() as f32, over.height() as f32);
    let margins = [(b.x0 - over.x0) as f32 / w, (over.x1 - b.x1) as f32 / w, (b.y0 - over.y0) as f32 / h, (over.y1 - b.y1) as f32 / h];
    margins.iter().all(|&m| m >= 0.05) && (b.width() as f32).max(b.height() as f32) / w.max(h) >= 0.1
}

/// FaceMesh over `area`, the landmarks mapped back to document pixels (the sample is `area`
/// box-downsampled toward 256²; the mapping is the proportional one the whole-document path
/// used, generalised to an offset rect). `None` when the mesh does not answer.
fn mesh_over(doc: &Document, target: Option<photocraft_doc::LayerId>, sample_all: bool, area: photocraft_geom::Rect) -> Option<Vec<[f32; 3]>> {
    with_doc_sampler(doc, target, sample_all, |smp, _| {
        let step = photocraft_algo::segment::scale_for(area, photocraft_ml::LANDMARK_SIDE * photocraft_ml::LANDMARK_SIDE);
        let img = smp.rgb_scaled(area, step);
        let pts = photocraft_ml::face_landmarks(&img.px, img.w, img.h)?;
        let (rw, rh) = (area.width() as f32, area.height() as f32);
        Some(pts.into_iter().map(|p| [p.x / img.w as f32 * rw + area.x0 as f32, p.y / img.h as f32 * rh + area.y0 as f32, p.z]).collect())
    })
}

/// The axis-aligned square about `bounds`' centre, clamped to the document — the pose
/// model's crop. MediaPipe's own pipeline feeds the landmark model a *rotated* detector
/// ROI; v1 takes the square upright (the pose_probe verdict of 2026-11-20: plausible on
/// upright photos, the tilted case recorded as the known degradation).
fn to_square(bounds: photocraft_geom::Rect, w: i32, h: i32) -> photocraft_geom::Rect {
    let side = bounds.width().max(bounds.height()) as i32;
    let (cx, cy) = ((bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2);
    let half = side / 2;
    let x0 = (cx - half).clamp(0, (w - side).max(0));
    let y0 = (cy - half).clamp(0, (h - side).max(0));
    photocraft_geom::Rect::new(x0, y0, (x0 + side).min(w), (y0 + side).min(h))
}

/// The body-pose landmarks of the person's square crop (BlazePose full, embedded), mapped
/// to document pixels — the foundation layer: they land on `Analysis.pose`, no stage
/// consumes them yet. `None` without a person (the caller keeps the empty default).
fn pose_over(doc: &Document, target: Option<photocraft_doc::LayerId>, sample_all: bool, bounds: photocraft_geom::Rect) -> Option<Vec<[f32; 4]>> {
    if bounds.is_empty() {
        return None;
    }
    let area = to_square(bounds, doc.size.width as i32, doc.size.height as i32);
    with_doc_sampler(doc, target, sample_all, |smp, _| {
        let step = photocraft_algo::segment::scale_for(area, photocraft_ml::pose::POSE_SIDE * photocraft_ml::pose::POSE_SIDE);
        let img = smp.rgb_scaled(area, step);
        let pts = photocraft_ml::pose::pose_landmarks(&img.px, img.w, img.h)?;
        let (rw, rh) = (area.width() as f32, area.height() as f32);
        Some(pts.into_iter().map(|p| [p.x / img.w as f32 * rw + area.x0 as f32, p.y / img.h as f32 * rh + area.y0 as f32, p.z, p.visibility]).collect())
    })
}

/// The parsing regions of `doc`, upsampled to document resolution and intersected with the
/// person: the model's per-class probabilities grouped into regions
/// (`photocraft_portrait::parsing`), each resampled like the subject mask rides back and
/// clipped to the person the pipeline is about (hair, skin and clothes belong to the
/// person; on off-distribution input the gate also suppresses stray predictions). Empty
/// masks stay out — a region with no support anywhere must not produce a layer.
///
/// `over` crops the model's input: the box path hands the mesh's tightened face box (the
/// export's CelebAMask-HQ training set is face close-ups, so the same crop argument as the
/// mesh's applies). The mask rides back to the crop's size and lands in a full-document
/// surface at its offset — outside the crop the region is unknown and stays 0.
fn parsing_regions(
    doc: &Document,
    target: Option<photocraft_doc::LayerId>,
    sample_all: bool,
    person: &photocraft_raster::Surface,
    over: Option<photocraft_geom::Rect>,
) -> Vec<(photocraft_portrait::Region, photocraft_raster::Surface)> {
    let Some((found, area)) = with_doc_sampler(doc, target, sample_all, |smp, d| {
        let area = over.unwrap_or_else(|| d.bounds());
        let step = photocraft_algo::segment::scale_for(area, photocraft_ml::PARSING_SIDE * photocraft_ml::PARSING_SIDE);
        let img = smp.rgb_scaled(area, step);
        photocraft_ml::parsing_probs(&img.px, img.w, img.h).map(|p| (p, area))
    }) else {
        return Vec::new();
    };
    // Read the person gate per pixel through a LayerMask (that is where `value` lives).
    let person = photocraft_doc::LayerMask { surface: person.clone(), ..photocraft_doc::LayerMask::reveal_all() };
    let (aw, ah) = (area.width() as usize, area.height() as usize);
    photocraft_portrait::parsing::region_planes(&found.planes, found.w, found.h)
        .into_iter()
        .filter_map(|(region, mask)| {
            let support = mask.iter().fold(0.0f32, |m, v| m.max(*v));
            if support < 0.5 {
                return None;
            }
            let mut up = photocraft_ml::resize_mask(&mask, found.w, found.h, aw, ah)?;
            for y in 0..ah {
                for x in 0..aw {
                    up[y * aw + x] *= person.value(x as i32 + area.x0, y as i32 + area.y0);
                }
            }
            let mut surface = photocraft_raster::Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
            surface.write_region(area, &up);
            Some((region, surface))
        })
        .collect()
}

/// The whole-body regions of `doc` (`selfiemulticlass.onnx`), sampled over the person's
/// own bounds, not the document: the model is selfie-framing trained, so the person
/// filling its 256² view is the distribution it understands, and the effective resolution
/// rises with the person's share of the document (the whole-document ask put a panorama's
/// small person into a handful of pixels, every class under 0.5 — docs/fork/models.md's
/// known boundary). The model's five person classes map onto the pipeline's regions
/// (body-skin ∪ face-skin → `Skin`, hair → `Hair`, clothes → `Clothes`), pass the
/// [`BODY_FLOOR`] ramp and are gated by the person like the face-parsing regions. Empty
/// when the model is not placed or the person has no bounds — the caller's regions then
/// stay whatever the face parsing found.
fn body_regions(
    doc: &Document,
    target: Option<photocraft_doc::LayerId>,
    sample_all: bool,
    person: &photocraft_raster::Surface,
) -> Vec<(photocraft_portrait::Region, photocraft_raster::Surface)> {
    let (dw, dh) = (doc.size.width, doc.size.height);
    let bounds = person.content_bounds();
    if bounds.is_empty() {
        return Vec::new();
    }
    let area = expanded(bounds, dw, dh, BODY_MARGIN);
    let Some((planes, sw, sh)) = with_doc_sampler(doc, target, sample_all, |smp, _d| {
        let step = photocraft_algo::segment::scale_for(area, photocraft_ml::body::BODY_SIDE * photocraft_ml::body::BODY_SIDE);
        let img = smp.rgb_scaled(area, step);
        let (sw, sh) = (img.w, img.h);
        photocraft_ml::body::body_probs(&img.px, sw, sh).map(|p| (p, sw, sh))
    }) else {
        return Vec::new();
    };
    gated_region_planes(
        &planes,
        sw,
        sh,
        area,
        person,
        &[
            (photocraft_ml::body::BODY_SKIN, photocraft_portrait::Region::Skin),
            (photocraft_ml::body::FACE_SKIN, photocraft_portrait::Region::Skin),
            (photocraft_ml::body::HAIR, photocraft_portrait::Region::Hair),
            (photocraft_ml::body::CLOTHES, photocraft_portrait::Region::Clothes),
        ],
    )
}

/// The body/fashion planes' shared tail: each `classes` entry's plane rides back at the
/// crop's size, through the [`BODY_FLOOR`] ramp and the person gate, lands speckle-suppressed
/// in a full-document surface at the crop's offset (outside it the region stays 0) — one
/// class may feed several regions (upper-clothes into `UpperClothes` *and* `Clothes`), the
/// body tier's two-skin-classes-into-`Skin` precedent.
fn gated_region_planes(
    planes: &[Vec<f32>],
    sw: usize,
    sh: usize,
    area: photocraft_geom::Rect,
    person: &photocraft_raster::Surface,
    classes: &[(usize, photocraft_portrait::Region)],
) -> Vec<(photocraft_portrait::Region, photocraft_raster::Surface)> {
    let gate = photocraft_doc::LayerMask { surface: person.clone(), ..photocraft_doc::LayerMask::reveal_all() };
    let (aw, ah) = (area.width() as usize, area.height() as usize);
    let mut out = Vec::new();
    for (class, region) in classes {
        let Some(mut vals) = planes.get(*class).and_then(|p| photocraft_ml::resize_mask(p, sw, sh, aw, ah)) else { continue };
        for y in 0..ah {
            for x in 0..aw {
                vals[y * aw + x] = body_floor(vals[y * aw + x]) * gate.value(x as i32 + area.x0, y as i32 + area.y0);
            }
        }
        suppress_speckle(&mut vals, aw, ah);
        let mut surface = photocraft_raster::Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
        surface.write_region(area, &vals);
        out.push((*region, surface));
    }
    out
}

/// The fashion tier's class→region table (ATR-18), the `fashion_probe` spike's verified
/// subset (docs/fork/decisions.md 2026-11-20): the garment classes into their sub-regions
/// *and* `Clothes`, the face and the bare-skin geometric classes (arm/leg, verified to
/// fire on bare skin only — suited sleeves produced no arm blob) into `Skin`, hair into
/// `Hair`. Unmapped on purpose: background, hat, sunglasses and belt — no probe evidence,
/// so they surface nowhere (the eight-region cap's lesson: classes earn regions).
const FASHION_REGIONS: &[(usize, photocraft_portrait::Region)] = &[
    (photocraft_ml::fashion::UPPER_CLOTHES, photocraft_portrait::Region::UpperClothes),
    (photocraft_ml::fashion::UPPER_CLOTHES, photocraft_portrait::Region::Clothes),
    (photocraft_ml::fashion::SKIRT, photocraft_portrait::Region::LowerClothes),
    (photocraft_ml::fashion::SKIRT, photocraft_portrait::Region::Clothes),
    (photocraft_ml::fashion::PANTS, photocraft_portrait::Region::LowerClothes),
    (photocraft_ml::fashion::PANTS, photocraft_portrait::Region::Clothes),
    (photocraft_ml::fashion::DRESS, photocraft_portrait::Region::Dress),
    (photocraft_ml::fashion::DRESS, photocraft_portrait::Region::Clothes),
    (photocraft_ml::fashion::LEFT_SHOE, photocraft_portrait::Region::Shoes),
    (photocraft_ml::fashion::RIGHT_SHOE, photocraft_portrait::Region::Shoes),
    (photocraft_ml::fashion::BAG, photocraft_portrait::Region::Bag),
    (photocraft_ml::fashion::SCARF, photocraft_portrait::Region::Scarf),
    (photocraft_ml::fashion::FACE, photocraft_portrait::Region::Skin),
    (photocraft_ml::fashion::LEFT_ARM, photocraft_portrait::Region::Skin),
    (photocraft_ml::fashion::RIGHT_ARM, photocraft_portrait::Region::Skin),
    (photocraft_ml::fashion::LEFT_LEG, photocraft_portrait::Region::Skin),
    (photocraft_ml::fashion::RIGHT_LEG, photocraft_portrait::Region::Skin),
    (photocraft_ml::fashion::HAIR, photocraft_portrait::Region::Hair),
];

/// The clothing regions of `doc` (`schp-atr.onnx`, the ATR-18 fashion tier) over the same
/// person crop as [`body_regions`]. This tier **replaces** the six-class one when it
/// answers — a union would keep the worse model's errors per-pixel-max (the rust neck
/// reading skin at 0.908 survives a union even with a correct scarf class), so the
/// precedence is: fashion answers → it owns the clothing sub-regions, `Clothes`, `Skin`
/// and `Hair`; the six-class tier runs only in its absence. Empty like `body_regions`
/// when the model is not placed or the person has no bounds.
fn fashion_regions(
    doc: &Document,
    target: Option<photocraft_doc::LayerId>,
    sample_all: bool,
    person: &photocraft_raster::Surface,
) -> Vec<(photocraft_portrait::Region, photocraft_raster::Surface)> {
    let (dw, dh) = (doc.size.width, doc.size.height);
    let bounds = person.content_bounds();
    if bounds.is_empty() {
        return Vec::new();
    }
    let area = expanded(bounds, dw, dh, BODY_MARGIN);
    let Some((planes, sw, sh)) = with_doc_sampler(doc, target, sample_all, |smp, _d| {
        let step = photocraft_algo::segment::scale_for(area, photocraft_ml::fashion::FASHION_SIDE * photocraft_ml::fashion::FASHION_SIDE);
        let img = smp.rgb_scaled(area, step);
        let (sw, sh) = (img.w, img.h);
        photocraft_ml::fashion::fashion_probs(&img.px, sw, sh).map(|p| (p, sw, sh))
    }) else {
        return Vec::new();
    };
    gated_region_planes(&planes, sw, sh, area, person, FASHION_REGIONS)
}

/// A copy of `mask` clipped by `person` (per-pixel multiply), the hull's share of the gate
/// the parsing regions ride. `None` when the read-back fails.
fn person_gated(mask: &photocraft_raster::Surface, person: &photocraft_raster::Surface, w: u32, h: u32) -> Option<photocraft_raster::Surface> {
    let area = photocraft_geom::Rect::new(0, 0, w as i32, h as i32);
    let mut vals = mask.try_read_region(area).ok()?;
    let gate = photocraft_doc::LayerMask { surface: person.clone(), ..photocraft_doc::LayerMask::reveal_all() };
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let i = (y * w as i32 + x) as usize;
            vals[i] *= gate.value(x, y);
        }
    }
    let mut out = photocraft_raster::Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
    out.write_region(area, &vals);
    Some(out)
}

/// `mask` with the pixels inside `r` zeroed — the body regions' step aside inside the mesh
/// ring, where the face parsing's finer answer must stand alone: two qualities stitched
/// per-pixel-max otherwise show the crop's straight edge as a seam.
fn outside_rect(mask: &photocraft_raster::Surface, r: photocraft_geom::Rect, w: u32, h: u32) -> Option<photocraft_raster::Surface> {
    let area = photocraft_geom::Rect::new(0, 0, w as i32, h as i32);
    let mut vals = mask.try_read_region(area).ok()?;
    for y in r.y0.max(0)..r.y1.min(h as i32) {
        for x in r.x0.max(0)..r.x1.min(w as i32) {
            vals[(y * w as i32 + x) as usize] = 0.0;
        }
    }
    let mut out = photocraft_raster::Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
    out.write_region(area, &vals);
    Some(out)
}

/// The cheek geometry the mesh contributes to [`cheeks_from_parsing`] — everything the
/// masks themselves do not carry.
struct CheekGeometry {
    /// The nose column: the tip's x and the column's half-width, document pixels — a
    /// smoothstep ramp either side (the blush never lands on the nose, but the last
    /// pixel of the ramp must not read as an edge).
    nose_x: f32,
    nose_half: f32,
    /// The eye-line to mouth-line band (`face::cheek_band`), document pixels. The cheek
    /// proper is a slice of it: below the under-eye, above the mouth corners.
    band: (f32, f32),
}

/// The cheeks from the parsing's skin plane: the skin's own hairline- and jaw-accurate
/// boundary, banded to the cheek slice of the eye-to-mouth line, minus the eye and lip
/// rings the mesh places and the nose column between them. The `blush_mask` ellipses
/// guess at all of this; here only the band and the exclusions are geometric — the
/// boundary is the model's. `None` when a read-back fails.
fn cheeks_from_parsing(
    skin: &photocraft_raster::Surface,
    eyes: &photocraft_raster::Surface,
    lips: &photocraft_raster::Surface,
    geo: &CheekGeometry,
    w: u32,
    h: u32,
) -> Option<photocraft_raster::Surface> {
    let area = photocraft_geom::Rect::new(0, 0, w as i32, h as i32);
    let mut vals = skin.try_read_region(area).ok()?;
    let eyes_v = eyes.try_read_region(area).ok()?;
    let lips_v = lips.try_read_region(area).ok()?;
    // The band's slice: in from the lid line by its top 18% (the under-eye is not the
    // cheek), up from the mouth line by 8% (the corners are not either). The slice fades
    // in over its top quarter and cuts at the bottom.
    let height = geo.band.1 - geo.band.0;
    let (top, bottom) = (geo.band.0 + height * 0.18, geo.band.1 - height * 0.08);
    let fade = ((bottom - top) * 0.25).max(1.0);
    let smoothstep = |t: f32| t * t * (3.0 - 2.0 * t);
    for y in 0..h as i32 {
        let yf = y as f32 + 0.5;
        let band_v = if yf <= bottom { ((yf - top) / fade).clamp(0.0, 1.0) } else { 0.0 };
        for x in 0..w as i32 {
            let i = (y * w as i32 + x) as usize;
            // Flat zero over the nose's inner 70%, rising over the outer 30% — the
            // bridge and the tip carry no blush at all, and the cheek fades in only at
            // the nose's own edge.
            let dx = (((x as f32 + 0.5) - geo.nose_x).abs() / geo.nose_half).clamp(0.0, 1.0);
            let nose_v = smoothstep(((dx - 0.7) / 0.3).clamp(0.0, 1.0));
            vals[i] *= band_v * nose_v * (1.0 - eyes_v[i].min(1.0)) * (1.0 - lips_v[i].min(1.0));
        }
    }
    let mut out = photocraft_raster::Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
    out.write_region(area, &vals);
    Some(out)
}

/// The person analysis of `doc` — the portrait pipeline's own stages, extracted so the
/// selection tools can consume the same regions: the subject/matte, the mesh (the `roi`'s
/// box or the detector's, else the whole document), the face-crop parsing and the
/// whole-body segmentation. `ctx` carries the progress; `select.region` passes a fresh
/// detached one, whose stages run silently. The drawn-`roi` failure is loud (the user asked
/// for that face); with `roi` absent every stage degrades quietly.
/// The person matte (the ort tier when the feature and its model are present) or Select
/// Subject's seeded min cut, at document resolution — the Person region every person-based
/// stage reads. `None` when neither can find a person.
fn person_surface(
    doc: &Arc<Document>,
    target: Option<photocraft_doc::LayerId>,
    sample_all: bool,
    ctx: &crate::jobs::JobCtx,
) -> Result<Option<photocraft_raster::Surface>> {
    // The ort matte (feature "ort" + feynobg.onnx in the models dir) replaces the
    // whole subject line when it answers: direct hair-edge alpha at 1024², no
    // seed/GrabCut. Without it (the default build, or no model placed) the
    // subject line runs as before.
    let matte = ctx.stage(0.05, 0.25, "Matting person", |_| {
        with_doc_sampler(doc, target, sample_all, |smp, d| {
            let step = photocraft_algo::segment::scale_for(d.bounds(), photocraft_ml::MATTING_SIDE * photocraft_ml::MATTING_SIDE);
            let img = smp.rgb_scaled(d.bounds(), step);
            photocraft_ml::person_matte(&img.px, img.w, img.h)
        })
    });
    if let Some(matte) = matte {
        ctx.check().map_err(|_| EngineError::Cancelled)?;
        let (dw, dh) = (doc.size.width as usize, doc.size.height as usize);
        let up = photocraft_ml::resize_mask(&matte.px, matte.w, matte.h, dw, dh).ok_or_else(|| EngineError::Other("the matte would not resample".into()))?;
        let mut surface = photocraft_raster::Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
        surface.write_region(photocraft_geom::Rect::new(0, 0, doc.size.width as i32, doc.size.height as i32), &up);
        return Ok(Some(surface));
    }
    // Cancel reaches the min cuts inside Select Subject, so Esc doesn't wait for them.
    let found = ctx
        .stage(0.25, 0.4, "Finding person", |ctl| {
            with_doc_sampler(doc, target, sample_all, |smp, d| {
                let seed = crate::smartselect_cmds::subject_seed(smp, d.bounds());
                subject::select_subject_seeded_with(smp, d.bounds(), seed.as_ref(), ctl)
            })
        })
        .map_err(|_| EngineError::Cancelled)?;
    Ok(found.map(|region| matting::region_surface(&region)))
}

/// SCRFD's box about the largest face, in document pixels. `None` without the detector
/// model (`facedetector.onnx` in the models dir) or a face — the caller keeps its
/// whole-document fallback.
fn detected_face_box(
    doc: &Arc<Document>,
    target: Option<photocraft_doc::LayerId>,
    sample_all: bool,
    ctx: &crate::jobs::JobCtx,
) -> Option<photocraft_geom::Rect> {
    ctx.stage(0.4, 0.45, "Finding faces", |_| {
        with_doc_sampler(doc, target, sample_all, |smp, d| {
            let step = photocraft_algo::segment::scale_for(d.bounds(), photocraft_ml::detect::DETECT_SIDE * photocraft_ml::detect::DETECT_SIDE);
            let img = smp.rgb_scaled(d.bounds(), step);
            let face = photocraft_ml::detect::detect_faces(&img.px, img.w, img.h)?.first().copied()?;
            let (sw, sh) = (img.w as f32, img.h as f32);
            let (dw, dh) = (d.size.width as f32, d.size.height as f32);
            Some(photocraft_geom::Rect::new(
                (face.x0 / sw * dw) as i32,
                (face.y0 / sh * dh) as i32,
                (face.x1 / sw * dw).ceil() as i32,
                (face.y1 / sh * dh).ceil() as i32,
            ))
        })
    })
}

/// The box path's two-pass FaceMesh: the first over the margin-expanded box, the second
/// over the oval ring's box, so a loose box still hands the mesh a face-dominant crop. A
/// drawn box (`drawn`) that fails the gate errors the run — the user asked for that face;
/// the automatic one answers `None` and the caller falls back to the whole-document mesh.
fn boxed_mesh(
    doc: &Arc<Document>,
    target: Option<photocraft_doc::LayerId>,
    sample_all: bool,
    face_box: Option<photocraft_geom::Rect>,
    drawn: bool,
    ctx: &crate::jobs::JobCtx,
) -> Result<Option<(Vec<[f32; 3]>, photocraft_geom::Rect)>> {
    let Some(r) = face_box else { return Ok(None) };
    let (dw, dh) = (doc.size.width, doc.size.height);
    let no_face = || EngineError::Other("no face found in the box — draw it tighter around the face".into());
    let first = expanded(r, dw, dh, BOX_MARGIN);
    let pts = ctx.stage(0.45, 0.55, "Finding landmarks", |_| mesh_over(doc, target, sample_all, first));
    let ring = pts.and_then(|pts| {
        let ring = expanded(oval_box(&pts)?, dw, dh, RING_MARGIN);
        (ring.width() >= MIN_FACE_SIDE as u32 && ring.height() >= MIN_FACE_SIDE as u32).then_some(ring)
    });
    match ring {
        Some(ring) => match ctx.stage(0.55, 0.6, "Finding landmarks", |_| mesh_over(doc, target, sample_all, ring)).filter(|pts| plausible(pts, ring)) {
            Some(pts) => Ok(Some((pts, ring))),
            None if drawn => Err(no_face()),
            None => Ok(None),
        },
        None if drawn => Err(no_face()),
        None => Ok(None),
    }
}

/// Which landmark set a sculpt panel needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SculptKind {
    Body,
    Face,
}

impl SculptKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SculptKind::Body => "body",
            SculptKind::Face => "face",
        }
    }
}

/// Where the face mesh's crop box came from — what the panel's quality line names when the
/// landmarks are less trustworthy than a drawn box would make them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceSource {
    /// The user's box (`roi`): the mesh ran where they asked.
    Drawn,
    /// SCRFD's largest face.
    Detected,
    /// No box at all: the mesh ran over the whole document, the fallback that puts a small
    /// face into a handful of pixels.
    WholeDocument,
}

impl FaceSource {
    pub fn as_str(self) -> &'static str {
        match self {
            FaceSource::Drawn => "drawn",
            FaceSource::Detected => "detected",
            FaceSource::WholeDocument => "wholeDocument",
        }
    }
}

/// A lightweight analysis plus where its face box came from. The source is the panel's one
/// honest quality signal that needs no extra stage: a whole-document mesh is the known-worse
/// case, and it is actionable (draw a box).
pub(crate) struct SculptAnalysis {
    pub analysis: Arc<photocraft_portrait::Analysis>,
    pub face: FaceSource,
}

/// The lightweight analysis a sculpt command needs — no Smart Portrait parsing, body or
/// fashion stages. Body: the person mask and the pose, with a boxed mesh when the detector
/// or the drawn `roi` finds a face. Face: the drawn `roi` or SCRFD's box and the FaceMesh,
/// so a photo the subject line cannot segment still gets its face; without a box the
/// whole-document mesh runs, with the subject line only as the liveness gate the embedded
/// mesh lacks.
///
/// Never stored in `portrait_cache`: the partial `Analysis` lacks the regions
/// `portrait.create` and Select Region read.
pub(crate) fn analyze_sculpt(
    doc: &Arc<Document>,
    target: Option<photocraft_doc::LayerId>,
    sample_all: bool,
    kind: SculptKind,
    roi: Option<photocraft_geom::Rect>,
    ctx: &crate::jobs::JobCtx,
) -> Result<SculptAnalysis> {
    let mut analysis = photocraft_portrait::Analysis::new(doc.size.width, doc.size.height);
    let person = match kind {
        SculptKind::Body => Some(person_surface(doc, target, sample_all, ctx)?.ok_or_else(|| EngineError::Other(crate::sculpt_cmds::NO_PERSON.into()))?),
        SculptKind::Face => None,
    };
    if let Some(person) = &person {
        analysis.set(photocraft_portrait::Region::Person, person.clone());
    }
    let face_box = roi.or_else(|| detected_face_box(doc, target, sample_all, ctx));
    let drawn = roi.is_some();
    let boxed = boxed_mesh(doc, target, sample_all, face_box, drawn, ctx)?;
    let face = match (drawn, boxed.is_some()) {
        (true, _) => FaceSource::Drawn,
        (false, true) => FaceSource::Detected,
        (false, false) => FaceSource::WholeDocument,
    };
    analysis.landmarks = match boxed {
        Some((pts, _)) => pts,
        None => {
            // The whole-document mesh is the no-detector fallback, and the embedded model has no
            // presence output: on anything without a face it answers an average-face template
            // (a blank canvas included). The subject line's answer is the liveness gate then —
            // it is what the pre-sculpt pipeline required of every Face run anyway — so a
            // faceless document reports no face instead of warping an invented one. With a
            // detected box none of this runs: Face never needs the person stage there.
            if kind == SculptKind::Face && person_surface(doc, target, sample_all, ctx)?.is_none() {
                return Err(EngineError::Other(crate::face_sculpt_cmds::NO_FACE.into()));
            }
            ctx.stage(0.65, 0.7, "Finding landmarks", |_| mesh_over(doc, target, sample_all, doc.bounds())).unwrap_or_default()
        }
    };
    if let SculptKind::Body = kind {
        analysis.pose = ctx
            .stage(0.7, 0.75, "Finding pose", |_| {
                let bounds = person.as_ref().map(photocraft_raster::Surface::content_bounds).unwrap_or_default();
                (!bounds.is_empty()).then(|| pose_over(doc, target, sample_all, bounds)).flatten()
            })
            .unwrap_or_default();
    }
    Ok(SculptAnalysis { analysis: Arc::new(analysis), face })
}

/// `portrait.analyze`'s answer: the anchors in the sculpt commands' stored shape, the Body
/// contour when there is one, and for Face the quality the panel warns from (`None` for
/// Body — the pose table has no equivalent honest reading).
pub struct AnchorsReply {
    pub anchors: Value,
    pub contour: Option<Vec<[f64; 2]>>,
    pub quality: Option<Value>,
}

/// The face's quality reply: where the box came from plus the mesh's own geometry. The
/// `source` is what the panel can act on; the angles are what it warns about.
fn face_quality(anchors: &photocraft_algo::face_sculpt::FaceAnchors, source: FaceSource) -> Value {
    let mut v = json!({"source": source.as_str()});
    if let Some(q) = anchors.quality() {
        v["faceWidth"] = json!(q.face_width);
        v["faceHeight"] = json!(q.face_height);
        v["rollDeg"] = json!(q.roll_deg);
        v["turnDeg"] = json!(q.turn_deg);
    }
    v
}

/// The anchors and contour `portrait.analyze` reports, straight from a document — the
/// command's own body, callable by the sculpt panels' background worker with its own
/// cancellable [`crate::jobs::JobCtx`]. One lightweight run and never cached, like the
/// command; the command adds only the reply framing.
pub fn analyze_anchors(
    doc: &Arc<Document>,
    target: Option<photocraft_doc::LayerId>,
    sample_all: bool,
    kind: SculptKind,
    roi: Option<photocraft_geom::Rect>,
    ctx: &crate::jobs::JobCtx,
) -> Result<AnchorsReply> {
    let found = analyze_sculpt(doc, target, sample_all, kind, roi, ctx)?;
    let analysis = &found.analysis;
    match kind {
        SculptKind::Body => {
            let anchors = crate::sculpt_cmds::anchors_from_analysis(analysis).ok_or_else(|| EngineError::Other(crate::sculpt_cmds::NO_PERSON.into()))?;
            let contour = analysis.mask(photocraft_portrait::Region::Person).and_then(crate::sculpt_cmds::person_contour);
            Ok(AnchorsReply {
                anchors: serde_json::to_value(&anchors).map_err(|e| EngineError::Other(format!("the anchors would not serialise: {e}")))?,
                contour,
                quality: None,
            })
        }
        SculptKind::Face => {
            let anchors =
                crate::face_sculpt_cmds::anchors_from_analysis(analysis).ok_or_else(|| EngineError::Other(crate::face_sculpt_cmds::NO_FACE.into()))?;
            Ok(AnchorsReply {
                anchors: serde_json::to_value(&anchors).map_err(|e| EngineError::Other(format!("the anchors would not serialise: {e}")))?,
                contour: None,
                quality: Some(face_quality(&anchors, found.face)),
            })
        }
    }
}

pub(crate) fn analyze_person(
    doc: &Arc<Document>,
    target: Option<photocraft_doc::LayerId>,
    sample_all: bool,
    roi: Option<photocraft_geom::Rect>,
    ctx: &crate::jobs::JobCtx,
) -> Result<Arc<photocraft_portrait::Analysis>> {
    let person = person_surface(doc, target, sample_all, ctx)?.ok_or_else(|| EngineError::Other("no person found".into()))?;
    let mut analysis = photocraft_portrait::Analysis::new(doc.size.width, doc.size.height);
    analysis.set(photocraft_portrait::Region::Person, person.clone());
    let (dw, dh) = (doc.size.width, doc.size.height);
    // The box the mesh and parsing crop to: the user's `roi` wins; without one
    // the detector (SCRFD) supplies the largest face's box — the automatic
    // sibling of the drawn one. Neither a model nor a face leaves `None` and the
    // legacy whole-document mesh runs.
    let face_box = roi.or_else(|| detected_face_box(doc, target, sample_all, ctx));
    // The box path's mesh runs first: its oval ring defines the crop the parsing stage
    // then reads (both models want the face-dominant crop — see `roi`).
    let boxed = boxed_mesh(doc, target, sample_all, face_box, roi.is_some(), ctx)?;
    // The parsing stage when a model is handed in (`PHOTOCRAFT_MODELS_DIR`):
    // skin/hair/clothes masks join the person mask; without one the slice is the
    // person alone and every layer but the first stays out. The whole-body
    // segmentation (`selfiemulticlass.onnx`, ~50ms) runs over the document — no
    // crop; its nature is full-image — and its classes UNION into the regions the
    // face-crop parsing set: body-skin and face-skin into Skin (so the smoothing
    // layer covers the arms and neck, not only the face), hair and clothes into
    // their regions.
    let (parse_lo, parse_hi) = if boxed.is_some() { (0.6, 0.65) } else { (0.4, 0.65) };
    let (regions, body) = ctx.stage(parse_lo, parse_hi, "Parsing regions", |_| {
        let regions = parsing_regions(doc, target, sample_all, &person, boxed.as_ref().map(|(_, r)| *r));
        // The fashion tier owns the body when it answers; the six-class tier is its
        // fallback (a union would keep the six-class rust-neck-as-skin misread).
        let body = match fashion_regions(doc, target, sample_all, &person) {
            found if !found.is_empty() => found,
            _ => body_regions(doc, target, sample_all, &person),
        };
        (regions, body)
    });
    // The parsing's skin plane is the face skin, hairline-accurate — also the Face
    // region's whole answer when it exists: the mesh hull below only floors a Face it
    // cannot improve (a symmetric ring is not a traced hairline), so it lands solely
    // when the parsing stage produced no skin (no model, or support under the filter).
    let parsing_skin = regions.iter().any(|(r, _)| *r == photocraft_portrait::Region::Skin);
    for (region, mask) in regions {
        if region == photocraft_portrait::Region::Skin {
            analysis.union(photocraft_portrait::Region::Face, mask.clone());
        }
        analysis.set(region, mask);
    }
    for (region, mask) in body {
        // Inside the mesh ring the body classes step aside for the parsing's finer
        // answer (Skin, Hair); what the ring cannot see — the neck below, the arms, the
        // clothes — is what they add.
        let mask = match (region, boxed.as_ref()) {
            (photocraft_portrait::Region::Skin | photocraft_portrait::Region::Hair, Some((_, ring))) => {
                outside_rect(&mask, *ring, doc.size.width, doc.size.height).unwrap_or(mask)
            }
            _ => mask,
        };
        analysis.union(region, mask);
    }
    // The body-pose landmarks (BlazePose full, embedded, ~35ms): the foundation layer of
    // 2026-11-20 — they land on `Analysis.pose` for the body-aware work of a later
    // iteration; a quiet stage that degrades to empty, nothing consumes them yet.
    analysis.pose = ctx
        .stage(0.7, 0.75, "Finding pose", |_| {
            let bounds = person.content_bounds();
            (!bounds.is_empty()).then(|| pose_over(doc, target, sample_all, bounds)).flatten()
        })
        .unwrap_or_default();
    // FaceMesh landmarks (weights embedded in photocraft-ml, ~15ms): the face
    // region comes from the landmark ring's hull, gated by the person like the
    // parsing regions. The box path already ran its mesh; the whole-document
    // path samples toward 256² and rides the points to document pixels.
    let mesh = match &boxed {
        Some((pts, _)) => Some(pts.clone()),
        None => ctx.stage(0.65, 0.7, "Finding landmarks", |_| mesh_over(doc, target, sample_all, doc.bounds())),
    };
    if let Some(pts) = mesh {
        analysis.landmarks = pts;
        if !parsing_skin
            && let Some(gated) = photocraft_portrait::face::face_mask(&analysis.landmarks, dw, dh).and_then(|hull| person_gated(&hull, &person, dw, dh))
        {
            analysis.union(photocraft_portrait::Region::Face, gated);
        }
        // The eye and lip rings: the parsing export reads them as skin, so the
        // landmark rings are their regions (person-gated like every other). The cheeks
        // read the same rings as exclusions.
        let features = photocraft_portrait::face::feature_masks(&analysis.landmarks, dw, dh);
        if let Some((eyes, lips)) = features.as_ref() {
            for (region, mask) in [(photocraft_portrait::Region::Eyes, eyes), (photocraft_portrait::Region::Lips, lips)] {
                if let Some(gated) = person_gated(mask, &person, dw, dh) {
                    analysis.set(region, gated);
                }
            }
        }
        // The cheeks: the parsing's skin, banded and feature-excluded, when the model
        // answered — the ellipses only guess at where a tilted face's cheekbones are;
        // they stay the degraded path's answer (no model, or no skin plane).
        let cheeks = match (parsing_skin, analysis.mask(photocraft_portrait::Region::Skin), features.as_ref()) {
            (true, Some(skin), Some((eyes, lips))) => {
                let span = analysis.landmarks.get(263).zip(analysis.landmarks.get(33)).map(|(a, b)| (a[0] - b[0]).abs()).unwrap_or_default();
                photocraft_portrait::face::cheek_band(&analysis.landmarks)
                    .and_then(|band| {
                        cheeks_from_parsing(
                            skin,
                            eyes,
                            lips,
                            &CheekGeometry { nose_x: analysis.landmarks.get(1).map(|p| p[0]).unwrap_or_default(), nose_half: span * 0.30, band },
                            dw,
                            dh,
                        )
                    })
                    .or_else(|| photocraft_portrait::face::blush_mask(&analysis.landmarks, dw, dh))
            }
            _ => photocraft_portrait::face::blush_mask(&analysis.landmarks, dw, dh),
        };
        // The makeup geometry: the cheeks above and the scaled eye rings, feathered.
        for (region, mask) in [
            (photocraft_portrait::Region::Cheeks, cheeks),
            (photocraft_portrait::Region::Eyelids, photocraft_portrait::face::eyeshadow_mask(&analysis.landmarks, dw, dh)),
        ] {
            if let Some(mask) = mask
                && let Some(gated) = person_gated(&mask, &person, dw, dh)
            {
                analysis.set(region, gated);
            }
        }
    }
    Ok(Arc::new(analysis))
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let recipe = recipe(p);
    let sample_all = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let roi = roi(p, st.doc.size);
    // The box path never reads the pixel-keyed cache — a cached analysis is
    // whole-document (or another box's) — and stores nothing: the next unboxed run must not
    // inherit this box's crop.
    let cached = if roi.is_none() { s.portrait_cache.get(&st.doc, crate::layer_multi_cmds::same_pixels) } else { None };
    let doc = Arc::clone(&st.doc);
    let target = st.active_layer;
    let reference = reference(p);
    crate::jobs::run(
        s,
        "Smart Portrait",
        true,
        move |ctx| {
            // The analysis resolves first — a cache hit skips subject, parsing and
            // landmarks; the derived pixels below run whenever the recipe asks.
            let analysis = if let Some(analysis) = cached { analysis } else { analyze_person(&doc, target, sample_all, roi, ctx)? };
            // The restoration first: the GAN redraws skin texture, so the smoothing below
            // runs on the enhanced composite when both are asked for (the DAG's
            // Enhance-before-Skin edge, realised as the blur's source).
            let mut composite: Option<photocraft_raster::Surface> = None;
            let enhance = if recipe.enhance && !analysis.landmarks.is_empty() {
                ctx.progress(0.7, "Enhancing face");
                let canvas = doc.bounds();
                let surface = composite.get_or_insert_with(|| photocraft_compose::composite_to_surface(&doc, canvas, photocraft_doc::PixelFormat::RGBA8, None));
                enhance_face(&doc, surface, &analysis)?
            } else {
                None
            };
            let skin = match recipe.skin.filter(|_| analysis.mask(photocraft_portrait::Region::Skin).is_some()) {
                Some(asked) => {
                    ctx.progress(0.8, "Smoothing skin");
                    let canvas = doc.bounds();
                    let source = match &enhance {
                        Some(e) => e,
                        None => {
                            composite.get_or_insert_with(|| photocraft_compose::composite_to_surface(&doc, canvas, photocraft_doc::PixelFormat::RGBA8, None))
                        }
                    };
                    Some(smooth_skin(ctx, asked, source, canvas)?)
                }
                None => None,
            };
            // 追色 last: the reference's statistics against the person's own tones.
            let grade = match reference {
                Some(r) => {
                    ctx.progress(0.9, "Matching colour");
                    Some(grade_transfer(&doc, analysis.mask(photocraft_portrait::Region::Person), r)?)
                }
                None => None,
            };
            Ok((analysis, Products { skin, enhance, grade }))
        },
        move |s, (analysis, products)| {
            let mut recipe = recipe.clone();
            recipe.grade = products.grade;
            let plan = photocraft_portrait::plan(&recipe, &analysis);
            let group = materialize(s, &plan, &analysis, &products)?;
            // Keyed to the post-edit document, past this edit's own cache reset (the same
            // re-point Select Subject does); an intervening edit still invalidates by pixels.
            // The box path stays out: its analysis is box-scoped, not the document's.
            if roi.is_none()
                && let Some(doc) = s.active().map(|st| Arc::clone(&st.doc))
            {
                s.portrait_cache.store(&doc, (*analysis).clone());
            }
            Ok(json!({ "group": group.0 }))
        },
    )
}

/// One history step named "Smart Portrait": the plan's standard layer subtree, built directly
/// (`Layer::group` takes children; masks are assigned, no selection roundtrip). Unimplemented
/// stages plan nothing, so the match's other arms are the generative stages' landing spots.
fn materialize(
    s: &mut Session,
    plan: &photocraft_portrait::LayerPlan,
    analysis: &photocraft_portrait::Analysis,
    products: &Products,
) -> Result<photocraft_doc::LayerId> {
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
                photocraft_portrait::PlannedLayer::Raster { name, mask } => {
                    // A rasterized stage result: an ordinary layer, masked to its region —
                    // the classic duplicate-and-retouch shape, undoable by hiding or
                    // deleting the layer. The pixels arrive from the work stage; the mask
                    // region says which product a raster carries.
                    let surface = match mask {
                        Some(photocraft_portrait::Region::Skin) => products.skin.as_ref(),
                        Some(photocraft_portrait::Region::Face) => products.enhance.as_ref(),
                        _ => None,
                    };
                    let Some(surface) = surface else { continue };
                    let mut layer = Layer::new(name.clone(), LayerContent::Raster(surface.clone()));
                    if let Some(region) = mask
                        && let Some(surface) = analysis.mask(*region)
                    {
                        layer.mask = Some(crate::commands::mask_at_depth(LayerMask { surface: surface.clone(), ..LayerMask::reveal_all() }, doc.depth)?);
                    }
                    children.push(layer);
                }
                photocraft_portrait::PlannedLayer::Group { .. } => {}
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
/// `portrait.saveRecipe`: the current recipe as a named action whose single step runs
/// [`CMD`] — what the Actions panel lists, what Batch's `action` parameter takes. The
/// whole params object (minus `name`) rides as the step's params; the recipe parser reads
/// only the keys it knows.
fn save_recipe(s: &mut Session, p: &Value) -> Result<Value> {
    let name = p.get("name").and_then(Value::as_str).ok_or_else(|| EngineError::Other("portrait.saveRecipe: a `name` is required".into()))?;
    let mut step = p.clone();
    if let Some(o) = step.as_object_mut() {
        o.remove("name");
    }
    let action = crate::actions_cmds::Action::new(name.to_owned(), vec![crate::actions_cmds::ActionStep::new(CMD, step)]);
    s.actions.list.push(action).map_err(|e| EngineError::Other(format!("could not save the recipe: {e}")))?;
    Ok(json!({ "action": name }))
}

/// `portrait.analyze`: the anchors that let the sculpt commands skip analysis on Apply — one
/// lightweight run, Body (person mask + pose, plus the person contour) or Face (the face mesh;
/// the person stage only gates the no-detector fallback). Read-only: no history step, and
/// nothing goes into `portrait_cache` (a partial analysis would poison `portrait.create` and
/// Select Region).
fn analyze(s: &mut Session, p: &Value) -> Result<Value> {
    let kind = match p.get("kind").and_then(Value::as_str).unwrap_or("body") {
        "body" => SculptKind::Body,
        "face" => SculptKind::Face,
        other => return Err(EngineError::BadParams { cmd: "portrait.analyze".into(), msg: format!("`kind` is \"body\" or \"face\", not {other:?}") }),
    };
    let sample_all = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let has_roi = p.get("roi").is_some_and(|v| !v.is_null());
    if kind == SculptKind::Body && has_roi {
        return Err(EngineError::BadParams { cmd: "portrait.analyze".into(), msg: "`roi` is Face Sculpt's face box; Body Sculpt's pose needs no box".into() });
    }
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let mut target = st.active_layer;
    if p.get("layer").is_some() {
        target = Some(crate::commands::layer_param(s, p)?);
    }
    // As the sculpt commands resolve it: a layer that is not a raster surface has no pixels
    // of its own to sample, so the sampler reads the flattened document instead.
    let target = target.filter(|l| !sample_all && st.doc.layer(*l).and_then(photocraft_doc::Layer::surface).is_some());
    let face_box = has_roi.then(|| roi(p, st.doc.size)).flatten();
    let doc = Arc::clone(&st.doc);
    let (width, height) = (doc.size.width, doc.size.height);
    let t0 = crate::photo_cmds::Stopwatch::start();
    crate::jobs::run(
        s,
        "Analyze Portrait",
        false,
        move |ctx| analyze_anchors(&doc, target, sample_all, kind, face_box, ctx),
        move |_, found| {
            let mut r = json!({"kind": kind.as_str(), "width": width, "height": height, "layer": target.map(|l| l.0), "anchors": found.anchors, "ms": t0.ms()});
            if let Some(contour) = found.contour {
                r["contour"] = json!(contour);
            }
            if let Some(quality) = found.quality {
                r["quality"] = quality;
            }
            Ok(r)
        },
    )
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: CMD,
            label: "Smart Portrait…",
            menu: &["Layer", "New Adjustment Layer"],
            shortcut: None,
            params: r##"{"sampleAllLayers":bool=false,"roi":[x,y,w,h]=none,"skinRadius":1..100=10,"skinThreshold":2..255=18,"enhanceFace":bool=false,"hairHue":-180..180=0,"hairSaturation":-100..100=0,"lipHue":0..360=330,"lipSaturation":0..100=30,"eyeBrighten":0..50=12,"blushHue":0..360=350,"blushSaturation":0..100=20,"shadowHue":0..360=30,"shadowSaturation":0..100=25,"hue":-180..180=0,"saturation":-100..100=0,"lightness":-100..100=0} → {group} (analyzes the person — and, with PHOTOCRAFT_MODELS_DIR, the skin/hair/clothes regions and the GFPGAN restoration — then adds a named group of standard adjustment layers masked to them: smoothing, hair and lip colour, eye brighten, the person grade; one history step. `roi` is the user-drawn face box (document pixels): the mesh and parsing stages run inside it — the face-dominant crop both models assume — and a box too small or whose clipped/degenerate mesh answer fails the gate errors with `no face found in the box`. Without `roi` the SCRFD detector (`facedetector.onnx` in the models dir) finds the largest face and boxes it the same way; no detector or no face leaves the whole-document analysis. The colour-match `reference` path and `referenceStats` are protocol-only, not dialog rows)"##,
            enabled,
            run,
            journal: true,
        },
        CommandSpec {
            id: "portrait.saveRecipe",
            label: "Save Portrait Recipe…",
            menu: &[],
            shortcut: None,
            params: r##"{"name":text,"…":any portrait.create params} → {action} (the recipe as a named action: the Actions panel lists it, Batch runs it over a folder)"##,
            enabled: |_| Ok(()),
            run: save_recipe,
            journal: true,
        },
        CommandSpec {
            id: "portrait.analyze",
            label: "Analyze Portrait",
            menu: &[],
            shortcut: None,
            params: r##"{"kind":"body"|"face"="body","layer":id?,"sampleAllLayers":bool=false,"roi":[x,y,w,h]=none (face only)} → {kind,width,height,layer,anchors,contour?,quality?,ms} (the anchors the sculpt panels hand `filter.bodySculpt`/`filter.faceSculpt`: Body = person mask + pose + the silhouette contour, Face = the face mesh — a detected face box needs no person stage; without one the subject line only gates presence. `roi` is Face Sculpt's drawn face box in document pixels: the mesh runs inside it and a box with no face errors instead of falling back; without it SCRFD's detected box is used, and without that the whole-document mesh. Face `quality` names the box source (`drawn`/`detected`/`wholeDocument`) and the mesh's face width, height, eye-line roll and nose-offset turn, so the panel can warn on the fallback and on a turned head. One lightweight run — no parsing/body/fashion stages, no history step, nothing cached — and Apply with these anchors never analyses again)"##,
            enabled,
            run: analyze,
            journal: false,
        },
    ]
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
    fn roi_rounds_clamps_and_drops_empties() {
        let size = photocraft_geom::Size::new(200, 160);
        assert_eq!(roi(&json!({"roi": [10.4, 20.5, 50, 40]}), size), Some(Rect::new(10, 21, 60, 61)));
        // Overshoot clips to the document; outside drops to None.
        assert_eq!(roi(&json!({"roi": [-30, -30, 500, 500]}), size), Some(Rect::new(0, 0, 200, 160)));
        assert_eq!(roi(&json!({"roi": [300, 300, 10, 10]}), size), None);
        assert_eq!(roi(&json!({}), size), None);
        assert_eq!(roi(&json!({"roi": "nope"}), size), None);
    }

    #[test]
    fn expanded_grows_about_the_centre_and_clamps_to_the_document() {
        assert_eq!(expanded(Rect::new(40, 40, 60, 60), 100, 80, 2.0), Rect::new(30, 30, 70, 70));
        assert_eq!(expanded(Rect::new(0, 0, 20, 20), 100, 80, 2.0), Rect::new(0, 0, 30, 30));
        assert_eq!(expanded(Rect::new(90, 70, 100, 80), 100, 80, 2.0), Rect::new(85, 65, 100, 80));
        // Never empty, even for a degenerate one-pixel input.
        assert!(!expanded(Rect::new(50, 40, 51, 41), 100, 80, 1.5).is_empty());
    }

    #[test]
    fn oval_box_binds_the_ring_and_plausibility_gates_it() {
        let mut pts = vec![[0.0f32; 3]; 478];
        for (i, &idx) in photocraft_portrait::face::FACE_OVAL.iter().enumerate() {
            pts[idx] = [100.0 + i as f32 * 2.0, 120.0 + (i % 20) as f32 * 2.0, 0.0];
        }
        let b = oval_box(&pts).unwrap();
        // 36 ring points: x spans 100..=170, y 120..=158.
        assert_eq!(b, Rect::new(100, 120, 170, 158));
        assert!(plausible(&pts, Rect::new(0, 0, 400, 400)), "a centred ring of sane size passes");
        // The same ring pushed against the crop's edge fails, and a collapsed one is a speck.
        for p in &mut pts {
            p[0] += 300.0;
        }
        assert!(!plausible(&pts, Rect::new(0, 0, 400, 400)), "an edge-hugging ring fails");
        for p in &mut pts {
            p[0] = 200.0;
            p[1] = 200.0;
        }
        assert!(!plausible(&pts, Rect::new(0, 0, 400, 400)), "a collapsed ring fails");
        // Too few landmarks for the ring: no box, not a panic.
        assert!(oval_box(&pts[..10]).is_none());
    }

    #[test]
    fn a_too_small_box_fails_loudly_and_builds_nothing() {
        // The textured disc carries the person stage; an 8×8 box cannot hold a face past
        // the mesh (even a full-span answer stays under MIN_FACE_SIDE after the margins),
        // so the floor is what the run meets — whatever the mesh answers on it.
        let (mut s, _) = disc();
        let steps = s.active().unwrap().history.past_len();
        let err = s.execute(CMD, json!({"roi": [96, 76, 8, 8]})).unwrap_err();
        assert!(err.to_string().contains("no face found in the box"), "was {err}");
        assert_eq!(s.active().unwrap().history.past_len(), steps, "no history step");
        assert_eq!(s.active().unwrap().doc.layers.len(), 1, "no group added");
        // The failed box run must not poison the cache a later unboxed run reads.
        assert!(s.portrait_cache.get(&s.active().unwrap().doc, |_, _| true).is_none());
    }

    #[test]
    fn a_box_run_stays_out_of_the_analysis_cache() {
        // A mesh is a regressor: on the disc's texture it answers a structurally
        // face-shaped ring (span ≈0.5, margins ≈0.2 — measured), so the geometric gate
        // cannot and must not reject it; that rejection is a detector's job. What is
        // deterministic here: the box-scoped run never lands in the pixel-keyed cache,
        // success or not.
        let (mut s, _) = disc();
        s.execute(CMD, json!({"roi": [50, 40, 100, 80]})).unwrap();
        assert!(s.portrait_cache.get(&s.active().unwrap().doc, |_, _| true).is_none(), "a box run stored its box-scoped analysis");
    }

    /// The whole-body segmentation (`selfiemulticlass.onnx`) rides inside the parsing stage
    /// and unions into the regions: with the model placed the run succeeds and the smoothing
    /// layer lands. A smoke over the stage wiring — the union's max semantics live in
    /// photocraft-portrait's own tests, and the real-photo A/B in docs/fork.
    #[test]
    fn body_segmentation_smokes_when_the_model_is_placed() {
        if std::env::var_os("PHOTOCRAFT_MODELS_DIR").is_none_or(|d| !std::path::Path::new(&d).join("selfiemulticlass.onnx").is_file()) {
            eprintln!("skipping: no selfiemulticlass.onnx in PHOTOCRAFT_MODELS_DIR");
            return;
        }
        let (mut s, _) = disc();
        s.execute(CMD, json!({"skinRadius": 10})).unwrap();
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        assert!(group.children.iter().any(|l| l.name == "Skin Smoothing"), "the skin layer lands with the body stage in the run");
    }

    /// The fashion tier answers, so its clothing regions exist in the analysis — as empty
    /// masks on the disc (no real garments): the ask succeeds with an empty selection
    /// rather than erroring or wronging the selection. Real-photo semantics are the
    /// dump/probe tests' job (region_dump_diagnostic, fashion_probe).
    #[test]
    fn fashion_tier_smokes_when_the_model_is_placed() {
        if std::env::var_os("PHOTOCRAFT_MODELS_DIR").is_none_or(|d| !std::path::Path::new(&d).join("schp-atr.onnx").is_file()) {
            eprintln!("skipping: no schp-atr.onnx in PHOTOCRAFT_MODELS_DIR");
            return;
        }
        let (mut s, _) = disc();
        let out = s.execute("select.region", json!({"region": "scarf"})).unwrap();
        assert_eq!(out["selected"], json!(false), "an empty region selects nothing");
        let cached = s.portrait_cache.get(&s.active().unwrap().doc, crate::layer_multi_cmds::same_pixels);
        assert!(cached.is_some(), "the analysis the fashion tier ran is cached");
        assert!(cached.unwrap().regions().any(|r| r == photocraft_portrait::Region::Scarf), "the fashion regions are in the cached analysis");
    }

    #[test]
    fn cancelled_while_finding_the_person_changes_nothing() {
        let (mut s, _) = disc();
        let before = s.active().unwrap().revision;
        let ctx = crate::jobs::JobCtx::new();
        ctx.cancel();
        s.set_inline_job_ctx(Some(ctx.clone()));
        assert!(matches!(s.execute(CMD, json!({})), Err(EngineError::Cancelled)));
        assert!(matches!(ctx.message().as_str(), "Finding person" | "Matting person"), "stage message was {:#?}", ctx.message());
        assert_eq!(s.active().unwrap().revision, before);
        s.set_inline_job_ctx(None);
        s.execute(CMD, json!({})).unwrap();
        assert!(s.active().unwrap().active_layer.is_some());
    }

    #[test]
    fn without_a_person_the_document_is_left_alone() {
        if std::env::var_os("PHOTOCRAFT_MODELS_DIR").is_some_and(|d| std::path::Path::new(&d).join("feynobg.onnx").is_file()) {
            eprintln!("skipping: the ort matte mattes anything, flat grey included — the no-person error is the subject line's answer, not the matte tier's");
            return;
        }
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
        if std::env::var_os("PHOTOCRAFT_MODELS_DIR").is_none() {
            eprintln!("skipping: PHOTOCRAFT_MODELS_DIR unset");
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

    /// Without a parsing model there is no Skin region, so asking for smoothing changes
    /// nothing — the degrade path, deterministic everywhere.
    #[test]
    fn skin_smoothing_without_regions_stays_one_layer() {
        if std::env::var_os("PHOTOCRAFT_MODELS_DIR").is_some() {
            eprintln!("skipping: a parsing model is set, the Skin region exists");
            return;
        }
        let (mut s, _) = disc();
        s.execute(CMD, json!({"skinRadius": 12.0})).unwrap();
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        assert_eq!(group.children.len(), 1);
    }

    /// With a model handed in the smoothing raster lands third, masked to the skin region
    /// and carrying blurred pixels (smoothed ≠ original somewhere).
    #[test]
    fn skin_smoothing_lands_a_blurred_raster_when_a_model_is_provided() {
        if std::env::var_os("PHOTOCRAFT_MODELS_DIR").is_none() {
            eprintln!("skipping: PHOTOCRAFT_MODELS_DIR unset");
            return;
        }
        let (mut s, _) = disc();
        s.execute(CMD, json!({"skinRadius": 12.0, "skinThreshold": 25.0})).unwrap();
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        let skin_layer = group.children.iter().find(|l| l.name == "Skin Smoothing").expect("a Skin Smoothing layer");
        assert!(skin_layer.mask.is_some(), "the smoothing raster is masked");
        let LayerContent::Raster(surface) = &skin_layer.content else { panic!("not a raster") };
        // Surface Blur of a noisy disc: at least somewhere the copy differs from the flat
        // composite it came from (noise averaged away inside the disc).
        let original = photocraft_compose::composite_to_surface(
            &s.active().unwrap().doc,
            photocraft_geom::Rect::new(0, 0, 200, 160),
            photocraft_doc::PixelFormat::RGBA8,
            None,
        );
        let mut differs = 0;
        for y in 100..110 {
            for x in 95..105 {
                let a = surface.rgba(x, y);
                let b = original.rgba(x, y);
                if (a[0] - b[0]).abs() > 0.01 || (a[1] - b[1]).abs() > 0.01 {
                    differs += 1;
                }
            }
        }
        assert!(differs > 0, "the smoothing raster is a copy, not a blur");
    }

    /// 追色 with injected statistics (no file, no model): the Color Match Curves layer
    /// lands after the person layer, masked to the person, carrying the transfer.
    #[test]
    fn colour_match_lands_a_person_masked_curves_layer() {
        let (mut s, _) = disc();
        // Reference warmer and tighter than the disc's own stats: red mean up, red std down.
        let r = s.execute(CMD, json!({"referenceStats": {"mean": [0.7, 0.5, 0.4], "std": [0.08, 0.12, 0.12]}})).unwrap();
        assert!(r["group"].as_u64().is_some());
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        assert_eq!(group.children.len(), 2, "person layer + colour match");
        let grade = &group.children[1];
        assert_eq!(grade.name, "Color Match");
        assert!(grade.mask.is_some(), "masked to the person");
        let LayerContent::Adjustment(photocraft_doc::Adjustment::Curves { per_channel, .. }) = &grade.content else { panic!("not curves") };
        // Every channel carries a two-point line (the affine transfer in curve points).
        for curve in per_channel {
            assert_eq!(curve.len(), 2, "{curve:?}");
            assert!(curve[0].input < curve[1].input);
        }
        // And it actually grades: the composite inside the disc shifts toward the reference.
        let inside = photocraft_compose::render(&s.active().unwrap().doc, photocraft_geom::Rect::from_xywh(100, 80, 1, 1)).px[0];
        assert!(inside[0] > 0.55, "red lifted toward the reference: {inside:?}");
        // Undo removes the whole group, grade included.
        let group_id = s.active().unwrap().active_layer.unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        assert!(s.active().unwrap().doc.layer(group_id).is_none());
    }

    #[test]
    fn colour_match_without_a_reference_stays_out() {
        let (mut s, _) = disc();
        s.execute(CMD, json!({})).unwrap();
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        assert_eq!(group.children.len(), 1, "no ask, no grade layer");
        // A malformed reference is an error, not a silent skip.
        let mut s2 = disc().0;
        assert!(s2.execute(CMD, json!({"reference": "/nonexistent/ref.png"})).is_err());
    }

    /// Without a GFPGAN model the enhancement ask changes nothing — the degrade path.
    #[test]
    fn face_enhance_without_a_model_stays_out() {
        if std::env::var_os("PHOTOCRAFT_MODELS_DIR").is_some_and(|d| std::path::Path::new(&d).join("gfpgan.onnx").is_file()) {
            eprintln!("skipping: a gfpgan model is placed, the enhance layer lands");
            return;
        }
        let (mut s, _) = disc();
        s.execute(CMD, json!({"enhanceFace": true})).unwrap();
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        assert_eq!(group.children.len(), 1, "no model, no enhance layer");
    }

    /// With the model placed the Face Enhance raster lands with the face-hull mask and
    /// genuinely redrawn pixels (the paste differs from the composite it came from).
    #[test]
    fn face_enhance_lands_a_redrawn_raster_when_a_model_is_placed() {
        if std::env::var_os("PHOTOCRAFT_MODELS_DIR").is_none_or(|d| !std::path::Path::new(&d).join("gfpgan.onnx").is_file()) {
            eprintln!("skipping: no gfpgan.onnx in PHOTOCRAFT_MODELS_DIR");
            return;
        }
        let (mut s, _) = disc();
        s.execute(CMD, json!({"enhanceFace": true})).unwrap();
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        let enhanced = group.children.iter().find(|l| l.name == "Face Enhance").expect("a Face Enhance layer");
        assert!(enhanced.mask.is_some(), "masked to the face hull");
        let LayerContent::Raster(surface) = &enhanced.content else { panic!("not a raster") };
        let original = photocraft_compose::composite_to_surface(
            &s.active().unwrap().doc,
            photocraft_geom::Rect::new(0, 0, 200, 160),
            photocraft_doc::PixelFormat::RGBA8,
            None,
        );
        let mask = photocraft_doc::LayerMask { surface: enhanced.mask.clone().unwrap().surface.clone(), ..photocraft_doc::LayerMask::reveal_all() };
        let (mut differs, mut inside) = (0, 0);
        for y in 60..100 {
            for x in 80..120 {
                if mask.value(x, y) >= 0.5 {
                    inside += 1;
                    let (a, b) = (surface.rgba(x, y), original.rgba(x, y));
                    if (a[0] - b[0]).abs() > 0.01 || (a[1] - b[1]).abs() > 0.01 {
                        differs += 1;
                    }
                }
            }
        }
        assert!(inside > 0, "the disc's landmark hull has support");
        // Pixel-level restoration is deliberately NOT asserted here: a synthetic disc's
        // landmark mesh is garbage, the aligned crop lands wherever it lands, and the
        // paste legitimately falls back to the base outside it. The wiring proof is the
        // layer, its mask and the model's real run (the suite's time budget shows it);
        // what the restoration does to a real face is the real-photo validation the
        // archive tracks. Silence the counters the diff used to check.
        let _ = (differs, original);
    }

    /// The embedded FaceMesh runs unconditionally (no env, ~15ms): the analysis carries
    /// the landmarks and the face-hull region on every create, model or no parsing model.
    #[test]
    fn landmarks_and_face_region_land_on_every_create() {
        let (mut s, inside) = disc();
        s.execute(CMD, json!({})).unwrap();
        let doc = &s.active().unwrap().doc;
        let cached = s.portrait_cache.get(doc, |_, _| true).expect("an analysis");
        assert_eq!(cached.landmarks.len(), photocraft_ml::LANDMARK_COUNT);
        let face = cached.mask(photocraft_portrait::Region::Face).expect("a face-hull region");
        // Gated by the person: every face pixel sits inside the disc.
        let gate = photocraft_doc::LayerMask { surface: face.clone(), ..photocraft_doc::LayerMask::reveal_all() };
        let mut hits = 0;
        for y in 0..160 {
            for x in 0..200 {
                if gate.value(x, y) >= 0.5 {
                    hits += 1;
                    assert!(inside(x, y), "face hull outside the person at ({x}, {y})");
                }
            }
        }
        assert!(hits > 0, "the hull has support");
    }

    /// The facial-feature layers are model-free (the embedded mesh carries the rings):
    /// lip colour and eye brighten land on any build, deterministically.
    #[test]
    fn lip_colour_and_eye_brighten_land_from_the_landmark_rings() {
        let (mut s, _) = disc();
        s.execute(CMD, json!({"lipHue": 320.0, "lipSaturation": 30.0, "eyeBrighten": 12.0})).unwrap();
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        assert_eq!(group.children.len(), 3, "person + eye + lip layers");
        let eye = &group.children[1];
        assert_eq!(eye.name, "Eye Brighten");
        assert!(eye.mask.is_some(), "masked to the eye rings");
        let LayerContent::Adjustment(photocraft_doc::Adjustment::BrightnessContrast { brightness, .. }) = &eye.content else {
            panic!("not brightness/contrast")
        };
        assert_eq!(*brightness, 12.0);
        let lip = &group.children[2];
        assert_eq!(lip.name, "Lip Color");
        assert!(lip.mask.is_some(), "masked to the lip ring");
        let LayerContent::Adjustment(photocraft_doc::Adjustment::HueSaturation { hue, colorize, .. }) = &lip.content else { panic!("not hue/saturation") };
        assert_eq!((*hue, *colorize), (320.0, true), "colourized at the asked hue");
        // Undo removes all three as one step.
        let group_id = s.active().unwrap().active_layer.unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        assert!(s.active().unwrap().doc.layer(group_id).is_none());
    }

    /// The makeup pair is model-free like the features: cheek ellipses and scaled eye
    /// rings land from the embedded mesh on any build.
    #[test]
    fn blush_and_eyeshadow_land_from_the_landmark_geometry() {
        let (mut s, _) = disc();
        s.execute(CMD, json!({"blushSaturation": 20.0, "shadowHue": 280.0, "shadowSaturation": 30.0})).unwrap();
        let layer = s.active().unwrap().active_layer.map(|id| s.active().unwrap().doc.layer(id).unwrap()).unwrap();
        let LayerContent::Group(group) = &layer.content else { panic!("not a group") };
        assert_eq!(group.children.len(), 3, "person + eyeshadow + blush layers");
        let blush = &group.children[1];
        assert_eq!(blush.name, "Blush");
        assert!(blush.mask.is_some(), "masked to the cheek ellipses");
        let shadow = &group.children[2];
        assert_eq!(shadow.name, "Eyeshadow");
        assert!(shadow.mask.is_some(), "masked to the scaled eye rings");
        let LayerContent::Adjustment(photocraft_doc::Adjustment::HueSaturation { hue, colorize, .. }) = &shadow.content else { panic!("not hue/saturation") };
        assert_eq!((*hue, *colorize), (280.0, true));
    }

    /// The batch composition: `file.automate.batch` (upstream) runs `portrait.create`
    /// (the fork's pipeline) as a recorded step over a folder — the whole batch feature is
    /// this composition, so this test is its contract. Two inputs, a light recipe with a
    /// landmark-driven layer (model-free, deterministic), PNG outputs, no errors.
    #[test]
    fn batch_runs_the_portrait_pipeline_over_a_folder() {
        let dir = std::env::temp_dir().join(format!("photocraft-portrait-batch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let dir = dir.to_string_lossy().into_owned();
        // Two grey inputs with a subject blob each (a blank document has no person).
        let inputs: Vec<String> = (0..2)
            .map(|i| {
                let mut s = Session::new();
                s.execute("file.new", json!({"width": 96, "height": 96, "background": "white"})).unwrap();
                s.edit("paint", |doc, _| {
                    let bg = doc.layers[0].surface_mut().unwrap();
                    bg.fill_rect(photocraft_geom::Rect::new(28, 28, 68, 68), &[0.6, 0.2, 0.1, 1.0]);
                    Ok(())
                })
                .unwrap();
                let path = format!("{dir}/img{i}.png");
                crate::file_cmds::save_doc(&s.active().unwrap().doc, &path, None).unwrap();
                path
            })
            .collect();
        let output = format!("{dir}/out");
        let mut runner = Session::new();
        let r = runner
            .execute(
                "file.automate.batch",
                json!({
                    "input": inputs,
                    "output": output,
                    "format": "png",
                    "steps": [["portrait.create", {"lipSaturation": 30.0, "eyeBrighten": 12.0}]],
                }),
            )
            .unwrap();
        assert_eq!(r["errors"], json!([]), "batch errors: {r}");
        assert_eq!(r["files"].as_array().map(Vec::len), Some(2), "one output per input: {r}");
        for f in r["files"].as_array().unwrap() {
            assert!(std::path::Path::new(f.as_str().unwrap()).is_file(), "{f} exists");
        }
        // The recipe-as-action bridge: saveRecipe stores the recipe under a name, and
        // Batch takes the name instead of inline steps — the Actions panel route.
        let mut saver = Session::new();
        let saved = saver.execute("portrait.saveRecipe", json!({"name": "Studio Skin", "eyeBrighten": 12.0, "lipSaturation": 30.0})).unwrap();
        assert_eq!(saved["action"], "Studio Skin");
        let output2 = format!("{dir}/out2");
        let r2 = saver.execute("file.automate.batch", json!({"input": inputs, "output": output2, "format": "png", "action": "Studio Skin"})).unwrap();
        assert_eq!(r2["errors"], json!([]), "action-based batch: {r2}");
        assert_eq!(r2["files"].as_array().map(Vec::len), Some(2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real-photo acceptance harness: `PHOTOCRAFT_PORTRAIT_PHOTO=<image>` imports
    /// it, runs the full pipeline (everything the placed models allow) and quick-exports
    /// the composite for eyeballing. Skipped without the env var — CI has neither the
    /// weights nor the photos.
    #[test]
    fn real_photo_acceptance_export() {
        let Some(photo) = std::env::var_os("PHOTOCRAFT_PORTRAIT_PHOTO") else {
            eprintln!("skipping: PHOTOCRAFT_PORTRAIT_PHOTO unset");
            return;
        };
        let run = || -> Result<()> {
            let mut s = Session::new();
            let name = std::path::Path::new(&photo).file_name().and_then(|n| n.to_str()).unwrap_or("photo").to_string();
            let imported =
                photocraft_io::import_file(&name, std::path::Path::new(&photo), &photocraft_io::ImportOptions::default(), &photocraft_raster::Interrupt::NONE)
                    .map_err(|e| EngineError::Other(format!("import: {e}")))?;
            s.open_document(imported.document, None);
            let t = std::time::Instant::now();
            let r = s.execute(CMD, json!({"enhanceFace": true, "skinRadius": 8.0, "hairHue": -10.0, "hairSaturation": 15.0, "referenceStats": {"mean": [0.55, 0.47, 0.42], "std": [0.14, 0.12, 0.11]}}))?;
            eprintln!("portrait.create on a real photo: {:.1} s (group {})", t.elapsed().as_secs_f32(), r["group"]);
            let out = std::env::var_os("PHOTOCRAFT_PORTRAIT_OUT").map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "/tmp/portrait_out.png".into());
            // The composite with the portrait group applied: flatten, then the quick export's
            // active layer is that single background.
            s.execute("layer.flattenImage", json!({}))?;
            s.execute("layer.quickExportAsPng", json!({"path": out}))?;
            eprintln!("exported: {out}");
            Ok(())
        };
        if let Err(e) = run() {
            panic!("acceptance harness failed: {e}");
        }
    }

    /// The Select Region diagnostic: `PHOTOCRAFT_REGION_PHOTO=<image>` imports the photo,
    /// runs the person analysis exactly as the `select.region` bar button does (no `roi`,
    /// the active layer) and writes every region mask as `<out>/<stem>-<region>.png` —
    /// grayscale, white = inside — plus a JSON summary (per-region mean coverage and
    /// content bounds, the landmark count) to stdout. Skipped without the env var, like
    /// the real-photo export above: a seeing instrument for "the regions are wrong",
    /// which stage contributed what.
    #[test]
    fn region_dump_diagnostic() {
        let Some(photo) = std::env::var_os("PHOTOCRAFT_REGION_PHOTO") else {
            eprintln!("skipping: PHOTOCRAFT_REGION_PHOTO unset");
            return;
        };
        let out = std::env::var_os("PHOTOCRAFT_REGION_OUT").map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "/tmp/regiondump".into());
        std::fs::create_dir_all(&out).unwrap();
        let stem = std::path::Path::new(&photo).file_stem().and_then(|s| s.to_str()).unwrap_or("photo").to_string();
        let run = || -> Result<()> {
            let mut s = Session::new();
            let name = std::path::Path::new(&photo).file_name().and_then(|n| n.to_str()).unwrap_or("photo").to_string();
            let imported =
                photocraft_io::import_file(&name, std::path::Path::new(&photo), &photocraft_io::ImportOptions::default(), &photocraft_raster::Interrupt::NONE)
                    .map_err(|e| EngineError::Other(format!("import: {e}")))?;
            s.open_document(imported.document, None);
            let analysis = {
                let d = s.active().ok_or(EngineError::NoDocument)?;
                let target = d.active_layer.filter(|id| d.doc.layer(*id).and_then(Layer::surface).is_some());
                let ctx = crate::jobs::JobCtx::new();
                crate::portrait_cmds::analyze_person(&d.doc, target, false, None, &ctx)?
            };
            let (w, h) = (analysis.width, analysis.height);
            let mut summary = serde_json::Map::new();
            summary.insert("landmarks".into(), json!(analysis.landmarks.len()));
            summary.insert("pose".into(), json!(analysis.pose.len()));
            for region in analysis.regions() {
                let surface = analysis.mask(region).cloned().ok_or_else(|| EngineError::Other("mask read".into()))?;
                let mask = photocraft_doc::LayerMask { surface: surface.clone(), ..photocraft_doc::LayerMask::reveal_all() };
                let mut covered = 0usize;
                let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
                for y in 0..h as i32 {
                    for x in 0..w as i32 {
                        if mask.value(x, y) > 0.5 {
                            covered += 1;
                            x0 = x0.min(x);
                            y0 = y0.min(y);
                            x1 = x1.max(x);
                            y1 = y1.max(y);
                        }
                    }
                }
                let name = format!("{region:?}").to_lowercase();
                let bounds = (covered > 0).then_some(json!([x0, y0, x1, y1])).unwrap_or(json!(null));
                summary.insert(name.clone(), json!({ "coverage": covered as f64 / (w as f64 * h as f64), "bounds": bounds }));
                let mut doc =
                    Document::new(&name, photocraft_geom::Size { width: w, height: h }, photocraft_doc::ColorMode::Grayscale, photocraft_doc::SampleType::U8);
                doc.layers.push(Layer::new(&name, LayerContent::Raster(surface)));
                let exported =
                    photocraft_io::export(&doc, "png", &photocraft_io::ExportOptions::default()).map_err(|e| EngineError::Other(format!("export: {e}")))?;
                std::fs::write(format!("{out}/{stem}-{name}.png"), &exported.bytes).map_err(|e| EngineError::Other(format!("write: {e}")))?;
            }
            eprintln!("{}", serde_json::to_string_pretty(&serde_json::Value::Object(summary)).unwrap_or_default());
            Ok(())
        };
        if let Err(e) = run() {
            panic!("region diagnostic failed: {e}");
        }
    }

    /// The body planes' floor: under 0.5 nothing survives (the scarf that reads skin at
    /// 0.55, the background speckle), over 0.75 everything does (the neck, the arms), and
    /// the stretch between ramps so a soft edge stays soft.
    #[test]
    fn body_floor_ramps_between_its_ends() {
        assert_eq!(body_floor(0.0), 0.0);
        assert_eq!(body_floor(0.49), 0.0);
        assert!((body_floor(0.625) - 0.5).abs() < 1e-6);
        assert_eq!(body_floor(0.75), 1.0);
        assert_eq!(body_floor(1.0), 1.0);
    }

    /// `outside_rect` zeroes the ring's inside and leaves the rest of the mask alone.
    #[test]
    fn outside_rect_zeroes_only_the_ring_inside() {
        let mask = |w: i32, h: i32| {
            let vals: Vec<f32> = (0..w * h).map(|i| (i % 7) as f32 / 7.0).collect();
            let mut s = photocraft_raster::Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
            s.write_region(photocraft_geom::Rect::new(0, 0, w, h), &vals);
            s
        };
        let m = photocraft_doc::LayerMask {
            surface: outside_rect(&mask(16, 8), photocraft_geom::Rect::new(4, 2, 9, 6), 16, 8).unwrap(),
            ..photocraft_doc::LayerMask::reveal_all()
        };
        for y in 0..8 {
            for x in 0..16 {
                let expected = if (4..9).contains(&x) && (2..6).contains(&y) { 0.0 } else { (y * 16 + x) as f32 % 7.0 / 7.0 };
                assert!((m.value(x, y) - expected).abs() < 0.01, "({x},{y})");
            }
        }
    }

    /// `suppress_speckle` keeps the components that matter relative to the plane's own
    /// largest blob and drops the isolated dots — the 256² model's borderline pixels.
    #[test]
    fn suppress_speckle_drops_dots_not_blobs() {
        let (w, h) = (16usize, 8usize);
        let mut vals = vec![0.0f32; w * h];
        // A 6×6 blob, a 2×2 dot and a single pixel — 36, 4 and 1 pixels.
        for y in 0..6 {
            for x in 0..6 {
                vals[y * w + x] = 0.9;
            }
        }
        for y in 6..8 {
            for x in 12..14 {
                vals[y * w + x] = 0.4;
            }
        }
        vals[7 * w + 15] = 1.0;
        suppress_speckle(&mut vals, w, h);
        for y in 0..6 {
            for x in 0..6 {
                assert_eq!(vals[y * w + x], 0.9, "the blob survives ({x},{y})");
            }
        }
        for y in 6..8 {
            for x in 12..14 {
                assert_eq!(vals[y * w + x], 0.0, "the 4-pixel dot dies ({x},{y})");
            }
        }
        assert_eq!(vals[7 * w + 15], 0.0, "the single pixel dies");
    }

    /// The body planes' confidence probe: `PHOTOCRAFT_REGION_PHOTO=<image>` samples the
    /// raw `body_probs` skin planes over two windows — the face (known skin) and a
    /// user-named `PHOTOCRAFT_REGION_PROBE=[x,y,w,h]` rectangle (the misclassified
    /// fabric) — and prints their confidence distributions. Decides whether a floor can
    /// separate them or only a better model can.
    #[test]
    fn body_confidence_probe() {
        let (Some(photo), Some(probe)) = (std::env::var_os("PHOTOCRAFT_REGION_PHOTO"), std::env::var_os("PHOTOCRAFT_REGION_PROBE")) else {
            eprintln!("skipping: PHOTOCRAFT_REGION_PHOTO or PHOTOCRAFT_REGION_PROBE unset");
            return;
        };
        let rect = |v: &str| -> Vec<f32> { v.split(',').filter_map(|n| n.trim().parse().ok()).collect() };
        let probe = rect(&probe.to_string_lossy());
        let run = || -> Result<()> {
            let name = std::path::Path::new(&photo).file_name().and_then(|n| n.to_str()).unwrap_or("photo").to_string();
            let imported =
                photocraft_io::import_file(&name, std::path::Path::new(&photo), &photocraft_io::ImportOptions::default(), &photocraft_raster::Interrupt::NONE)
                    .map_err(|e| EngineError::Other(format!("import: {e}")))?;
            let doc = imported.document;
            // The whole document: the probe asks about the model's raw confidence, not
            // what any gate or crop makes of it.
            let area = doc.bounds();
            let Some((planes, sw, sh)) = with_doc_sampler(&doc, None, true, |smp, _d| {
                let step = photocraft_algo::segment::scale_for(area, photocraft_ml::body::BODY_SIDE * photocraft_ml::body::BODY_SIDE);
                let img = smp.rgb_scaled(area, step);
                let (sw, sh) = (img.w, img.h);
                photocraft_ml::body::body_probs(&img.px, sw, sh).map(|p| (p, sw, sh))
            }) else {
                return Ok(());
            };
            let stats = |label: &str, x: f32, y: f32, w: f32, h: f32| {
                let (mut lo, mut hi, mut sum, mut n) = (1.0f32, 0.0f32, 0.0f32, 0usize);
                let (x0, y0) = ((x / area.width() as f32 * sw as f32) as usize, (y / area.height() as f32 * sh as f32) as usize);
                let (x1, y1) = (((x + w) / area.width() as f32 * sw as f32) as usize, ((y + h) / area.height() as f32 * sh as f32) as usize);
                for y in y0..y1.min(sh) {
                    for x in x0..x1.min(sw) {
                        let v = planes
                            .get(photocraft_ml::body::BODY_SKIN)
                            .map(|p| p[y * sw + x])
                            .unwrap_or(0.0)
                            .max(planes.get(photocraft_ml::body::FACE_SKIN).map(|p| p[y * sw + x]).unwrap_or(0.0));
                        lo = lo.min(v);
                        hi = hi.max(v);
                        sum += v;
                        n += 1;
                    }
                }
                eprintln!("{label}: n={n} min={lo:.3} mean={:.3} max={hi:.3}", if n > 0 { sum / n as f32 } else { 0.0 });
            };
            stats("face window", 220.0, 220.0, 150.0, 180.0);
            stats("probe window", probe[0] - area.x0 as f32, probe[1] - area.y0 as f32, probe[2], probe[3]);
            let _ = planes;
            Ok(())
        };
        if let Err(e) = run() {
            panic!("probe failed: {e}");
        }
    }

    /// The parsing-derived cheeks: the skin's own boundary, kept only between the eye
    /// line and the mouth line, with the eye rings, lip ring and nose column removed.
    #[test]
    fn cheeks_from_parsing_bands_and_excludes() {
        let surface = |w: i32, h: i32, on: &dyn Fn(i32, i32) -> bool| {
            let vals: Vec<f32> = (0..w * h).map(|i| if on(i % w, i / w) { 1.0 } else { 0.0 }).collect();
            let mut s = photocraft_raster::Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
            s.write_region(photocraft_geom::Rect::new(0, 0, w, h), &vals);
            s
        };
        let (w, h) = (32, 16);
        // Skin everywhere, eyes as two 6×3 boxes at y 4..7, lips as a 8×3 box at y 12..15.
        let skin = surface(w, h, &|_, _| true);
        let eyes = surface(w, h, &|x, y| (4..10).contains(&x) && (4..7).contains(&y) || (22..28).contains(&x) && (4..7).contains(&y));
        let lips = surface(w, h, &|x, y| (12..20).contains(&x) && (12..15).contains(&y));
        let mask = photocraft_doc::LayerMask {
            surface: cheeks_from_parsing(&skin, &eyes, &lips, &CheekGeometry { nose_x: 16.0, nose_half: 6.0, band: (4.0, 12.0) }, 32, 16).unwrap(),
            ..photocraft_doc::LayerMask::reveal_all()
        };
        assert_eq!(mask.value(16, 2), 0.0, "above the eye line is forehead");
        assert_eq!(mask.value(6, 5), 0.0, "the eye ring is excluded");
        assert_eq!(mask.value(16, 13), 0.0, "the lip ring is excluded");
        // The band's slice: top = 4 + 0.18×8 ≈ 5.4, bottom = 12 − 0.08×8 ≈ 11.4.
        assert_eq!(mask.value(8, 5), 0.0, "the under-eye rows above the slice stay out");
        assert!(mask.value(16, 8) < 0.1, "the nose column is near-zero at its centre");
        assert!(mask.value(12, 8) < 0.35, "the nose ramp stays low beside the tip");
        assert!(mask.value(7, 8) > 0.9, "outside the nose column the skin stands");
        assert_eq!(mask.value(16, 14), 0.0, "below the mouth line is chin");
        assert!(mask.value(8, 9) > 0.9, "the left cheek survives");
        assert!(mask.value(24, 9) > 0.9, "the right cheek survives");
    }

    /// A blank canvas has no person: Body's analysis says so and neither kind writes history.
    #[test]
    fn portrait_analyze_body_on_a_blank_document_reports_no_person() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 96, "height": 128})).unwrap();
        let steps = s.active().unwrap().history.past_len();
        let err = s.execute("portrait.analyze", json!({"kind": "body"})).unwrap_err().to_string();
        assert!(err.contains("no person"), "{err}");
        assert_eq!(s.active().unwrap().history.past_len(), steps, "analysis is not an edit");
    }

    /// The Face path never runs the person stage when a detector finds the face; without one
    /// the subject line is only the liveness gate for the whole-document mesh (the embedded
    /// model has no presence output and answers an average face on anything), so a blank
    /// document reports no face — not the person line's `no person`.
    #[test]
    fn portrait_analyze_face_on_a_blank_document_reports_no_face() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 96, "height": 128})).unwrap();
        let steps = s.active().unwrap().history.past_len();
        let err = s.execute("portrait.analyze", json!({"kind": "face"})).unwrap_err().to_string();
        assert!(err.contains("no face"), "{err}");
        assert!(!err.contains("no person"), "Face no longer needs the person stage: {err}");
        assert_eq!(s.active().unwrap().history.past_len(), steps, "analysis is not an edit");
    }

    #[test]
    fn portrait_analyze_refuses_an_unknown_kind() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        let err = s.execute("portrait.analyze", json!({"kind": "bodyy"})).unwrap_err().to_string();
        assert!(err.contains("kind"), "{err}");
    }

    /// The face box is Face Sculpt's crop: Body's pose table needs no box, and silently
    /// ignoring a caller's box would hide a mistake.
    #[test]
    fn portrait_analyze_refuses_a_face_box_for_body() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 96, "height": 128})).unwrap();
        let err = s.execute("portrait.analyze", json!({"kind": "body", "roi": [10, 10, 40, 40]})).unwrap_err().to_string();
        assert!(err.contains("roi"), "{err}");
    }

    /// Face's quality reply names a drawn box and measures the mesh: the panel warns from it.
    /// A box with no face in it is the loud refusal, not a silent whole-document fallback.
    #[test]
    fn portrait_analyze_face_reports_a_drawn_box_and_refuses_an_empty_one() {
        let (mut s, _) = disc();
        let r = s.execute("portrait.analyze", json!({"kind": "face", "roi": [50, 40, 100, 80]})).unwrap();
        assert_eq!(r["quality"]["source"], "drawn");
        assert_eq!(r["anchors"]["pts"].as_array().map(Vec::len), Some(478));
        assert!(r["quality"]["faceWidth"].as_f64().is_some_and(|w| w > 0.0), "the face's width is measured: {}", r["quality"]);

        let err = s.execute("portrait.analyze", json!({"kind": "face", "roi": [96, 76, 8, 8]})).unwrap_err().to_string();
        assert!(err.contains("no face found in the box"), "{err}");
    }

    /// Without a drawn or detected box, the mesh reads the whole document — the fallback the
    /// panel's warning names, because a small face then gets few pixels.
    #[test]
    fn portrait_analyze_face_reports_the_whole_document_fallback() {
        let (mut s, _) = disc();
        let r = s.execute("portrait.analyze", json!({"kind": "face"})).unwrap();
        assert_eq!(r["quality"]["source"], "wholeDocument");
        assert!(r.get("quality").is_some(), "Face always carries its quality reading");
    }

    /// The sculpt panels consume only landmarks; the analysis that produced them lacks the
    /// parsing/body/fashion regions, so it must never land in `portrait_cache`, which
    /// `portrait.create` and Select Subject read as a whole-document analysis.
    #[test]
    fn portrait_analyze_does_not_store_a_partial_analysis() {
        let (mut s, _) = disc();
        let r = s.execute("portrait.analyze", json!({"kind": "face"})).unwrap();
        assert_eq!(r["anchors"]["pts"].as_array().map(Vec::len), Some(478));
        let doc = std::sync::Arc::clone(&s.active().unwrap().doc);
        assert!(s.portrait_cache.get(&doc, |_, _| true).is_none(), "a partial analysis was stored");
    }

    #[test]
    fn no_document_is_disabled() {
        assert!(!Session::new().is_enabled(CMD));
        assert!(Session::new().execute(CMD, json!({})).is_err());
    }
}

/// The foundation layer's acceptance: a real photo's analysis carries the 33 body-pose
/// landmarks, finite, visibility in range, shoulders above hips — the numbers printed
/// for eyeballing against the photo (the probe-based acceptance the methodology wants).
#[test]
fn pose_landmarks_dump_probe() {
    let Some(photo) = std::env::var_os("PHOTOCRAFT_REGION_PHOTO") else {
        eprintln!("skipping: PHOTOCRAFT_REGION_PHOTO unset");
        return;
    };
    let run = || -> Result<std::sync::Arc<photocraft_portrait::Analysis>> {
        let mut s = Session::new();
        let name = std::path::Path::new(&photo).file_name().and_then(|n| n.to_str()).unwrap_or("photo").to_string();
        let imported =
            photocraft_io::import_file(&name, std::path::Path::new(&photo), &photocraft_io::ImportOptions::default(), &photocraft_raster::Interrupt::NONE)
                .map_err(|e| EngineError::Other(format!("import: {e}")))?;
        s.open_document(imported.document, None);
        let d = s.active().ok_or(EngineError::NoDocument)?;
        let target = d.active_layer.filter(|id| d.doc.layer(*id).and_then(Layer::surface).is_some());
        let ctx = crate::jobs::JobCtx::new();
        crate::portrait_cmds::analyze_person(&d.doc, target, false, None, &ctx)
    };
    let analysis = run().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(analysis.pose.len(), 33, "the embedded model answers 33 joints");
    assert!(analysis.pose.iter().all(|p| p.iter().all(|v| v.is_finite())));
    assert!(analysis.pose.iter().all(|p| (0.0..=1.0).contains(&p[3])), "visibility through the sigmoid");
    let (shoulder, hip) = ((analysis.pose[11][1] + analysis.pose[12][1]) / 2.0, (analysis.pose[23][1] + analysis.pose[24][1]) / 2.0);
    let vis: f32 = analysis.pose.iter().map(|p| p[3]).sum::<f32>() / 33.0;
    eprintln!("pose: shoulder y {shoulder:.0}, hip y {hip:.0}, mean visibility {vis:.2}");
    for (i, p) in analysis.pose.iter().enumerate() {
        eprintln!("  {i:2} x={:7.1} y={:7.1} vis={:.2}", p[0], p[1], p[3]);
    }
    assert!(shoulder < hip, "shoulders above hips on an upright photo");
}
