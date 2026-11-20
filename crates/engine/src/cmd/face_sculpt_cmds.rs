//! Filter › Face Sculpt: the face panel over FaceMesh — the body panel's twin
//! ([`crate::sculpt_cmds`]), down to the plumbing: the same portrait-cache analysis, the
//! same params-embedded anchors (the 478 mesh points, so a stored smart filter re-renders
//! with no analysis at composite time), the same run_on_layer ride for a linked mask, the
//! same opt-out of the live preview and the remembered values. The parts table lives in
//! [`photocraft_algo::face_sculpt`] over the four primitives the body table defined.

use photocraft_algo::contour;
use photocraft_algo::face_sculpt::{FaceAnchors, FacePreset, FaceSides, FaceSliders, FaceSpec, SYMMETRY_PAIRS, face_field};
use photocraft_algo::liquify::{apply_liquify, auto_cell};
use photocraft_doc::LayerContent;
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::photo_cmds::Stopwatch;
use crate::{EngineError, Result, Session};

pub const FACE_SCULPT: &str = "filter.faceSculpt";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: FACE_SCULPT.into(), msg: msg.into() }
}

fn slider(p: &Value, key: &str) -> Result<f32> {
    let raw = match p.get(key) {
        None => 0.0,
        Some(v) => v.as_f64().ok_or_else(|| bad(format!("`{key}` must be a number in -100..100")))?,
    };
    Ok(raw.clamp(-100.0, 100.0) as f32)
}

fn sliders(p: &Value) -> Result<FaceSliders> {
    Ok(FaceSliders {
        face_slim: slider(p, "faceSlim")?,
        temples: slider(p, "temples")?,
        cheekbones: slider(p, "cheekbones")?,
        jaw: slider(p, "jaw")?,
        face_size: slider(p, "faceSize")?,
        v_face: slider(p, "vFace")?,
        face_width: slider(p, "faceWidth")?,
        hairline: slider(p, "hairline")?,
        forehead_width: slider(p, "foreheadWidth")?,
        forehead_height: slider(p, "foreheadHeight")?,
        philtrum: slider(p, "philtrum")?,
        mid_face: slider(p, "midFace")?,
        lower_face: slider(p, "lowerFace")?,
        chin_length: slider(p, "chinLength")?,
        chin_point: slider(p, "chinPoint")?,
        chin_retract: slider(p, "chinRetract")?,
        eye_size: slider(p, "eyeSize")?,
        pupil_size: slider(p, "pupilSize")?,
        eye_height: slider(p, "eyeHeight")?,
        eye_width: slider(p, "eyeWidth")?,
        eye_spacing: slider(p, "eyeSpacing")?,
        inner_corner: slider(p, "innerCorner")?,
        outer_corner: slider(p, "outerCorner")?,
        lower_lid: slider(p, "lowerLid")?,
        eye_tilt: slider(p, "eyeTilt")?,
        eye_position: slider(p, "eyePosition")?,
        nose_size: slider(p, "noseSize")?,
        nose_height: slider(p, "noseHeight")?,
        nose_shift: slider(p, "noseShift")?,
        nose_bridge: slider(p, "noseBridge")?,
        nose_wings: slider(p, "noseWings")?,
        nose_tip: slider(p, "noseTip")?,
        mouth_size: slider(p, "mouthSize")?,
        mouth_width: slider(p, "mouthWidth")?,
        mouth_height: slider(p, "mouthHeight")?,
        mouth_shift: slider(p, "mouthShift")?,
        gum_smile: slider(p, "gumSmile")?,
        lip_tilt: slider(p, "lipTilt")?,
        upper_lip: slider(p, "upperLip")?,
        lower_lip: slider(p, "lowerLip")?,
        brow_head: slider(p, "browHead")?,
        brow_tail: slider(p, "browTail")?,
        brow_thickness: slider(p, "browThickness")?,
        brow_spacing: slider(p, "browSpacing")?,
        brow_tilt: slider(p, "browTilt")?,
        brow_shape: slider(p, "browShape")?,
        brow_height: slider(p, "browHeight")?,
        preset: match p.get("preset").and_then(Value::as_str) {
            None | Some("none") => FacePreset::None,
            Some(name) => {
                FacePreset::parse(name).ok_or_else(|| bad(format!("unknown preset \"{name}\" (none|classic|youthful|heroic|juvenile|round|long|square)")))?
            }
        },
        preset_strength: p.get("presetStrength").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, 100.0) as f32,
        sides: sides_of(p)?,
    })
}

/// One optional side override: absent is `None` (the side follows its base slider),
/// a non-number is refused.
fn side_opt(p: &Value, key: &str) -> Result<Option<f32>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let x = v.as_f64().ok_or_else(|| bad(format!("`{key}` must be a number in -100..100")))?;
            Ok(Some(x.clamp(-100.0, 100.0) as f32))
        }
    }
}

/// The per-side overrides, the poor man's lock: a side left unset follows its base slider,
/// so the pair moves together until one side is given its own value.
fn sides_of(p: &Value) -> Result<FaceSides> {
    let mut s = FaceSides::default();
    let pair = |l: &str, r: &str, slot: &mut [Option<f32>; 2]| -> Result<()> {
        slot[0] = side_opt(p, l)?;
        slot[1] = side_opt(p, r)?;
        Ok(())
    };
    pair("templesLeft", "templesRight", &mut s.temples)?;
    pair("cheekbonesLeft", "cheekbonesRight", &mut s.cheekbones)?;
    pair("eyeSizeLeft", "eyeSizeRight", &mut s.eye_size)?;
    pair("pupilSizeLeft", "pupilSizeRight", &mut s.pupil_size)?;
    pair("eyeHeightLeft", "eyeHeightRight", &mut s.eye_height)?;
    pair("eyeWidthLeft", "eyeWidthRight", &mut s.eye_width)?;
    pair("eyeSpacingLeft", "eyeSpacingRight", &mut s.eye_spacing)?;
    pair("innerCornerLeft", "innerCornerRight", &mut s.inner_corner)?;
    pair("outerCornerLeft", "outerCornerRight", &mut s.outer_corner)?;
    pair("lowerLidLeft", "lowerLidRight", &mut s.lower_lid)?;
    pair("eyeTiltLeft", "eyeTiltRight", &mut s.eye_tilt)?;
    pair("eyePositionLeft", "eyePositionRight", &mut s.eye_position)?;
    pair("browHeadLeft", "browHeadRight", &mut s.brow_head)?;
    pair("browTailLeft", "browTailRight", &mut s.brow_tail)?;
    pair("browThicknessLeft", "browThicknessRight", &mut s.brow_thickness)?;
    pair("browSpacingLeft", "browSpacingRight", &mut s.brow_spacing)?;
    pair("browTiltLeft", "browTiltRight", &mut s.brow_tilt)?;
    pair("browShapeLeft", "browShapeRight", &mut s.brow_shape)?;
    pair("browHeightLeft", "browHeightRight", &mut s.brow_height)?;
    pair("mouthWidthLeft", "mouthWidthRight", &mut s.mouth_width)?;
    pair("lipTiltLeft", "lipTiltRight", &mut s.lip_tilt)?;
    Ok(s)
}

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

/// The stored form: sliders, `anchors` (the mesh points) and `meshSize`. `None` when the
/// anchors are absent or malformed.
fn stored_spec(p: &Value, canvas: Rect) -> Option<(FaceSpec, f64)> {
    let anchors: FaceAnchors = serde_json::from_value(p.get("anchors")?.clone()).ok()?;
    Some((FaceSpec { sliders: sliders(p).ok()?, anchors }, cell(p, canvas).ok()?))
}

/// The mesh anchors of an analysis: the 478 points as `x, y` document pixels. `None` when
/// the mesh stage never answered (no face found).
pub(crate) fn anchors_from_analysis(a: &photocraft_portrait::Analysis) -> Option<FaceAnchors> {
    if a.landmarks.len() != photocraft_algo::face_sculpt::MESH_COUNT {
        return None;
    }
    let pts = a.landmarks.iter().map(|q| [q[0], q[1], q[2]]).collect();
    Some(FaceAnchors { doc: [a.width as f32, a.height as f32], pts })
}

/// One 3D塑颜 angle: the slider's ±100 maps to ±8° (yaw, pitch) or ±6° (roll) — the
/// range where the orthographic read of a turn stays honest.
fn angle_of(p: &Value, key: &str, max_deg: f64) -> f64 {
    p.get(key).and_then(Value::as_f64).unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0 * max_deg
}

/// The symmetry amount (对称液化·五官, one-directional 0..100).
fn symmetry_of(p: &Value) -> f32 {
    p.get("symmetry").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, 100.0) as f32
}

/// The smoothing amount (AI 面部流畅, one-directional 0..100).
fn smooth_of(p: &Value) -> f32 {
    p.get("smooth").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, 100.0) as f32
}

fn contour_of(p: &Value) -> Option<Vec<[f64; 2]>> {
    serde_json::from_value::<Vec<[f64; 2]>>(p.get("contour")?.clone()).ok().filter(|c| c.len() >= 8)
}

/// The face's contour: the FACE_OVAL ring, ordered and closed — the same list the head
/// anchor's centroid comes from.
fn oval_ring(anchors: &FaceAnchors) -> Vec<[f64; 2]> {
    photocraft_portrait::face::FACE_OVAL.iter().map(|&i| [anchors.pts[i][0] as f64, anchors.pts[i][1] as f64]).collect()
}

fn stamp_ring(field: &mut photocraft_algo::liquify::LiquifyField, pts: &[[f64; 2]], smooth: f32) {
    let len: f64 = pts.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum::<f64>()
        + (pts[0][0] - pts[pts.len() - 1][0]).hypot(pts[0][1] - pts[pts.len() - 1][1]);
    contour::stamp(field, pts, len / pts.len() as f64, f64::from(smooth));
}

fn face_sculpt(s: &mut Session, p: &Value) -> Result<Value> {
    let values = sliders(p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let id = crate::commands::layer_param(s, p)?;
    let smooth = smooth_of(p);
    let symmetry = symmetry_of(p);
    let poses = angle_of(p, "poseYaw", 8.0) + angle_of(p, "posePitch", 8.0) + angle_of(p, "poseRoll", 6.0);
    if values == FaceSliders::default() && smooth <= 0.0 && symmetry <= 0.0 && poses.abs() < 1e-6 && values.preset == FacePreset::None {
        return Ok(json!({"layer": id.0, "changed": false}));
    }
    let cell = cell(p, canvas)?;
    let all_layers = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let (doc, analysis) = {
        let d = s.active().ok_or(EngineError::NoDocument)?;
        let target = d.active_layer.filter(|l| !all_layers && d.doc.layer(*l).and_then(photocraft_doc::Layer::surface).is_some());
        match s.portrait_cache.get(&d.doc, crate::layer_multi_cmds::same_pixels) {
            Some(hit) => (std::sync::Arc::clone(&d.doc), hit),
            None => {
                let ctx = crate::jobs::JobCtx::new();
                let analysis = crate::portrait_cmds::analyze_person(&d.doc, target, all_layers, None, &ctx)?;
                (std::sync::Arc::clone(&d.doc), analysis)
            }
        }
    };
    s.portrait_cache.store(&doc, (*analysis).clone());
    let Some(anchors) = anchors_from_analysis(&analysis) else {
        return Err(EngineError::Other("no face found — Face Sculpt needs the face-mesh landmarks".into()));
    };
    let t0 = Stopwatch::start();
    let ring = oval_ring(&anchors);
    let spec = FaceSpec { sliders: values, anchors };
    let (mut field, mut report) = face_field(&spec, canvas, cell);
    // 3D塑颜: the mesh's own depth rotates — the far side compresses, the near opens.
    let (yaw, pitch, roll) = (angle_of(p, "poseYaw", 8.0), angle_of(p, "posePitch", 8.0), angle_of(p, "poseRoll", 6.0));
    if yaw.abs() + pitch.abs() + roll.abs() > 1e-6 {
        let pts3: Vec<[f64; 3]> = spec.anchors.pts.iter().map(|q| [q[0] as f64, q[1] as f64, q[2] as f64]).collect();
        let eye_w = (pts3[263][0] - pts3[362][0]).abs().max(8.0);
        contour::pose_stamp(&mut field, &pts3, yaw, pitch, roll, eye_w * 1.5);
        report.applied.push("pose");
    }
    // 对称液化: the paired features align toward each other's mirror about the face's
    // axis — pure geometry of the stored mesh, so the re-render replays it for free.
    if symmetry > 0.0 {
        let pts: Vec<[f64; 2]> = spec.anchors.pts.iter().map(|q| [q[0] as f64, q[1] as f64]).collect();
        let axis = [(pts[454][0] + pts[234][0]) * 0.5, (pts[10][1] + pts[152][1]) * 0.5];
        let up = [pts[10][0] - pts[152][0], pts[10][1] - pts[152][1]];
        let eye_w = (pts[263][0] - pts[362][0]).abs().max(8.0);
        contour::align_pairs(&mut field, &pts, SYMMETRY_PAIRS, axis, up, f64::from(symmetry), eye_w * 2.0);
        report.applied.push("symmetry");
    }
    // AI 面部流畅: the face oval's own contour, low-passed and stamped over the parts.
    if smooth > 0.0 {
        match contour_of(p).or_else(|| Some(ring.clone())) {
            Some(pts) => {
                stamp_ring(&mut field, &pts, smooth);
                report.applied.push("smooth");
            }
            None => report.skipped.push(("smooth", "no face contour to relax")),
        }
    }
    let smart = matches!(s.active().and_then(|d| d.doc.layer(id)).map(|l| &l.content), Some(LayerContent::Smart(_)));
    if field.is_identity() && !smart {
        return Ok(json!({"layer": id.0, "changed": false, "parts": parts(&report)}));
    }
    let stored_contour = if smooth > 0.0 { Some(contour_of(p).unwrap_or_else(|| ring.clone())) } else { None };
    let mut stored = p.clone();
    if let Value::Object(m) = &mut stored {
        m.insert("anchors".into(), serde_json::to_value(&spec.anchors).map_err(|e| bad(format!("the anchors would not serialise: {e}")))?);
        if let Some(pts) = &stored_contour {
            m.insert("contour".into(), serde_json::to_value(pts).map_err(|e| bad(format!("the contour would not serialise: {e}")))?);
        }
    }
    let r = crate::distort_cmds::run_on_layer(
        s,
        FACE_SCULPT,
        "Face Sculpt",
        &stored,
        false,
        &|surf, sel, _| {
            let out = apply_liquify(surf, &field);
            Ok(match sel {
                Some(sel) => crate::distort_cmds::mix_by_selection(surf, &out, sel)?,
                None => out,
            })
        },
        Some(&|mask, _| Ok(apply_liquify(mask, &field))),
    )?;
    let mut r = r;
    r["ms"] = json!(t0.ms());
    r["maxDisplacement"] = json!(field.max_displacement());
    r["parts"] = parts(&report);
    Ok(r)
}

fn parts(report: &photocraft_algo::face_sculpt::FaceReport) -> Value {
    json!({
        "applied": report.applied,
        "skipped": report.skipped.iter().map(|(name, why)| json!({"part": name, "why": why})).collect::<Vec<_>>(),
    })
}

/// Smart-filter hook: re-applies a stored Face Sculpt to a rendered smart object.
pub(crate) fn apply_to_surface(params: &Value, surf: &Surface, canvas: Rect) -> Option<Surface> {
    let (spec, cell) = stored_spec(params, canvas)?;
    let (mut field, _) = face_field(&spec, canvas, cell);
    let (yaw, pitch, roll) = (angle_of(params, "poseYaw", 8.0), angle_of(params, "posePitch", 8.0), angle_of(params, "poseRoll", 6.0));
    if yaw.abs() + pitch.abs() + roll.abs() > 1e-6 {
        let pts3: Vec<[f64; 3]> = spec.anchors.pts.iter().map(|q| [q[0] as f64, q[1] as f64, q[2] as f64]).collect();
        let eye_w = (pts3[263][0] - pts3[362][0]).abs().max(8.0);
        contour::pose_stamp(&mut field, &pts3, yaw, pitch, roll, eye_w * 1.5);
    }
    let symmetry = symmetry_of(params);
    if symmetry > 0.0 {
        let pts: Vec<[f64; 2]> = spec.anchors.pts.iter().map(|q| [q[0] as f64, q[1] as f64]).collect();
        let axis = [(pts[454][0] + pts[234][0]) * 0.5, (pts[10][1] + pts[152][1]) * 0.5];
        let up = [pts[10][0] - pts[152][0], pts[10][1] - pts[152][1]];
        let eye_w = (pts[263][0] - pts[362][0]).abs().max(8.0);
        contour::align_pairs(&mut field, &pts, SYMMETRY_PAIRS, axis, up, f64::from(symmetry), eye_w * 2.0);
    }
    if let Some(pts) = contour_of(params) {
        stamp_ring(&mut field, &pts, smooth_of(params));
    }
    Some(apply_liquify(surf, &field))
}

pub fn specs() -> Vec<CommandSpec> {
    const P: &str = r##"{"faceSlim":-100..100=0,"temples":-100..100=0,"cheekbones":-100..100=0,"jaw":-100..100=0,"faceSize":-100..100=0,"vFace":-100..100=0,"faceWidth":-100..100=0,"hairline":-100..100=0,"foreheadWidth":-100..100=0,"foreheadHeight":-100..100=0,"philtrum":-100..100=0,"midFace":-100..100=0,"lowerFace":-100..100=0,"chinLength":-100..100=0,"chinPoint":-100..100=0,"chinRetract":-100..100=0,"eyeSize":-100..100=0,"pupilSize":-100..100=0,"eyeHeight":-100..100=0,"eyeWidth":-100..100=0,"eyeSpacing":-100..100=0,"innerCorner":-100..100=0,"outerCorner":-100..100=0,"lowerLid":-100..100=0,"eyeTilt":-100..100=0,"eyePosition":-100..100=0,"noseSize":-100..100=0,"noseHeight":-100..100=0,"noseShift":-100..100=0,"noseBridge":-100..100=0,"noseWings":-100..100=0,"noseTip":-100..100=0,"mouthSize":-100..100=0,"mouthWidth":-100..100=0,"mouthHeight":-100..100=0,"mouthShift":-100..100=0,"gumSmile":-100..100=0,"lipTilt":-100..100=0,"upperLip":-100..100=0,"lowerLip":-100..100=0,"browHead":-100..100=0,"browTail":-100..100=0,"browThickness":-100..100=0,"browSpacing":-100..100=0,"browTilt":-100..100=0,"browShape":-100..100=0,"browHeight":-100..100=0,"templesLeft":-100..100=0,"templesRight":-100..100=0,"cheekbonesLeft":-100..100=0,"cheekbonesRight":-100..100=0,"eyeSizeLeft":-100..100=0,"eyeSizeRight":-100..100=0,"pupilSizeLeft":-100..100=0,"pupilSizeRight":-100..100=0,"eyeHeightLeft":-100..100=0,"eyeHeightRight":-100..100=0,"eyeWidthLeft":-100..100=0,"eyeWidthRight":-100..100=0,"eyeSpacingLeft":-100..100=0,"eyeSpacingRight":-100..100=0,"innerCornerLeft":-100..100=0,"innerCornerRight":-100..100=0,"outerCornerLeft":-100..100=0,"outerCornerRight":-100..100=0,"lowerLidLeft":-100..100=0,"lowerLidRight":-100..100=0,"eyeTiltLeft":-100..100=0,"eyeTiltRight":-100..100=0,"eyePositionLeft":-100..100=0,"eyePositionRight":-100..100=0,"browHeadLeft":-100..100=0,"browHeadRight":-100..100=0,"browTailLeft":-100..100=0,"browTailRight":-100..100=0,"browThicknessLeft":-100..100=0,"browThicknessRight":-100..100=0,"browSpacingLeft":-100..100=0,"browSpacingRight":-100..100=0,"browTiltLeft":-100..100=0,"browTiltRight":-100..100=0,"browShapeLeft":-100..100=0,"browShapeRight":-100..100=0,"browHeightLeft":-100..100=0,"browHeightRight":-100..100=0,"mouthWidthLeft":-100..100=0,"mouthWidthRight":-100..100=0,"lipTiltLeft":-100..100=0,"lipTiltRight":-100..100=0,"preset":"none|classic|youthful|heroic|juvenile|round|long|square"="none","presetStrength":0..100=60 (a named bundle of the sliders, stacked with the manual ones; each Left/Right pair stays linked until one side is given its own value — the lock, without the widget),"poseYaw":-100..100=0,"posePitch":-100..100=0,"poseRoll":-100..100=0 (3D塑颜: the mesh's own depth rotates — ±100 is ±8° yaw/pitch, ±6° roll),"symmetry":0..100=0 (对称液化·五官: the paired features align toward each other's mirror about the face's axis),"smooth":0..100=0 (AI 面部流畅: the face oval contour, low-passed and relaxed; the ring is embedded from the mesh at apply time, like the anchors) (the face-mesh analysis runs first — cached like Select Region, seconds the first time; each slider's positive direction is its label's action),"anchors":{…}? (embedded from that analysis and stored with the smart filter so re-renders replay exactly; never typed),"meshSize":px?,"sampleAllLayers":bool=false,"layer":id?} — a selection limits the effect; on a smart object it becomes a smart filter"##;
    vec![CommandSpec {
        id: FACE_SCULPT,
        label: "Face Sculpt…",
        menu: &["Filter"],
        shortcut: None,
        params: P,
        enabled: crate::filters::has_filterable_layer,
        journal: true,
        run: face_sculpt,
    }]
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::Session;

    #[test]
    fn zero_sliders_change_nothing_without_an_analysis() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 96, "height": 128, "background": "white"})).unwrap();
        let r = s.execute(FACE_SCULPT, json!({})).unwrap();
        assert_eq!(r, json!({"layer": s.active().unwrap().active_layer.unwrap().0, "changed": false}));
    }

    #[test]
    fn a_personless_photo_reports_no_face() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 96, "height": 128, "background": "white"})).unwrap();
        let err = s.execute(FACE_SCULPT, json!({"eyeSize": 30})).unwrap_err().to_string();
        // A blank image the analysis refuses before its mesh stage ("no person"); a photo
        // with a person but no detectable face would answer the mesh stage's own wording.
        assert!(err.contains("no person") || err.contains("no face"), "{err}");
        assert!(s.execute("edit.undo", json!({})).is_err());
    }

    #[test]
    fn stored_params_replay_the_same_field() {
        let mut pts = vec![[256.0f32, 240.0f32, 0.0f32]; photocraft_algo::face_sculpt::MESH_COUNT];
        for &(i, x, y) in &[(10usize, 256.0, 96.0), (152, 256.0, 392.0), (454, 376.0, 248.0), (234, 136.0, 248.0)] {
            pts[i] = [x as f32, y as f32, 0.0];
        }
        let anchors = FaceAnchors { doc: [512.0, 512.0], pts };
        let params = json!({"eyeSize": -25.0, "faceSlim": 40.0, "anchors": serde_json::to_value(&anchors).unwrap(), "meshSize": 4});
        let bounds = Rect::new(0, 0, 512, 512);
        let (spec, _) = stored_spec(&params, bounds).unwrap();
        assert_eq!(spec.anchors, anchors, "the anchors survive the params round-trip");
        let f1 = face_field(&spec, bounds, 4.0).0;
        let f2 = face_field(&stored_spec(&params, bounds).unwrap().0, bounds, 4.0).0;
        assert!(!f1.is_identity());
        assert_eq!(f1, f2, "the same params must rebuild the same field");
        assert!(stored_spec(&json!({"eyeSize": -25.0}), bounds).is_none());
    }

    /// The real-photo acceptance: `PHOTOCRAFT_SCULPT_PHOTO=<image>` (the same env as the
    /// body panel's — one photo, both panels), the group table for every slider alone at
    /// +100, a footprint check (Eye Size moves the eye band and not the chin), the
    /// smart-object replay and an export for eyeballing (`/tmp/face_sculpt_out.png`).
    /// Skipped without the env var, like the body harness.
    #[test]
    fn real_photo_face_sculpt_diagnostic() {
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
            let (canvas, analysis, layer) = {
                let d = s.active().ok_or(EngineError::NoDocument)?;
                let target = d.active_layer.filter(|l| d.doc.layer(*l).and_then(photocraft_doc::Layer::surface).is_some());
                let ctx = crate::jobs::JobCtx::new();
                let analysis = crate::portrait_cmds::analyze_person(&d.doc, target, false, None, &ctx)?;
                let layer =
                    target.and_then(|l| d.doc.layer(l)).and_then(|l| l.surface()).cloned().ok_or_else(|| EngineError::Other("no pixel layer".into()))?;
                (d.doc.bounds(), analysis, layer)
            };
            let anchors = anchors_from_analysis(&analysis).ok_or_else(|| EngineError::Other("no face mesh in the photo".into()))?;
            let cell = auto_cell(canvas);
            let one = |name: &str, v: f32| {
                eprintln!(
                    "{name:16} max {:.1}px",
                    face_field(&FaceSpec { sliders: with_one(name, v), anchors: anchors.clone() }, canvas, cell).0.max_displacement()
                )
            };
            for name in [
                "faceSlim",
                "temples",
                "cheekbones",
                "jaw",
                "faceSize",
                "vFace",
                "faceWidth",
                "hairline",
                "foreheadWidth",
                "foreheadHeight",
                "philtrum",
                "midFace",
                "lowerFace",
                "chinLength",
                "chinPoint",
                "chinRetract",
                "eyeSize",
                "pupilSize",
                "eyeHeight",
                "eyeWidth",
                "eyeSpacing",
                "innerCorner",
                "outerCorner",
                "lowerLid",
                "eyeTilt",
                "eyePosition",
                "noseSize",
                "noseHeight",
                "noseShift",
                "noseBridge",
                "noseWings",
                "noseTip",
                "mouthSize",
                "mouthWidth",
                "mouthHeight",
                "mouthShift",
                "gumSmile",
                "lipTilt",
                "upperLip",
                "lowerLip",
                "browHead",
                "browTail",
                "browThickness",
                "browSpacing",
                "browTilt",
                "browShape",
                "browHeight",
            ] {
                one(name, 100.0);
            }
            // The footprint gauge: Eye Size moves its band and not the chin's.
            let eye = [anchors.pts[468][0], anchors.pts[468][1]];
            let chin = [anchors.pts[152][0], anchors.pts[152][1]];
            let changed = |before: &Surface, after: &Surface, c: [f32; 2], r: i32| -> f64 {
                let region = Rect::new(c[0] as i32 - r, c[1] as i32 - r, c[0] as i32 + r, c[1] as i32 + r);
                let (a, b) = (before.read_region(region), after.read_region(region));
                let n = before.format().channels().min(3);
                a.chunks_exact(n).zip(b.chunks_exact(n)).filter(|(x, y)| (0..n).any(|k| (x[k] - y[k]).abs() > 4.0 / 255.0)).count() as f64 / a.len() as f64
                    * n as f64
            };
            let (eye_field, _) = face_field(&FaceSpec { sliders: with_one("eyeSize", -100.0), anchors: anchors.clone() }, canvas, cell);
            let warped = apply_liquify(&layer, &eye_field);
            let r = ((anchors.pts[454][0] - anchors.pts[234][0]).abs() * 0.07) as i32;
            let (eye_band, chin_band) = (changed(&layer, &warped, eye, r.max(6)), changed(&layer, &warped, chin, r.max(6)));
            eprintln!("eyeSize footprint: eye {eye_band:.3}, chin {chin_band:.3}");
            assert!(eye_band > 0.02, "Eye Size must move the eye band: {eye_band:.3}");
            assert!(chin_band < eye_band / 3.0, "the warp must stay off the chin: {chin_band:.3} vs {eye_band:.3}");
            // The smart object stores the mesh and replays bit-identically.
            s.execute("select.all", json!({}))?;
            s.execute("layer.smartObjects.convertToSmartObject", json!({}))?;
            let r = s.execute(FACE_SCULPT, json!({"eyeSize": -20, "faceSlim": 25, "smooth": 60}))?;
            assert!(r["parts"]["applied"].as_array().is_some_and(|a| a.iter().any(|v| v == "smooth")), "the smooth pass must run: {r}");
            assert_eq!(r["changed"], json!(true), "{r}");
            let (params, source) = {
                let d = s.active().ok_or(EngineError::NoDocument)?;
                let id = d.active_layer.ok_or(EngineError::Other("no active layer".into()))?;
                let photocraft_doc::LayerContent::Smart(sm) = &d.doc.layer(id).ok_or(EngineError::NoLayer(id))?.content else { unreachable!("just converted") };
                (sm.smart_filters[0].params.clone(), sm.cache.clone().ok_or_else(|| EngineError::Other("no rendered source".into()))?)
            };
            assert!(params.get("anchors").is_some(), "the mesh must ride the stored params");
            let a = crate::filters::apply_filter_to_surface(FACE_SCULPT, &params, &source, source.content_bounds(), None, canvas)?
                .ok_or_else(|| EngineError::Other("the re-render path refused the stored filter".into()))?;
            let b = crate::filters::apply_filter_to_surface(FACE_SCULPT, &params, &source, source.content_bounds(), None, canvas)?
                .ok_or_else(|| EngineError::Other("the re-render path refused the stored filter".into()))?;
            assert_eq!(a.read_region(a.content_bounds()), b.read_region(b.content_bounds()), "re-render must be bit-identical");
            s.execute("layer.flattenImage", json!({}))?;
            let out = "/tmp/face_sculpt_out.png";
            s.execute("layer.quickExportAsPng", json!({"path": out}))?;
            eprintln!("exported: {out}");
            Ok(())
        };
        if let Err(e) = run() {
            panic!("face sculpt diagnostic failed: {e}");
        }
    }

    /// One slider at `v`, the rest zero.
    fn with_one(name: &str, v: f32) -> FaceSliders {
        let mut s = FaceSliders::default();
        match name {
            "faceSlim" => s.face_slim = v,
            "temples" => s.temples = v,
            "cheekbones" => s.cheekbones = v,
            "jaw" => s.jaw = v,
            "faceSize" => s.face_size = v,
            "vFace" => s.v_face = v,
            "faceWidth" => s.face_width = v,
            "hairline" => s.hairline = v,
            "foreheadWidth" => s.forehead_width = v,
            "foreheadHeight" => s.forehead_height = v,
            "philtrum" => s.philtrum = v,
            "midFace" => s.mid_face = v,
            "lowerFace" => s.lower_face = v,
            "chinLength" => s.chin_length = v,
            "chinPoint" => s.chin_point = v,
            "chinRettract" => s.chin_retract = v,
            "eyeSize" => s.eye_size = v,
            "pupilSize" => s.pupil_size = v,
            "eyeHeight" => s.eye_height = v,
            "eyeWidth" => s.eye_width = v,
            "eyeSpacing" => s.eye_spacing = v,
            "innerCorner" => s.inner_corner = v,
            "outerCorner" => s.outer_corner = v,
            "lowerLid" => s.lower_lid = v,
            "eyeTilt" => s.eye_tilt = v,
            "eyePosition" => s.eye_position = v,
            "noseSize" => s.nose_size = v,
            "noseHeight" => s.nose_height = v,
            "noseShift" => s.nose_shift = v,
            "noseBridge" => s.nose_bridge = v,
            "noseWings" => s.nose_wings = v,
            "noseTip" => s.nose_tip = v,
            "mouthSize" => s.mouth_size = v,
            "mouthWidth" => s.mouth_width = v,
            "mouthHeight" => s.mouth_height = v,
            "mouthShift" => s.mouth_shift = v,
            "gumSmile" => s.gum_smile = v,
            "lipTilt" => s.lip_tilt = v,
            "upperLip" => s.upper_lip = v,
            "lowerLip" => s.lower_lip = v,
            "browHead" => s.brow_head = v,
            "browTail" => s.brow_tail = v,
            "browThickness" => s.brow_thickness = v,
            "browSpacing" => s.brow_spacing = v,
            "browTilt" => s.brow_tilt = v,
            "browShape" => s.brow_shape = v,
            "browHeight" => s.brow_height = v,
            _ => {}
        }
        s
    }
}
