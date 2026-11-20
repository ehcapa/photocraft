//! SCRFD face detection (insightface det_10g, ~16MB, `PHOTOCRAFT_MODELS_DIR/facedetector.onnx`):
//! the crop stage's automatic face finder — the box the user can also draw by hand. Native
//! only, tract tier; every failure (unset, missing file, bad graph) caches as `None` and the
//! caller falls back to its no-detector path.
//!
//! Preprocessing contract (the insightface reference): letterboxed to 640² — the long side
//! scaled to 640, the short side padded symmetrically with the normaliser's floor — NCHW,
//! `(px·255 − 127.5) / 128`. The graph answers nine tensors: per stride (8, 16, 32) the
//! anchor scores `[n,1]` (post-sigmoid in this export), the ltrb distance deltas `[n,4]`
//! and the five-keypoint deltas `[n,10]`, anchors flattened `[row, col, 2-per-cell]` on a
//! shared centre `(col + 0.5)·stride, (row + 0.5)·stride` (the distance representation
//! needs no anchor size). Decoding and NMS run here; the boxes ride back through the
//! letterbox to the caller's pixels.

use std::sync::OnceLock;

use tract_onnx::prelude::*;
use tract_onnx::tract_core::prelude::DatumType;
use tract_onnx::tract_hir::infer::InferenceFact;

/// The letterbox side the export runs at (`1×3×640×640`, static in the plan).
pub const DETECT_SIDE: usize = 640;

/// One detected face: `x0..x1 × y0..y1` in the caller's pixels and the insightface
/// five keypoints (eyes, nose, mouth corners) — the alignment input the GFPGAN
/// ecosystem standardised on.
#[derive(Clone, Copy, Debug)]
pub struct FaceDet {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
    pub kps: [[f32; 2]; 5],
}

impl FaceDet {
    pub fn area(&self) -> f32 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }
}

/// The compiled plan, `into_runnable`'s `Arc` (the sharing proof the other tiers carry).
type Plan = std::sync::Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>;

const _: fn() = {
    fn shared<T: Send + Sync>() {}
    shared::<Plan>
};

static PLAN: OnceLock<Option<Plan>> = OnceLock::new();

/// The process-wide plan from `PHOTOCRAFT_MODELS_DIR/facedetector.onnx`, built on first use.
/// A missing or broken model is cached as `None`: re-reading would re-pay the same miss.
fn plan() -> Option<&'static Plan> {
    PLAN.get_or_init(build).as_ref()
}

fn build() -> Option<Plan> {
    let path = super::models::locate(&super::models::FACEDETECTOR)?;
    super::models::matches(&path, &super::models::FACEDETECTOR);
    let proto = tract_onnx::onnx().model_for_path(&path).ok()?;
    // The export's input is dynamic `1×3×h×w`; tract needs the shape pinned. The letterbox
    // always presents 640², so static is also the fastest shape to plan.
    let proto = proto.with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, [1usize, 3, DETECT_SIDE, DETECT_SIDE])).ok()?;
    let typed = proto.into_typed().ok()?;
    let typed = typed.clone().into_optimized().unwrap_or(typed);
    typed.into_runnable().ok()
}

const STRIDES: [usize; 3] = [8, 16, 32];
const SCORE_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.4;

/// The faces of `px` (`w × h` row-major RGB, `0..=1`), largest first, or `None` without a
/// usable model. No face is `Some(vec![])` — a model that ran and found nothing is an
/// answer, not a failure.
pub fn detect_faces(px: &[[f32; 3]], w: usize, h: usize) -> Option<Vec<FaceDet>> {
    if w == 0 || h == 0 || px.len() != w * h {
        return None;
    }
    let plan = plan()?;
    // Letterbox: scale the long side to 640, centre the rest on the normaliser's floor.
    let scale = DETECT_SIDE as f32 / w.max(h) as f32;
    let (tw, th) = ((w as f32 * scale).round() as usize, (h as f32 * scale).round() as usize);
    let (ox, oy) = ((DETECT_SIDE - tw) as f32 * 0.5, (DETECT_SIDE - th) as f32 * 0.5);
    let scaled = super::resample::resize::<3>(px, w, h, tw, th);
    let mut input = vec![(0.0f32 - 127.5) / 128.0; 3 * DETECT_SIDE * DETECT_SIDE];
    let (fy, fx) = (oy as usize, ox as usize);
    for c in 0..3 {
        let dst = &mut input[c * DETECT_SIDE * DETECT_SIDE..];
        for y in 0..th {
            for x in 0..tw {
                dst[(y + fy) * DETECT_SIDE + x + fx] = (scaled[y * tw + x][c] * 255.0 - 127.5) / 128.0;
            }
        }
    }
    let tensor = Tensor::from_shape(&[1, 3, DETECT_SIDE, DETECT_SIDE], &input).ok()?;
    let outputs = plan.run(tvec![tensor.into()]).ok()?;
    let flat: Vec<Vec<f32>> = outputs.iter().map(|o| o.to_plain_array_view::<f32>().map(|v| v.iter().copied().collect()).unwrap_or_default()).collect();
    if flat.len() != 9 || flat.iter().any(|v| v.is_empty()) {
        return None;
    }
    // Back through the letterbox: 640-space minus the offset, over the scale.
    let back = |v: f32, o: f32| (v - o) / scale;
    let mut faces = Vec::new();
    for level in 0..3 {
        let (stride, cells) = (STRIDES[level], DETECT_SIDE / STRIDES[level]);
        let (scores, boxes, kps) = (&flat[level], &flat[3 + level], &flat[6 + level]);
        for i in 0..scores.len() {
            if scores[i] < SCORE_THRESHOLD {
                continue;
            }
            let cell = i / 2;
            let s = stride as f32;
            let (cx, cy) = ((cell % cells) as f32 + 0.5, (cell / cells) as f32 + 0.5);
            let d = |k: usize| boxes[i * 4 + k] * s;
            let face = FaceDet {
                x0: back(cx * s - d(0), ox).max(0.0),
                y0: back(cy * s - d(1), oy).max(0.0),
                x1: back(cx * s + d(2), ox).min(w as f32),
                y1: back(cy * s + d(3), oy).min(h as f32),
                score: scores[i],
                kps: std::array::from_fn(|k| {
                    [back(cx * s + kps[i * 10 + k * 2] * s, ox).clamp(0.0, w as f32), back(cy * s + kps[i * 10 + k * 2 + 1] * s, oy).clamp(0.0, h as f32)]
                }),
            };
            if face.x1 > face.x0 && face.y1 > face.y0 {
                faces.push(face);
            }
        }
    }
    Some(nms(faces))
}

fn iou(a: &FaceDet, b: &FaceDet) -> f32 {
    let (ix, iy) = ((a.x1.min(b.x1) - a.x0.max(b.x0)).max(0.0), (a.y1.min(b.y1) - a.y0.max(b.y0)).max(0.0));
    let inter = ix * iy;
    inter / (a.area() + b.area() - inter + f32::EPSILON)
}

/// Greedy NMS: largest first, dropping everything it overlaps past the threshold.
fn nms(mut faces: Vec<FaceDet>) -> Vec<FaceDet> {
    faces.sort_by(|a, b| b.area().total_cmp(&a.area()));
    let mut kept = Vec::new();
    while !faces.is_empty() {
        let face = faces.remove(0);
        faces.retain(|other| iou(&face, other) <= NMS_THRESHOLD);
        kept.push(face);
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_keeps_the_larger_of_two_overlapping_and_both_disjoint() {
        let f = |x0, y0, s| FaceDet { x0, y0, x1: x0 + s, y1: y0 + s, score: 0.9, kps: [[0.0; 2]; 5] };
        let kept = nms(vec![f(10.0, 10.0, 100.0), f(30.0, 30.0, 70.0), f(400.0, 400.0, 60.0)]);
        assert_eq!(kept.len(), 2, "the small face inside the big one folds away");
        assert!((kept[0].area() - 100.0 * 100.0).abs() < 1.0, "largest first");
        assert!(kept[1].x0 >= 400.0);
    }
}
