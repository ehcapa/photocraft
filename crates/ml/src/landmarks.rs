//! The FaceMesh plan (478 landmarks incl. iris): build once, run per image, weights
//! embedded (5.1MB, under the commit line — the u2netp precedent). Native only.
//!
//! Input contract (docs/fork/2026-11-18-portrait-pipeline.md, spike of 2026-11-18):
//! 0..1 RGB stretched to 256², **NHWC** `[1, 256, 256, 3]`; the output is 1434 floats
//! (`478 × (x, y, z)`) in 0..256 input-pixel space, rescaled here to the caller's size and
//! to 0..1 normalised coordinates.

use std::sync::OnceLock;

use tract_onnx::prelude::*;

/// The side the export expects (`1×256×256×3`).
pub const LANDMARK_SIDE: usize = 256;
/// MediaPipe FaceMesh with the iris refinement: 478 points.
pub const LANDMARK_COUNT: usize = 478;

type Plan = std::sync::Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>;

const _: fn() = {
    fn shared<T: Send + Sync>() {}
    let proof: fn() = shared::<Plan>;
    proof
};

static PLAN: OnceLock<Option<Plan>> = OnceLock::new();

fn plan() -> Option<&'static Plan> {
    PLAN.get_or_init(build).as_ref()
}

fn build() -> Option<Plan> {
    let bytes: &[u8] = include_bytes!("../data/facemesh.onnx");
    let typed = tract_onnx::onnx().model_for_read(&mut &*bytes).ok()?.into_typed().ok()?;
    let typed = typed.clone().into_optimized().unwrap_or(typed);
    typed.into_runnable().ok()
}

/// The 478 landmarks of `px` (`w × h` row-major RGB, `0..=1`), normalised to that size.
/// `None` when the samples do not match the size or inference failed.
pub fn face_landmarks(px: &[[f32; 3]], w: usize, h: usize) -> Option<Vec<super::Landmark>> {
    if w == 0 || h == 0 || px.len() != w * h {
        return None;
    }
    let plan = plan()?;
    let scaled = super::resample::resize::<3>(px, w, h, LANDMARK_SIDE, LANDMARK_SIDE);
    // NHWC [1, 256, 256, 3], values 0..1 as the export expects.
    let mut input = vec![0.0f32; 3 * LANDMARK_SIDE * LANDMARK_SIDE];
    for (i, p) in scaled.iter().enumerate() {
        for c in 0..3 {
            input[i * 3 + c] = p[c];
        }
    }
    let tensor = Tensor::from_shape(&[1, LANDMARK_SIDE, LANDMARK_SIDE, 3], &input).ok()?;
    let outputs = plan.run(tvec![tensor.into()]).ok()?;
    let raw: Vec<f32> = outputs.into_iter().next()?.to_plain_array_view::<f32>().ok()?.iter().copied().collect();
    if raw.len() != LANDMARK_COUNT * 3 {
        return None;
    }
    // 0..256 input-pixel space → the caller's normalised space.
    let side = LANDMARK_SIDE as f32;
    Some(raw.chunks(3).map(|c| super::Landmark { x: c[0] / side * w as f32, y: c[1] / side * h as f32, z: c[2] }).collect())
}
