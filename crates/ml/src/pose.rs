//! The body-pose landmarks (MediaPipe Pose Landmarks Detector, BlazePose full: 33 body
//! joints, weights embedded at `data/poselandmark.onnx`) — the foundation layer of
//! docs/fork/2026-11-20: the points land on [`Analysis`](photocraft_portrait::Analysis)
//! for the body-aware work of a later iteration; nothing consumes them yet beyond the
//! diagnostics. Native only, tract tier, ~35 ms a run.
//!
//! Preprocessing contract (the export's own README, probe-confirmed 2026-11-20): NHWC
//! `[1, 256, 256, 3]`, `0..1` RGB, stretched — the caller hands the **square person crop**
//! (MediaPipe's own pipeline rotates its detector ROI first; v1 takes the axis-aligned
//! square about the person matte's bounds upright, the pose_probe verdict: plausible on
//! upright photos). The answer's first output is `[1, 195]` = 39 × (x, y, z, visibility,
//! presence) in 256-crop pixels, indices 0–32 the real body joints; the visibility field
//! is a **raw logit** — the sigmoid is ours to apply (a probe reads 8.5 on visible joints,
//! −0.8 on occluded ones).

use std::sync::OnceLock;

use tract_onnx::prelude::*;

/// The side the export expects (`1×256×256×3`).
pub const POSE_SIDE: usize = 256;
/// The 33 body joints (0 nose … 32 right foot index); the export carries 39 rows, the
/// last six auxiliary.
pub const POSE_COUNT: usize = 33;
/// The export's landmark rows (33 body + 6 auxiliary).
pub const RAW_COUNT: usize = 39;
/// The fields per landmark row: x, y, z, visibility (raw logit), presence.
pub const POSE_STRIDE: usize = 5;

/// The compiled plan, `into_runnable`'s `Arc` (the sharing proof the other tiers carry).
type Plan = std::sync::Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>;

const _: fn() = {
    fn shared<T: Send + Sync>() {}
    shared::<Plan>
};

static PLAN: OnceLock<Option<Plan>> = OnceLock::new();

/// The process-wide plan over the embedded weights, built on first use. A load failure is
/// cached as `None`: re-reading would re-pay the same miss.
fn plan() -> Option<&'static Plan> {
    PLAN.get_or_init(build).as_ref()
}

fn build() -> Option<Plan> {
    let bytes: &[u8] = include_bytes!("../data/poselandmark.onnx");
    let proto = tract_onnx::onnx().model_for_read(&mut &*bytes).ok()?;
    let proto = proto.with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, [1usize, POSE_SIDE, POSE_SIDE, 3])).ok()?;
    let typed = proto.into_typed().ok()?;
    let typed = typed.clone().into_optimized().unwrap_or(typed);
    typed.into_runnable().ok()
}

/// One body landmark: `x`/`y` in the caller's pixel space, `z` MediaPipe's rough depth
/// relative to the hip centre, `visibility` 0..=1 (the export's logit through the sigmoid).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyLandmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub visibility: f32,
}

/// The 33 body landmarks of the square person crop `px` (`w × h` row-major RGB, `0..=1`),
/// in the crop's own pixel space — the caller maps them to the document. `None` when this
/// build has no model or inference failed; the model always answers 33 rows (FaceMesh's
/// semantics).
pub fn pose_landmarks(px: &[[f32; 3]], w: usize, h: usize) -> Option<Vec<BodyLandmark>> {
    if w == 0 || h == 0 || px.len() != w * h {
        return None;
    }
    let plan = plan()?;
    let scaled = super::resample::resize::<3>(px, w, h, POSE_SIDE, POSE_SIDE);
    // NHWC, 0..1 — the tflite heritage's contract, FaceMesh's own.
    let mut input = vec![0.0f32; 3 * POSE_SIDE * POSE_SIDE];
    for (i, p) in scaled.iter().enumerate() {
        for c in 0..3 {
            input[i * 3 + c] = p[c];
        }
    }
    let tensor = Tensor::from_shape(&[1, POSE_SIDE, POSE_SIDE, 3], &input).ok()?;
    let outputs = plan.run(tvec![tensor.into()]).ok()?;
    // The first output is the [1, 39×5] screen-landmark row (the export's four further
    // outputs — presence, segmentation, heatmap, world landmarks — are not ours to keep).
    let raw: Vec<f32> = outputs.into_iter().next()?.to_plain_array_view::<f32>().ok()?.iter().copied().collect();
    if raw.len() != RAW_COUNT * POSE_STRIDE {
        return None;
    }
    Some(
        (0..POSE_COUNT)
            .map(|i| BodyLandmark {
                x: raw[i * POSE_STRIDE] / POSE_SIDE as f32 * w as f32,
                y: raw[i * POSE_STRIDE + 1] / POSE_SIDE as f32 * h as f32,
                z: raw[i * POSE_STRIDE + 2],
                // The export's raw logit; the graph's own post-processing applies the sigmoid.
                visibility: 1.0 / (1.0 + (-raw[i * POSE_STRIDE + 3]).exp()),
            })
            .collect(),
    )
}
