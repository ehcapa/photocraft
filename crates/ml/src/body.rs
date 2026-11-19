//! The whole-body six-class segmentation plan (MediaPipe selfie multiclass over its 18-class
//! fashion namesake's six portrait classes): build once, run per image, from a runtime path.
//! Native only, tract tier — the ONNX conversion of the tflite original, whose
//! `TRANSPOSE_CONV` tract-tflite cannot translate (docs/fork/models.md, spike of 2026-11-19).
//!
//! Preprocessing contract: the tflite heritage shows — **NHWC** `[1, 256, 256, 3]`, `0..1`
//! RGB, stretched onto the square (the aspect distortion is undone on the way out, like the
//! parsing model). The answer is `[1, 256, 256, 6]`, the **class axis last**, softmax already
//! baked into this export — both verified on the acceptance photo (the class-last point the
//! spike's first dump got wrong: plane-major reading scrambles every class into ~18% noise).

use std::sync::OnceLock;

use tract_onnx::prelude::*;

/// The side the export expects (`1×256×256×3`).
pub const BODY_SIDE: usize = 256;
/// The six portrait classes, in the export's channel order.
pub const BODY_CLASSES: usize = 6;
/// The channel of each class: 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes,
/// 5 others (accessories).
pub const HAIR: usize = 1;
pub const BODY_SKIN: usize = 2;
pub const FACE_SKIN: usize = 3;
pub const CLOTHES: usize = 4;

/// The compiled plan, `into_runnable`'s `Arc` (the sharing proof the other tiers carry).
type Plan = std::sync::Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>;

const _: fn() = {
    fn shared<T: Send + Sync>() {}
    shared::<Plan>
};

static PLAN: OnceLock<Option<Plan>> = OnceLock::new();

/// The process-wide plan from `PHOTOCRAFT_MODELS_DIR/selfiemulticlass.onnx`, built on first
/// use. A missing or broken model is cached as `None`: re-reading would re-pay the same miss.
fn plan() -> Option<&'static Plan> {
    PLAN.get_or_init(build).as_ref()
}

fn build() -> Option<Plan> {
    let path = super::models::locate(&super::models::SELFIE)?;
    super::models::matches(&path, &super::models::SELFIE);
    let proto = tract_onnx::onnx().model_for_path(&path).ok()?;
    let typed = proto.into_typed().ok()?;
    let typed = typed.clone().into_optimized().unwrap_or(typed);
    typed.into_runnable().ok()
}

/// The body segmentation of `px` (`w × h` row-major RGB, `0..=1`): per-class probabilities at
/// the caller's size, or `None` without a usable model. `planes[class]` is the `w × h`
/// probability plane (background excluded by the caller — the five person classes are what
/// the pipeline consumes; the background channel is 1 − the rest on-distribution anyway).
pub fn body_probs(px: &[[f32; 3]], w: usize, h: usize) -> Option<Vec<Vec<f32>>> {
    if w == 0 || h == 0 || px.len() != w * h {
        return None;
    }
    let plan = plan()?;
    let scaled = super::resample::resize::<3>(px, w, h, BODY_SIDE, BODY_SIDE);
    // NHWC, 0..1 — the tflite heritage's contract.
    let mut input = vec![0.0f32; 3 * BODY_SIDE * BODY_SIDE];
    for (i, p) in scaled.iter().enumerate() {
        for c in 0..3 {
            input[i * 3 + c] = p[c];
        }
    }
    let tensor = Tensor::from_shape(&[1, BODY_SIDE, BODY_SIDE, 3], &input).ok()?;
    let outputs = plan.run(tvec![tensor.into()]).ok()?;
    let raw: Vec<f32> = outputs.into_iter().next()?.to_plain_array_view::<f32>().ok()?.iter().copied().collect();
    if raw.len() != BODY_CLASSES * BODY_SIDE * BODY_SIDE {
        return None;
    }
    // Class-last NHWC → per-class planes at the caller's size (the background plane rides
    // back too and is dropped by the caller; resize per class like the mask resampler).
    (0..BODY_CLASSES)
        .map(|c| {
            let plane: Vec<f32> = (0..BODY_SIDE * BODY_SIDE).map(|i| raw[i * BODY_CLASSES + c]).collect();
            super::resize_mask(&plane, BODY_SIDE, BODY_SIDE, w, h)
        })
        .collect::<Option<Vec<_>>>()
}
