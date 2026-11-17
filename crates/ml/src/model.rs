//! The U²-Netᵖ plan: build once, run per image. Native only (see the crate docs).

use std::sync::OnceLock;

use tract_onnx::prelude::*;

use super::MODEL_PX;

/// The weights, embedded at build time (Apache-2.0; `data/LICENSE-u2netp.txt`).
static BYTES: &[u8] = include_bytes!("../data/u2netp.onnx");

/// rembg's normalisation of the exported graph (`rembg/sessions/base.py`): divide by the
/// image's own maximum, then per-channel ImageNet statistics. The original U²-Net repository
/// divides by 255 instead; these exported weights match rembg's scheme, which the
/// `matches_rembg` parity test locks in. Keep the three together so a switch is one edit.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The compiled plan (`into_runnable` hands out an `Arc`). tract plans run through `&self`
/// (per-call state) and the graph they hold is `Send + Sync` — asserted below, so the
/// `OnceLock` is sound however callers thread.
type Plan = std::sync::Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>;

const _: fn() = {
    fn shared<T: Send + Sync>() {}
    let proof: fn() = shared::<Plan>;
    proof
};

static PLAN: OnceLock<Option<Plan>> = OnceLock::new();

/// The process-wide plan, building it on first use. A model that fails to parse or plan is
/// cached as `None`: re-parsing on every call would re-pay the same failure each time.
fn plan() -> Option<&'static Plan> {
    PLAN.get_or_init(build).as_ref()
}

fn build() -> Option<Plan> {
    let typed = tract_onnx::onnx().model_for_read(&mut &*BYTES).ok()?.into_typed().ok()?;
    // Optimisation is best-effort: the unoptimised typed model computes the same answers.
    let typed = typed.clone().into_optimized().unwrap_or(typed);
    typed.into_runnable().ok()
}

/// The subject-saliency map of `px` (`w × h` row-major RGB, `0..=1`), resampled back to `w × h`.
pub(super) fn run(px: &[[f32; 3]], w: usize, h: usize) -> Option<Vec<f32>> {
    if w == 0 || h == 0 || px.len() != w * h {
        return None;
    }
    let side = super::MODEL_SIDE;
    let plan = plan()?;
    // rembg stretches the image onto the square input (no letterbox); the aspect distortion
    // is undone on the way out, and the coarse seed only feeds GrabCut and the re-cut.
    let scaled = super::resample::resize::<3>(px, w, h, side, side);
    let max = scaled.iter().fold(0.0f32, |m, p| m.max(p[0]).max(p[1]).max(p[2])).max(1e-6);
    // NCHW f32 `[1, 3, side, side]` with the constants above.
    let mut input = vec![0.0f32; 3 * MODEL_PX];
    for (i, p) in scaled.iter().enumerate() {
        for c in 0..3 {
            input[c * MODEL_PX + i] = (p[c] / max - MEAN[c]) / STD[c];
        }
    }
    let tensor = Tensor::from_shape(&[1, 3, side, side], &input).ok()?;
    let outputs = plan.run(tvec![tensor.into()]).ok()?;
    // The d1 head: `[1, 1, side, side]`, min-max normalised like rembg's predict(). A flat
    // output (range ~0) becomes 0.5 rather than a division by zero.
    let raw: Vec<f32> = outputs.into_iter().next()?.to_plain_array_view::<f32>().ok()?.iter().copied().collect();
    if raw.len() != MODEL_PX {
        return None;
    }
    let (lo, hi) = raw.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let range = hi - lo;
    let norm: Vec<[f32; 1]> = raw.iter().map(|v| [if range > 1e-6 { (v - lo) / range } else { 0.5 }]).collect();
    let mask = super::resample::resize::<1>(&norm, side, side, w, h);
    Some(mask.into_iter().map(|p| p[0]).collect())
}
