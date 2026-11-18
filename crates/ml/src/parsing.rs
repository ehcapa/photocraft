//! The face-parsing plan (BiSeNet over CelebAMask-HQ, 19 classes): build once, run per
//! image, from a runtime path. Native only (see the crate docs).
//!
//! Unlike the subject model the weights are not embedded — 52.6MB sits past the commit
//! line — so the plan comes from `PHOTOCRAFT_MODELS_DIR/faceparser.onnx` (models.rs) and
//! every failure (unset, missing file, bad graph) caches as `None`: the portrait pipeline
//! runs its person slice without region masks and nothing else changes.
//!
//! Preprocessing contract (docs/fork/2026-11-18-portrait-pipeline.md, spike of 2026-11-18):
//! this export family bakes its normalisation into the graph — feed **raw 0–255** RGB,
//! stretched to 512², NCHW. The README-style ImageNet normalisation collapses it to a
//! person-shaped blob; the cross-export agreement test pins the raw contract.

use std::sync::OnceLock;

use tract_onnx::prelude::*;

/// The side the export expects (`1×3×512×512`).
pub const PARSING_SIDE: usize = 512;
/// CelebAMask-HQ classes the graph outputs.
pub const PARSING_CLASSES: usize = 19;
const PARSING_PX: usize = PARSING_SIDE * PARSING_SIDE;

/// The compiled plan, `into_runnable`'s `Arc` (same sharing proof as the subject model).
type Plan = std::sync::Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>;

const _: fn() = {
    fn shared<T: Send + Sync>() {}
    let proof: fn() = shared::<Plan>;
    proof
};

static PLAN: OnceLock<Option<Plan>> = OnceLock::new();

/// The process-wide plan from `PHOTOCRAFT_MODELS_DIR/faceparser.onnx`, built on first use.
/// A missing or broken model is cached as `None`: re-reading would re-pay the same miss.
fn plan() -> Option<&'static Plan> {
    PLAN.get_or_init(build).as_ref()
}

fn build() -> Option<Plan> {
    // PHOTOCRAFT_MODELS_DIR/faceparser.onnx is the one placement; a fingerprint mismatch
    // warns and proceeds (models.rs).
    let path = super::models::locate(&super::models::FACEPARSER)?;
    super::models::matches(&path, &super::models::FACEPARSER);
    let proto = tract_onnx::onnx().model_for_path(&path).ok()?;
    let typed = proto.into_typed().ok()?;
    let typed = typed.clone().into_optimized().unwrap_or(typed);
    typed.into_runnable().ok()
}

/// Per-class probabilities of `px` (`w × h` row-major RGB, `0..=1`), or `None` without a
/// usable model. The softmax denominators are computed in one pass over the 19 logits.
pub fn parsing_probs(px: &[[f32; 3]], w: usize, h: usize) -> Option<super::ParsingProbs> {
    if w == 0 || h == 0 || px.len() != w * h {
        return None;
    }
    let plan = plan()?;
    // Stretched onto the square input like the subject model (the aspect distortion is
    // undone on the way out); raw 0–255 per the contract above.
    let scaled = super::resample::resize::<3>(px, w, h, PARSING_SIDE, PARSING_SIDE);
    let mut input = vec![0.0f32; 3 * PARSING_PX];
    for (i, p) in scaled.iter().enumerate() {
        for c in 0..3 {
            input[c * PARSING_PX + i] = p[c] * 255.0;
        }
    }
    let tensor = Tensor::from_shape(&[1, 3, PARSING_SIDE, PARSING_SIDE], &input).ok()?;
    let outputs = plan.run(tvec![tensor.into()]).ok()?;
    let raw: Vec<f32> = outputs.into_iter().next()?.to_plain_array_view::<f32>().ok()?.iter().copied().collect();
    if raw.len() != PARSING_CLASSES * PARSING_PX {
        return None;
    }
    // Softmax over the class axis of `[1, 19, side, side]`: the max and the denominator are
    // per pixel across the 19 classes (a per-plane max lets exp run positive and overflow).
    let exps: Vec<f32> = {
        let mut exps = vec![0.0f32; raw.len()];
        for i in 0..PARSING_PX {
            let mut max = f32::MIN;
            for c in 0..PARSING_CLASSES {
                max = max.max(raw[c * PARSING_PX + i]);
            }
            let mut total = 0.0f32;
            for c in 0..PARSING_CLASSES {
                let e = (raw[c * PARSING_PX + i] - max).exp();
                exps[c * PARSING_PX + i] = e;
                total += e;
            }
            let total = total.max(1e-6);
            for c in 0..PARSING_CLASSES {
                exps[c * PARSING_PX + i] /= total;
            }
        }
        exps
    };
    let mut planes = Vec::with_capacity(PARSING_CLASSES);
    for c in 0..PARSING_CLASSES {
        let norm: Vec<[f32; 1]> = exps[c * PARSING_PX..(c + 1) * PARSING_PX].iter().map(|v| [*v]).collect();
        planes.push(super::resample::resize::<1>(&norm, PARSING_SIDE, PARSING_SIDE, w, h).into_iter().map(|p| p[0]).collect());
    }
    Some(super::ParsingProbs { planes, w, h })
}

/// A single-plane mask resample exposed for the engine's region masks (model resolution →
/// document resolution), the same filter the subject mask rides back on.
pub fn resize_mask(px: &[f32], w: usize, h: usize, to_w: usize, to_h: usize) -> Option<Vec<f32>> {
    if w == 0 || h == 0 || px.len() != w * h {
        return None;
    }
    let interleaved: Vec<[f32; 1]> = px.iter().map(|v| [*v]).collect();
    Some(super::resample::resize::<1>(&interleaved, w, h, to_w, to_h).into_iter().map(|p| p[0]).collect())
}
