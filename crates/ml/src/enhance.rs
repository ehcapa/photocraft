//! The GFPGAN face-restoration plan: build once, run per aligned crop, from the runtime
//! models dir. Native only (the tract tier — GFPGAN is conv-dominated and runs ~1.4s
//! here, no ort needed).
//!
//! Input contract (docs/fork/2026-11-18-portrait-pipeline.md, spike of 2026-11-18):
//! **0..1** RGB NCHW `[1, 3, 512, 512]` — this export bakes no normalisation, unlike the
//! parsing export's raw 0–255; the two contracts are deliberately separate constants.
//! The output is 0..1-ish (a GAN may dip negative; callers clamp).

use std::sync::OnceLock;

use tract_onnx::prelude::*;

const ENHANCE_SIDE: usize = super::ENHANCE_SIDE;
const ENHANCE_PX: usize = ENHANCE_SIDE * ENHANCE_SIDE;

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
    let path = super::models::locate(&super::models::GFPGAN)?;
    super::models::matches(&path, &super::models::GFPGAN);
    let typed = tract_onnx::onnx().model_for_path(&path).ok()?.into_typed().ok()?;
    let typed = typed.clone().into_optimized().unwrap_or(typed);
    typed.into_runnable().ok()
}

/// The restored face of an **aligned** `512 × 512` crop (`px`, row-major RGB, `0..=1`),
/// clamped back to `0..=1`. `None` without a usable model or on any failure — the
/// enhancement layer simply does not appear.
pub fn enhance_face(px: &[[f32; 3]]) -> Option<Vec<[f32; 3]>> {
    if px.len() != ENHANCE_PX {
        return None;
    }
    let plan = plan()?;
    let mut input = vec![0.0f32; 3 * ENHANCE_PX];
    for (i, p) in px.iter().enumerate() {
        for c in 0..3 {
            input[c * ENHANCE_PX + i] = p[c].clamp(0.0, 1.0);
        }
    }
    let tensor = Tensor::from_shape(&[1, 3, ENHANCE_SIDE, ENHANCE_SIDE], &input).ok()?;
    let outputs = plan.run(tvec![tensor.into()]).ok()?;
    let raw: Vec<f32> = outputs.into_iter().next()?.to_plain_array_view::<f32>().ok()?.iter().copied().collect();
    if raw.len() != 3 * ENHANCE_PX {
        return None;
    }
    Some((0..ENHANCE_PX).map(|i| [raw[i].clamp(0.0, 1.0), raw[ENHANCE_PX + i].clamp(0.0, 1.0), raw[2 * ENHANCE_PX + i].clamp(0.0, 1.0)]).collect())
}
