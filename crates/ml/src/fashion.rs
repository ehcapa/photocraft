//! The ATR-18 fashion parsing (SegFormer-B2 trained on ATR, `schp-atr.onnx`): the clothing
//! tier of the body stage — the classes `selfiemulticlass`'s one `clothes` blob splits into
//! (docs/fork/decisions.md 2026-11-20 named SCHP ATR-18 as the tier; this export is the
//! one that passed the `fashion_probe` spike of 2026-11-20: tract-loadable, ~0.5 s/run,
//! class semantics window-verified, the arm/leg geometric classes firing on bare skin
//! only). Native only, tract tier.
//!
//! Preprocessing contract (the export's own `preprocessor_config.json`, probe-confirmed):
//! **NCHW** `[1, 3, 512, 512]`, `0..1` RGB **ImageNet-normalized** (mean 0.485/0.456/0.406,
//! std 0.229/0.224/0.225), stretched onto the square — the aspect distortion is undone on
//! the way out. The answer is `[1, 18, 128, 128]` raw logits, class axis **first**: the
//! softmax over that axis is applied here, like the body model's (a probe reads −13…9 —
//! impossible softmax output).

use std::sync::OnceLock;

use tract_onnx::prelude::*;

/// The side the export expects (`1×3×512×512`).
pub const FASHION_SIDE: usize = 512;
/// The ATR-18 classes, in the export's channel order.
pub const FASHION_CLASSES: usize = 18;
/// The channel of each class: 0 background, 1 hat, 2 hair, 3 sunglasses, 4 upper-clothes,
/// 5 skirt, 6 pants, 7 dress, 8 belt, 9 left-shoe, 10 right-shoe, 11 face, 12 left-leg,
/// 13 right-leg, 14 left-arm, 15 right-arm, 16 bag, 17 scarf.
pub const HAT: usize = 1;
pub const HAIR: usize = 2;
pub const UPPER_CLOTHES: usize = 4;
pub const SKIRT: usize = 5;
pub const PANTS: usize = 6;
pub const DRESS: usize = 7;
pub const LEFT_SHOE: usize = 9;
pub const RIGHT_SHOE: usize = 10;
pub const FACE: usize = 11;
pub const LEFT_LEG: usize = 12;
pub const RIGHT_LEG: usize = 13;
pub const LEFT_ARM: usize = 14;
pub const RIGHT_ARM: usize = 15;
pub const BAG: usize = 16;
pub const SCARF: usize = 17;

/// The compiled plan, `into_runnable`'s `Arc` (the sharing proof the other tiers carry).
type Plan = std::sync::Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>;

const _: fn() = {
    fn shared<T: Send + Sync>() {}
    shared::<Plan>
};

static PLAN: OnceLock<Option<Plan>> = OnceLock::new();

/// The process-wide plan from `PHOTOCRAFT_MODELS_DIR/schp-atr.onnx`, built on first use.
/// A missing or broken model is cached as `None`: re-reading would re-pay the same miss.
fn plan() -> Option<&'static Plan> {
    PLAN.get_or_init(build).as_ref()
}

fn build() -> Option<Plan> {
    let path = super::models::locate(&super::models::SCHP)?;
    super::models::matches(&path, &super::models::SCHP);
    let proto = tract_onnx::onnx().model_for_path(&path).ok()?;
    // The export's input is dynamic; pin the 512² NCHW fact before into_typed or the conv
    // analysis fails (the SCRFD export's precedent).
    let proto = proto.with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, [1usize, 3, FASHION_SIDE, FASHION_SIDE])).ok()?;
    let typed = proto.into_typed().ok()?;
    let typed = typed.clone().into_optimized().unwrap_or(typed);
    typed.into_runnable().ok()
}

/// The fashion parsing of `px` (`w × h` row-major RGB, `0..=1`): per-class probabilities at
/// the caller's size, or `None` without a usable model. `planes[class]` is the `w × h`
/// probability plane; the caller drops the classes its region table does not name.
pub fn fashion_probs(px: &[[f32; 3]], w: usize, h: usize) -> Option<Vec<Vec<f32>>> {
    if w == 0 || h == 0 || px.len() != w * h {
        return None;
    }
    let plan = plan()?;
    let scaled = super::resample::resize::<3>(px, w, h, FASHION_SIDE, FASHION_SIDE);
    // NCHW, 0..1 ImageNet-normalized — the export's own contract.
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let mut input = vec![0.0f32; 3 * FASHION_SIDE * FASHION_SIDE];
    for (i, p) in scaled.iter().enumerate() {
        for c in 0..3 {
            input[c * FASHION_SIDE * FASHION_SIDE + i] = (p[c] - MEAN[c]) / STD[c];
        }
    }
    let tensor = Tensor::from_shape(&[1, 3, FASHION_SIDE, FASHION_SIDE], &input).ok()?;
    let outputs = plan.run(tvec![tensor.into()]).ok()?;
    let raw: Vec<f32> = outputs.into_iter().next()?.to_plain_array_view::<f32>().ok()?.iter().copied().collect();
    // Class axis first: `[1, 18, oh, ow]` — the spatial side follows from the count.
    if raw.len() % FASHION_CLASSES != 0 {
        return None;
    }
    let stride = raw.len() / FASHION_CLASSES;
    let side = match isqrt(stride) {
        Some(side) if side * side == stride => side,
        _ => return None,
    };
    // Softmax over the class axis is OURS to apply (the export emits raw logits), per
    // spatial position with max subtraction for stability — the body model's routine.
    let mut probs: Vec<f32> = raw;
    for i in 0..stride {
        let mut max = f32::NEG_INFINITY;
        for c in 0..FASHION_CLASSES {
            max = max.max(probs[c * stride + i]);
        }
        let mut sum = 0.0f32;
        for c in 0..FASHION_CLASSES {
            probs[c * stride + i] = (probs[c * stride + i] - max).exp();
            sum += probs[c * stride + i];
        }
        if sum > 0.0 {
            for c in 0..FASHION_CLASSES {
                probs[c * stride + i] /= sum;
            }
        }
    }
    // Class-first NCHW → per-class planes at the caller's size (stride-`stride` extraction,
    // then the mask resampler; the 128² side is the model's own output resolution).
    (0..FASHION_CLASSES)
        .map(|c| {
            let plane: Vec<f32> = (0..stride).map(|i| probs[c * stride + i]).collect();
            super::resize_mask(&plane, side, side, w, h)
        })
        .collect::<Option<Vec<_>>>()
}

/// The integer square root of `n`, `None` when it is not a perfect square.
fn isqrt(n: usize) -> Option<usize> {
    let root = (n as f64).sqrt() as usize;
    (root * root == n).then_some(root)
}
