//! The ort-backed matting tier (feature `ort`): the heavy attention models tract cannot
//! run interactively — FeyNoBg/BiRefNet, 91s in tract, ~10s under onnxruntime here
//! (docs/fork/2026-11-18-portrait-pipeline.md, the L2.5 decision). Off by default: the
//! tract tier stays the pure-Rust path for everything it runs well, and wasm never sees
//! this module.
//!
//! The matte rides the same manifest discipline as every runtime model
//! (`PHOTOCRAFT_MODELS_DIR/feynobg.onnx`, fingerprint checked, absent = `None`), and the
//! same preprocessing the reference inference script documents: ImageNet-normalized RGB
//! at 1024², NCHW; the graph answers a sigmoid alpha in 0..1.

use std::sync::{Mutex, OnceLock};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use super::SubjectMask;

const MATTING_SIDE: usize = super::MATTING_SIDE;
const MATTING_PX: usize = MATTING_SIDE * MATTING_SIDE;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// `Session::run` takes `&mut self`; one process-wide session behind a lock is exactly
/// the single-shot usage the matting stage makes of it.
static SESSION: OnceLock<Option<Mutex<Session>>> = OnceLock::new();

fn session() -> Option<&'static Mutex<Session>> {
    SESSION
        .get_or_init(|| {
            let path = super::models::locate(&super::models::FEYNOBG)?;
            super::models::matches(&path, &super::models::FEYNOBG);
            let env = ort::init().build().ok()?;
            let built = Session::builder(&env)
                .ok()?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .ok()?
                .with_intra_threads(8)
                .ok()?
                .commit_from_file(&path)
                .ok()?;
            Some(Mutex::new(built))
        })
        .as_ref()
}

/// The soft person matte of `px` (`w × h` row-major RGB, `0..=1`), resampled to the
/// caller's size. `None` without the feature's model or on any failure — the caller
/// falls back to the subject line (u2netp seed → GrabCut).
pub fn person_matte(px: &[[f32; 3]], w: usize, h: usize) -> Option<SubjectMask> {
    if w == 0 || h == 0 || px.len() != w * h {
        return None;
    }
    let mut guard = session()?.lock().ok()?;
    let scaled = super::resample::resize::<3>(px, w, h, MATTING_SIDE, MATTING_SIDE);
    let mut input = vec![0.0f32; 3 * MATTING_PX];
    for (i, p) in scaled.iter().enumerate() {
        for c in 0..3 {
            input[c * MATTING_PX + i] = (p[c] - MEAN[c]) / STD[c];
        }
    }
    let tensor = Tensor::from_array(([1usize, 3, MATTING_SIDE, MATTING_SIDE], input)).ok()?;
    let outputs = guard.run(ort::inputs![tensor]).ok()?;
    let (_, alpha) = outputs[0].try_extract_tensor::<f32>().ok()?;
    if alpha.len() != MATTING_PX {
        return None;
    }
    // A binary-ish pass over the softmax-free sigmoid output: high confidence in, low out,
    // the soft band in between is exactly the hair the matte exists for.
    let norm: Vec<[f32; 1]> = alpha.iter().map(|v| [v.clamp(0.0, 1.0)]).collect();
    let mask = super::resample::resize::<1>(&norm, MATTING_SIDE, MATTING_SIDE, w, h);
    Some(SubjectMask { px: mask.into_iter().map(|p| p[0]).collect(), w, h })
}
