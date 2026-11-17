//! `photocraft-ml`: on-device subject detection for Select Subject.
//!
//! U²-Netᵖ (X. Qin, P. Zhang, F. Wu, D. Dai, "U²-Net: Going Deeper with Nested U-Structure for
//! Salient Object Detection", Pattern Recognition 2020; weights `data/u2netp.onnx`,
//! Apache-2.0) run through tract, the pure-Rust ONNX runtime — no C library, no network
//! access, nothing leaves the process. The classical saliency pipeline in `photocraft-algo`
//! stays the fallback: inference is compiled out on wasm (the web build's size gate), and any
//! failure answers `None`, which the caller turns into the classical path.
//!
//! Never panics: image-derived sizes are checked against the sample count before use, and a
//! model that fails to load is cached as `None` rather than re-parsed on every call.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

/// Whether this build can run the model at all (`false` on wasm, whose size gate excludes it).
pub const AVAILABLE: bool = cfg!(not(target_arch = "wasm32"));

/// The network's square input side.
pub const MODEL_SIDE: usize = 320;
/// The network's input pixel count.
pub const MODEL_PX: usize = MODEL_SIDE * MODEL_SIDE;

/// A coarse subject-probability map over the image it was computed from: row-major, `0..=1`.
#[derive(Clone, Debug, PartialEq)]
pub struct SubjectMask {
    pub px: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

/// The salient-subject mask of `px` (`w × h` row-major RGB, `0..=1`), returned at the caller's
/// size. `None` when this build has no model (wasm), the samples do not match the size, or
/// inference failed — callers fall back to the classical heuristic.
pub fn subject_mask(px: &[[f32; 3]], w: usize, h: usize) -> Option<SubjectMask> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let px = model::run(px, w, h)?;
        Some(SubjectMask { px, w, h })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (px, w, h);
        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod model;

mod resample;

#[cfg(test)]
mod tests;
