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

pub use landmarks::{LANDMARK_COUNT, LANDMARK_SIDE};

/// The ort tier's matting input side (1024), exported unconditionally so callers scale
/// samplers without a feature gate; only the run itself needs feature "ort".
pub const MATTING_SIDE: usize = crate::ort_side::MATTING_SIDE;

/// The GFPGAN enhancement input side (512), unconditional like [`MATTING_SIDE`].
pub const ENHANCE_SIDE: usize = crate::ort_side::ENHANCE_SIDE;
pub use parsing::{PARSING_CLASSES, PARSING_SIDE};

/// One FaceMesh landmark: `x`/`y` in the caller's pixel space, `z` a rough depth relative
/// to the head centre (MediaPipe's convention).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// The face-parsing result: one probability plane per class, at the caller's size.
pub struct ParsingProbs {
    pub planes: Vec<Vec<f32>>,
    pub w: usize,
    pub h: usize,
}

/// The face-parsing probabilities of `px` (`w × h` row-major RGB, `0..=1`): one plane per
/// CelebAMask-HQ class, returned at the caller's size. `None` without a usable
/// `PHOTOCRAFT_MODELS_DIR` model (always on wasm) — the portrait pipeline then runs
/// its person slice without region masks.
pub fn parsing_probs(px: &[[f32; 3]], w: usize, h: usize) -> Option<ParsingProbs> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        parsing::parsing_probs(px, w, h)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (px, w, h);
        None
    }
}

/// The 478 FaceMesh landmarks of `px` (`w × h` row-major RGB, `0..=1`), normalised to that
/// size. `None` on a size mismatch or inference failure — the pipeline's alignment stage
/// simply stays empty.
pub fn face_landmarks(px: &[[f32; 3]], w: usize, h: usize) -> Option<Vec<Landmark>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        landmarks::face_landmarks(px, w, h)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (px, w, h);
        None
    }
}

/// The restored face of an aligned `512 × 512` crop (row-major RGB, `0..=1`), clamped
/// to `0..=1`. `None` without `PHOTOCRAFT_MODELS_DIR/gfpgan.onnx` (or on failure) — the
/// enhancement layer does not appear. Input contract: **0..1** (this export bakes no
/// normalisation), unlike the parsing export's raw 0–255.
pub fn enhance_face(px: &[[f32; 3]]) -> Option<Vec<[f32; 3]>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        enhance::enhance_face(px)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = px;
        None
    }
}

/// The soft person matte of `px` (`w × h` row-major RGB, `0..=1`) through the ort tier
/// (feature `ort` + `PHOTOCRAFT_MODELS_DIR/feynobg.onnx`): hair-edge alpha at 1024²,
/// ~10s a run. `None` without the feature or the model — callers fall back to the
/// subject line. The tract tier stays the default everywhere it is good enough.
pub fn person_matte(px: &[[f32; 3]], w: usize, h: usize) -> Option<SubjectMask> {
    #[cfg(all(not(target_arch = "wasm32"), feature = "ort"))]
    {
        ort_matting::person_matte(px, w, h)
    }
    #[cfg(not(all(not(target_arch = "wasm32"), feature = "ort")))]
    {
        let _ = (px, w, h);
        None
    }
}

/// A single-plane `0..=1` mask resampled to `to_w × to_h` (the subject mask's filter).
pub fn resize_mask(px: &[f32], w: usize, h: usize, to_w: usize, to_h: usize) -> Option<Vec<f32>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        parsing::resize_mask(px, w, h, to_w, to_h)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (px, w, h, to_w, to_h);
        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod model;

#[cfg(not(target_arch = "wasm32"))]
pub mod landmarks;

/// The ort-tier side constant, feature-free (see [`MATTING_SIDE`]).
mod ort_side {
    pub const MATTING_SIDE: usize = 1024;
    pub const ENHANCE_SIDE: usize = 512;
}

#[cfg(not(target_arch = "wasm32"))]
mod enhance;

#[cfg(all(not(target_arch = "wasm32"), feature = "ort"))]
mod ort_matting;

/// Runtime model files: resolution over env overrides and `PHOTOCRAFT_MODELS_DIR`, and
/// fingerprint checks against the manifest. Pure path/hash logic, no network — see
/// docs/fork/models.md.
pub mod models;

#[cfg(not(target_arch = "wasm32"))]
pub mod parsing;

/// SCRFD face detection: the automatic face box (native only, tract tier).
#[cfg(not(target_arch = "wasm32"))]
pub mod detect;

mod resample;

#[cfg(test)]
mod tests;
