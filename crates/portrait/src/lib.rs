//! `photocraft-portrait`: the fork's smart-portrait orchestrator
//! (docs/fork/2026-11-18-portrait-pipeline.md).
//!
//! Pure planning and caching: the engine runs the analysis (masks arrive here as data, the
//! way `subject_seed` hands the subject mask to GrabCut) and materializes the resulting plan
//! as standard layers — an adjustment layer carrying a generated pixel mask inside a named
//! group. No new `Adjustment` variant and no new layer kind, so a document the pipeline
//! touched stays a plain Photoshop-shaped document, and nothing here conflicts with an
//! upstream sync.
//!
//! Never panics: masks are sized against the document before use, and a cache entry that no
//! longer matches the document is simply a miss.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod align;
pub mod analysis;
pub mod cache;
pub mod face;
pub mod parsing;
pub mod plan;
pub mod recipe;

pub use analysis::Analysis;
pub use cache::Cache;
pub use plan::{LayerPlan, PlannedLayer, plan};
pub use recipe::{Grade, HueSat, Recipe, Region, Skin, StageId};
