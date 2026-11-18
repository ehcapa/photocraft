//! The analysis the pipeline's stages produce, kept apart from the recipe that asked for it.

use std::collections::BTreeMap;

use photocraft_raster::Surface;

use crate::recipe::Region;

/// Per-region analysis, valid for exactly the document pixels it was computed from. Masks are
/// grayscale [`Surface`]s at document resolution (0 = outside, 1 = inside) — pixels as data:
/// this crate never runs a model, the engine hands the masks in the way `subject_seed` hands
/// the subject mask to GrabCut.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub width: u32,
    pub height: u32,
    /// The FaceMesh landmarks (`x`, `y` in document pixels, `z` a rough depth), empty when
    /// the stage did not run — the alignment and face-hull stages read them.
    pub landmarks: Vec<[f32; 3]>,
    masks: BTreeMap<Region, Surface>,
}

impl Analysis {
    pub fn new(width: u32, height: u32) -> Self {
        Analysis { width, height, landmarks: Vec::new(), masks: BTreeMap::new() }
    }

    /// Records `mask` as the analysis of `region`. The caller sizes it; the plan builder
    /// reads masks back by region only, so a mismatched size surfaces as wrong pixels in the
    /// layer mask, never as a panic here.
    pub fn set(&mut self, region: Region, mask: Surface) {
        self.masks.insert(region, mask);
    }

    pub fn mask(&self, region: Region) -> Option<&Surface> {
        self.masks.get(&region)
    }

    pub fn regions(&self) -> impl Iterator<Item = Region> + '_ {
        self.masks.keys().copied()
    }
}
