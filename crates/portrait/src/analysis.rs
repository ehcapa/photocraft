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

    /// Records `mask` into `region` where the region has no mask yet, else the per-pixel
    /// maximum of the two — the body segmentation's answer joins the face parsing's over the
    /// same region (skin of the arms unions into skin of the face) instead of replacing it.
    /// A mask that cannot be read back lands as `set` would land it.
    pub fn union(&mut self, region: Region, mask: Surface) {
        let Some(old) = self.masks.get(&region) else {
            self.masks.insert(region, mask);
            return;
        };
        let area = photocraft_geom::Rect::new(0, 0, self.width as i32, self.height as i32);
        let (Ok(mut vals), Ok(gate)) = (mask.try_read_region(area), old.try_read_region(area)) else {
            self.masks.insert(region, mask);
            return;
        };
        for (v, g) in vals.iter_mut().zip(gate) {
            *v = v.max(g);
        }
        let mut out = Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
        out.write_region(area, &vals);
        self.masks.insert(region, out);
    }

    pub fn mask(&self, region: Region) -> Option<&Surface> {
        self.masks.get(&region)
    }

    pub fn regions(&self) -> impl Iterator<Item = Region> + '_ {
        self.masks.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(w: u32, h: u32, value: impl Fn(i32, i32) -> f32 + Copy) -> Surface {
        let area = photocraft_geom::Rect::new(0, 0, w as i32, h as i32);
        let vals: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| value(x as i32, y as i32))).collect();
        let mut s = Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
        s.write_region(area, &vals);
        s
    }

    /// `union` is the body segmentation's landing: the arms' skin joins the face's without
    /// replacing it — the per-pixel maximum, and a first mask lands like `set`.
    #[test]
    fn union_takes_the_per_pixel_maximum() {
        let mut a = Analysis::new(8, 4);
        a.union(Region::Skin, mask(8, 4, |x, _| if x < 4 { 0.8 } else { 0.0 }));
        a.union(Region::Skin, mask(8, 4, |x, _| if x >= 2 && x < 6 { 0.5 } else { 0.0 }));
        let m = photocraft_doc::LayerMask { surface: a.mask(Region::Skin).unwrap().clone(), ..photocraft_doc::LayerMask::reveal_all() };
        // Left half stays the face's 0.8, the join keeps 0.5, the far right stays 0
        // (GRAY8 quantises, so approximate).
        let close = |a: f32, b: f32| (a - b).abs() < 0.01;
        assert!(close(m.value(0, 0), 0.8));
        assert!(close(m.value(4, 0), 0.5));
        assert!(close(m.value(6, 0), 0.0));
        // A region unioned without an existing mask lands whole.
        a.union(Region::Hair, mask(8, 4, |_, y| if y == 0 { 1.0 } else { 0.0 }));
        let hair = photocraft_doc::LayerMask { surface: a.mask(Region::Hair).unwrap().clone(), ..photocraft_doc::LayerMask::reveal_all() };
        assert_eq!(hair.value(0, 0), 1.0);
        assert_eq!(hair.value(0, 1), 0.0);
    }
}
