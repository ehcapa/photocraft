//! The CelebAMask-HQ class → [`Region`] mapping and the plane grouping that turns the
//! model's per-class probabilities into per-region masks.
//!
//! Only the classes the pipeline has a use for are mapped; the rest of the 19 are ignored.
//! 1 (skin) and 16 (hair) were confirmed positionally on a real portrait (spike of
//! 2026-11-18); 15 (cloth) follows zllrunning's canonical order and is still to be
//! verified — small classes (eyes, brows, lips) map once a labelled sample pins the exact
//! order of this export's 19th channel.

use crate::recipe::Region;

/// The mapping, in class-index order. A region takes the per-pixel maximum over its member
/// classes' probabilities (overlapping classes keep the stronger claim).
pub const CLASS_REGIONS: &[(u8, Region)] = &[(1, Region::Skin), (15, Region::Clothes), (16, Region::Hair)];

/// The per-region masks of `planes` (one probability plane per class, `w × h` row-major):
/// each mapped region's plane is the pointwise maximum of its member classes. Regions with
/// no mapped class come out absent, which the plan builder reads as "not analyzed".
pub fn region_planes(planes: &[Vec<f32>], w: usize, h: usize) -> Vec<(Region, Vec<f32>)> {
    let len = w * h;
    CLASS_REGIONS
        .iter()
        .filter_map(|&(class, region)| {
            let plane = planes.get(usize::from(class))?;
            if plane.len() != len {
                return None;
            }
            let mut mask = plane.clone();
            for &(also, same) in CLASS_REGIONS {
                if same == region
                    && let Some(extra) = planes.get(usize::from(also))
                    && extra.len() == len
                {
                    for (m, v) in mask.iter_mut().zip(extra) {
                        *m = m.max(*v);
                    }
                }
            }
            Some((region, mask))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(len: usize, v: f32) -> Vec<f32> {
        vec![v; len]
    }

    #[test]
    fn groups_member_classes_by_max_and_skips_unmapped() {
        // 17 fake classes at 2×2: only 1 (skin), 15 (cloth) and 16 (hair) carry signal.
        let (w, h) = (2, 2);
        let mut planes = vec![plane(4, 0.0); 17];
        planes[1] = plane(4, 0.2);
        planes[15] = plane(4, 0.5);
        planes[16] = plane(4, 0.9);
        let grouped = region_planes(&planes, w, h);
        assert_eq!(grouped.len(), 3, "only mapped regions appear");
        for (region, mask) in &grouped {
            let want = match *region {
                Region::Skin => 0.2,
                Region::Clothes => 0.5,
                Region::Hair => 0.9,
                _ => unreachable!(),
            };
            assert!(mask.iter().all(|v| (*v - want).abs() < 1e-6), "{region:?}");
        }
    }

    #[test]
    fn a_wrong_sized_plane_drops_its_region() {
        let mut planes = vec![plane(4, 0.0); 17];
        planes[1] = plane(4, 0.7);
        planes[16] = plane(9, 0.9); // 3×3 against a 2×2 caller
        let grouped = region_planes(&planes, 2, 2);
        // A wrong-sized plane drops its region; a correctly-sized all-zero one stays —
        // "analyzed, found nothing" — and the engine skips empty masks when it builds the
        // analysis, so no layer appears for them.
        assert_eq!(grouped, vec![(Region::Skin, plane(4, 0.7)), (Region::Clothes, plane(4, 0.0))]);
    }
}
