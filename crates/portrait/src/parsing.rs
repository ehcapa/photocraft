//! The CelebAMask-HQ class → [`Region`] mapping and the plane grouping that turns the
//! model's per-class probabilities into per-region masks.
//!
//! The classes the pipeline has a use for, verified 2026-11-18 by landmark-anchored
//! coverage probes over two real portraits (crates/engine/examples/class_probe.rs, the
//! table lives in the fork archive): 1 = skin, 16 = cloth, 17 = hair and 18 = hair/hat
//! (long hair lands in 18, short in 17 — the union is the robust Hair). The export does
//! NOT separate eyes or lips (they read as skin), so those regions come from the FaceMesh
//! landmark rings instead (`crate::face`); 15 (canonical necklace) never fires.

use crate::recipe::Region;

/// The mapping, in class-index order. A region takes the per-pixel maximum over its member
/// classes' probabilities (overlapping classes keep the stronger claim).
pub const CLASS_REGIONS: &[(u8, Region)] = &[(1, Region::Skin), (16, Region::Clothes), (17, Region::Hair), (18, Region::Hair)];

/// The per-region masks of `planes` (one probability plane per class, `w × h` row-major):
/// each mapped region's plane is the pointwise maximum of its member classes. Regions with
/// no mapped class come out absent, which the plan builder reads as "not analyzed".
pub fn region_planes(planes: &[Vec<f32>], w: usize, h: usize) -> Vec<(Region, Vec<f32>)> {
    let len = w * h;
    // One entry per distinct region (hair's 17 ∪ 18 members max into one mask); a class
    // with a wrong-sized plane drops out of its region, not the region itself.
    let mut out: Vec<(Region, Vec<f32>)> = Vec::new();
    for &(class, region) in CLASS_REGIONS {
        let Some(plane) = planes.get(usize::from(class)) else { continue };
        if plane.len() != len {
            continue;
        }
        match out.iter_mut().find(|(r, _)| *r == region) {
            Some((_, mask)) => {
                for (m, v) in mask.iter_mut().zip(plane) {
                    *m = m.max(*v);
                }
            }
            None => out.push((region, plane.clone())),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(len: usize, v: f32) -> Vec<f32> {
        vec![v; len]
    }

    #[test]
    fn groups_member_classes_by_max_and_skips_unmapped() {
        // 19 fake classes at 2×2: 1 (skin), 16 (cloth), and hair's union 17 ∪ 18 — long
        // hair lands in 18, short in 17, the region takes the max of both.
        let (w, h) = (2, 2);
        let mut planes = vec![plane(4, 0.0); 19];
        planes[1] = plane(4, 0.2);
        planes[16] = plane(4, 0.5);
        planes[17] = plane(4, 0.3);
        planes[18] = plane(4, 0.9);
        let grouped = region_planes(&planes, w, h);
        assert_eq!(grouped.len(), 3, "only mapped regions appear");
        for (region, mask) in &grouped {
            let want = match *region {
                Region::Skin => 0.2,
                Region::Clothes => 0.5,
                Region::Hair => 0.9,
                other => panic!("unmapped region {other:?} grouped"),
            };
            assert!(mask.iter().all(|v| (*v - want).abs() < 1e-6), "{region:?}");
        }
    }

    #[test]
    fn a_wrong_sized_plane_drops_out_of_a_union() {
        let mut planes = vec![plane(4, 0.0); 19];
        planes[1] = plane(4, 0.7);
        planes[17] = plane(9, 0.9); // 3×3 against a 2×2 caller: drops out of the union
        planes[18] = plane(4, 0.4); // but 18 keeps Hair alive at its own level
        let grouped = region_planes(&planes, 2, 2);
        // Clothes stays as "analyzed, found nothing" (its plane is sized and all zero);
        // the engine skips empty masks when it builds the analysis.
        assert_eq!(grouped, vec![(Region::Skin, plane(4, 0.7)), (Region::Clothes, plane(4, 0.0)), (Region::Hair, plane(4, 0.4))]);
    }
}
