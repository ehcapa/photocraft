//! The face hull: the Face region derived from FaceMesh landmarks rather than the parsing
//! model's small classes (whose exact order is still unverified). MediaPipe's FACE_OVAL
//! ring of 36 landmark indices outlines the face; its convex hull, filled, is the mask.

use photocraft_raster::Surface;

use crate::recipe::Region;

/// MediaPipe's canonical FACE_OVAL indices into the 478-landmark vector.
pub const FACE_OVAL: &[usize] = &[
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
];

/// The convex hull of `pts` (monotone chain, counter-clockwise). Collinear points drop.
fn convex_hull(mut pts: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
    let cross = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let build = |src: &[(f32, f32)]| {
        let mut hull: Vec<(f32, f32)> = Vec::with_capacity(src.len());
        for &p in src {
            while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(p);
        }
        hull
    };
    let lower = build(&pts);
    pts.reverse();
    let upper = build(&pts);
    [lower, upper].concat()
}

/// The filled face mask of `landmarks` (`x`, `y` in pixels of a `w × h` image): the convex
/// hull of the FACE_OVAL ring, scanline-filled over its bounding box. `None` when the
/// landmark vector is too short for the ring.
pub fn face_mask(landmarks: &[[f32; 3]], w: u32, h: u32) -> Option<Surface> {
    if landmarks.len() <= *FACE_OVAL.iter().max()? || w == 0 || h == 0 {
        return None;
    }
    let ring: Vec<(f32, f32)> = FACE_OVAL.iter().map(|&i| (landmarks[i][0], landmarks[i][1])).collect();
    let hull = convex_hull(ring);
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (x, y) in &hull {
        x0 = x0.min(*x);
        x1 = x1.max(*x);
        y0 = y0.min(*y);
        y1 = y1.max(*y);
    }
    // The hull's x extent only clamps the crossings below; the scanline walks y.
    let (_, y0) = (x0.floor().max(0.0) as i32, y0.floor().max(0.0) as i32);
    let (_, y1) = (x1.ceil().min(w as f32) as i32, y1.ceil().min(h as f32) as i32);
    let mut mask = Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
    // Each scanline enters and leaves the convex hull once: the x-span between the two
    // edge crossings (a half-open top, closed bottom — the usual pixel-centre convention).
    for y in y0..y1 {
        let cy = y as f32 + 0.5;
        let (mut enter, mut leave) = (f32::MAX, f32::MIN);
        let n = hull.len();
        for i in 0..n {
            let (ax, ay) = hull[i];
            let (bx, by) = hull[(i + 1) % n];
            if (ay - cy) * (by - cy) < 0.0 || (ay == by && ay == cy) {
                let t = (cy - ay) / (by - ay);
                let x = ax + t * (bx - ax);
                if x < enter {
                    enter = x;
                }
                if x > leave {
                    leave = x;
                }
            }
        }
        if leave <= enter {
            continue;
        }
        let (xa, xb) = (enter.ceil().max(0.0) as i32, leave.floor().min(w as f32) as i32);
        if xb <= xa {
            continue;
        }
        let span = vec![1.0f32; (xb - xa) as usize];
        mask.write_region(photocraft_geom::Rect::new(xa, y, xb, y + 1), &span);
    }
    Some(mask)
}

/// The region a face mask is: the Face.
pub const FACE_REGION: Region = Region::Face;

#[cfg(test)]
mod tests {
    use super::*;

    fn surface_value(s: &Surface, x: i32, y: i32) -> f32 {
        s.rgba(x, y)[0]
    }

    /// A diamond of landmarks around (50, 40): hull fills the diamond, corners of the
    /// bounding box stay empty.
    #[test]
    fn hull_fills_the_ring_and_only_the_ring() {
        // 478 landmarks, all parked at the centre; the FACE_OVAL ones form the diamond.
        let mut pts = vec![[50.0f32, 40.0, 0.0]; 478];
        let diamond = [(50.0f32, 10.0), (80.0, 40.0), (50.0, 70.0), (20.0, 40.0)];
        for (slot, p) in FACE_OVAL.iter().zip(diamond.iter().cycle()) {
            pts[*slot] = [p.0, p.1, 0.0];
        }
        let mask = face_mask(&pts, 100, 80).expect("a mask");
        assert!(surface_value(&mask, 50, 40) > 0.5, "centre inside");
        assert!(surface_value(&mask, 50, 15) > 0.5, "upper area inside the diamond");
        assert!(surface_value(&mask, 50, 5) < 0.5, "above the tip outside");
        assert!(surface_value(&mask, 10, 40) < 0.5, "left of the edge outside");
        assert!(surface_value(&mask, 78, 40) > 0.5, "inside the right edge");
    }

    #[test]
    fn a_short_landmark_vector_is_none() {
        assert!(face_mask(&[[0.0; 3]; 10], 100, 80).is_none());
        assert!(face_mask(&[], 0, 0).is_none());
    }
}
