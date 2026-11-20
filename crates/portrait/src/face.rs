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

/// The eye and lip rings: the parsing export reads eyes and lips as skin, so these
/// regions are the landmark rings themselves (the AR-makeup standard) — filled exactly
/// like the face hull.
pub const LEFT_EYE_RING: &[usize] = &[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
pub const RIGHT_EYE_RING: &[usize] = &[263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
pub const LIPS_RING: &[usize] = &[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];

/// FaceMesh's cheek points (left, right): the blush ellipses' centres.
pub const CHEEKS: [usize; 2] = [205, 425];

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
    ring_mask(landmarks, FACE_OVAL, w, h)
}

/// [`face_mask`] for any landmark ring: hull-filled (rings are near-convex; the hull
/// bridges small concavities like the eye bridge). `None` when too short for the ring.
pub fn ring_mask(landmarks: &[[f32; 3]], ring_idx: &[usize], w: u32, h: u32) -> Option<Surface> {
    if landmarks.len() <= *ring_idx.iter().max()? || w == 0 || h == 0 {
        return None;
    }
    let ring: Vec<(f32, f32)> = ring_idx.iter().map(|&i| (landmarks[i][0], landmarks[i][1])).collect();
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

/// The eyes and lips of `landmarks`: both eye rings unioned into one surface, and the
/// lips ring alone. `None` on a too-short vector; absent (empty) masks are the caller's
/// signal the face did not carry them.
pub fn feature_masks(landmarks: &[[f32; 3]], w: u32, h: u32) -> Option<(Surface, Surface)> {
    let left = ring_mask(landmarks, LEFT_EYE_RING, w, h)?;
    let right = ring_mask(landmarks, RIGHT_EYE_RING, w, h)?;
    let lips = ring_mask(landmarks, LIPS_RING, w, h)?;
    // Union: both eye hulls into one mask.
    let area = photocraft_geom::Rect::new(0, 0, w as i32, h as i32);
    let mut lv = left.try_read_region(area).ok()?;
    let rv = right.try_read_region(area).ok()?;
    for (l, r) in lv.iter_mut().zip(rv.iter()) {
        *l = l.max(*r);
    }
    let mut eyes = Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
    eyes.write_region(area, &lv);
    Some((eyes, lips))
}

/// A light box feather (three radius-1 passes ≈ a small Gaussian): makeup masks need soft
/// edges — a hard ellipse or scaled ring reads as a sticker on the skin.
fn feather(a: &mut [f32], w: u32, h: u32) {
    for _ in 0..3 {
        let mut next = vec![0.0f32; a.len()];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let (mut sum, mut n) = (0.0f32, 0.0f32);
                for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1), (0, 0)] {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if (0..w as i32).contains(&xx) && (0..h as i32).contains(&yy) {
                        sum += a[yy as usize * w as usize + xx as usize];
                        n += 1.0;
                    }
                }
                next[y * w as usize + x] = sum / n;
            }
        }
        a.copy_from_slice(&next);
    }
}

/// The cheek band's lines from the mesh: the lower lids' mean y to the inner lips' mean y
/// (document pixels) — above the first line is brow and forehead, below the second is
/// chin. `None` when the landmarks run short.
pub fn cheek_band(landmarks: &[[f32; 3]]) -> Option<(f32, f32)> {
    Some(((landmarks.get(145)?[1] + landmarks.get(374)?[1]) * 0.5, (landmarks.get(13)?[1] + landmarks.get(14)?[1]) * 0.5))
}

/// The blush mask: an ellipse per cheek point, sized off the eye span, feathered. The
/// union of the pair; `None` on a too-short landmark vector.
pub fn blush_mask(landmarks: &[[f32; 3]], w: u32, h: u32) -> Option<Surface> {
    let need = CHEEKS.iter().chain(LEFT_EYE_RING.iter()).chain(RIGHT_EYE_RING.iter()).max()?;
    if landmarks.len() <= *need || w == 0 || h == 0 {
        return None;
    }
    let span = (landmarks[263][0] - landmarks[33][0]).abs();
    let (rx, ry) = (span * 0.42, span * 0.30);
    let mut plane = vec![0.0f32; (w * h) as usize];
    for &cheek in CHEEKS.iter() {
        let (cx, cy) = (landmarks[cheek][0], landmarks[cheek][1]);
        for y in ((cy - ry).floor().max(0.0) as usize)..((cy + ry).ceil().min(h as f32) as usize) {
            for x in ((cx - rx).floor().max(0.0) as usize)..((cx + rx).ceil().min(w as f32) as usize) {
                let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                if (dx / rx).powi(2) + (dy / ry).powi(2) <= 1.0 {
                    plane[y * w as usize + x] = 1.0;
                }
            }
        }
    }
    feather(&mut plane, w, h);
    let mut surface = Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
    surface.write_region(photocraft_geom::Rect::new(0, 0, w as i32, h as i32), &plane);
    Some(surface)
}

/// The eyeshadow mask: each eye ring scaled 1.6× about its centre, filled, clipped to the
/// lid (the rows above the eye centre by a fifth of the ring's height), the pair unioned
/// and feathered.
pub fn eyeshadow_mask(landmarks: &[[f32; 3]], w: u32, h: u32) -> Option<Surface> {
    let scaled = |ring: &[usize]| -> Option<Vec<[f32; 3]>> {
        if landmarks.len() <= *ring.iter().max()? {
            return None;
        }
        let pts: Vec<[f32; 2]> = ring.iter().map(|&i| [landmarks[i][0], landmarks[i][1]]).collect();
        let (mut cx, mut cy) = (0.0f32, 0.0f32);
        for p in &pts {
            cx += p[0];
            cy += p[1];
        }
        let (cx, cy) = (cx / pts.len() as f32, cy / pts.len() as f32);
        // The lid only: every scaled point clamps at the eye centre (the feather below
        // spreads a couple of px onto the lower lash line, no further).
        Some(pts.iter().map(|p| [cx + (p[0] - cx) * 1.6, (cy + (p[1] - cy) * 1.6).min(cy + 2.0), 0.0]).collect())
    };
    let left = scaled(LEFT_EYE_RING)?;
    let right = scaled(RIGHT_EYE_RING)?;
    let mut plane = vec![0.0f32; (w * h) as usize];
    for ring in [&left, &right] {
        let filled = ring_mask(ring, &(0..ring.len()).collect::<Vec<_>>(), w, h)?;
        let area = photocraft_geom::Rect::new(0, 0, w as i32, h as i32);
        let v = filled.try_read_region(area).ok()?;
        for (p, v) in plane.iter_mut().zip(v) {
            *p = p.max(v);
        }
    }
    feather(&mut plane, w, h);
    let mut surface = Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[0.0]);
    surface.write_region(photocraft_geom::Rect::new(0, 0, w as i32, h as i32), &plane);
    Some(surface)
}

/// The region a face mask is: the Face.
pub const FACE_REGION: Region = Region::Face;

#[cfg(test)]
mod tests {
    use super::*;

    /// `cheek_band` averages the lower lids' line and the inner lips' line, and asks for
    /// exactly the landmarks it reads (a short vector answers None, not a panic).
    #[test]
    fn cheek_band_reads_the_lids_and_lips_lines() {
        let mut pts = vec![[0.0f32, 0.0, 0.0]; 478];
        pts[145][1] = 10.0;
        pts[374][1] = 12.0;
        pts[13][1] = 30.0;
        pts[14][1] = 34.0;
        assert_eq!(cheek_band(&pts), Some((11.0, 32.0)));
        assert_eq!(cheek_band(&pts[..200]), None);
    }

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

    /// Cheek ellipses cover their centres and stay inside the eye-span-sized box;
    /// the eyeshadow sits above the eye rings and never below the eye centre.
    #[test]
    fn blush_and_eyeshadow_geometry() {
        // 478 landmarks parked at the centre; the anchors placed around a face at (100, 60).
        let mut pts = vec![[100.0f32, 60.0, 0.0]; 478];
        pts[33] = [70.0, 40.0, 0.0];
        pts[263] = [130.0, 40.0, 0.0];
        pts[205] = [75.0, 70.0, 0.0];
        pts[425] = [125.0, 70.0, 0.0];
        let mut place = |ring: &[usize], quad: [(f32, f32); 4]| {
            for (slot, p) in ring.iter().zip(quad.iter().cycle()) {
                pts[*slot] = [p.0, p.1, 0.0];
            }
        };
        place(LEFT_EYE_RING, [(70.0, 35.0), (85.0, 35.0), (85.0, 45.0), (70.0, 45.0)]);
        place(RIGHT_EYE_RING, [(115.0, 35.0), (130.0, 35.0), (130.0, 45.0), (115.0, 45.0)]);
        let blush = blush_mask(&pts, 200, 120).expect("a blush mask");
        let gate = photocraft_doc::LayerMask { surface: blush, ..photocraft_doc::LayerMask::reveal_all() };
        assert!(gate.value(75, 70) > 0.9 && gate.value(125, 70) > 0.9, "the cheek centres are covered");
        assert!(gate.value(100, 10) < 0.1, "nowhere near the forehead");
        let shadow = eyeshadow_mask(&pts, 200, 120).expect("an eyeshadow mask");
        let gate = photocraft_doc::LayerMask { surface: shadow, ..photocraft_doc::LayerMask::reveal_all() };
        // Above the scaled ring's lower clip (eye centre y=40): rows 30-38 carry shadow.
        let mut any = false;
        for y in 28..39 {
            if gate.value(77, y) > 0.0 || gate.value(122, y) > 0.0 {
                any = true;
            }
        }
        assert!(any, "the lids above the eyes carry shadow");
        assert!(gate.value(77, 48) == 0.0 && gate.value(122, 48) == 0.0, "nothing below the eye centre and its feather");
    }

    #[test]
    fn a_short_landmark_vector_is_none() {
        assert!(face_mask(&[[0.0; 3]; 10], 100, 80).is_none());
        assert!(face_mask(&[], 0, 0).is_none());
    }
}
