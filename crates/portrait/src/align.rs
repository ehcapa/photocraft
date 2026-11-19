//! The face alignment geometry: FaceMesh landmarks → the FFHQ-512 similarity transform
//! GFPGAN's reference preprocessing defines, the crop it samples, and the paste back.
//! Pure pixel math on plain buffers — no raster dependencies beyond arithmetic.

/// FaceMesh indices of the five alignment points: eye centres (corner means — robust
/// without the iris refinement), nose tip, mouth corners.
pub const LEFT_EYE: [usize; 2] = [33, 133];
pub const RIGHT_EYE: [usize; 2] = [362, 263];
pub const NOSE: usize = 1;
pub const MOUTH: [usize; 2] = [61, 291];

/// facexlib's FFHQ-512 template: where the five points sit in an aligned crop.
#[allow(clippy::excessive_precision)]
const TEMPLATE: [[f32; 2]; 5] =
    [[0.315_462_42, 0.469_849_03], [0.682_292_58, 0.469_849_03], [0.500_224_47, 0.621_868_87], [0.349_498_86, 0.743_231_25], [0.650_501_14, 0.743_231_25]];

/// A similarity transform (rotation + uniform scale + translation) fitted least-squares
/// over the five points, source → template pixels of a `side × side` crop.
#[derive(Clone, Copy, Debug)]
pub struct Similarity {
    pub a: f32,
    pub b: f32,
    pub tx: f32,
    pub ty: f32,
}

impl Similarity {
    /// Where source point `(x, y)` lands: `(a·x − b·y + tx, b·x + a·y + ty)`.
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (self.a * x - self.b * y + self.tx, self.b * x + self.a * y + self.ty)
    }

    /// The exact inverse (a similarity is closed under inversion).
    pub fn inverse(&self) -> Similarity {
        let det = self.a * self.a + self.b * self.b;
        if det.abs() < 1e-12 {
            return *self;
        }
        // inv([[a, -b], [b, a]]) = [[a, b], [-b, a]] / det; the translation follows.
        let (ia, ib) = (self.a / det, -self.b / det);
        Similarity { a: ia, b: ib, tx: -(ia * self.tx - ib * self.ty), ty: -(self.b / det * self.tx + self.a / det * self.ty) }
    }
}

/// The five source points of `landmarks` (document pixels), or `None` when the vector is
/// too short for the indices.
pub fn five_points(landmarks: &[[f32; 3]]) -> Option<[[f32; 2]; 5]> {
    let need = LEFT_EYE.iter().chain(RIGHT_EYE.iter()).chain([NOSE].iter()).chain(MOUTH.iter()).max()?;
    if landmarks.len() <= *need {
        return None;
    }
    let mean = |idx: &[usize]| -> [f32; 2] {
        let mut s = [0.0f32; 2];
        for &i in idx {
            s[0] += landmarks[i][0];
            s[1] += landmarks[i][1];
        }
        [s[0] / idx.len() as f32, s[1] / idx.len() as f32]
    };
    Some([
        mean(&LEFT_EYE),
        mean(&RIGHT_EYE),
        [landmarks[NOSE][0], landmarks[NOSE][1]],
        [landmarks[MOUTH[0]][0], landmarks[MOUTH[0]][1]],
        [landmarks[MOUTH[1]][0], landmarks[MOUTH[1]][1]],
    ])
}

/// The least-squares similarity mapping `src` onto the FFHQ-512 template at `side`.
pub fn fit(src: &[[f32; 2]; 5], side: usize) -> Similarity {
    let dst: Vec<(f32, f32)> = TEMPLATE.iter().map(|p| (p[0] * side as f32, p[1] * side as f32)).collect();
    // Closed-form similarity least squares (the four scalars solve one 2×2 system built
    // from the point covariances).
    let (mut sx, mut sy, mut dx, mut dy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let n = src.len() as f32;
    for (s, d) in src.iter().zip(dst.iter()) {
        sx += s[0];
        sy += s[1];
        dx += d.0;
        dy += d.1;
    }
    let (mx, my, mdx, mdy) = (sx / n, sy / n, dx / n, dy / n);
    // The complex-form similarity least squares: c = Σu·conj(v)/Σ|u|² with u, v the
    // centred points, then t = d̄ − c·s̄. Exact on a similarity-transformed template.
    let (mut num_r, mut num_i, mut den) = (0.0f32, 0.0f32, 0.0f32);
    for (s, d) in src.iter().zip(dst.iter()) {
        let (ux, uy) = (s[0] - mx, s[1] - my);
        let (vx, vy) = (d.0 - mdx, d.1 - mdy);
        num_r += ux * vx + uy * vy;
        num_i += uy * vx - ux * vy;
        den += ux * ux + uy * uy;
    }
    if den < 1e-9 {
        // Degenerate (all points equal): identity with the centroid moved — the paste
        // back then lands where the face was, unenhanced geometry.
        return Similarity { a: 1.0, b: 0.0, tx: mdx - mx, ty: mdy - my };
    }
    let (a, b) = (num_r / den, num_i / den);
    Similarity { a, b, tx: mdx - (a * mx - b * my), ty: mdy - (b * mx + a * my) }
}

/// Samples the aligned `side × side` crop of `px` (`w × h` row-major RGB) through
/// `forward` (document → crop): bilinear, clamped at the borders.
pub fn sample_crop(px: &[[f32; 3]], w: usize, h: usize, forward: &Similarity, side: usize) -> Vec<[f32; 3]> {
    let inverse = forward.inverse();
    let at = |x: f32, y: f32| -> [f32; 3] {
        if !(0.0..=w as f32).contains(&x) || !(0.0..=h as f32).contains(&y) {
            return [0.0; 3];
        }
        let (x0, y0) = (x.floor().clamp(0.0, (w - 1) as f32), y.floor().clamp(0.0, (h - 1) as f32));
        let (x1, y1) = ((x0 + 1.0).min((w - 1) as f32), (y0 + 1.0).min((h - 1) as f32));
        let (fx, fy) = (x - x0, y - y0);
        let p = |xx: usize, yy: usize| px[yy * w + xx];
        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| [a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t, a[2] * (1.0 - t) + b[2] * t];
        let top = lerp(p(x0 as usize, y0 as usize), p(x1 as usize, y0 as usize), fx);
        let bot = lerp(p(x0 as usize, y1 as usize), p(x1 as usize, y1 as usize), fx);
        lerp(top, bot, fy)
    };
    (0..side)
        .flat_map(|cy| (0..side).map(move |cx| (cx, cy)))
        .map(|(cx, cy)| {
            // Crop centre + 0.5 → document through the inverse.
            let (sx, sy) = inverse.apply(cx as f32 + 0.5, cy as f32 + 0.5);
            at(sx - 0.5, sy - 0.5)
        })
        .collect()
}

/// Pastes the enhanced crop back over `base` (`w × h` row-major RGB): each document
/// pixel inside `alpha` takes the crop's bilinear sample through `forward`, blended by a
/// lightly feathered `alpha` (a small separable box blur — the hull's hard edge would
/// show as a seam against GFPGAN's redrawn texture). Out-of-crop samples keep the base.
pub fn paste_back(enhanced: &[[f32; 3]], side: usize, forward: &Similarity, base: &[[f32; 3]], w: usize, h: usize, alpha: &[f32]) -> Vec<[f32; 3]> {
    // Feather: three box passes of radius 1 ≈ a small Gaussian.
    let soft = {
        let mut a = alpha.to_vec();
        for _ in 0..3 {
            let mut next = vec![0.0f32; a.len()];
            for y in 0..h {
                for x in 0..w {
                    let (mut sum, mut n) = (0.0f32, 0.0f32);
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)] {
                        let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                        if (0..w as i32).contains(&xx) && (0..h as i32).contains(&yy) {
                            sum += a[yy as usize * w + xx as usize];
                            n += 1.0;
                        }
                    }
                    next[y * w + x] = sum / n;
                }
            }
            a = next;
        }
        a
    };
    let at = |x: f32, y: f32| -> [f32; 3] {
        let (x0, y0) = (x.floor().clamp(0.0, (side - 1) as f32), y.floor().clamp(0.0, (side - 1) as f32));
        let (x1, y1) = ((x0 + 1.0).min((side - 1) as f32), (y0 + 1.0).min((side - 1) as f32));
        let (fx, fy) = (x - x0, y - y0);
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let top = enhanced[y0 as usize * side + x0 as usize][c] * (1.0 - fx) + enhanced[y0 as usize * side + x1 as usize][c] * fx;
            let bot = enhanced[y1 as usize * side + x0 as usize][c] * (1.0 - fx) + enhanced[y1 as usize * side + x1 as usize][c] * fx;
            out[c] = top * (1.0 - fy) + bot * fy;
        }
        out
    };
    base.iter()
        .enumerate()
        .zip(soft.iter())
        .map(|((i, b), a)| {
            let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            let (cx, cy) = forward.apply(x, y);
            if *a <= 0.0 {
                return *b;
            }
            if !(0.0..=side as f32).contains(&cx) || !(0.0..=side as f32).contains(&cy) {
                return *b;
            }
            let e = at(cx - 0.5, cy - 0.5);
            [e[0] * a + b[0] * (1.0 - a), e[1] * a + b[1] * (1.0 - a), e[2] * a + b[2] * (1.0 - a)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_points_need_the_full_mesh() {
        assert!(five_points(&[[0.0; 3]; 300]).is_none());
        assert!(five_points(&[[0.0; 3]; 478]).is_some());
    }

    #[test]
    fn a_translated_identity_face_maps_cleanly() {
        // A face whose five points sit exactly one template apart, translated by (100, 50)
        // and scaled ×2: the fit must recover it, and the crop's centre sample must come
        // from the face's centre pixel.
        let side = 512usize;
        let t = TEMPLATE.map(|p| [p[0] * side as f32 * 2.0 + 100.0, p[1] * side as f32 * 2.0 + 50.0]);
        let fit = fit(&t, side);
        let back = fit.apply(t[2][0], t[2][1]);
        assert!((back.0 - side as f32 * 0.500_224_5).abs() < 1.0, "nose to template: {back:?}");
        assert!((back.1 - side as f32 * 0.621_868_9).abs() < 1.0, "nose to template: {back:?}");
        // The inverse round-trips.
        let inv = fit.inverse();
        let rt = inv.apply(back.0, back.1);
        assert!((rt.0 - t[2][0]).abs() < 1e-2 && (rt.1 - t[2][1]).abs() < 1e-2);
    }

    #[test]
    fn crop_samples_the_source_bilinearly() {
        // A 4×4 ramp image: the crop at identity-ish transform reads the same ramp.
        let (w, h) = (4, 4);
        let px: Vec<[f32; 3]> = (0..w * h).map(|i| [(i % w) as f32 / 3.0; 3]).collect();
        let forward = Similarity { a: 1.0, b: 0.0, tx: 0.0, ty: 0.0 };
        let crop = sample_crop(&px, w, h, &forward, 4);
        assert_eq!(crop.len(), 16);
        assert!((crop[0][0] - px[0][0]).abs() < 1e-4);
        assert!((crop[15][0] - px[15][0]).abs() < 1e-4, "the corner samples map to itself");
    }
}
