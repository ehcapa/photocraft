//! Separable triangle-filter resampling for interleaved f32 images.
//!
//! One kernel covers both directions: downscaling averages over the covered source span (a
//! tent of the source scale, so a 2× downscale weighs sources by their overlap), upscaling
//! interpolates linearly between the two nearest sources. rembg feeds the network
//! Lanczos-resized inputs; the box is close enough for the coarse seed this mask becomes
//! (GrabCut and the full-resolution re-cut refine the boundary downstream), which the rembg
//! parity test quantifies.

/// Per-destination weights along one axis: `(source index, weight)` pairs summing to 1.
/// Both lengths must be nonzero (callers check sizes before resampling).
fn axis(src: usize, dst: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = src as f32 / dst as f32;
    // Kernel radius: the source span a destination pixel covers when downscaling, one source
    // pixel's reach when upscaling.
    let radius = scale.max(1.0);
    let mut out = Vec::with_capacity(dst);
    for d in 0..dst {
        // The destination pixel's centre in source coordinates.
        let centre = (d as f32 + 0.5) * scale - 0.5;
        let lo = (centre - radius).ceil().max(0.0) as usize;
        let hi = ((centre + radius).floor() + 1.0).min(src as f32) as usize;
        let mut ws = Vec::with_capacity((hi - lo).max(1));
        for s in lo..hi {
            let dist = (s as f32 - centre).abs();
            let w = 1.0 - dist / radius;
            if w > 0.0 {
                ws.push((s, w));
            }
        }
        let total: f32 = ws.iter().map(|(_, w)| *w).sum();
        if total > 1e-6 {
            for (_, w) in &mut ws {
                *w /= total;
            }
        }
        out.push(ws);
    }
    out
}

/// Resamples `src` (`sw × sh`, interleaved `N`-channel f32) to `dw × dh`. Horizontal then
/// vertical pass; every destination value is a normalised weighted average of its sources.
pub(super) fn resize<const N: usize>(src: &[[f32; N]], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<[f32; N]> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let wx = axis(sw, dw);
    let wy = axis(sh, dh);
    // Horizontal pass: sh rows of dw samples.
    let mut mid = vec![[0.0f32; N]; dw * sh];
    for y in 0..sh {
        for (x, ws) in wx.iter().enumerate() {
            let mut acc = [0.0f32; N];
            for &(s, wt) in ws {
                let p = &src[y * sw + s];
                for c in 0..N {
                    acc[c] += p[c] * wt;
                }
            }
            mid[y * dw + x] = acc;
        }
    }
    // Vertical pass over the intermediate image.
    let mut out = vec![[0.0f32; N]; dw * dh];
    for (y, ws) in wy.iter().enumerate() {
        for x in 0..dw {
            let mut acc = [0.0f32; N];
            for &(s, wt) in ws {
                let p = &mid[s * dw + x];
                for c in 0..N {
                    acc[c] += p[c] * wt;
                }
            }
            out[y * dw + x] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::resize;

    #[test]
    fn identity_returns_the_input() {
        let src = vec![[0.25f32, 0.5], [0.75, 1.0], [1.25, 1.5], [1.75, 2.0]];
        assert_eq!(resize::<2>(&src, 2, 2, 2, 2), src);
    }

    #[test]
    fn downscale_averages_and_upscale_interpolates() {
        // 4 → 2 with a tent of radius 2 centred on 0.5 and 2.5: sources 0,1,2 weigh 3:3:1 and
        // 1,2,3 weigh 1:3:3, so the pair average gains a quarter-share of the next source.
        let src: Vec<[f32; 1]> = (0..4).map(|v| [v as f32]).collect();
        let down = resize::<1>(&src, 4, 1, 2, 1);
        assert!((down[0][0] - 1.25 / 1.75).abs() < 1e-4 && (down[1][0] - 4.0 / 1.75).abs() < 1e-4);
        // 2 → 4 interpolates linearly at half-pixel centres: the ends hold, and the quarter
        // points weigh their nearer source 3:1.
        let up = resize::<1>(&down, 2, 1, 4, 1);
        assert!((up[0][0] - down[0][0]).abs() < 1e-4 && (up[3][0] - down[1][0]).abs() < 1e-4);
        assert!((up[1][0] - (0.75 * down[0][0] + 0.25 * down[1][0])).abs() < 1e-4);
        assert!((up[2][0] - (0.25 * down[0][0] + 0.75 * down[1][0])).abs() < 1e-4);
    }

    #[test]
    fn constant_input_stays_constant() {
        let src = vec![[0.3f32, 0.6, 0.9]; 12 * 8];
        let out = resize::<3>(&src, 12, 8, 3, 2);
        assert!(out.iter().all(|p| (p[0] - 0.3).abs() < 1e-5 && (p[1] - 0.6).abs() < 1e-5 && (p[2] - 0.9).abs() < 1e-5));
    }

    #[test]
    fn fractional_scales_keep_extremes_bounded() {
        // 7 → 3 and 3 → 7 (non-integer ratios): every value stays within the input range and
        // the total mass is preserved (a flat image stays flat within the extremes).
        let src: Vec<[f32; 1]> = (0..7).map(|v| [v as f32 / 6.0]).collect();
        let down = resize::<1>(&src, 7, 1, 3, 1);
        assert!(down.iter().all(|p| p[0] >= -1e-5 && p[0] <= 1.0 + 1e-5));
        let up = resize::<1>(&down, 3, 1, 7, 1);
        assert!(up.iter().all(|p| p[0] >= -1e-5 && p[0] <= 1.0 + 1e-5));
        assert!((up[0][0] - down[0][0]).abs() < 1e-4);
    }

    #[test]
    fn square_input_to_the_model_side_is_flat_for_flat() {
        let src = vec![[0.5f32; 3]; 100 * 60];
        let out = resize::<3>(&src, 100, 60, 320, 320);
        assert_eq!(out.len(), 320 * 320);
        assert!(out.iter().all(|p| (p[0] - 0.5).abs() < 1e-4));
    }
}
