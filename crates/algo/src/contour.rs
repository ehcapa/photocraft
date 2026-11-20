//! The panels' smoothness pass: 身型流畅 for the body, AI 面部流畅 for the face — PixCake's
//! contour relaxers, and the one part of either panel that is not a landmark warp: a
//! silhouette's own bumps (the trapezius slope, an arm's bulge, a jutting jaw) are read off
//! a contour, low-passed, and the difference stamped back into the field. The contour
//! arrives as a plain polyline (the body's from the person mask's rims, the face's from the
//! FACE_OVAL ring), so the caller stores it in the params and a stored smart filter
//! replays the same smoothing with no mask and no analysis — the panels' anchors rule.

use crate::liquify::{LiquifyField, falloff};

/// How far each contour point's correction reaches into its neighbourhood, in multiples of
/// the contour's own point spacing: the smoothing scale follows the feature it relaxes.
const REACH: f64 = 6.0;

/// Smooths `pts` in place with `strength` (0..100) passes of the binomial kernel
/// `[1, 2, 1] / 4` (a closed contour wraps; an open one holds its ends). The kernel is the
/// classic contour-relaxation low-pass: one pass barely moves a point toward its
/// neighbours' mean, twenty flattens a bulge to its base.
pub fn low_pass(pts: &mut [[f64; 2]], closed: bool, strength: f64) {
    let passes = (strength.clamp(0.0, 100.0) * 0.20).round() as usize; // 0..20
    let n = pts.len();
    if n < 3 || passes == 0 {
        return;
    }
    let at = |i: isize, pts: &[[f64; 2]]| -> [f64; 2] {
        let i = if closed { (i.rem_euclid(n as isize)) as usize } else { i.clamp(0, n as isize - 1) as usize }.min(n - 1);
        pts[i]
    };
    for _ in 0..passes {
        let prev: Vec<[f64; 2]> = pts.to_vec();
        for i in 0..n {
            let (a, b, c) = (at(i as isize - 1, &prev), prev[i], at(i as isize + 1, &prev));
            pts[i] = [(a[0] + 2.0 * b[0] + c[0]) * 0.25, (a[1] + 2.0 * b[1] + c[1]) * 0.25];
        }
    }
}

/// Stamps a contour's relaxation into `field`: each point's correction (its low-passed
/// position minus itself — inward where the contour bulges, outward where it dips) moves
/// its neighbourhood on a falloff disc of `REACH` spacings, the correction's own scale. The
/// stamps superpose; a straight run corrects nothing, a bulge pulls its whole rim in.
///
/// The forward motion is the correction itself (a bulge's rim moves inward), so the
/// inverse-map displacement points against it.
pub fn stamp(field: &mut LiquifyField, pts: &[[f64; 2]], spacing: f64, strength: f64) {
    if pts.len() < 3 {
        return;
    }
    let mut smoothed = pts.to_vec();
    low_pass(&mut smoothed, true, strength);
    let reach = (spacing * REACH).max(6.0);
    let inv = 1.0 / field.cell;
    let (bx, by) = (f64::from(field.bounds.x0) + 0.5, f64::from(field.bounds.y0) + 0.5);
    // Scattered interpolation, not a sum: overlapping discs carry coherent corrections, and
    // adding them amplified a 1.6 px conservation drift into a 7.4 px shove. Each node
    // takes the corrections' falloff-weighted mean.
    let mut num = vec![[0.0f64; 2]; field.w * field.h];
    let mut den = vec![0.0f64; field.w * field.h];
    for (p, q) in pts.iter().zip(&smoothed) {
        let delta = [q[0] - p[0], q[1] - p[1]];
        let mag0 = delta[0].hypot(delta[1]);
        if mag0 < 1e-9 {
            continue;
        }
        let (i0, j0, i1, j1) = (
            (((p[0] - reach - bx) * inv).floor().max(0.0)) as usize,
            (((p[1] - reach - by) * inv).floor().max(0.0)) as usize,
            (((p[0] + reach - bx) * inv).ceil().min(field.w as f64 - 1.0)).max(0.0) as usize,
            (((p[1] + reach - by) * inv).ceil().min(field.h as f64 - 1.0)).max(0.0) as usize,
        );
        for j in j0..=j1 {
            for i in i0..=i1 {
                let n = field.node_pos(i, j);
                let rho = (n[0] - p[0]).hypot(n[1] - p[1]);
                if rho >= reach {
                    continue;
                }
                let lat = falloff((rho / reach) * (rho / reach), 0.6);
                let idx = j * field.w + i;
                num[idx][0] += delta[0] * lat;
                num[idx][1] += delta[1] * lat;
                den[idx] += lat;
            }
        }
    }
    for (d, (nv, w)) in field.d.iter_mut().zip(num.iter().zip(&den)) {
        if *w > 1e-9 {
            d[0] -= (nv[0] / w) as f32;
            d[1] -= (nv[1] / w) as f32;
        }
    }
    relax(field, 2);
}

/// Reflects `p` about the line through `a` along unit `u`.
fn reflect(p: [f64; 2], a: [f64; 2], u: [f64; 2]) -> [f64; 2] {
    let d = [p[0] - a[0], p[1] - a[1]];
    let along = d[0] * u[0] + d[1] * u[1];
    let perp = [d[0] - along * u[0], d[1] - along * u[1]];
    [a[0] + along * u[0] - perp[0], a[1] + along * u[1] - perp[1]]
}

/// 对称液化: aligns landmark `pairs` toward symmetry about the line (`axis` point, `dir`
/// unit direction) — each pair's two points move toward each other's mirror by `strength`
/// (0..100; 100 is full alignment, at both ends simultaneously), each correction stamped on
/// a falloff disc of `reach`. Pure geometry of the points and the axis: a stored filter
/// replays it from the anchors alone, no extra data.
pub fn align_pairs(field: &mut LiquifyField, pts: &[[f64; 2]], pairs: &[(usize, usize)], axis: [f64; 2], dir: [f64; 2], strength: f64, reach: f64) {
    let n = len_unit(dir);
    if n < 1e-9 || pairs.is_empty() || pts.len() < 8 {
        return;
    }
    let u = [dir[0] / n, dir[1] / n];
    let t = strength.clamp(0.0, 100.0) / 100.0;
    if t <= 0.0 {
        return;
    }
    let mut items: Vec<([f64; 2], [f64; 2])> = Vec::with_capacity(2 * pairs.len());
    for &(i, j) in pairs {
        let (pi, pj) = (pts[i], pts[j]);
        let (mi, mj) = (reflect(pi, axis, u), reflect(pj, axis, u));
        // At strength 100 the two points land on each other's mirror: symmetric by
        // construction, whichever side was off.
        items.push((pi, [(mj[0] - pi[0]) * t, (mj[1] - pi[1]) * t]));
        items.push((pj, [(mi[0] - pj[0]) * t, (mi[1] - pj[1]) * t]));
    }
    stamp_deltas(field, &items, reach);
}

/// Stamps explicit `(point, correction)` pairs — the forward motion is the correction, the
/// inverse-map displacement points against it — as a falloff-weighted mean with one
/// relaxation pass, the contour stamp's own rules.
pub fn stamp_deltas(field: &mut LiquifyField, items: &[([f64; 2], [f64; 2])], reach: f64) {
    if items.is_empty() || reach < 4.0 {
        return;
    }
    let inv = 1.0 / field.cell;
    let (bx, by) = (f64::from(field.bounds.x0) + 0.5, f64::from(field.bounds.y0) + 0.5);
    let mut num = vec![[0.0f64; 2]; field.w * field.h];
    let mut den = vec![0.0f64; field.w * field.h];
    for &(p, delta) in items {
        let mag0 = delta[0].hypot(delta[1]);
        if mag0 < 1e-9 {
            continue;
        }
        let (i0, j0, i1, j1) = (
            (((p[0] - reach - bx) * inv).floor().max(0.0)) as usize,
            (((p[1] - reach - by) * inv).floor().max(0.0)) as usize,
            (((p[0] + reach - bx) * inv).ceil().min(field.w as f64 - 1.0)).max(0.0) as usize,
            (((p[1] + reach - by) * inv).ceil().min(field.h as f64 - 1.0)).max(0.0) as usize,
        );
        for j in j0..=j1 {
            for i in i0..=i1 {
                let n = field.node_pos(i, j);
                let rho = (n[0] - p[0]).hypot(n[1] - p[1]);
                if rho >= reach {
                    continue;
                }
                let lat = falloff((rho / reach) * (rho / reach), 0.6);
                let idx = j * field.w + i;
                num[idx][0] += delta[0] * lat;
                num[idx][1] += delta[1] * lat;
                den[idx] += lat;
            }
        }
    }
    for (d, (nv, w)) in field.d.iter_mut().zip(num.iter().zip(&den)) {
        if *w > 1e-9 {
            d[0] -= (nv[0] / w) as f32;
            d[1] -= (nv[1] / w) as f32;
        }
    }
    relax(field, 1);
}

fn len_unit(d: [f64; 2]) -> f64 {
    d[0].hypot(d[1])
}

/// Relaxes the field toward its 8-neighbour mean — Liquify's Smooth brush over the stamped
/// corrections, whose own gradient (a bump's correction curve) can step past the cell even
/// when every correction is small.
fn relax(field: &mut LiquifyField, passes: usize) {
    let (w, h) = (field.w, field.h);
    for _ in 0..passes {
        let prev = field.d.clone();
        for j in 0..h {
            for i in 0..w {
                let (mut acc, mut n) = ([0.0f32; 2], 0.0f32);
                for (di, dj) in [(-1isize, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                    let (x, y) = (i as isize + di, j as isize + dj);
                    if (0..w as isize).contains(&x) && (0..h as isize).contains(&y) {
                        let v = prev[y as usize * w + x as usize];
                        acc[0] += v[0] / 8.0;
                        acc[1] += v[1] / 8.0;
                        n += 1.0 / 8.0;
                    }
                }
                let d = &mut field.d[j * w + i];
                if n > 0.0 {
                    d[0] += (acc[0] / n - d[0]) * 0.5;
                    d[1] += (acc[1] / n - d[1]) * 0.5;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_geom::Rect;

    fn bounds() -> Rect {
        Rect::new(0, 0, 256, 256)
    }

    /// A wobbly circle: radius 80 with a 10 px bulge at one point.
    fn wobbly_circle() -> (Vec<[f64; 2]>, f64) {
        let n = 96;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|k| {
                let t = k as f64 / n as f64 * std::f64::consts::TAU;
                // A single-point bump is high-frequency; the low-pass must not chase it all
                // the way, so the bump spans a tenth of the ring.
                let bump = 10.0 * (-t * 5.0).cos().exp().max(0.0) * (t.cos() + 1.0) * 0.5;
                let r = 80.0 + if t < 1.2 { 10.0 * (t / 1.2 * std::f64::consts::PI).sin() } else { bump * 0.0 };
                [128.0 + r * t.cos(), 128.0 + r * t.sin()]
            })
            .collect();
        let spacing = std::f64::consts::TAU * 80.0 / n as f64;
        (pts, spacing)
    }

    #[test]
    fn the_bulge_pulls_in_and_the_far_side_stays() {
        let (pts, spacing) = wobbly_circle();
        let mut field = LiquifyField::new(bounds(), 2.0);
        stamp(&mut field, &pts, spacing, 60.0);
        assert!(!field.is_identity());
        // The bulge sits at angle 0.6π/2… the bump spans t∈[0, 1.2], peaking t=0.6: outward
        // from the centre at (128,128). Its rim must move inward: the field there points
        // outward (the inverse map of an inward correction).
        let peak_t = 0.6f64;
        let p = [128.0 + 90.0 * peak_t.cos(), 128.0 + 90.0 * peak_t.sin()];
        let d = field.sample(p[0], p[1]);
        let outward = [p[0] - 128.0, p[1] - 128.0];
        let dot = d[0] * outward[0] + d[1] * outward[1];
        assert!(dot > 0.0, "the bulge's correction must read inward: {d:?} at {p:?}");
        // The opposite side (t = π) has no bump of its own: it carries only the low-pass's
        // small centroid-conservation drift (the bump's mass redistributes around the
        // closed ring), which must be a fraction of the bump's own correction.
        let far = field.sample(48.0, 128.0);
        assert!(far[0].abs() < 1.5, "the far side only drifts: {far:?}");
    }

    #[test]
    fn zero_strength_stamps_nothing() {
        let (pts, spacing) = wobbly_circle();
        let mut field = LiquifyField::new(bounds(), 2.0);
        stamp(&mut field, &pts, spacing, 0.0);
        assert!(field.is_identity());
    }

    #[test]
    fn the_same_contour_stamps_the_same_field() {
        let (pts, spacing) = wobbly_circle();
        let mut a = LiquifyField::new(bounds(), 2.0);
        let mut b = LiquifyField::new(bounds(), 2.0);
        stamp(&mut a, &pts, spacing, 45.0);
        stamp(&mut b, &pts, spacing, 45.0);
        assert_eq!(a, b);
    }

    #[test]
    fn the_stamp_stays_under_the_fold_guard() {
        let (pts, spacing) = wobbly_circle();
        let mut field = LiquifyField::new(bounds(), 2.0);
        stamp(&mut field, &pts, spacing, 100.0);
        let mut worst = 0.0f32;
        for j in 0..field.h {
            for i in 0..field.w {
                let d = field.d[j * field.w + i];
                if i + 1 < field.w {
                    let r = field.d[j * field.w + i + 1];
                    worst = worst.max((d[0] - r[0]).abs().max((d[1] - r[1]).abs()));
                }
                if j + 1 < field.h {
                    let dn = field.d[(j + 1) * field.w + i];
                    worst = worst.max((d[0] - dn[0]).abs().max((d[1] - dn[1]).abs()));
                }
            }
        }
        assert!((worst as f64) < 2.0, "adjacent nodes step {worst:.2}px past the 2px cell");
    }
}
