//! The panels' smoothness pass: 身型流畅 for the body, AI 面部流畅 for the face — PixCake's
//! contour relaxers, and the one part of either panel that is not a landmark warp: a
//! silhouette's own bumps (the trapezius slope, an arm's bulge, a jutting jaw) are read off
//! a contour, low-passed, and the difference stamped back into the field. The contour
//! arrives as a plain polyline (the body's from the person mask's rims, the face's from the
//! FACE_OVAL ring), so the caller stores it in the params and a stored smart filter
//! replays the same smoothing with no mask and no analysis — the panels' anchors rule.

use crate::liquify::{LiquifyField, falloff};

/// How far each contour point's correction reaches into its neighbourhood, in multiples of
/// the contour's own point spacing: the smoothing scale follows the feature it relaxes. It
/// stays a few spacings wide so the stamp reads as the outline relaxing, not as a brush disc
/// drawn over the photo; it must also stay above the falloff's peak slope times the largest
/// correction allowed below (about 1.9 × `MAX_CORRECTION`) or the warp folds.
const REACH: f64 = 3.0;

/// A contour point this close to the field's border lies on a frame edge, where the silhouette
/// is cut by the photo rather than shaped by the subject: it is held out of the smoothing, so a
/// corner at the frame cannot lever the whole side of the subject inward.
const BORDER: f64 = 0.75;

/// The smallest pose visibility a joint needs before a symmetry pass may move it. Below this
/// the pose is guessing (a hand out of frame, a shoulder behind the head point), and aligning
/// the guess would warp whatever the guess happened to land on.
const SEEN: f32 = 0.5;

/// How much of the stamp's disc a symmetry correction may use. A pair needing more than this is
/// not an asymmetry to smooth but a different pose — one arm raised, a hand on the hip — and
/// teleporting it would drag the whole disc with it.
const ALIGN_LIMIT: f64 = 0.35;

/// How far a contour point may move in one smoothing pass, in multiples of the contour's own
/// spacing. A correction beyond this is not a bump the low-pass relaxed: it is the silhouette
/// changing plan — a handbag jutting out, the gap between two legs, the person mask fusing with
/// a crowd — and stamping it drags a disc of the photo with it (the circular smears the smooth
/// slider drew from a real mask). Such points are left where they are: the pass relaxes the
/// outline's bumps and leaves its real concavities alone.
const MAX_CORRECTION: f64 = 1.0;

/// The largest step between neighbouring contour points that is still one contour, in multiples
/// of the point spacing. A bigger step is a cut — where the tracked silhouette ends (the mask
/// fused with the background, the frame clipped it) or where two limbs cross — and the low-pass
/// must hold its ends there instead of averaging across the cut: a closed-ring smoother saw the
/// chord across such a cut as a bump and dragged its corners hundreds of pixels.
const MAX_STEP: f64 = 2.5;

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
/// its neighbourhood on a falloff disc of `REACH` spacings, the correction's own scale.
/// Overlapping stamps blend as a weighted mean; a straight run corrects nothing, a bulge pulls
/// its whole rim in.
///
/// The forward motion is the correction itself (a bulge's rim moves inward), so the
/// inverse-map displacement points against it.
///
/// A closed contour with no cut is smoothed whole. One that touches the frame edge, or whose
/// points step farther than [`MAX_STEP`] (a cut where the tracked silhouette ends or two limbs
/// cross), is split there into open runs first: a cut is not a silhouette feature, and letting
/// the closed-ring low-pass see it pulled the frame-clipped side of a portrait inward by more
/// than a hundred pixels, and dragged the corners of a stopped body contour hundreds of pixels —
/// circular bulges, not a smoother outline. An open run holds its ends, so the correction fades
/// to nothing at a cut.
///
/// Returns how many corrections were left out because they exceeded [`MAX_CORRECTION`]; the
/// caller reports the pass approximate when any were, instead of showing a large warp as a
/// faithful smoother.
pub fn stamp(field: &mut LiquifyField, pts: &[[f64; 2]], spacing: f64, strength: f64) -> usize {
    stamp_contour(field, pts, spacing, strength, false)
}

/// [`stamp`], with the whole contour's own shift and uniform shrink removed first — the
/// face outline's case. A closed convex ring low-passed toward its neighbours' chords
/// shrinks every point toward the interior: twenty passes on a face oval moved its rim
/// inward by sixty pixels on a real portrait, and stamped as-is the smooth slider read as
/// a smaller face, dragging the eyes and brows along with the rim. Removing the similarity
/// the low-pass applied to the ring (its mean shift and its scale about the centroid) leaves
/// its local relief — the bumps the slider exists to relax — and nothing else. A contour
/// that already is smooth therefore relaxes to nothing, as it should.
pub fn stamp_relief(field: &mut LiquifyField, pts: &[[f64; 2]], spacing: f64, strength: f64) -> usize {
    stamp_contour(field, pts, spacing, strength, true)
}

fn stamp_contour(field: &mut LiquifyField, pts: &[[f64; 2]], spacing: f64, strength: f64, relief: bool) -> usize {
    if pts.len() < 3 || !spacing.is_finite() || spacing <= 1e-9 {
        return 0;
    }
    let reach = (spacing * REACH).max(6.0);
    let limit = spacing * MAX_CORRECTION;
    let step = spacing * MAX_STEP;
    let n = pts.len();
    let gap = |i: usize| (pts[(i + 1) % n][0] - pts[i][0]).hypot(pts[(i + 1) % n][1] - pts[i][1]) > step;
    // A cut precedes every run. Starting at one keeps a run from wrapping across the seam.
    let start = (0..n).find(|&i| on_border(field, pts[i]) || gap(i) || gap((i + n - 1) % n));
    let mut items: Vec<([f64; 2], [f64; 2])> = Vec::new();
    let mut dropped = 0;
    match start {
        None => {
            // No cut anywhere: the contour closes on itself and is smoothed as one ring.
            let mut smoothed = pts.to_vec();
            low_pass(&mut smoothed, true, strength);
            if relief {
                steady(pts, &mut smoothed);
            }
            dropped = push_corrections(&mut items, pts, &smoothed, limit);
        }
        Some(start) => {
            let mut run: Vec<[f64; 2]> = Vec::new();
            for k in 0..n {
                let i = (start + k) % n;
                if on_border(field, pts[i]) {
                    dropped += smooth_run(&mut items, &run, strength, limit, relief);
                    run.clear();
                    continue;
                }
                run.push(pts[i]);
                if gap(i) {
                    dropped += smooth_run(&mut items, &run, strength, limit, relief);
                    run.clear();
                }
            }
            dropped += smooth_run(&mut items, &run, strength, limit, relief);
        }
    }
    stamp_corrections(field, &items, reach, 2);
    dropped
}

/// Removes from `smoothed` the translation and uniform scale that best map `origin` onto
/// it, so a relaxation's whole-shape drift is not stamped as a correction. Points missing
/// from either list, or a degenerate origin, leave `smoothed` as it is.
fn steady(origin: &[[f64; 2]], smoothed: &mut [[f64; 2]]) {
    let n = origin.len();
    if n < 3 || smoothed.len() != n {
        return;
    }
    let inv = 1.0 / n as f64;
    let c = origin.iter().fold([0.0f64; 2], |a, p| [a[0] + p[0] * inv, a[1] + p[1] * inv]);
    let (mut sxx, mut sxy, mut t) = (0.0f64, 0.0f64, [0.0f64; 2]);
    for (p, q) in origin.iter().zip(smoothed.iter()) {
        let x = [p[0] - c[0], p[1] - c[1]];
        let y = [q[0] - c[0], q[1] - c[1]];
        sxx += x[0] * x[0] + x[1] * x[1];
        sxy += x[0] * y[0] + x[1] * y[1];
        t[0] += y[0] * inv;
        t[1] += y[1] * inv;
    }
    if sxx <= 1e-12 {
        return;
    }
    let s = sxy / sxx;
    let s = if s.is_finite() && s > 0.0 && s < 2.0 { s } else { 1.0 };
    for (p, q) in origin.iter().zip(smoothed.iter_mut()) {
        let x = [p[0] - c[0], p[1] - c[1]];
        q[0] -= (s - 1.0) * x[0] + t[0];
        q[1] -= (s - 1.0) * x[1] + t[1];
    }
}

/// A point cut by the frame, held out of the contour smoothing.
fn on_border(field: &LiquifyField, p: [f64; 2]) -> bool {
    let b = field.bounds;
    p[0] <= f64::from(b.x0) + BORDER || p[0] >= f64::from(b.x1) - BORDER || p[1] <= f64::from(b.y0) + BORDER || p[1] >= f64::from(b.y1) - BORDER
}

/// The corrections of one open contour run, low-passed with its ends held. Returns how many
/// exceeded the limit and were left out.
fn smooth_run(items: &mut Vec<([f64; 2], [f64; 2])>, run: &[[f64; 2]], strength: f64, limit: f64, relief: bool) -> usize {
    if run.len() < 3 {
        return 0;
    }
    let mut smoothed = run.to_vec();
    low_pass(&mut smoothed, false, strength);
    if relief {
        steady(run, &mut smoothed);
    }
    push_corrections(items, run, &smoothed, limit)
}

fn push_corrections(items: &mut Vec<([f64; 2], [f64; 2])>, pts: &[[f64; 2]], smoothed: &[[f64; 2]], limit: f64) -> usize {
    let mut dropped = 0;
    for (p, q) in pts.iter().zip(smoothed) {
        let delta = [q[0] - p[0], q[1] - p[1]];
        let mag = delta[0].hypot(delta[1]);
        if mag < 1e-9 {
            continue;
        }
        if mag > limit {
            dropped += 1;
            continue;
        }
        items.push((*p, delta));
    }
    dropped
}

/// Stamps `(point, correction)` pairs as a falloff-weighted mean and relaxes the field.
fn stamp_corrections(field: &mut LiquifyField, items: &[([f64; 2], [f64; 2])], reach: f64, passes: usize) {
    let inv = 1.0 / field.cell;
    let (bx, by) = (f64::from(field.bounds.x0) + 0.5, f64::from(field.bounds.y0) + 0.5);
    // Scattered interpolation, not a sum: overlapping discs carry coherent corrections, and
    // adding them amplified a 1.6 px conservation drift into a 7.4 px shove. Each node
    // takes the corrections' falloff-weighted mean.
    let mut num = vec![[0.0f64; 2]; field.w * field.h];
    let mut den = vec![0.0f64; field.w * field.h];
    for &(p, delta) in items {
        if delta[0].hypot(delta[1]) < 1e-9 {
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
    relax(field, passes);
}

/// 3D塑颜: rotates the mesh's own 3D points about their centre (yaw about the vertical
/// axis, pitch about the horizontal, roll about the view axis; degrees) and stamps each
/// point's projected displacement — the far side of the face compresses and the near side
/// opens exactly as a small turn would move it. FaceMesh's z is the depth the mesh ships
/// with; the angles stay small (the caller maps its slider to a few degrees), where the
/// orthographic projection is honest.
pub fn pose_stamp(field: &mut LiquifyField, pts: &[[f64; 3]], yaw_deg: f64, pitch_deg: f64, roll_deg: f64, reach: f64) {
    if pts.len() < 8 || (yaw_deg.abs() < 1e-6 && pitch_deg.abs() < 1e-6 && roll_deg.abs() < 1e-6) {
        return;
    }
    let n = pts.len() as f64;
    let c = pts.iter().fold([0.0f64; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n]);
    let (y0, p0, r0) = (yaw_deg.to_radians(), pitch_deg.to_radians(), roll_deg.to_radians());
    let (sy, cy) = y0.sin_cos();
    let (sp, cp) = p0.sin_cos();
    let (sr, cr) = r0.sin_cos();
    let items: Vec<([f64; 2], [f64; 2])> = pts
        .iter()
        .map(|p| {
            let l = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            // yaw (about y, image-down): x picks up z, z gives back x.
            let x1 = l[0] * cy + l[2] * sy;
            let z1 = -l[0] * sy + l[2] * cy;
            // pitch (about x): y picks up z.
            let y1 = l[1] * cp - z1 * sp;
            // roll (about the view axis): the 2D rotation.
            let xr = x1 * cr - y1 * sr;
            let yr = x1 * sr + y1 * cr;
            ([p[0], p[1]], [xr - l[0], yr - l[1]])
        })
        .collect();
    stamp_deltas(field, &items, reach);
}

/// Reflects `p` about the line through `a` along unit `u`.
fn reflect(p: [f64; 2], a: [f64; 2], u: [f64; 2]) -> [f64; 2] {
    let d = [p[0] - a[0], p[1] - a[1]];
    let along = d[0] * u[0] + d[1] * u[1];
    let perp = [d[0] - along * u[0], d[1] - along * u[1]];
    [a[0] + along * u[0] - perp[0], a[1] + along * u[1] - perp[1]]
}

/// The pairs [`align_pairs`] may move: both joints seen by the pose, and the mirror correction
/// no larger than [`ALIGN_LIMIT`] of the stamp's `reach`. A pair failing either is not a touch
/// of asymmetry — it is a different pose (one arm raised, a hand on the hip) or a joint the
/// pose guessed — and aligning it would drag a disc of the photo across the frame, which is
/// what turned the symmetry slider into circular smears on a real photo.
pub fn alignable_pairs(pts: &[[f64; 2]], visibility: &[f32], pairs: &[(usize, usize)], axis: [f64; 2], dir: [f64; 2], reach: f64) -> Vec<(usize, usize)> {
    let n = len_unit(dir);
    if n < 1e-9 || reach < 4.0 {
        return Vec::new();
    }
    let u = [dir[0] / n, dir[1] / n];
    let limit = reach * ALIGN_LIMIT;
    let seen = |i: usize| visibility.get(i).copied().unwrap_or(0.0) >= SEEN;
    pairs
        .iter()
        .copied()
        .filter(|&(i, j)| {
            let (Some(&pi), Some(&pj)) = (pts.get(i), pts.get(j)) else {
                return false;
            };
            seen(i) && seen(j) && mirror_error(pi, pj, axis, u) <= limit
        })
        .collect()
}

/// The pairs a symmetry pass may align when every point is trusted — the face mesh always
/// names its own features. `limit` is in document pixels: a pair whose mirror correction at
/// full strength is larger is not an asymmetry to smooth but a different geometry — on a
/// turned or tilted head the reflected feature is nowhere near its counterpart — and aligning
/// it would drag a disc of the photo across the face. The face caller scales the limit to its
/// own width; a pair that fails is left where it is.
pub fn symmetric_pairs(pts: &[[f64; 2]], pairs: &[(usize, usize)], axis: [f64; 2], dir: [f64; 2], limit: f64) -> Vec<(usize, usize)> {
    let n = len_unit(dir);
    if n < 1e-9 || limit.is_nan() || limit <= 0.0 {
        return Vec::new();
    }
    let u = [dir[0] / n, dir[1] / n];
    pairs
        .iter()
        .copied()
        .filter(|&(i, j)| {
            let (Some(&pi), Some(&pj)) = (pts.get(i), pts.get(j)) else {
                return false;
            };
            mirror_error(pi, pj, axis, u) <= limit
        })
        .collect()
}

/// The larger of a pair's two mirror corrections: how far each point would move to land on
/// the other's reflection.
fn mirror_error(pi: [f64; 2], pj: [f64; 2], axis: [f64; 2], u: [f64; 2]) -> f64 {
    let (mi, mj) = (reflect(pi, axis, u), reflect(pj, axis, u));
    (mj[0] - pi[0]).hypot(mj[1] - pi[1]).max((mi[0] - pj[0]).hypot(mi[1] - pj[1]))
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
    stamp_corrections(field, items, reach, 1);
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
    fn a_yaw_compresses_the_far_side_and_opens_the_near() {
        // A small 3D plate of points: near (z +10), far (z −10), both off the centre.
        let pts: Vec<[f64; 3]> = (0..24)
            .map(|k| {
                let t = k as f64 / 24.0 * std::f64::consts::TAU;
                [128.0 + 40.0 * t.cos(), 128.0 + 40.0 * t.sin(), 10.0 * t.cos()]
            })
            .collect();
        let mut field = LiquifyField::new(bounds(), 2.0);
        pose_stamp(&mut field, &pts, 5.0, 0.0, 0.0, 24.0);
        assert!(!field.is_identity());
        // Yaw moves +x content by ~z·sin: the near-right point (t=0: z=10) travels right,
        // the far-left (t=π: z=−10) travels right too but by the opposite sign — sample the
        // two ends and check they went opposite ways (one opens, one compresses).
        let near = field.sample(pts[0][0], pts[0][1]);
        let far = field.sample(pts[12][0], pts[12][1]);
        assert!(near[0] < 0.0, "the near point's correction points left (opens left for a right turn): {near:?}");
        assert!(far[0] > 0.0, "the far point opposes it: {far:?}");
        // Zero angles stamp nothing.
        let mut flat = LiquifyField::new(bounds(), 2.0);
        pose_stamp(&mut flat, &pts, 0.0, 0.0, 0.0, 24.0);
        assert!(flat.is_identity());
    }

    #[test]
    fn zero_strength_stamps_nothing() {
        let (pts, spacing) = wobbly_circle();
        let mut field = LiquifyField::new(bounds(), 2.0);
        stamp(&mut field, &pts, spacing, 0.0);
        assert!(field.is_identity());
    }

    /// A ring with no relief — a plain circle sampled as sparsely as a face oval — must not
    /// move at all under the relief stamp. The plain stamp shrinks it, because the low-pass
    /// pulls every point toward its neighbours' chords; on a face that read as a smaller
    /// face. This is the behaviour the face's smooth slider depends on.
    #[test]
    fn the_relief_stamp_leaves_a_smooth_ring_alone() {
        let n = 36;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|k| {
                let t = k as f64 / n as f64 * std::f64::consts::TAU;
                [128.0 + 80.0 * t.cos(), 128.0 + 80.0 * t.sin()]
            })
            .collect();
        let spacing = std::f64::consts::TAU * 80.0 / n as f64;
        let mut plain = LiquifyField::new(bounds(), 2.0);
        stamp(&mut plain, &pts, spacing, 100.0);
        let mut relief = LiquifyField::new(bounds(), 2.0);
        stamp_relief(&mut relief, &pts, spacing, 100.0);
        assert!(plain.max_displacement() > 4.0, "the plain low-pass shrinks the ring: {}", plain.max_displacement());
        assert!(relief.max_displacement() < 0.5, "a smooth ring relaxes to nothing: {}", relief.max_displacement());
    }

    /// The relief stamp is not a no-op: a local bump on an otherwise smooth ring still
    /// pulls in.
    #[test]
    fn the_relief_stamp_still_relaxes_a_bump() {
        let n = 96;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|k| {
                let t = k as f64 / n as f64 * std::f64::consts::TAU;
                let r = 80.0 + if (k as isize - 24).abs() <= 1 { 6.0 } else { 0.0 };
                [128.0 + r * t.cos(), 128.0 + r * t.sin()]
            })
            .collect();
        let spacing = std::f64::consts::TAU * 80.0 / n as f64;
        let mut field = LiquifyField::new(bounds(), 2.0);
        stamp_relief(&mut field, &pts, spacing, 100.0);
        let bulge = pts[24];
        let d = field.sample(bulge[0], bulge[1]);
        let r = (bulge[0] - 128.0).hypot(bulge[1] - 128.0);
        let outward = [(bulge[0] - 128.0) / r, (bulge[1] - 128.0) / r];
        assert!(d[0] * outward[0] + d[1] * outward[1] > 0.5, "the bump's correction must read inward: {d:?}");
    }

    #[test]
    fn only_pairs_within_the_limit_are_symmetric() {
        let (pts, _) = raised_arm();
        let close = symmetric_pairs(&pts, &[(11, 12), (15, 16)], [130.0, 100.0], [0.0, 1.0], 40.0);
        assert_eq!(close, vec![(11, 12)], "the raised wrist pair is a different geometry, not an asymmetry");
        let all = symmetric_pairs(&pts, &[(11, 12), (15, 16)], [130.0, 100.0], [0.0, 1.0], 1000.0);
        assert_eq!(all.len(), 2, "a generous limit keeps both pairs");
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

    /// A pose with the left arm raised: the wrists are nowhere near mirrored, the rest is
    /// close. The axis is the vertical x = 130.
    fn raised_arm() -> (Vec<[f64; 2]>, Vec<f32>) {
        let mut pts = vec![[0.0; 2]; 33];
        let mut vis = vec![0.0f32; 33];
        for &(i, x, y) in &[
            (11usize, 100.0, 100.0),
            (12, 160.0, 106.0),
            (13, 105.0, 150.0),
            (14, 155.0, 155.0),
            (15, 108.0, 60.0),
            (16, 158.0, 240.0),
            (23, 105.0, 220.0),
            (24, 155.0, 220.0),
            (25, 110.0, 300.0),
            (26, 150.0, 300.0),
            (27, 112.0, 380.0),
            (28, 148.0, 380.0),
        ] {
            pts[i] = [x, y];
            vis[i] = 1.0;
        }
        (pts, vis)
    }

    #[test]
    fn only_a_pair_near_its_mirror_is_alignable() {
        let (pts, vis) = raised_arm();
        let pairs = alignable_pairs(&pts, &vis, &[(11, 12), (15, 16)], [130.0, 100.0], [0.0, 1.0], 40.0);
        assert_eq!(pairs, vec![(11, 12)], "the raised wrist pair is a different pose, not an asymmetry");
    }

    #[test]
    fn a_pair_the_pose_did_not_see_is_not_alignable() {
        let (pts, mut vis) = raised_arm();
        vis[16] = 0.2;
        let pairs = alignable_pairs(&pts, &vis, &[(15, 16)], [130.0, 100.0], [0.0, 1.0], 40.0);
        assert!(pairs.is_empty(), "a guessed joint must not be moved: {pairs:?}");
    }

    #[test]
    fn a_contour_cut_by_the_frame_does_not_pull_the_subject_inward() {
        // A subject rectangle whose top and left sides lie on the frame: its contour runs
        // along x = 0.5 and y = 0.5. Low-passing the closed ring used to round those frame
        // corners and drag the top-left neighbourhood inward by tens of pixels.
        let mut pts = Vec::new();
        let (x0, y0, x1, y1) = (0.5, 0.5, 200.5, 200.5);
        for k in 0..=20 {
            let t = k as f64 / 20.0;
            pts.push([x0 + (x1 - x0) * t, y0]);
        }
        for k in 1..=20 {
            let t = k as f64 / 20.0;
            pts.push([x1, y0 + (y1 - y0) * t]);
        }
        for k in 1..=20 {
            let t = k as f64 / 20.0;
            pts.push([x1 - (x1 - x0) * t, y1]);
        }
        for k in 1..=19 {
            let t = k as f64 / 20.0;
            pts.push([x0, y1 - (y1 - y0) * t]);
        }
        let mut field = LiquifyField::new(bounds(), 2.0);
        stamp(&mut field, &pts, 10.0, 100.0);
        let near = field.sample(20.0, 20.0);
        assert!(near[0].hypot(near[1]) < 2.0, "the frame corner must not drag the subject: {near:?}");
        let corner = field.sample(198.0, 198.0);
        assert!(corner[0].hypot(corner[1]) > 0.5, "the interior corner still smooths: {corner:?}");
    }

    /// A contour that jumps to another edge — the person mask fusing into the crowd behind the
    /// subject — is a cut, not a bump: the low-pass holds at the jump instead of averaging
    /// across it, so no disc of the photo is dragged hundreds of pixels.
    #[test]
    fn a_contour_break_is_held_not_smoothed_across() {
        let (mut pts, spacing) = wobbly_circle();
        pts[0] = [pts[0][0] + 200.0, pts[0][1]];
        let mut field = LiquifyField::new(bounds(), 2.0);
        stamp(&mut field, &pts, spacing, 100.0);
        let d = field.sample(pts[0][0], pts[0][1]);
        assert!(d[0].hypot(d[1]) < spacing * 4.0, "the break was stamped as a bump: {d:?}");
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
