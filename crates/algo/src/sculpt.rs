//! Filter › Body Sculpt: the landmark-anchored displacement field of whole-body shaping —
//! the first consumer of the body-pose foundation (each part is placed by BlazePose's joints,
//! the head by the face oval), the pixel work stays [`crate::liquify`]'s.
//!
//! The field is written analytically, never through Liquify's brush machinery. A body-scale
//! part is a 200-400 px brush, and a dab chain at that scale is thousands of dabs whose
//! pressure × rate × spacing composition makes slider → displacement an emergent function
//! (uncalibratable) and re-pays that cost on every smart-filter re-render. Instead each part
//! is one pass over its node bounding box writing the closed-form displacement, so a slider
//! maps to its rim pull first-order linearly, and the parts stay a table of data — the face
//! panel of a later iteration adds rows, not code paths.
//!
//! Freeze is never used: [`crate::liquify::apply_liquify`] blends its Catmull-Rom sample
//! toward bilinear wherever the freeze mask is set, so a freeze band edge becomes a visible
//! interpolation kink. Each part's lateral profile is its own [`falloff`] (C1-flat at the
//! rim), which bounds the part without any mask.
//!
//! Everything the field needs travels in the params: the anchors (the 33 joints, the head,
//! the chin) are embedded when the command applies, so a stored smart filter re-renders from
//! data alone, with no analysis at composite time. [`sculpt_field`] is a pure function of
//! the spec — the same spec builds the same field bit for bit.

use crate::liquify::{LiquifyField, falloff};
use photocraft_geom::Rect;
use serde::{Deserialize, Serialize};

/// The BlazePose joints the anchors carry (the same 33 the analysis stores; restated so this
/// crate stays independent of the ml one — the layering rule).
pub const POSE_COUNT: usize = 33;

/// The joints that must be visible (the model's sigmoidged confidence) for a part to run.
const VISIBILITY: f64 = 0.5;

/// How far each part's full effect reaches toward its rim (Liquify's Density, 0..1): the
/// body parts want a broad plateau with the feather at the rim only.
const DENSITY: f64 = 0.6;

/// The sixteen shaping sliders, each −100..100. A slider's positive direction is its label's
/// action: `waist` narrows, `hips` rounds, `leg_length` lengthens; zero leaves the part out.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SculptSliders {
    pub slim_body: f32,
    pub waist: f32,
    pub belly: f32,
    pub arms: f32,
    pub thighs: f32,
    pub calves: f32,
    pub shoulders: f32,
    pub hips: f32,
    pub bust: f32,
    pub head_size: f32,
    pub neck: f32,
    pub height: f32,
    pub leg_length: f32,
    pub waist_length: f32,
    pub torso_length: f32,
    pub neck_length: f32,
}

/// The head anchor: the face-oval centre and a radius that covers the hair.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SculptHead {
    pub centre: [f32; 2],
    pub radius: f32,
}

/// Everything a stored field needs to rebuild: the 33 body-pose joints (`x, y` in document
/// pixels, the visibility the model sigmoidged), the head anchor, the chin and the document
/// size the points belong to. The command embeds this in its params at apply time — the
/// re-render path has no analysis to ask.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SculptAnchors {
    pub doc: [f32; 2],
    pub pose: Vec<[f32; 3]>,
    pub head: SculptHead,
    pub chin: [f32; 2],
}

/// The sliders plus the anchors they shape: the whole input of [`sculpt_field`].
#[derive(Clone, Debug, PartialEq)]
pub struct SculptSpec {
    pub sliders: SculptSliders,
    pub anchors: SculptAnchors,
}

/// What a build did: the rows whose slider asked and landmarks allowed, and the rows that
/// asked but were left out (a limb hidden from the pose, anchors too close to name an axis)
/// with the reason — the command surfaces this, so a slider that does nothing explains
/// itself instead of reading as a dead control.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SculptReport {
    pub applied: Vec<&'static str>,
    pub skipped: Vec<(&'static str, &'static str)>,
}

// ---------- the frame: the anchors as f64 geometry ----------

/// The anchors resolved once per build: joints as `[x, y]` pairs with their visibility, the
/// head and the chin. Every part's shape is derived from this.
struct Frame {
    p: Vec<[f64; 2]>,
    vis: Vec<f64>,
    chin: [f64; 2],
    head_c: [f64; 2],
    head_r: f64,
}

impl Frame {
    /// `None` when the anchors are not a pose this crate understands (a joint count other
    /// than 33, or non-finite points): the caller reports and applies nothing rather than
    /// warping from garbage.
    fn new(a: &SculptAnchors) -> Option<Frame> {
        if a.pose.len() != POSE_COUNT || !a.doc.iter().chain(&a.chin).chain(a.head.centre.iter()).all(|v| v.is_finite()) {
            return None;
        }
        let (p, vis): (Vec<_>, Vec<_>) = a.pose.iter().map(|q| ([q[0] as f64, q[1] as f64], q[2] as f64)).unzip();
        if p.iter().flatten().chain(vis.iter()).any(|v| !v.is_finite()) {
            return None;
        }
        let head_r = if a.head.radius.is_finite() && a.head.radius > 0.0 { a.head.radius as f64 } else { 0.0 };
        Some(Frame { p, vis, chin: [a.chin[0] as f64, a.chin[1] as f64], head_c: [a.head.centre[0] as f64, a.head.centre[1] as f64], head_r })
    }

    fn pt(&self, i: usize) -> [f64; 2] {
        self.p[i]
    }

    fn visible(&self, i: usize) -> bool {
        self.vis[i] >= VISIBILITY
    }

    fn mid(&self, i: usize, j: usize) -> [f64; 2] {
        scale(add(self.pt(i), self.pt(j)), 0.5)
    }

    fn shoulder_mid(&self) -> [f64; 2] {
        self.mid(11, 12)
    }

    fn hip_mid(&self) -> [f64; 2] {
        self.mid(23, 24)
    }

    fn ankle_mid(&self) -> [f64; 2] {
        self.mid(27, 28)
    }

    fn shoulder_w(&self) -> f64 {
        norm(sub(self.pt(12), self.pt(11)))
    }

    fn hip_w(&self) -> f64 {
        norm(sub(self.pt(24), self.pt(23)))
    }

    /// Shoulder mid to hip mid, the torso's own length.
    fn torso_len(&self) -> f64 {
        norm(sub(self.hip_mid(), self.shoulder_mid()))
    }

    /// The person's lateral half-reach: the widest bony width, a little beyond.
    fn person_w(&self) -> f64 {
        self.shoulder_w().max(self.hip_w()) * 0.75
    }

    /// The one lateral width every length ramp uses: wide enough that its plateau (the rigid
    /// first 60%, [`write_ramp`]) covers the whole silhouette including hanging arms (the
    /// wrist sits at ~0.65 of it on a standing pose), so what rides a stretch rides whole.
    fn ramp_width(&self) -> f64 {
        self.person_w() * 2.0
    }

    /// Unit vector down the torso (shoulder mid → hip mid).
    fn u_torso(&self) -> [f64; 2] {
        let d = sub(self.hip_mid(), self.shoulder_mid());
        let n = norm(d);
        if n < 1e-9 { [0.0, 1.0] } else { scale(d, 1.0 / n) }
    }

    /// Unit vector along the shoulder line (joint 12 toward 11).
    fn u_shoulder(&self) -> [f64; 2] {
        let d = sub(self.pt(11), self.pt(12));
        let n = norm(d);
        if n < 1e-9 { [1.0, 0.0] } else { scale(d, 1.0 / n) }
    }

    fn head_radius(&self) -> f64 {
        if self.head_r > 0.0 {
            self.head_r
        } else {
            // No head anchor worth the name: half the shoulder width stands in for the
            // skull (the shoulders carry the head; the relation is stable across builds).
            self.shoulder_w() * 0.60
        }
    }

    // The three shapes, as the parts table asks for them. Each returns `None` on a
    // degenerate anchor (a segment too short to name a direction), which the caller reports.

    fn contract(&self, a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64) -> Option<Shape> {
        (norm(sub(b, a)) >= 4.0 && width >= 4.0 && band.1 > band.0).then_some(Shape::Contract { a, b, band, width, gain })
    }

    fn radial(&self, c: [f64; 2], radius: f64, gain: f64) -> Option<Shape> {
        (radius >= 4.0).then_some(Shape::Radial { c, radius, gain })
    }

    fn ramp(&self, a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64) -> Option<Shape> {
        (norm(sub(b, a)) >= 4.0 && width >= 4.0 && band.1 > band.0).then_some(Shape::Ramp { a, b, band, width, gain })
    }
}

/// One part's resolved geometry.
enum Shape {
    /// Scale the lateral distance to a segment: the displacement grows from nothing on the
    /// axis to the rim pull at the falloff's shoulder, inside a longitudinal window over
    /// `band` (t along the segment) with smoothstep ramps at its ends. `gain` is the rim
    /// pull at slider 100 as a fraction of `width`; the field points outward (the inverse
    /// map samples farther out, which reads as contraction), so a negative gain expands.
    Contract { a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64 },
    /// The same scaling about one centre: the head, the bust, the hips.
    Radial { c: [f64; 2], radius: f64, gain: f64 },
    /// Stretch along a→b by `gain` of the segment's length: the smoothstep ramp is zero
    /// before the band, full at and beyond it — everything beyond `b` rides rigidly (the
    /// torso rides a leg stretch, the head rides a neck stretch), then tapers off over
    /// t∈[1.5, 2.5] so the ride ends smoothly instead of at a seam. Laterally the falloff
    /// is about the axis line over `width`.
    Ramp { a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64 },
}

/// One row of the parts table: the report name, the slider it answers to, the joints that
/// must be visible, and the shape derived from the frame. The table is the feature — a new
/// part is a row, and the later face panel is a second table over a face frame.
struct Part {
    name: &'static str,
    slider: fn(&SculptSliders) -> f32,
    needs: &'static [usize],
    shape: fn(&Frame) -> Option<Shape>,
}

/// The parts, table order = build order (the superposition is order-stable, which is part of
/// the determinism the smart filter relies on).
const PARTS: &[Part] = &[
    Part {
        name: "slim body",
        slider: |s| s.slim_body,
        needs: &[11, 12, 23, 24, 27, 28],
        shape: |f| f.contract(f.shoulder_mid(), f.ankle_mid(), (0.02, 0.98), f.shoulder_w() * 0.65, 0.05),
    },
    Part {
        name: "waist",
        slider: |s| s.waist,
        needs: &[11, 12, 23, 24],
        shape: |f| f.contract(f.shoulder_mid(), f.hip_mid(), (0.30, 0.78), f.shoulder_w() * 0.60, 0.10),
    },
    Part {
        name: "belly",
        slider: |s| s.belly,
        needs: &[11, 12, 23, 24],
        shape: |f| f.contract(f.shoulder_mid(), f.hip_mid(), (0.55, 0.95), f.shoulder_w() * 0.62, 0.08),
    },
    Part {
        name: "shoulders",
        slider: |s| s.shoulders,
        needs: &[11, 12],
        shape: |f| {
            let a = add(f.shoulder_mid(), scale(f.u_torso(), -0.12 * f.torso_len()));
            let b = add(f.shoulder_mid(), scale(f.u_torso(), 0.28 * f.torso_len()));
            f.contract(a, b, (0.0, 1.0), f.shoulder_w() * 0.75, 0.07)
        },
    },
    Part { name: "left upper arm", slider: |s| s.arms, needs: &[11, 13], shape: |f| arm(f, 11, 13, 0.10) },
    Part { name: "left forearm", slider: |s| s.arms, needs: &[13, 15], shape: |f| arm(f, 13, 15, 0.10) },
    Part { name: "right upper arm", slider: |s| s.arms, needs: &[12, 14], shape: |f| arm(f, 12, 14, 0.10) },
    Part { name: "right forearm", slider: |s| s.arms, needs: &[14, 16], shape: |f| arm(f, 14, 16, 0.10) },
    Part { name: "left thigh", slider: |s| s.thighs, needs: &[23, 25], shape: |f| limb(f, 23, 25, 0.42, 0.10) },
    Part { name: "right thigh", slider: |s| s.thighs, needs: &[24, 26], shape: |f| limb(f, 24, 26, 0.42, 0.10) },
    Part { name: "left calf", slider: |s| s.calves, needs: &[25, 27], shape: |f| limb(f, 25, 27, 0.32, 0.08) },
    Part { name: "right calf", slider: |s| s.calves, needs: &[26, 28], shape: |f| limb(f, 26, 28, 0.32, 0.08) },
    // The hips and the bust expand on their positive direction (the labels mean fuller), so
    // their gains are negative: the field points inward, which reads as expansion.
    Part { name: "left hip", slider: |s| s.hips, needs: &[23, 24], shape: |f| hip(f, 23) },
    Part { name: "right hip", slider: |s| s.hips, needs: &[23, 24], shape: |f| hip(f, 24) },
    Part { name: "left bust", slider: |s| s.bust, needs: &[11, 12, 23, 24], shape: |f| bust(f, -1.0) },
    Part { name: "right bust", slider: |s| s.bust, needs: &[11, 12, 23, 24], shape: |f| bust(f, 1.0) },
    Part { name: "head", slider: |s| s.head_size, needs: &[], shape: |f| f.radial(f.head_c, f.head_radius() * 1.40, 0.09) },
    Part { name: "neck", slider: |s| s.neck, needs: &[11, 12], shape: |f| f.contract(f.chin, f.shoulder_mid(), (0.10, 0.95), f.shoulder_w() * 0.36, 0.08) },
    Part {
        name: "height",
        slider: |s| s.height,
        needs: &[11, 12, 23, 24, 27, 28],
        shape: |f| f.ramp(f.ankle_mid(), f.shoulder_mid(), (0.05, 0.95), f.ramp_width(), 0.045),
    },
    Part {
        name: "legs",
        slider: |s| s.leg_length,
        needs: &[23, 24, 27, 28],
        shape: |f| f.ramp(f.ankle_mid(), f.hip_mid(), (0.02, 0.98), f.ramp_width(), 0.07),
    },
    Part {
        name: "waist length",
        slider: |s| s.waist_length,
        needs: &[11, 12, 23, 24],
        shape: |f| {
            let b = lerp(f.hip_mid(), f.shoulder_mid(), 0.5);
            f.ramp(f.hip_mid(), b, (0.05, 0.95), f.ramp_width(), 0.09)
        },
    },
    Part {
        name: "torso length",
        slider: |s| s.torso_length,
        needs: &[11, 12, 23, 24],
        shape: |f| f.ramp(f.hip_mid(), f.shoulder_mid(), (0.05, 0.95), f.ramp_width(), 0.045),
    },
    Part { name: "neck length", slider: |s| s.neck_length, needs: &[11, 12], shape: |f| f.ramp(f.shoulder_mid(), f.chin, (0.05, 0.95), f.ramp_width(), 0.25) },
];

/// One limb segment (`i`→`j`): the brush width is a fraction of the bone's own length, so
/// the part scales with the person in the frame — a floor keeps a foreshortened segment
/// from vanishing.
fn limb(f: &Frame, i: usize, j: usize, width_frac: f64, gain: f64) -> Option<Shape> {
    let len = norm(sub(f.pt(j), f.pt(i)));
    f.contract(f.pt(i), f.pt(j), (0.05, 0.95), (len * width_frac).max(12.0), gain)
}

/// An arm segment, a limb with a slimmer fraction.
fn arm(f: &Frame, i: usize, j: usize, gain: f64) -> Option<Shape> {
    limb(f, i, j, 0.32, gain)
}

/// One hip: a radial expansion centred a little outside and below the joint, where the
/// curve the slider means actually lives.
fn hip(f: &Frame, i: usize) -> Option<Shape> {
    let out = sub(f.pt(i), f.hip_mid());
    let out_n = norm(out);
    let out = if out_n < 1e-9 { [0.0, 0.0] } else { scale(out, 1.0 / out_n) };
    let c = add(add(f.pt(i), scale(out, 0.10 * f.shoulder_w())), scale(f.u_torso(), 0.06 * f.torso_len()));
    f.radial(c, (f.hip_w() * 0.55).max(20.0), -0.08)
}

/// One side of the bust: a radial expansion on the torso axis a third of the way down from
/// the shoulders, offset to that side.
fn bust(f: &Frame, side: f64) -> Option<Shape> {
    let base = lerp(f.shoulder_mid(), f.hip_mid(), 0.32);
    let c = add(base, scale(f.u_shoulder(), side * f.shoulder_w() * 0.18));
    f.radial(c, f.shoulder_w() * 0.30, -0.10)
}

// ---------- the build ----------

/// The length family's shared fold budget: their |Δ|s may sum to this multiple of the ramp
/// width before the family is scaled down together. Five ramps stacked at full slider, each
/// with its own taper, would shear the silhouette edge past the cell size (the field folds);
/// one budget for the family keeps any combination an order below it while leaving every
/// slider alone well inside — the worst solo ramp is ~0.14 of the width.
const RAMP_BUDGET: f64 = 0.20;

/// The field of `spec` over `bounds` (`cell` document pixels per node, [`LiquifyField::new`]'s
/// own clamps apply). Pure and deterministic: the same spec, bounds and cell build the same
/// field bit for bit, which is what a stored smart filter replays on.
pub fn sculpt_field(spec: &SculptSpec, bounds: Rect, cell: f64) -> (LiquifyField, SculptReport) {
    let mut field = LiquifyField::new(bounds, cell);
    let mut report = SculptReport::default();
    let Some(frame) = Frame::new(&spec.anchors) else { return (field, report) };
    // Resolve every asked part before writing any: the length family shares one fold budget
    // ([`RAMP_BUDGET`]), which needs all their magnitudes up front.
    let mut asked: Vec<(&'static Part, f64, Shape)> = Vec::new();
    for part in PARTS {
        let slider = (part.slider)(&spec.sliders);
        if slider.abs() < 0.5 {
            continue; // the slider is at zero: nothing was asked of this row
        }
        if !part.needs.iter().all(|&i| frame.visible(i)) {
            report.skipped.push((part.name, "the pose does not see those landmarks"));
            continue;
        }
        let Some(shape) = (part.shape)(&frame) else {
            report.skipped.push((part.name, "the anchors are too close to name an axis"));
            continue;
        };
        asked.push((part, f64::from(slider / 100.0).clamp(-1.0, 1.0), shape));
    }
    let budget = RAMP_BUDGET * frame.ramp_width();
    let ramp_sum: f64 = asked.iter().map(|&(_, k, ref s)| ramp_magnitude(s, k)).sum();
    let ramp_scale = if ramp_sum > budget && ramp_sum > 0.0 { budget / ramp_sum } else { 1.0 };
    for (part, k, shape) in asked {
        match shape {
            Shape::Contract { a, b, band, width, gain } => write_contract(&mut field, a, b, band, width, gain, k),
            Shape::Radial { c, radius, gain } => write_radial(&mut field, c, radius, gain, k),
            Shape::Ramp { a, b, band, width, gain } => write_ramp(&mut field, a, b, band, width, gain, k * ramp_scale),
        }
        report.applied.push(part.name);
    }
    (field, report)
}

/// A ramp's |Δ| in pixels — the fold budget's currency.
fn ramp_magnitude(s: &Shape, k: f64) -> f64 {
    match s {
        Shape::Ramp { a, b, gain, .. } => gain.abs() * k.abs() * norm(sub(*b, *a)),
        _ => 0.0,
    }
}

/// The inclusive node range covering the document rectangle, `None` when it misses the grid
/// (NaN-safe: a NaN bound fails the comparisons and touches nothing).
fn span(field: &LiquifyField, x0: f64, y0: f64, x1: f64, y1: f64) -> Option<(usize, usize, usize, usize)> {
    let (bx, by) = (f64::from(field.bounds.x0) + 0.5, f64::from(field.bounds.y0) + 0.5);
    let inv = 1.0 / field.cell;
    let (wmax, hmax) = (field.w as f64 - 1.0, field.h as f64 - 1.0);
    let (gx0, gy0) = (((x0 - bx) * inv).floor().max(0.0), ((y0 - by) * inv).floor().max(0.0));
    let (gx1, gy1) = (((x1 - bx) * inv).ceil().min(wmax), ((y1 - by) * inv).ceil().min(hmax));
    if !(gx1 >= 0.0 && gy1 >= 0.0 && gx0 <= wmax && gy0 <= hmax && gx0 <= gx1 && gy0 <= gy1) {
        return None;
    }
    Some((gx0 as usize, gy0 as usize, gx1 as usize, gy1 as usize))
}

/// A contraction about a segment. The displacement at a node grows from nothing on the axis
/// to the rim pull (`gain · k · width`) at the falloff's shoulder — the bones barely move,
/// the contour does — and points outward, so the inverse map samples farther out and the
/// result reads as pulled in.
fn write_contract(field: &mut LiquifyField, a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64, k: f64) {
    let ab = sub(b, a);
    let len = norm(ab);
    let u = scale(ab, 1.0 / len);
    let (lo, hi) = (band.0.min(band.1), band.1.max(band.0));
    let ramp = (hi - lo) * 0.25;
    let (min_x, max_x) = (a[0].min(b[0]) - width, a[0].max(b[0]) + width);
    let (min_y, max_y) = (a[1].min(b[1]) - width, a[1].max(b[1]) + width);
    let Some((i0, j0, i1, j1)) = span(field, min_x, min_y, max_x, max_y) else { return };
    let scale_mag = gain * k * width;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let p = field.node_pos(i, j);
            let t = dot(sub(p, a), u) / len;
            let q = add(a, scale(u, t.clamp(0.0, 1.0) * len));
            let d = sub(p, q);
            let rho = norm(d);
            if rho >= width || rho < 1e-9 {
                continue;
            }
            let lat = falloff((rho / width) * (rho / width), DENSITY);
            let long = smoothstep(lo, lo + ramp, t) * (1.0 - smoothstep(hi - ramp, hi, t));
            if long <= 0.0 {
                continue;
            }
            let mag = scale_mag * lat * long * (rho / width);
            let idx = j * field.w + i;
            field.d[idx][0] += (d[0] / rho * mag) as f32;
            field.d[idx][1] += (d[1] / rho * mag) as f32;
        }
    }
}

/// The contraction about one centre (the head, the bust, the hips): the same profile with
/// the lateral distance measured from `c`.
fn write_radial(field: &mut LiquifyField, c: [f64; 2], radius: f64, gain: f64, k: f64) {
    let Some((i0, j0, i1, j1)) = span(field, c[0] - radius, c[1] - radius, c[0] + radius, c[1] + radius) else { return };
    let scale_mag = gain * k * radius;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let d = sub(field.node_pos(i, j), c);
            let rho = norm(d);
            if rho >= radius || rho < 1e-9 {
                continue;
            }
            let lat = falloff((rho / radius) * (rho / radius), DENSITY);
            let mag = scale_mag * lat * (rho / radius);
            let idx = j * field.w + i;
            field.d[idx][0] += (d[0] / rho * mag) as f32;
            field.d[idx][1] += (d[1] / rho * mag) as f32;
        }
    }
}

/// An axial stretch. The forward motion is `Δ · s(t) · u` (content moves toward `b` by up to
/// `Δ = gain · k · len`), so the inverse-map node displacement points back down the axis;
/// everything beyond `b` rides the full `Δ` and tapers off over t∈[1.5, 2.5]. Backward
/// mapping cannot leave holes; the failure mode is folding.
///
/// The lateral profile is a plateau, not a falloff: rigid over the first 60% of `width` —
/// the whole silhouette rides one piece, arms included — then a smoothstep feather to
/// nothing over the remaining 40%, where only background pays the shear. A falloff here
/// put its steepest slope at the silhouette's edge, and the five length ramps stacked that
/// slope (measured: 6.4 px per 4 px cell); the plateau moves it out and the family's shared
/// budget ([`RAMP_BUDGET`]) caps how much there is to shear.
fn write_ramp(field: &mut LiquifyField, a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64, k: f64) {
    let ab = sub(b, a);
    let len = norm(ab);
    let u = scale(ab, 1.0 / len);
    let (lo, hi) = (band.0.min(band.1), band.1.max(band.0));
    let delta = gain * k * len;
    // The ride reaches 2.6 lengths beyond b before the taper ends.
    let far = add(b, scale(u, len * 2.6));
    let min_x = a[0].min(b[0]).min(far[0]) - width;
    let max_x = a[0].max(b[0]).max(far[0]) + width;
    let min_y = a[1].min(b[1]).min(far[1]) - width;
    let max_y = a[1].max(b[1]).max(far[1]) + width;
    let Some((i0, j0, i1, j1)) = span(field, min_x, min_y, max_x, max_y) else { return };
    for j in j0..=j1 {
        for i in i0..=i1 {
            let p = field.node_pos(i, j);
            let t = dot(sub(p, a), u) / len;
            let s = smoothstep(lo, hi, t) * (1.0 - smoothstep(1.5, 2.5, t));
            if s <= 0.0 {
                continue;
            }
            let q = add(a, scale(u, t * len));
            let rho = norm(sub(p, q));
            if rho >= width {
                continue;
            }
            let lat = 1.0 - smoothstep(0.6, 1.0, rho / width);
            let mag = delta * s * lat;
            let idx = j * field.w + i;
            field.d[idx][0] -= (u[0] * mag) as f32;
            field.d[idx][1] -= (u[1] * mag) as f32;
        }
    }
}

// ---------- the small maths ----------

fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn scale(a: [f64; 2], k: f64) -> [f64; 2] {
    [a[0] * k, a[1] * k]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn norm(a: [f64; 2]) -> f64 {
    a[0].hypot(a[1])
}

fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    add(a, scale(sub(b, a), t))
}

/// Hermite smoothstep, flat at both ends (C1): the part windows' ramp, so no window edge is
/// a seam.
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_geom::Rect;

    const CELL: f64 = 4.0;

    fn bounds() -> Rect {
        Rect::new(0, 0, 512, 768)
    }

    /// A synthetic standing figure, joint by joint (BlazePose indices): every part has clean
    /// anatomy to anchor on and the geometry is pure data, so these tests run everywhere.
    fn standing() -> SculptAnchors {
        let mut pose = vec![[0.0f32; 3]; POSE_COUNT];
        let joints: &[(usize, f64, f64)] = &[
            (0, 256.0, 92.0), // nose
            (11, 204.0, 180.0),
            (12, 308.0, 180.0), // shoulders
            (13, 192.0, 276.0),
            (14, 320.0, 276.0), // elbows
            (15, 188.0, 372.0),
            (16, 324.0, 372.0), // wrists
            (23, 222.0, 400.0),
            (24, 292.0, 400.0), // hips
            (25, 226.0, 540.0),
            (26, 288.0, 540.0), // knees
            (27, 230.0, 680.0),
            (28, 284.0, 680.0), // ankles
        ];
        for &(i, x, y) in joints {
            pose[i] = [x as f32, y as f32, 1.0];
        }
        SculptAnchors { doc: [512.0, 768.0], pose, head: SculptHead { centre: [256.0, 88.0], radius: 52.0 }, chin: [256.0, 138.0] }
    }

    fn spec(sliders: SculptSliders) -> SculptSpec {
        SculptSpec { sliders, anchors: standing() }
    }

    fn build(sliders: SculptSliders) -> (LiquifyField, SculptReport) {
        sculpt_field(&spec(sliders), bounds(), CELL)
    }

    #[test]
    fn zero_sliders_build_the_identity_field() {
        let (field, report) = build(SculptSliders::default());
        assert!(field.is_identity());
        assert!(report.applied.is_empty() && report.skipped.is_empty());
    }

    #[test]
    fn the_same_spec_builds_the_same_field_twice() {
        let sliders = SculptSliders { waist: 40.0, leg_length: 60.0, bust: -30.0, ..SculptSliders::default() };
        let (a, ra) = build(sliders);
        let (b, rb) = build(sliders);
        assert_eq!(a, b);
        assert_eq!(ra, rb);
    }

    #[test]
    fn every_slider_alone_moves_the_field() {
        for sliders in [
            SculptSliders { slim_body: 100.0, ..SculptSliders::default() },
            SculptSliders { waist: 100.0, ..SculptSliders::default() },
            SculptSliders { belly: 100.0, ..SculptSliders::default() },
            SculptSliders { arms: 100.0, ..SculptSliders::default() },
            SculptSliders { thighs: 100.0, ..SculptSliders::default() },
            SculptSliders { calves: 100.0, ..SculptSliders::default() },
            SculptSliders { shoulders: 100.0, ..SculptSliders::default() },
            SculptSliders { hips: 100.0, ..SculptSliders::default() },
            SculptSliders { bust: 100.0, ..SculptSliders::default() },
            SculptSliders { head_size: 100.0, ..SculptSliders::default() },
            SculptSliders { neck: 100.0, ..SculptSliders::default() },
            SculptSliders { height: 100.0, ..SculptSliders::default() },
            SculptSliders { leg_length: 100.0, ..SculptSliders::default() },
            SculptSliders { waist_length: 100.0, ..SculptSliders::default() },
            SculptSliders { torso_length: 100.0, ..SculptSliders::default() },
            SculptSliders { neck_length: 100.0, ..SculptSliders::default() },
        ] {
            let (field, report) = build(sliders);
            assert!(!field.is_identity(), "the slider left the field untouched");
            assert!(report.skipped.is_empty(), "a clean pose skipped {:?}", report.skipped);
        }
    }

    #[test]
    fn the_waist_slider_pulls_the_rim_in_proportion_to_its_value() {
        // A point on the waist band's rim, to the axis's right. The inverse map points
        // outward there (the rim samples farther out, which reads as pulled in).
        let (x, y) = (294.0, 299.0);
        let m = |v: f32| {
            let (field, _) = build(SculptSliders { waist: v, ..SculptSliders::default() });
            let d = field.sample(x, y);
            d[0].hypot(d[1])
        };
        let (m25, m50, m100) = (m(25.0), m(50.0), m(100.0));
        assert!(m100 > 1.0, "waist at 100 moved the rim by only {m100:.2}px");
        for (lo, hi, name) in [(m25, m50, "25→50"), (m50, m100, "50→100")] {
            let ratio = hi / lo;
            assert!((1.6..=2.4).contains(&ratio), "{name} grew {ratio:.2}×, not ~2×");
        }
        let (field, _) = build(SculptSliders { waist: 100.0, ..SculptSliders::default() });
        let right = field.sample(x, y);
        let left = field.sample(219.0, 299.0);
        assert!(right[0] > 0.5 && left[0] < -0.5, "the field must point outward on both rims ({right:?} {left:?})");
    }

    #[test]
    fn head_size_moves_the_head_rim_and_spares_the_centre_and_the_corners() {
        let (field, _) = build(SculptSliders { head_size: 100.0, ..SculptSliders::default() });
        let rim = field.sample(308.0, 88.0); // head centre + radius, to the right
        assert!(rim[0] > 0.5, "the head rim must move outward (shrink): {rim:?}");
        let centre = field.sample(256.0, 88.0);
        assert!(centre[0].abs() < 0.05 && centre[1].abs() < 0.05, "the centre stays: {centre:?}");
        let corner = field.sample(10.0, 10.0);
        assert!(corner[0].abs() < 1e-6 && corner[1].abs() < 1e-6, "the corner stays: {corner:?}");
    }

    #[test]
    fn the_neck_length_band_stretches_and_the_head_rides() {
        let (field, _) = build(SculptSliders { neck_length: 100.0, ..SculptSliders::default() });
        // Above the chin, inside the ride: the inverse map points down the axis (the head
        // shows the source from below, which reads as moved up).
        let ride = field.sample(256.0, 108.0);
        assert!(ride[1] > 5.0, "the head must ride the stretch: {ride:?}");
        // Below the band's start, and outside the lateral width, nothing moves.
        let below = field.sample(256.0, 300.0);
        assert!(below[0].abs() < 0.05 && below[1].abs() < 0.05, "below the band stays: {below:?}");
        // Past the ramp's full lateral width — the head's side rides within it, the far
        // background does not.
        let beside = field.sample(430.0, 88.0);
        assert!(beside[0].abs() < 0.05 && beside[1].abs() < 0.05, "past the width stays: {beside:?}");
    }

    #[test]
    fn hidden_shoulders_skip_their_rows_and_say_why() {
        let mut anchors = standing();
        for i in [11usize, 12] {
            anchors.pose[i][2] = 0.1;
        }
        let s = SculptSpec { sliders: SculptSliders { shoulders: 50.0, arms: 50.0, head_size: 50.0, ..SculptSliders::default() }, anchors };
        let (_, report) = sculpt_field(&s, bounds(), CELL);
        let skipped: Vec<_> = report.skipped.iter().map(|(n, _)| *n).collect();
        assert!(skipped.contains(&"shoulders") && skipped.contains(&"left upper arm"), "{skipped:?}");
        assert!(report.applied.contains(&"head"), "the head needs no shoulder: {:?}", report.applied);
        assert!(report.skipped.iter().all(|(_, why)| why.contains("landmarks")), "{skipped:?}");
    }

    /// The largest displacement step between adjacent nodes anywhere in the grid — the fold
    /// guard: backward mapping leaves no holes, but a field steeper than the cell size folds
    /// over itself.
    fn max_adjacent_step(field: &LiquifyField) -> f64 {
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
        worst as f64
    }

    #[test]
    fn every_slider_at_full_superposes_smoothly() {
        for sign in [1.0f32, -1.0f32] {
            let sliders = SculptSliders {
                slim_body: 100.0 * sign,
                waist: 100.0 * sign,
                belly: 100.0 * sign,
                arms: 100.0 * sign,
                thighs: 100.0 * sign,
                calves: 100.0 * sign,
                shoulders: 100.0 * sign,
                hips: 100.0 * sign,
                bust: 100.0 * sign,
                head_size: 100.0 * sign,
                neck: 100.0 * sign,
                height: 100.0 * sign,
                leg_length: 100.0 * sign,
                waist_length: 100.0 * sign,
                torso_length: 100.0 * sign,
                neck_length: 100.0 * sign,
            };
            let (field, report) = build(sliders);
            assert!(report.skipped.is_empty(), "{:?}", report.skipped);
            let step = max_adjacent_step(&field);
            assert!(step < CELL, "adjacent nodes step {step:.2}px past the {CELL}px cell — the field folds");
        }
    }
}
