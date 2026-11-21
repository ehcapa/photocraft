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

/// The body's symmetry pairs (left joint, right joint) 对称液化 aligns about the torso
/// axis: shoulders, elbows, wrists, hips, knees, ankles.
pub const SYMMETRY_PAIRS: &[(usize, usize)] = &[(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)];

/// The joints that must be visible (the model's sigmoidged confidence) for a part to run.
const VISIBILITY: f64 = 0.5;

/// How much of a limb's own bone the person silhouette must cover for the limb to shape the
/// covered part when one of its joints falls outside the silhouette: half the bone. Below
/// that the pose's bone is mostly guesswork, and the row is skipped with its reason.
const LIMB_COVERED: f64 = 0.45;

/// Why a row that ran anyway is reported approximate.
const INFERRED_REASON: &str = "the pose barely sees those landmarks; the position is inferred from the silhouette";
const COVERED_REASON: &str = "a landmark lies outside the person silhouette; the covered part of the limb is used";
const WAIST_PRIOR_REASON: &str = "the outline shows no waist, so the row follows the hip joints; the garment's waistband sits at or just above them";

/// How far each part's full effect reaches toward its rim (Liquify's Density, 0..1): the
/// body parts want a broad plateau with the feather at the rim only.
const DENSITY: f64 = 0.6;

/// How far a [`Shape::Profile`]'s envelope reaches beyond the measured half-width at its
/// level. The lateral falloff peaks near half its width, so twice the measured contour puts
/// the peak on the visible rim (the fitted rows' own 2.05).
const PROFILE_REACH: f64 = 2.05;

/// The fraction of a limb's bone each of its two ends reaches past the joint, so the thigh's
/// cap and the calf's overlap at the knee instead of both dying there.
const LIMB_JOINT_OVERLAP: f64 = 0.18;

/// The coupling length per pixel of local displacement: the slimming family's weight decays
/// to nothing over twice the displacement it carries. The smoothstep's steepest slope is
/// 1.5/L, so the weight contributes at most 0.75 to the map's slope — under the fold limit
/// of 1 — at every node, however many rows stack there.
const COUPLING_LENGTH: f64 = 2.0;

/// The shortest coupling length (document px): tiny displacements still get a smooth handoff
/// instead of a hard clip.
const COUPLING_MIN: f64 = 8.0;

/// The twenty-one shaping sliders, each −100..100. A slider's positive direction is its
/// label's action: `waist` narrows, `hips` rounds, `leg_length` lengthens; zero leaves the
/// part out.
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
    /// The composition sliders this crate never reads as rows — `ai_sculpt` (a gentle
    /// whole-body slim) and `swan_neck` (slim + lengthen + square the shoulders) blend
    /// into the parts they name inside [`sculpt_field`] (see [`SculptSliders::composed`]),
    /// so the run path and the stored-params replay compose identically.
    pub ai_sculpt: f32,
    pub straight_shoulders: f32,
    pub swan_neck: f32,
    pub bust_lift: f32,
    pub double_chin: f32,
}

impl SculptSliders {
    /// The sliders after the composition family blends in: AI塑形 adds a gentle whole-body
    /// slim across the shaping parts (the rounder zones — hips, bust, head — stay out; the
    /// slider means the silhouette, not the figure), 天鹅颈 adds its neck trio. The blend
    /// happens here, inside the one pure spec → field function, so a stored filter's raw
    /// sliders compose to the same field bit for bit.
    pub fn composed(self) -> Self {
        let blend = |own: f32, extra: f32| (own + extra).clamp(-100.0, 100.0);
        let (ai, swan) = (self.ai_sculpt, self.swan_neck);
        Self {
            slim_body: blend(self.slim_body, ai * 0.60),
            waist: blend(self.waist, ai * 0.50),
            belly: blend(self.belly, ai * 0.40),
            arms: blend(self.arms, ai * 0.50),
            thighs: blend(self.thighs, ai * 0.50),
            calves: blend(self.calves, ai * 0.35),
            shoulders: blend(self.shoulders, ai * 0.25),
            neck: blend(self.neck, swan * 0.60),
            neck_length: blend(self.neck_length, swan * 0.40),
            straight_shoulders: blend(self.straight_shoulders, swan * 0.50),
            ..self
        }
    }
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

/// The eight limb segments [`BodyFit::limbs`] answers for, in its order: upper arm, forearm,
/// thigh and calf on the pose's left, then the same on its right. The table's rows and the
/// analysis that fills the fit both read this list, so the two cannot disagree.
pub const LIMB_SEGMENTS: [(usize, usize); 8] = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (24, 26), (25, 27), (26, 28)];

/// One sampled row of the person silhouette: its y and the horizontal runs the mask fills.
/// Several runs because a raised arm or a gap between the legs is a separate run; the field's
/// clip reads them all, and the torso fit takes the run that holds the torso axis.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BodyRow {
    pub y: f32,
    pub runs: Vec<[f32; 2]>,
}

/// One torso level the slimming rows act on, at `t` along shoulder-mid → hip-mid, with the
/// silhouette's half-widths on each side of the axis. Half-widths, not absolute x: the levels
/// travel with the anchors, and the axis is rebuilt from them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LevelFit {
    pub t: f32,
    pub left: f32,
    pub right: f32,
}

/// One limb's half-thickness read from the person mask at its bone, and the fraction of the
/// bone the mask actually covers (a cropped or skirt-hidden end clips the row's band instead
/// of killing it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LimbFit {
    pub radius: f32,
    pub t0: f32,
    pub t1: f32,
}

/// The person silhouette fitted to the body's own geometry: torso levels, limb thickness and
/// the rows the field clips to. `portrait.analyze` computes it once from the person mask; the
/// command stores it in its params, so a smart filter replays the same field without the mask.
/// Without one (old filters, or anchors sent alone) the parts fall back to joint spacing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BodyFit {
    pub rows: Vec<BodyRow>,
    pub waist: LevelFit,
    pub belly: LevelFit,
    pub hip: LevelFit,
    pub bust: LevelFit,
    pub limbs: [LimbFit; 8],
    /// Whether each pose joint lies inside the silhouette: an occluded joint inside it is
    /// still a usable estimate; one outside it makes the row suspect, so it is skipped unless
    /// it is a limb whose covered bone span is long enough to shape. A `Vec` rather than
    /// `[bool; POSE_COUNT]` because serde has no array impl past 32.
    pub inside: Vec<bool>,
    /// The waist row fell back to the hip joints because the mask showed no crease, rather
    /// than to a measured waist: the field reports the waist and belly as approximate so a
    /// garment whose outline hides the cinch is not mistaken for an exact measurement.
    #[serde(default)]
    pub waist_prior: bool,
}

impl Default for BodyFit {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            waist: LevelFit::default(),
            belly: LevelFit::default(),
            hip: LevelFit::default(),
            bust: LevelFit::default(),
            limbs: std::array::from_fn(|_| LimbFit::default()),
            inside: vec![false; POSE_COUNT],
            waist_prior: false,
        }
    }
}

impl BodyFit {
    /// Whether the fit is usable: finite, ordered rows within the size the field will read.
    /// A malformed or hostile `fit` from params is ignored (the parts fall back to the joints)
    /// rather than trusted or allowed to size an allocation.
    pub fn is_valid(&self) -> bool {
        self.rows.len() >= 2
            && self.rows.len() <= 4096
            && self.inside.len() == POSE_COUNT
            && self.rows.windows(2).all(|w| w[0].y <= w[1].y)
            && self
                .rows
                .iter()
                .all(|r| r.y.is_finite() && r.runs.len() <= 128 && r.runs.iter().all(|run| run.iter().all(|v| v.is_finite()) && run[0] <= run[1]))
            && [self.waist, self.belly, self.hip, self.bust]
                .iter()
                .all(|l| l.t.is_finite() && l.left.is_finite() && l.right.is_finite() && l.left >= 0.0 && l.right >= 0.0)
            && self.limbs.iter().all(|l| {
                l.radius.is_finite() && l.t0.is_finite() && l.t1.is_finite() && l.radius >= 0.0 && (0.0..=1.0).contains(&l.t0) && (0.0..=1.0).contains(&l.t1)
            })
    }

    /// The widest half-width the levels measured: the scale the contract/radial fold budget
    /// is set against.
    fn max_half(&self) -> f64 {
        [self.waist, self.belly, self.hip, self.bust].iter().map(|l| f64::from(l.left.max(l.right))).fold(0.0, f64::max)
    }
}

/// The sliders plus the anchors they shape: the whole input of [`sculpt_field`].
#[derive(Clone, Debug, PartialEq)]
pub struct SculptSpec {
    pub sliders: SculptSliders,
    pub anchors: SculptAnchors,
    /// The person-mask fit `portrait.analyze` measured, when the caller has one.
    pub fit: Option<BodyFit>,
}

/// What a build did: the rows whose slider asked and landmarks allowed, and the rows that
/// asked but were left out (a limb hidden from the pose, anchors too close to name an axis)
/// with the reason — the command surfaces this, so a slider that does nothing explains
/// itself instead of reading as a dead control. `approximate` names the rows that ran on an
/// inference — a landmark the pose barely saw (an occluded knee inside the silhouette), or
/// a limb shaped over the part of its bone the silhouette covers when a joint fell outside
/// it: they did something, but the position is inferred.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SculptReport {
    pub applied: Vec<&'static str>,
    pub approximate: Vec<(&'static str, &'static str)>,
    pub skipped: Vec<(&'static str, &'static str)>,
}

// ---------- the frame: the anchors as f64 geometry ----------

/// The anchors resolved once per build: joints as `[x, y]` pairs with their visibility, the
/// head and the chin. Every part's shape is derived from this.
struct Frame<'a> {
    p: Vec<[f64; 2]>,
    vis: Vec<f64>,
    chin: [f64; 2],
    head_c: [f64; 2],
    head_r: f64,
    /// The person-mask fit, when one came with the spec and passed its sanity checks.
    fit: Option<&'a BodyFit>,
}

impl<'a> Frame<'a> {
    /// `None` when the anchors are not a pose this crate understands (a joint count other
    /// than 33, or non-finite points): the caller reports and applies nothing rather than
    /// warping from garbage. A `fit` that fails [`BodyFit::is_valid`] is ignored, not
    /// trusted: the parts then use joint spacing, as an older stored filter would.
    fn new(a: &SculptAnchors, fit: Option<&'a BodyFit>) -> Option<Frame<'a>> {
        if a.pose.len() != POSE_COUNT || !a.doc.iter().chain(&a.chin).chain(a.head.centre.iter()).all(|v| v.is_finite()) {
            return None;
        }
        let (p, vis): (Vec<_>, Vec<_>) = a.pose.iter().map(|q| ([q[0] as f64, q[1] as f64], q[2] as f64)).unzip();
        if p.iter().flatten().chain(vis.iter()).any(|v| !v.is_finite()) {
            return None;
        }
        let head_r = if a.head.radius.is_finite() && a.head.radius > 0.0 { a.head.radius as f64 } else { 0.0 };
        Some(Frame {
            p,
            vis,
            chin: [a.chin[0] as f64, a.chin[1] as f64],
            head_c: [a.head.centre[0] as f64, a.head.centre[1] as f64],
            head_r,
            fit: fit.filter(|f| f.is_valid()),
        })
    }

    fn pt(&self, i: usize) -> [f64; 2] {
        self.p[i]
    }

    fn visible(&self, i: usize) -> bool {
        self.vis[i] >= VISIBILITY
    }

    /// Whether a part may run and how strongly, plus the approximate reason when it runs on
    /// an inference. With a fit, a joint inside the silhouette is usable even when the pose's
    /// visibility is low (occlusion, not absence): it runs at 55–100% and reports as
    /// approximate below the visibility gate. A joint the silhouette does not contain is the
    /// pose guessing outside the body, and the row is skipped with that reason — except for a
    /// limb, whose covered bone span the caller may still use. Without a fit the old
    /// all-or-nothing gate stands.
    fn confidence(&self, needs: &[usize]) -> Result<(f64, Option<&'static str>), &'static str> {
        let Some(fit) = self.fit else {
            return if needs.iter().all(|&i| self.visible(i)) { Ok((1.0, None)) } else { Err("the pose does not see those landmarks") };
        };
        let mut min_vis = 1.0f64;
        for &i in needs {
            if !fit.inside.get(i).copied().unwrap_or(false) {
                return Err("the landmark is outside the person silhouette");
            }
            min_vis = min_vis.min(self.vis[i].clamp(0.0, 1.0));
        }
        if min_vis >= VISIBILITY { Ok((1.0, None)) } else { Ok((0.55 + 0.45 * min_vis / VISIBILITY, Some(INFERRED_REASON))) }
    }

    /// The fraction of the limb `i`→`j`'s own bone the person silhouette covers, when the fit
    /// measured that limb: the band [`limb`] shapes, nothing when the fit is absent or the
    /// limb unmeasured.
    fn limb_covered(&self, i: usize, j: usize) -> Option<f64> {
        let fit = self.fit?;
        let idx = LIMB_SEGMENTS.iter().position(|&(a, b)| a == i && b == j)?;
        let limb = fit.limbs.get(idx)?;
        (limb.radius > 1.0).then(|| (f64::from(limb.t1) - f64::from(limb.t0)).clamp(0.0, 1.0))
    }

    /// A torso level's two half-widths, each at least a few pixels so a fold in the
    /// silhouette cannot collapse a row to nothing.
    fn level_half(&self, level: &LevelFit) -> (f64, f64) {
        (f64::from(level.left).max(4.0), f64::from(level.right).max(4.0))
    }

    /// The torso's width profile as `(t, half-width)` knots sorted by `t`: the fit's four
    /// measured levels (bust, waist, belly, hip), or the joint-spacing estimate when there is
    /// no fit. Sorted here, not trusted: `t` comes from params, and the profile writer blends
    /// between neighbours.
    fn torso_levels(&self) -> [(f64, f64); 4] {
        let mut levels = match self.fit {
            Some(fit) => [&fit.bust, &fit.waist, &fit.belly, &fit.hip].map(|l| {
                let half = f64::from(l.left.max(l.right)).max(4.0);
                (f64::from(l.t), half)
            }),
            None => {
                let half = self.shoulder_w() * 0.425;
                [(0.0, half), (0.30, half), (0.70, half), (1.0, half)]
            }
        };
        levels.sort_by(|a, b| a.0.total_cmp(&b.0));
        levels
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

    fn profile(&self, a: [f64; 2], b: [f64; 2], levels: [(f64, f64); 4], kernel: Kernel, gain: f64) -> Option<Shape> {
        (norm(sub(b, a)) >= 4.0).then_some(Shape::Profile { a, b, levels, kernel, gain })
    }

    fn ramp(&self, a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64) -> Option<Shape> {
        (norm(sub(b, a)) >= 4.0 && width >= 4.0 && band.1 > band.0).then_some(Shape::Ramp { a, b, band, width, gain })
    }

    fn shift(&self, a: [f64; 2], b: [f64; 2], dir: [f64; 2], band: (f64, f64), width: f64, gain: f64) -> Option<Shape> {
        (norm(sub(b, a)) >= 4.0 && norm(dir) > 1e-9 && width >= 4.0 && band.1 > band.0).then_some(Shape::Shift { a, b, dir, band, width, gain })
    }
}

/// The longitudinal profile of a [`Shape::Profile`]: how much of the row's pull a level of
/// the torso receives, as a function of `t` along the shoulder→hip segment. Both kernels are
/// C1 with zero slope at their ends, so a row's effect blends into its neighbours' instead of
/// stopping at a window edge — the isolated-band dent a hard window produced.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Kernel {
    /// A raised cosine: 1 at `centre`, 0 with zero slope at `centre ± half`. The waist and
    /// belly are bumps: strongest on their measured row, tapering continuously toward the
    /// ribs and the hips.
    Bump { centre: f64, half: f64 },
    /// Smoothstep up over `ease`, full between `lo + ease` and `hi - ease`, smoothstep down.
    /// The whole-torso rows (slim body) use it: a broad plateau, not a band.
    Plateau { lo: f64, hi: f64, ease: f64 },
}

impl Kernel {
    fn at(&self, t: f64) -> f64 {
        match *self {
            Kernel::Bump { centre, half } => {
                if half <= 1e-9 {
                    return 1.0;
                }
                let u = ((t - centre) / half).clamp(-1.0, 1.0);
                0.5 * (1.0 + (std::f64::consts::PI * u).cos())
            }
            Kernel::Plateau { lo, hi, ease } => {
                let ease = ease.max(0.0);
                smoothstep(lo, lo + ease, t) * (1.0 - smoothstep(hi - ease, hi, t))
            }
        }
    }
}

/// One part's resolved geometry — shared with the face table ([`crate::face_sculpt`]):
/// the panel's parts are rows over these primitives, nothing more.
pub(crate) enum Shape {
    /// Scale the lateral distance to a segment: the displacement grows from nothing on the
    /// axis to the rim pull at the falloff's shoulder, inside a longitudinal window over
    /// `band` (t along the segment) with smoothstep ramps at its ends. `gain` is the rim
    /// pull at slider 100 as a fraction of `width`; the field points outward (the inverse
    /// map samples farther out, which reads as contraction), so a negative gain expands.
    Contract { a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64 },
    /// A contraction whose lateral width and longitudinal strength follow the torso's own
    /// measured width profile: `levels` are `(t, half-width)` knots sorted by `t` (the fit's
    /// bust, waist, belly and hip), and the envelope at any `t` reaches [`PROFILE_REACH`]
    /// times the interpolated half-width, so the pull lands on the visible contour at every
    /// level. `kernel` sets how strongly the row acts at each level; one continuous shape
    /// replaces the per-level bands, so waist, belly and slim compose without dents.
    Profile { a: [f64; 2], b: [f64; 2], levels: [(f64, f64); 4], kernel: Kernel, gain: f64 },
    /// The same scaling about one centre: the head, the bust, the hips.
    Radial { c: [f64; 2], radius: f64, gain: f64 },
    /// Stretch along a→b by `gain` of the segment's length: the smoothstep ramp is zero
    /// before the band, full at and beyond it — everything beyond `b` rides rigidly (the
    /// torso rides a leg stretch, the head rides a neck stretch), then tapers off over
    /// t∈[1.5, 2.5] so the ride ends smoothly instead of at a seam. Laterally the falloff
    /// is about the axis line over `width`.
    Ramp { a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64 },
    /// Translate a band rigidly along a fixed direction: the whole zone rides `Δ =
    /// gain · k · len` (len = the segment's own length as the scale), laterally a
    /// plateau-plus-feather like a ramp's (the core rides whole), longitudinally the
    /// smoothstep window over `band`. The inverse-map displacement points against `dir`.
    Shift { a: [f64; 2], b: [f64; 2], dir: [f64; 2], band: (f64, f64), width: f64, gain: f64 },
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

impl Part {
    /// The limb segment this row shapes, when it is one (its two needed joints name a bone).
    /// A limb can keep shaping the covered part of its bone when a joint falls outside the
    /// person silhouette; a torso row cannot (its envelope needs both ends).
    fn limb(&self) -> Option<(usize, usize)> {
        match *self.needs {
            [i, j] if LIMB_SEGMENTS.contains(&(i, j)) => Some((i, j)),
            _ => None,
        }
    }

    /// Whether the row stretches the whole figure from the ankles: Height and Leg Length. These
    /// move everything above their lower joint, so a feet position the pose only guessed would
    /// stretch the wrong part of the frame at a strength the user cannot see — on a portrait
    /// whose legs are out of frame they moved the picture by a handful of pixels and read as a
    /// dead slider. Unlike a local shaping row, they need the pose to have *seen* both feet.
    fn needs_both_feet(&self) -> bool {
        self.needs.contains(&27) && self.needs.contains(&28)
    }
}

/// Whether a row is placed off the fitted silhouette when there is one, and so is only
/// approximate when there is none: the torso width profile and the hip and bust radials.
/// Limb and shoulder rows are joint geometry in either case.
fn fit_placed(name: &str) -> bool {
    matches!(name, "slim body" | "waist" | "belly" | "left hip" | "right hip" | "left bust" | "right bust")
}

/// The parts, table order = build order (the superposition is order-stable, which is part of
/// the determinism the smart filter relies on).
const PARTS: &[Part] = &[
    Part {
        name: "slim body",
        slider: |s| s.slim_body,
        // The torso only: the limbs have their own rows, and a contract about the torso axis
        // would pull both legs toward the centre instead of slimming each. The old row also
        // needed the ankles, so a cropped or skirt-hidden pair killed the whole slider.
        needs: &[11, 12, 23, 24],
        shape: |f| {
            // One continuous envelope down the whole torso, its width following the measured
            // contour at every level and its strength a broad plateau: the slim slider reads
            // as the silhouette narrowing, not as bands that end in seams.
            let levels = f.torso_levels();
            let gain = if f.fit.is_some() { 0.14 } else { 0.10 };
            f.profile(f.shoulder_mid(), f.hip_mid(), levels, Kernel::Plateau { lo: 0.0, hi: 1.05, ease: 0.25 }, gain)
        },
    },
    Part {
        name: "waist",
        slider: |s| s.waist,
        needs: &[11, 12, 23, 24],
        shape: |f| {
            // A bump centred on the measured waist, wide enough to blend into the ribs above
            // and the hips below: the old ±8.5% window acted on a narrow band and left the
            // torso dented above and below it. The level's `t` still places the peak; the
            // tail is what makes the slider read as a continuous narrowing.
            // Without a fit nothing measured the torso, so the pose's hip row stands in: on a
            // clothed subject the hip joints mark the visible hip line, and a garment's
            // waistband sits at or just above it, so the slider lands where the photo cinches.
            // The old anatomical middles (0.54, then 0.62) assumed bare anatomy; under any
            // garment they put the slider on the ribs, which is what made the fallback read
            // as squeezing the chest.
            let centre = f.fit.map_or(1.0, |fit| f64::from(fit.waist.t));
            let gain = if f.fit.is_some() { 0.18 } else { 0.10 };
            f.profile(f.shoulder_mid(), f.hip_mid(), f.torso_levels(), Kernel::Bump { centre, half: 0.26 }, gain)
        },
    },
    Part {
        name: "belly",
        slider: |s| s.belly,
        needs: &[11, 12, 23, 24],
        shape: |f| {
            // The measured row when the silhouette has one. Without a fit the lower abdomen
            // sits about a tenth of the torso above the hip line the waist fallback uses, so
            // the belly reads above the waistband and neither fallback touches the ribs.
            let centre = f.fit.map_or(0.90, |fit| f64::from(fit.belly.t));
            let gain = if f.fit.is_some() { 0.17 } else { 0.08 };
            f.profile(f.shoulder_mid(), f.hip_mid(), f.torso_levels(), Kernel::Bump { centre, half: 0.22 }, gain)
        },
    },
    Part {
        name: "shoulders",
        slider: |s| s.shoulders,
        needs: &[11, 12],
        shape: |f| {
            let a = add(f.shoulder_mid(), scale(f.u_torso(), -0.12 * f.torso_len()));
            let b = add(f.shoulder_mid(), scale(f.u_torso(), 0.28 * f.torso_len()));
            // The contract's lateral falloff peaks near half its width from the axis, and the
            // visible shoulder line (a sleeve, the deltoid) sits well outside the joint span:
            // the old 0.75 of that span put the peak inside the shoulder and left the rim
            // almost still, so the slider read as a dent rather than a narrower shoulder.
            // The ramp width is the one reach that covers the whole silhouette, arms included.
            f.contract(a, b, (0.0, 1.0), f.ramp_width(), 0.07)
        },
    },
    // Straight shoulders: the trapezius slope contracts toward the shoulder line itself
    // (a horizontal axis), which reads as the line flattening — the slope's rim pulls down
    // to it the same way a waist's rim pulls in.
    Part {
        name: "straight shoulders",
        slider: |s| s.straight_shoulders,
        needs: &[11, 12],
        shape: |f| f.contract(f.pt(11), f.pt(12), (0.0, 1.0), f.shoulder_w() * 0.38, 0.08),
    },
    Part { name: "left upper arm", slider: |s| s.arms, needs: &[11, 13], shape: |f| arm(f, 11, 13, 0.26) },
    Part { name: "left forearm", slider: |s| s.arms, needs: &[13, 15], shape: |f| arm(f, 13, 15, 0.26) },
    Part { name: "right upper arm", slider: |s| s.arms, needs: &[12, 14], shape: |f| arm(f, 12, 14, 0.26) },
    Part { name: "right forearm", slider: |s| s.arms, needs: &[14, 16], shape: |f| arm(f, 14, 16, 0.26) },
    Part { name: "left thigh", slider: |s| s.thighs, needs: &[23, 25], shape: |f| limb(f, 23, 25, 0.42, 0.24) },
    Part { name: "right thigh", slider: |s| s.thighs, needs: &[24, 26], shape: |f| limb(f, 24, 26, 0.42, 0.24) },
    Part { name: "left calf", slider: |s| s.calves, needs: &[25, 27], shape: |f| limb(f, 25, 27, 0.32, 0.22) },
    Part { name: "right calf", slider: |s| s.calves, needs: &[26, 28], shape: |f| limb(f, 26, 28, 0.32, 0.22) },
    // The hips and the bust expand on their positive direction (the labels mean fuller), so
    // their gains are negative: the field points inward, which reads as expansion.
    Part { name: "left hip", slider: |s| s.hips, needs: &[23, 24], shape: |f| hip(f, 23) },
    Part { name: "right hip", slider: |s| s.hips, needs: &[23, 24], shape: |f| hip(f, 24) },
    Part { name: "left bust", slider: |s| s.bust, needs: &[11, 12, 23, 24], shape: |f| bust(f, -1.0) },
    Part { name: "right bust", slider: |s| s.bust, needs: &[11, 12, 23, 24], shape: |f| bust(f, 1.0) },
    // The bust-lift band rides up the torso axis (the third of it the bust occupies).
    Part {
        name: "bust lift",
        slider: |s| s.bust_lift,
        needs: &[11, 12, 23, 24],
        shape: |f| {
            let a = lerp(f.shoulder_mid(), f.hip_mid(), 0.18);
            let b = lerp(f.shoulder_mid(), f.hip_mid(), 0.50);
            f.shift(a, b, scale(f.u_torso(), -1.0), (0.05, 0.95), f.shoulder_w() * 0.70, 0.07)
        },
    },
    Part { name: "head", slider: |s| s.head_size, needs: &[], shape: |f| f.radial(f.head_c, f.head_radius() * 1.40, 0.09) },
    Part { name: "neck", slider: |s| s.neck, needs: &[11, 12], shape: |f| f.contract(f.chin, f.shoulder_mid(), (0.10, 0.95), f.shoulder_w() * 0.36, 0.08) },
    // Double chin: the submental bulge contracts toward the short under-chin axis —
    // PixCake files this under flaw removal; the liquify approximation rides the panel.
    Part {
        name: "double chin",
        slider: |s| s.double_chin,
        needs: &[11, 12],
        shape: |f| {
            let b = lerp(f.chin, f.shoulder_mid(), 0.30);
            f.contract(f.chin, b, (0.05, 0.95), f.head_radius() * 0.85, 0.09)
        },
    },
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

/// One limb segment (`i`→`j`). With a fit, the brush width is twice the thickness the person
/// mask measured at the bone and the band is the part of the bone the mask covers — the
/// field's peak lands on the limb's own surface instead of dying before it, and a cropped or
/// skirt-hidden end shortens the row rather than skipping it. Without a fit the old
/// bone-length fraction stands.
fn limb(f: &Frame, i: usize, j: usize, width_frac: f64, gain: f64) -> Option<Shape> {
    // Every limb row must reach past its joints: a band that ramps to zero exactly at the
    // joint left the knee and elbow still while the bones either side moved, reading as a
    // pinch; overlapping the neighbouring bone's cap keeps the chain continuous.
    if let Some(fit) = f.fit
        && let Some(idx) = LIMB_SEGMENTS.iter().position(|&(a, b)| a == i && b == j)
    {
        let l = fit.limbs[idx];
        let radius = f64::from(l.radius);
        let (t0, t1) = (f64::from(l.t0).clamp(0.0, 0.98), f64::from(l.t1).clamp(0.02, 1.0));
        if radius > 1.0 && t1 - t0 >= 0.10 {
            let band = (t0 - LIMB_JOINT_OVERLAP, t1 + LIMB_JOINT_OVERLAP);
            return f.contract(f.pt(i), f.pt(j), band, (2.0 * radius).max(16.0), gain);
        }
    }
    let len = norm(sub(f.pt(j), f.pt(i)));
    let band = (0.05 - LIMB_JOINT_OVERLAP, 0.95 + LIMB_JOINT_OVERLAP);
    f.contract(f.pt(i), f.pt(j), band, (len * width_frac).max(12.0), gain)
}

/// An arm segment, a limb with a slimmer fraction.
fn arm(f: &Frame, i: usize, j: usize, gain: f64) -> Option<Shape> {
    limb(f, i, j, 0.32, gain)
}

/// One hip: a radial expansion placed halfway between the torso axis and the fitted hip
/// contour, with the envelope reaching twice that distance so its peak lands on the visible
/// curve. Without a fit it keeps the joint-spacing estimate.
fn hip(f: &Frame, i: usize) -> Option<Shape> {
    if let Some(fit) = f.fit {
        let (left, right) = f.level_half(&fit.hip);
        let side = if f.pt(i)[0] < f.hip_mid()[0] { -1.0 } else { 1.0 };
        let half = if side < 0.0 { left } else { right };
        let axis = lerp(f.shoulder_mid(), f.hip_mid(), f64::from(fit.hip.t).clamp(0.0, 1.1));
        let c = [axis[0] + side * 0.45 * half, axis[1]];
        return f.radial(c, (1.10 * half).max(18.0), -0.20);
    }
    let out = sub(f.pt(i), f.hip_mid());
    let out_n = norm(out);
    let out = if out_n < 1e-9 { [0.0, 0.0] } else { scale(out, 1.0 / out_n) };
    let c = add(add(f.pt(i), scale(out, 0.10 * f.shoulder_w())), scale(f.u_torso(), 0.06 * f.torso_len()));
    f.radial(c, (f.hip_w() * 0.55).max(20.0), -0.08)
}

/// One side of the bust: the same fitted construction at the bust level, on the torso axis
/// a third of the way down from the shoulders. Without a fit it keeps the joint-spacing
/// estimate.
fn bust(f: &Frame, side: f64) -> Option<Shape> {
    if let Some(fit) = f.fit {
        let (left, right) = f.level_half(&fit.bust);
        let half = if side < 0.0 { left } else { right };
        let axis = lerp(f.shoulder_mid(), f.hip_mid(), f64::from(fit.bust.t).clamp(0.0, 1.0));
        let c = [axis[0] + side * 0.45 * half, axis[1]];
        return f.radial(c, (1.10 * half).max(18.0), -0.18);
    }
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

/// The contract/radial family's shared fold budget: their peak |Δ|s may sum to this multiple
/// of the fitted torso half-width before the family scales down together. The slimming rows
/// all act around the same axis and stack (waist + belly + slim at once), so the budget is
/// what keeps a full-slider stack an order below the cell size.
const SHAPE_BUDGET: f64 = 0.35;

/// The field of `spec` over `bounds` (`cell` document pixels per node, [`LiquifyField::new`]'s
/// own clamps apply). Pure and deterministic: the same spec, bounds and cell build the same
/// field bit for bit, which is what a stored smart filter replays on.
pub fn sculpt_field(spec: &SculptSpec, bounds: Rect, cell: f64) -> (LiquifyField, SculptReport) {
    let mut field = LiquifyField::new(bounds, cell);
    let mut report = SculptReport::default();
    let Some(frame) = Frame::new(&spec.anchors, spec.fit.as_ref()) else { return (field, report) };
    // Resolve every asked part before writing any: the length family shares one fold budget
    // ([`RAMP_BUDGET`]) and the slimming family another ([`SHAPE_BUDGET`]), which needs all
    // their magnitudes up front.
    let mut asked: Vec<(&'static Part, f64, Shape, Option<&'static str>)> = Vec::new();
    let sliders = spec.sliders.composed();
    for part in PARTS {
        let slider = (part.slider)(&sliders);
        if slider.abs() < 0.5 {
            continue; // the slider is at zero: nothing was asked of this row
        }
        if part.needs_both_feet() && !(frame.visible(27) && frame.visible(28)) {
            report.skipped.push((part.name, "the pose does not see both feet, so the photo does not show the legs to stretch"));
            continue;
        }
        let (confidence, approximate) = match frame.confidence(part.needs) {
            Ok(found) => found,
            Err(why) => {
                // A limb's joint can sit a few pixels outside the matte while its bone runs
                // inside it (an arm folded behind the head, a calf against a skirt). The
                // pose must still see the joints; then shape the covered part and say the
                // row is approximate rather than drop a limb the photo plainly shows.
                let covered = part.limb().filter(|_| part.needs.iter().all(|&k| frame.visible(k))).and_then(|(i, j)| frame.limb_covered(i, j));
                if covered.is_some_and(|c| c >= LIMB_COVERED) {
                    (1.0, Some(COVERED_REASON))
                } else {
                    report.skipped.push((part.name, why));
                    continue;
                }
            }
        };
        let Some(shape) = (part.shape)(&frame) else {
            report.skipped.push((part.name, "the anchors are too close to name an axis"));
            continue;
        };
        asked.push((part, f64::from(slider / 100.0).clamp(-1.0, 1.0) * confidence, shape, approximate));
    }
    // Without a fitted silhouette every torso row places itself from the joints alone: the
    // photo's own outline was not measurable (a matte fused with the background), so the
    // region is approximate and the report says so rather than showing it as measured.
    if spec.fit.is_none() {
        for (part, ..) in &asked {
            if fit_placed(part.name) {
                report.approximate.push((part.name, "the photo's background blends with the body, so the region follows the pose, not the visible outline"));
            }
        }
    }
    // The fit measured no crease and placed the waist from the hip joints instead: the rows
    // that row feeds are approximate, and the report says so rather than showing them as a
    // measured waist.
    if spec.fit.as_ref().is_some_and(|fit| fit.waist_prior) {
        for (part, ..) in &asked {
            if matches!(part.name, "slim body" | "waist" | "belly") {
                report.approximate.push((part.name, WAIST_PRIOR_REASON));
            }
        }
    }
    let budget = RAMP_BUDGET * frame.ramp_width();
    let ramp_sum: f64 = asked.iter().map(|&(_, k, ref s, _)| ramp_magnitude(s, k)).sum();
    let ramp_scale = if ramp_sum > budget && ramp_sum > 0.0 { budget / ramp_sum } else { 1.0 };
    let shape_budget = SHAPE_BUDGET * frame.fit.map_or(frame.person_w(), |fit| fit.max_half().max(24.0));
    let shape_sum: f64 = asked.iter().map(|&(_, k, ref s, _)| shape_magnitude(s, k)).sum();
    let shape_scale = if shape_sum > shape_budget && shape_sum > 0.0 { shape_budget / shape_sum } else { 1.0 };
    // The shaping family is built into its own field first: its participation weight has to
    // scale the family's *total* displacement at each node, not each row's separately, so a
    // stack of rows contributes one bounded slope. Length ramps and shifts keep their own
    // lateral plateau (they move the silhouette and the background together to avoid a seam)
    // and are written straight into the field.
    let mut shaping = LiquifyField::new(bounds, cell);
    for (part, k, shape, approximate) in asked {
        let k = match &shape {
            Shape::Contract { .. } | Shape::Radial { .. } | Shape::Profile { .. } => k * shape_scale,
            Shape::Ramp { .. } => k * ramp_scale,
            Shape::Shift { .. } => k,
        };
        if let Some(why) = approximate {
            report.approximate.push((part.name, why));
        }
        match shape {
            Shape::Ramp { .. } | Shape::Shift { .. } => write_shape(&mut field, &shape, k, None),
            _ => write_shape(&mut shaping, &shape, k, None),
        }
        report.applied.push(part.name);
    }
    couple_to_person(&mut field, &shaping, spec.fit.as_ref().filter(|fit| fit.is_valid()));
    (field, report)
}

/// Adds the shaping family to the field, coupled to the person silhouette: inside the person
/// the whole displacement stands; outside, it decays to nothing over twice its own local
/// magnitude ([`COUPLING_LENGTH`]). The silhouette and the background beside it therefore
/// move together — a slim drags the near background in, a fuller hip pushes it out — while
/// the far background stays, and no slider stack can fold the map on the weight alone. This
/// replaces the old near-binary person clip, whose short fixed feather left the field frozen
/// just outside the outline (visible as a ridge along it) and could not move the background
/// out at all.
fn couple_to_person(field: &mut LiquifyField, shaping: &LiquifyField, fit: Option<&BodyFit>) {
    let n = field.w * field.h;
    let Some(fit) = fit else {
        for idx in 0..n.min(shaping.d.len()).min(field.d.len()) {
            field.d[idx][0] += shaping.d[idx][0];
            field.d[idx][1] += shaping.d[idx][1];
        }
        return;
    };
    if shaping.w != field.w || shaping.h != field.h || shaping.d.len() != n {
        // The two fields are built from the same bounds and cell; anything else is a bug,
        // and dropping the shapes would be worse than running them uncoupled.
        for idx in 0..n.min(shaping.d.len()) {
            field.d[idx][0] += shaping.d[idx][0];
            field.d[idx][1] += shaping.d[idx][1];
        }
        return;
    }
    let dist = silhouette_distance(fit, field);
    for idx in 0..n {
        let d = shaping.d[idx];
        let m = f64::from(d[0]).hypot(f64::from(d[1]));
        if !m.is_finite() || m <= 0.0 {
            continue; // zero, or a NaN the frame's checks should have stopped
        }
        let l = (COUPLING_LENGTH * m).max(COUPLING_MIN);
        let w = 1.0 - smoothstep(0.0, l, f64::from(dist[idx]));
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        field.d[idx][0] += (f64::from(d[0]) * w) as f32;
        field.d[idx][1] += (f64::from(d[1]) * w) as f32;
    }
}

/// A ramp's |Δ| in pixels — the fold budget's currency.
fn ramp_magnitude(s: &Shape, k: f64) -> f64 {
    match s {
        Shape::Ramp { a, b, gain, .. } => gain.abs() * k.abs() * norm(sub(*b, *a)),
        _ => 0.0,
    }
}

/// A contract/profile/radial's peak |Δ| in pixels, the same currency for the shaping family
/// (the lateral falloff's shoulder is where the profile's peak sits, at ~0.35 of its width).
fn shape_magnitude(s: &Shape, k: f64) -> f64 {
    match *s {
        Shape::Contract { width, gain, .. } => gain.abs() * k.abs() * width * 0.35,
        Shape::Profile { levels, gain, .. } => {
            let half = levels.iter().map(|&(_, h)| h).fold(0.0, f64::max).max(4.0);
            gain.abs() * k.abs() * PROFILE_REACH * half * 0.35
        }
        Shape::Radial { radius, gain, .. } => gain.abs() * k.abs() * radius * 0.35,
        _ => 0.0,
    }
}

/// The per-node distance from the node to the fitted person silhouette (0 inside it), in
/// document pixels. Between two sampled rows the two rows' horizontal distances are blended
/// with a smoothstep, so the distance is smooth along y instead of jumping at the row spacing
/// (the old nearest-row clip turned that jump into a visible ridge); the smoothstep's zero
/// slope at each row also keeps its own y-slope bounded. Beyond the first and last rows the
/// vertical distance counts. One 1-2-1 pass along y softens the remaining kink where a run
/// starts or ends (the top of a hand beside the hip), which is the one place the two rows'
/// distances can disagree sharply.
fn silhouette_distance(fit: &BodyFit, field: &LiquifyField) -> Vec<f32> {
    let n = field.w * field.h;
    let mut out = vec![0.0f32; n];
    let (Some(first), Some(last)) = (fit.rows.first(), fit.rows.last()) else { return out };
    let (top, bottom) = (f64::from(first.y), f64::from(last.y));
    // Rows and field node rows are both ordered by y, so the bracketing rows are found by one
    // forward cursor instead of a scan per node — a 24 MP field is a million nodes.
    let mut near = 0usize;
    for j in 0..field.h {
        let y = field.node_pos(0, j)[1];
        if !y.is_finite() {
            continue;
        }
        while near + 1 < fit.rows.len() && f64::from(fit.rows[near + 1].y) <= y {
            near += 1;
        }
        let (r0, r1, t) = if y < top {
            (&fit.rows[0], &fit.rows[0], 0.0)
        } else if y >= bottom {
            let last = &fit.rows[fit.rows.len() - 1];
            (last, last, 0.0)
        } else {
            let y0 = f64::from(fit.rows[near].y);
            let y1 = f64::from(fit.rows[near + 1].y);
            let t = if y1 > y0 { ((y - y0) / (y1 - y0)).clamp(0.0, 1.0) } else { 0.0 };
            (&fit.rows[near], &fit.rows[near + 1], t)
        };
        let dy = if y < top {
            top - y
        } else if y > bottom {
            y - bottom
        } else {
            0.0
        };
        let blend = smoothstep(0.0, 1.0, t);
        for i in 0..field.w {
            let x = field.node_pos(i, j)[0];
            if !x.is_finite() {
                out[j * field.w + i] = f32::INFINITY;
                continue;
            }
            let dx0 = runs_distance(r0, x);
            let dx1 = runs_distance(r1, x);
            let dx = dx0 + (dx1 - dx0) * blend;
            out[j * field.w + i] = dx.hypot(dy) as f32;
        }
    }
    let mut softened = out.clone();
    for j in 0..field.h {
        let up = j.saturating_sub(1) * field.w;
        let down = (j + 1).min(field.h - 1) * field.w;
        let here = j * field.w;
        for i in 0..field.w {
            softened[here + i] = (out[up + i] + 2.0 * out[here + i] + out[down + i]) * 0.25;
        }
    }
    softened
}

/// The horizontal distance from `x` to the nearest of a row's runs, 0 inside one.
fn runs_distance(row: &BodyRow, x: f64) -> f64 {
    let mut best = f64::INFINITY;
    for run in &row.runs {
        let (x0, x1) = (f64::from(run[0]), f64::from(run[1]));
        if x >= x0 && x <= x1 {
            return 0.0;
        }
        best = best.min((x - x0).abs().min((x - x1).abs()));
    }
    best
}

/// A per-node weight a part's displacement is multiplied by, as `(i, j) -> 0..=1`: the face
/// table's clip to the face oval and its protection of the eyes and lips, and the body's
/// feathered clip to the fitted person silhouette. Length ramps pass `None`: their plateau
/// must carry the silhouette and the background together.
pub(crate) type NodeWeight<'a> = &'a dyn Fn(usize, usize) -> f32;

/// The field one part writes into and the weight it writes through — the pair every
/// primitive needs, bundled so the writers stay under the argument-count lint.
struct Write<'f, 'w> {
    field: &'f mut LiquifyField,
    weight: Option<NodeWeight<'w>>,
}

/// Writes one resolved shape at slider fraction `k` — the whole primitive surface, shared
/// by both parts tables.
pub(crate) fn write_shape(field: &mut LiquifyField, shape: &Shape, k: f64, weight: Option<NodeWeight<'_>>) {
    let mut write = Write { field, weight };
    match *shape {
        Shape::Contract { a, b, band, width, gain } => write_contract(&mut write, a, b, band, width, gain, k),
        Shape::Profile { a, b, levels, kernel, gain } => write_profile(&mut write, a, b, levels, &kernel, gain, k),
        Shape::Radial { c, radius, gain } => write_radial(&mut write, c, radius, gain, k),
        Shape::Ramp { a, b, band, width, gain } => write_ramp(&mut write, a, b, band, width, gain, k),
        Shape::Shift { a, b, dir, band, width, gain } => write_shift(&mut write, a, b, dir, band, width, gain * k),
    }
}

/// The weight at `(i, j)`, 1 without one. A zero weight skips the node before its shape
/// maths runs, so a clipped-away face part costs only the lookup.
#[inline]
fn weight_at(weight: Option<NodeWeight<'_>>, i: usize, j: usize) -> f64 {
    weight.map_or(1.0, |w| f64::from(w(i, j).clamp(0.0, 1.0)))
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
fn write_contract(write: &mut Write<'_, '_>, a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64, k: f64) {
    let weight = write.weight;
    let field = &mut *write.field;
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
            let w = weight_at(weight, i, j);
            if w <= 0.0 {
                continue;
            }
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
            let mag = scale_mag * lat * long * (rho / width) * w;
            let idx = j * field.w + i;
            field.d[idx][0] += (d[0] / rho * mag) as f32;
            field.d[idx][1] += (d[1] / rho * mag) as f32;
        }
    }
}

/// A contraction whose envelope and longitudinal strength follow the torso's own measured
/// width profile ([`Shape::Profile`]): at every `t` the lateral width is [`PROFILE_REACH`]
/// times the interpolated measured half-width — so the pull lands on the visible contour at
/// that level — while `kernel` sets how much of the row's pull reaches there. The width is
/// blended between level knots with a smoothstep, so the envelope has no kink at a level and
/// the rows compose without the seam an isolated band left.
fn write_profile(write: &mut Write<'_, '_>, a: [f64; 2], b: [f64; 2], levels: [(f64, f64); 4], kernel: &Kernel, gain: f64, k: f64) {
    let weight = write.weight;
    let field = &mut *write.field;
    let ab = sub(b, a);
    let len = norm(ab);
    let u = scale(ab, 1.0 / len);
    let max_half = levels.iter().map(|&(_, h)| h).fold(0.0, f64::max).max(4.0);
    let reach = PROFILE_REACH * max_half;
    let (min_x, max_x) = (a[0].min(b[0]) - reach, a[0].max(b[0]) + reach);
    let (min_y, max_y) = (a[1].min(b[1]) - reach, a[1].max(b[1]) + reach);
    let Some((i0, j0, i1, j1)) = span(field, min_x, min_y, max_x, max_y) else { return };
    for j in j0..=j1 {
        for i in i0..=i1 {
            let w = weight_at(weight, i, j);
            if w <= 0.0 {
                continue;
            }
            let p = field.node_pos(i, j);
            let t = dot(sub(p, a), u) / len;
            let long = kernel.at(t);
            if long <= 0.0 {
                continue;
            }
            let width = PROFILE_REACH * profile_half(&levels, t).max(4.0);
            let q = add(a, scale(u, t.clamp(0.0, 1.0) * len));
            let d = sub(p, q);
            let rho = norm(d);
            if rho >= width || rho < 1e-9 {
                continue;
            }
            let lat = falloff((rho / width) * (rho / width), DENSITY);
            let mag = gain * k * width * lat * (rho / width) * long * w;
            let idx = j * field.w + i;
            field.d[idx][0] += (d[0] / rho * mag) as f32;
            field.d[idx][1] += (d[1] / rho * mag) as f32;
        }
    }
}

/// The measured half-width at `t`, blended between the neighbouring knots with a smoothstep
/// (zero slope at each knot) and clamped to the end knots outside them. `levels` is sorted
/// by `t`.
fn profile_half(levels: &[(f64, f64); 4], t: f64) -> f64 {
    let (t0, h0) = levels[0];
    if !t.is_finite() || t <= t0 {
        return h0;
    }
    for pair in levels.windows(2) {
        let (a, ha) = pair[0];
        let (b, hb) = pair[1];
        if t <= b {
            let s = if b > a { smoothstep(0.0, 1.0, (t - a) / (b - a)) } else { 1.0 };
            return ha + (hb - ha) * s;
        }
    }
    levels[3].1
}

/// The contraction about one centre (the head, the bust, the hips): the same profile with
/// the lateral distance measured from `c`.
fn write_radial(write: &mut Write<'_, '_>, c: [f64; 2], radius: f64, gain: f64, k: f64) {
    let weight = write.weight;
    let field = &mut *write.field;
    let Some((i0, j0, i1, j1)) = span(field, c[0] - radius, c[1] - radius, c[0] + radius, c[1] + radius) else { return };
    let scale_mag = gain * k * radius;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let w = weight_at(weight, i, j);
            if w <= 0.0 {
                continue;
            }
            let d = sub(field.node_pos(i, j), c);
            let rho = norm(d);
            if rho >= radius || rho < 1e-9 {
                continue;
            }
            let lat = falloff((rho / radius) * (rho / radius), DENSITY);
            let mag = scale_mag * lat * (rho / radius) * w;
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
fn write_ramp(write: &mut Write<'_, '_>, a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64, k: f64) {
    let weight = write.weight;
    let field = &mut *write.field;
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
            let w = weight_at(weight, i, j);
            if w <= 0.0 {
                continue;
            }
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
            let mag = delta * s * lat * w;
            let idx = j * field.w + i;
            field.d[idx][0] -= (u[0] * mag) as f32;
            field.d[idx][1] -= (u[1] * mag) as f32;
        }
    }
}

/// A rigid band translate ([`Shape::Shift`]): the displacement's magnitude is uniform over
/// the plateau (the zone rides whole, the bust-lift's point) with the ramp family's
/// plateau-plus-feather laterally and the contract family's smoothstep window
/// longitudinally. The inverse map points against `dir`, so the content reads as moved
/// along it; the ride is `gain` of the segment's own length.
fn write_shift(write: &mut Write<'_, '_>, a: [f64; 2], b: [f64; 2], dir: [f64; 2], band: (f64, f64), width: f64, gain: f64) {
    let weight = write.weight;
    let field = &mut *write.field;
    let ab = sub(b, a);
    let len = norm(ab);
    let u = scale(ab, 1.0 / len);
    let dir = scale(dir, 1.0 / norm(dir));
    let delta = gain * len;
    let (lo, hi) = (band.0.min(band.1), band.1.max(band.0));
    let ramp = (hi - lo) * 0.25;
    let (min_x, max_x) = (a[0].min(b[0]) - width, a[0].max(b[0]) + width);
    let (min_y, max_y) = (a[1].min(b[1]) - width, a[1].max(b[1]) + width);
    let Some((i0, j0, i1, j1)) = span(field, min_x, min_y, max_x, max_y) else { return };
    for j in j0..=j1 {
        for i in i0..=i1 {
            let w = weight_at(weight, i, j);
            if w <= 0.0 {
                continue;
            }
            let p = field.node_pos(i, j);
            let t = dot(sub(p, a), u) / len;
            let q = add(a, scale(u, t.clamp(0.0, 1.0) * len));
            let rho = norm(sub(p, q));
            if rho >= width {
                continue;
            }
            let long = smoothstep(lo, lo + ramp, t) * (1.0 - smoothstep(hi - ramp, hi, t));
            if long <= 0.0 {
                continue;
            }
            let lat = 1.0 - smoothstep(0.6, 1.0, rho / width);
            let mag = delta * long * lat * w;
            let idx = j * field.w + i;
            field.d[idx][0] -= (dir[0] * mag) as f32;
            field.d[idx][1] -= (dir[1] * mag) as f32;
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
pub(crate) fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
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
        SculptSpec { sliders, anchors: standing(), fit: None }
    }

    /// A person fit for [`standing`]: a silhouette a little wider than the joint centres, the
    /// levels measured off it, and each limb's half-thickness. The rows carry the arms as
    /// their own runs, as a real mask does.
    fn fitted() -> BodyFit {
        let a = standing();
        let arm = |y: f32| -> (f32, f32) {
            let t = ((y - 180.0) / 192.0).clamp(0.0, 1.0);
            (a.pose[11][0] + (a.pose[15][0] - a.pose[11][0]) * t, a.pose[12][0] + (a.pose[16][0] - a.pose[12][0]) * t)
        };
        let rows = (120..=700)
            .step_by(20)
            .map(|y| {
                let half = 60.0 + (y - 120) as f32 * 0.06;
                let mut runs = vec![[257.0 - half, 257.0 + half]];
                if (180..=372).contains(&y) {
                    let (l, r) = arm(y as f32);
                    runs.push([l - 26.0, l + 26.0]);
                    runs.push([r - 26.0, r + 26.0]);
                }
                BodyRow { y: y as f32, runs }
            })
            .collect();
        BodyFit {
            rows,
            waist: LevelFit { t: 0.525, left: 70.0, right: 76.0 },
            belly: LevelFit { t: 0.70, left: 80.0, right: 86.0 },
            hip: LevelFit { t: 1.0, left: 90.0, right: 96.0 },
            bust: LevelFit { t: 0.30, left: 60.0, right: 66.0 },
            limbs: [
                LimbFit { radius: 22.0, t0: 0.05, t1: 0.95 },
                LimbFit { radius: 18.0, t0: 0.05, t1: 0.95 },
                LimbFit { radius: 22.0, t0: 0.05, t1: 0.95 },
                LimbFit { radius: 18.0, t0: 0.05, t1: 0.95 },
                LimbFit { radius: 40.0, t0: 0.05, t1: 0.95 },
                LimbFit { radius: 40.0, t0: 0.05, t1: 0.95 },
                LimbFit { radius: 26.0, t0: 0.05, t1: 0.95 },
                LimbFit { radius: 26.0, t0: 0.05, t1: 0.95 },
            ],
            inside: vec![true; POSE_COUNT],
            waist_prior: false,
        }
    }

    fn fitted_spec(sliders: SculptSliders) -> SculptSpec {
        SculptSpec { sliders, anchors: standing(), fit: Some(fitted()) }
    }

    fn build(sliders: SculptSliders) -> (LiquifyField, SculptReport) {
        sculpt_field(&spec(sliders), bounds(), CELL)
    }

    /// The field's largest displacement and the document point it sits at.
    fn field_peak(field: &LiquifyField) -> (f32, [f64; 2]) {
        let (mut peak, mut at) = (0.0f32, [0.0f64; 2]);
        for j in 0..field.h {
            for i in 0..field.w {
                let d = field.d[j * field.w + i];
                let m = d[0].hypot(d[1]);
                if m > peak {
                    peak = m;
                    at = field.node_pos(i, j);
                }
            }
        }
        (peak, at)
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
            SculptSliders { straight_shoulders: 100.0, ..SculptSliders::default() },
            SculptSliders { bust_lift: 100.0, ..SculptSliders::default() },
            SculptSliders { double_chin: 100.0, ..SculptSliders::default() },
            // The composition sliders have no rows of their own: alone they still move the
            // field through the parts they blend into (the blend runs inside sculpt_field).
            SculptSliders { ai_sculpt: 100.0, ..SculptSliders::default() },
            SculptSliders { swan_neck: 100.0, ..SculptSliders::default() },
        ] {
            let (field, report) = build(sliders);
            assert!(!field.is_identity(), "the slider left the field untouched");
            assert!(report.skipped.is_empty(), "a clean pose skipped {:?}", report.skipped);
        }
    }

    /// Without a fitted silhouette the waist falls back to the pose's hip row: on a clothed
    /// subject the hip joints mark the visible hip line, and the garment's waistband sits at
    /// or just above it. The old fallbacks sat at 0.54, then 0.62, of the shoulder→hip span —
    /// on the ribs — which is why the slider read as squeezing the chest whenever the matte
    /// could not measure a waist.
    #[test]
    fn without_a_fit_the_waist_peak_sits_on_the_hip_line() {
        let (field, report) = build(SculptSliders { waist: 100.0, ..SculptSliders::default() });
        assert!(report.applied.contains(&"waist"), "{report:?}");
        let (peak, at) = field_peak(&field);
        assert!(peak > 2.0, "the waist must pull: {peak:.2}px");
        let t = (at[1] - 180.0) / (400.0 - 180.0); // shoulder mid → hip mid
        assert!((0.90..=1.10).contains(&t), "the waist peak t={t:.2} left the hip line");
        // The ribs the old fallback squeezed stay still.
        let chest = field.sample(306.0, 180.0 + 0.62 * 220.0);
        assert!(chest[0].hypot(chest[1]) < 0.5, "the chest must stay still: {chest:?}");
    }

    /// The belly fallback follows the waist one down: with no fit it sits on the lower
    /// abdomen just above the hip line, so the rows read as belly above, waist at the band.
    /// The old 0.81 row started inside the ribs.
    #[test]
    fn without_a_fit_the_belly_peak_sits_on_the_lower_abdomen() {
        let (field, report) = build(SculptSliders { belly: 100.0, ..SculptSliders::default() });
        assert!(report.applied.contains(&"belly"), "{report:?}");
        let (peak, at) = field_peak(&field);
        assert!(peak > 1.0, "the belly must pull: {peak:.2}px");
        let t = (at[1] - 180.0) / (400.0 - 180.0); // shoulder mid → hip mid
        assert!((0.84..=0.96).contains(&t), "the belly peak t={t:.2} left the lower abdomen");
        let ribs = field.sample(306.0, 180.0 + 0.60 * 220.0);
        assert!(ribs[0].hypot(ribs[1]) < 0.5, "the ribs must stay still: {ribs:?}");
    }

    /// A torso row placed from the joints alone is approximate and must say so; with a
    /// fitted silhouette the same row is measured and must not. Presenting the fallback as
    /// measured is what made "the slider is precise" impossible to trust.
    #[test]
    fn a_missing_silhouette_names_the_torso_rows_approximate() {
        let (_, loose) = build(SculptSliders { waist: 100.0, ..SculptSliders::default() });
        assert!(loose.approximate.iter().any(|(n, _)| *n == "waist"), "{loose:?}");
        assert!(!loose.skipped.iter().any(|(n, _)| *n == "waist"), "{loose:?}");
        let (_, fitted) = sculpt_field(&fitted_spec(SculptSliders { waist: 100.0, ..SculptSliders::default() }), bounds(), CELL);
        assert!(fitted.approximate.iter().all(|(n, _)| *n != "waist"), "{fitted:?}");
    }

    /// A fit whose waist came from the hip joints rather than a measured crease says so: the
    /// waist and belly rows it places are approximate. Presenting a row placed from the pose
    /// as a measured one is what let the slider sit on the ribs and still read as precise.
    #[test]
    fn a_waist_placed_from_the_hip_joints_is_named_approximate() {
        let mut fit = fitted();
        fit.waist_prior = true;
        let sliders = SculptSliders { waist: 100.0, belly: 60.0, ..SculptSliders::default() };
        let (_, report) = sculpt_field(&SculptSpec { sliders, anchors: standing(), fit: Some(fit) }, bounds(), CELL);
        assert!(report.approximate.iter().any(|(n, why)| *n == "waist" && why.contains("hip joints")), "{report:?}");
        assert!(report.approximate.iter().any(|(n, _)| *n == "belly"), "{report:?}");
    }

    /// The shoulders slider narrows the shoulder the photo shows — the sleeve and the
    /// deltoid beside the joint — not just the strip between the joints. The old falloff
    /// was narrower than that visible width, so its peak sat inside the joint and the rim
    /// beside it barely moved: the slider read as a dent.
    #[test]
    fn the_shoulders_slider_moves_the_whole_shoulder_width() {
        let (field, report) = build(SculptSliders { shoulders: 100.0, ..SculptSliders::default() });
        assert!(report.applied.contains(&"shoulders"), "{report:?}");
        let y = 200.0; // inside the contract's band
        let joint = field.sample(308.0, y)[0].abs();
        let sleeve = field.sample(338.0, y)[0].abs();
        assert!(joint > 1.0, "the joint must move: {joint:.2}px");
        assert!(sleeve > 0.3 * joint, "the sleeve beside the joint must move with it: {sleeve:.2}px vs {joint:.2}px");
    }

    #[test]
    fn the_waist_slider_pulls_the_rim_in_proportion_to_its_value() {
        // A point on the waist band's rim, to the axis's right, just above the hip row the
        // fallback centres on. The inverse map points outward there (the rim samples farther
        // out, which reads as pulled in).
        let (x, y) = (294.0, 396.0);
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
        let left = field.sample(219.0, 396.0);
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
    fn the_fitted_waist_pulls_in_with_the_near_background_and_the_far_background_stays() {
        let (field, report) = sculpt_field(&fitted_spec(SculptSliders { waist: 100.0, ..SculptSliders::default() }), bounds(), CELL);
        assert!(report.applied.contains(&"waist"), "{report:?}");
        // The measured waist contour: the fitted envelope reaches past it, so the rim moves
        // (the old joint-spacing width stopped short of it).
        let rim = field.sample(326.0, 295.0);
        assert!(rim[0] > 1.0, "the fitted waist must pull its own contour in: {rim:?}");
        // The background beside the contour is coupled to the person: it rides the pull at
        // nearly the rim's strength. The old fixed feather moved it at a fraction of the
        // rim's strength and froze everything past the feather, which read as a ridge along
        // the silhouette.
        let near = field.sample(180.0, 295.0);
        assert!(near[0].abs() > 0.7 * rim[0].abs(), "the near background must move with the contour (near {near:?}, rim {rim:?})");
        assert!(near[0] < 0.0, "the near background outside the left rim moves outward (with the pull): {near:?}");
        // The far background does not move: the coupling decays over the local displacement.
        let far = field.sample(150.0, 295.0);
        assert!(far[0].abs() < 0.5 && far[1].abs() < 0.5, "the far background stays: {far:?}");
    }

    /// Fattening moves the silhouette *out*: a pixel just outside the measured hip contour
    /// samples inward far enough to show the body, and the background beside it is pushed
    /// along. The old near-binary clip froze everything past its short feather, so a fuller
    /// hip could only widen the body inside its own outline.
    #[test]
    fn the_fitted_hip_expansion_pushes_the_near_background_out() {
        let (field, _) = sculpt_field(&fitted_spec(SculptSliders { hips: 100.0, ..SculptSliders::default() }), bounds(), CELL);
        // The measured hip row's right run ends at x≈334; 4 px outside, the inverse map must
        // sample at least that far inward (toward the joint at x=292), which reads as the
        // contour expanding out to this pixel.
        let just_outside = field.sample(338.0, 400.0);
        assert!(just_outside[0] < -4.2, "the hip must expand past its measured contour: {just_outside:?}");
        let beside = field.sample(344.0, 400.0);
        assert!(beside[0] < 0.0, "the near background is pushed outward with the hip: {beside:?}");
        let far = field.sample(380.0, 400.0);
        assert!(far[0].abs() < 0.5 && far[1].abs() < 0.5, "the far background stays: {far:?}");
    }

    #[test]
    fn the_fitted_levels_land_on_their_measured_rows() {
        let belly = sculpt_field(&fitted_spec(SculptSliders { belly: 100.0, ..SculptSliders::default() }), bounds(), CELL).0;
        let waist = sculpt_field(&fitted_spec(SculptSliders { waist: 100.0, ..SculptSliders::default() }), bounds(), CELL).0;
        let at_belly = belly.sample(300.0, 334.0);
        assert!(at_belly[0] > 1.0, "the belly must pull its measured row in: {at_belly:?}");
        // Each row's own measured row is its strongest; the other row's pull there is the
        // continuous taper, not full strength.
        assert!(at_belly[0] > 2.0 * waist.sample(300.0, 334.0)[0].abs(), "the belly row belongs to the belly");
        let at_waist = waist.sample(300.0, 296.0);
        assert!(at_waist[0] > 1.0, "the waist must pull its measured row in: {at_waist:?}");
        assert!(at_waist[0] > 2.0 * belly.sample(300.0, 296.0)[0].abs(), "the waist row belongs to the waist");
        // The hips expand on the joint line, not on a skirt's widest hem row.
        let hips = sculpt_field(&fitted_spec(SculptSliders { hips: 100.0, ..SculptSliders::default() }), bounds(), CELL).0;
        let hip = hips.sample(330.0, 400.0);
        assert!(hip[0] < -1.0, "the hips must push their own contour out: {hip:?}");
        let bust = sculpt_field(&fitted_spec(SculptSliders { bust: 100.0, ..SculptSliders::default() }), bounds(), CELL).0;
        let chest = bust.sample(320.0, 246.0);
        assert!(chest[0] < -1.0, "the bust must push the chest out: {chest:?}");
    }

    /// The waist and belly rows taper across the whole torso: a tenth of the torso above and
    /// below the waist row the slider still pulls at a good fraction of its peak. The old
    /// ±8.5% band stopped at zero just outside the level, which read as a dent that the ribs
    /// and hips did not follow.
    #[test]
    fn the_waist_taper_reaches_above_and_below_its_row() {
        let (field, _) = sculpt_field(&fitted_spec(SculptSliders { waist: 100.0, ..SculptSliders::default() }), bounds(), CELL);
        let at = |y: f64| field.sample(300.0, y)[0];
        let peak = at(296.0);
        assert!(peak > 1.0, "the waist row pulls: {peak:.2}");
        let above = at(270.0); // t ≈ 0.41, toward the ribs
        let below = at(320.0); // t ≈ 0.64, toward the belly
        assert!(above > 0.4 * peak, "the pull must carry up toward the ribs (above {above:.2}, peak {peak:.2})");
        assert!(below > 0.4 * peak, "the pull must carry down toward the hips (below {below:.2}, peak {peak:.2})");
    }

    /// The knee is one continuous surface: a point on the outer knee must move under the
    /// thigh slider and under the calf slider. Each limb band used to ramp to zero exactly at
    /// its joint, leaving a dead ring at the knee and elbow that read as a pinch.
    #[test]
    fn the_knee_moves_under_the_thigh_and_the_calf() {
        let thigh = sculpt_field(&fitted_spec(SculptSliders { thighs: 100.0, ..SculptSliders::default() }), bounds(), CELL).0;
        let calf = sculpt_field(&fitted_spec(SculptSliders { calves: 100.0, ..SculptSliders::default() }), bounds(), CELL).0;
        let knee = 200.0; // the outer knee contour: 26 px out from the joint at x=226
        let by_thigh = thigh.sample(knee, 540.0)[0].abs();
        let by_calf = calf.sample(knee, 540.0)[0].abs();
        assert!(by_thigh > 0.5, "the knee must move under the thigh: {by_thigh:.2}px");
        assert!(by_calf > 0.5, "the knee must move under the calf: {by_calf:.2}px");
    }

    #[test]
    fn the_fitted_limbs_pull_their_own_surface_in() {
        let (field, report) = sculpt_field(&fitted_spec(SculptSliders { arms: 100.0, ..SculptSliders::default() }), bounds(), CELL);
        assert_eq!(report.applied.len(), 4, "{:?}", report.applied);
        let arm = field.sample(176.0, 228.0);
        assert!(arm[0] < -1.0, "the upper arm's own rim must pull toward its bone: {arm:?}");
        let thigh = sculpt_field(&fitted_spec(SculptSliders { thighs: 100.0, ..SculptSliders::default() }), bounds(), CELL).0;
        let leg = thigh.sample(184.0, 470.0);
        assert!(leg[0] < -1.0, "the thigh's own rim must pull toward its bone: {leg:?}");
    }

    #[test]
    fn the_fitted_slim_body_needs_no_ankles() {
        let mut anchors = standing();
        for i in [27usize, 28] {
            anchors.pose[i][2] = 0.0;
        }
        let spec = SculptSpec { sliders: SculptSliders { slim_body: 100.0, ..SculptSliders::default() }, anchors, fit: Some(fitted()) };
        let (field, report) = sculpt_field(&spec, bounds(), CELL);
        assert!(report.applied.contains(&"slim body"), "{report:?}");
        assert!(report.skipped.is_empty(), "{report:?}");
        let rim = field.sample(320.0, 300.0);
        assert!(rim[0] > 1.0, "the fitted slim body must pull the torso contour in: {rim:?}");
    }

    #[test]
    fn an_occluded_joint_inside_the_silhouette_runs_reduced_and_says_so() {
        let mut anchors = standing();
        anchors.pose[25][2] = 0.2; // the left knee: barely seen, but the mask contains it
        let visible = sculpt_field(&fitted_spec(SculptSliders { thighs: 100.0, ..SculptSliders::default() }), bounds(), CELL).0;
        let spec = SculptSpec { sliders: SculptSliders { thighs: 100.0, ..SculptSliders::default() }, anchors, fit: Some(fitted()) };
        let (field, report) = sculpt_field(&spec, bounds(), CELL);
        assert!(report.applied.contains(&"left thigh"), "{:?}", report.applied);
        assert!(report.approximate.iter().any(|(n, _)| *n == "left thigh"), "{:?}", report.approximate);
        let occluded = field.sample(184.0, 470.0)[0].abs();
        let full = visible.sample(184.0, 470.0)[0].abs();
        assert!(occluded > 0.0 && occluded < full, "the occluded knee runs at reduced strength: {occluded:.2} vs {full:.2}");
    }

    #[test]
    fn a_joint_outside_a_mostly_uncovered_limb_skips_its_row_with_the_reason() {
        let mut fit = fitted();
        fit.inside[25] = false;
        fit.limbs[4].t1 = 0.30; // the knee is off the matte and most of the bone with it
        let spec = SculptSpec { sliders: SculptSliders { thighs: 100.0, ..SculptSliders::default() }, anchors: standing(), fit: Some(fit) };
        let (_, report) = sculpt_field(&spec, bounds(), CELL);
        let why = report.skipped.iter().find(|(n, _)| *n == "left thigh").map(|(_, w)| *w);
        assert_eq!(why, Some("the landmark is outside the person silhouette"));
        assert!(report.applied.contains(&"right thigh"), "{:?}", report.applied);
    }

    /// The matte can miss a joint by a few pixels while the limb's bone still runs through
    /// the person — an arm folded behind the head against the hair. The visible limb must
    /// still slim, over the covered part of its bone, and the report must say it is inferred.
    #[test]
    fn a_limb_joint_just_outside_the_matte_still_shapes_its_covered_part() {
        let mut fit = fitted();
        fit.inside[25] = false; // the left knee, just off the silhouette
        let spec = SculptSpec { sliders: SculptSliders { thighs: 100.0, ..SculptSliders::default() }, anchors: standing(), fit: Some(fit) };
        let (field, report) = sculpt_field(&spec, bounds(), CELL);
        assert!(report.applied.contains(&"left thigh"), "{report:?}");
        assert!(report.skipped.iter().all(|(n, _)| *n != "left thigh"), "{report:?}");
        assert!(report.approximate.iter().any(|(n, why)| *n == "left thigh" && why.contains("covered part")), "{report:?}");
        let rim = field.sample(184.0, 470.0);
        assert!(rim[0].abs() > 1.0, "the covered thigh surface must still move: {rim:?}");
    }

    #[test]
    fn an_invalid_fit_is_ignored_not_trusted() {
        let mut bad = fitted();
        bad.rows.clear();
        let sliders = SculptSliders { waist: 100.0, ..SculptSliders::default() };
        let a = sculpt_field(&SculptSpec { sliders, anchors: standing(), fit: Some(bad) }, bounds(), CELL).0;
        let b = build(SculptSliders { waist: 100.0, ..SculptSliders::default() }).0;
        assert_eq!(a, b, "a malformed fit must fall back to the joint spacing");
    }

    #[test]
    fn the_fitted_slimming_stack_does_not_fold() {
        let sliders = SculptSliders {
            slim_body: 100.0,
            waist: 100.0,
            belly: 100.0,
            arms: 100.0,
            thighs: 100.0,
            calves: 100.0,
            shoulders: 100.0,
            hips: 100.0,
            bust: 100.0,
            head_size: 100.0,
            neck: 100.0,
            height: 100.0,
            leg_length: 100.0,
            waist_length: 100.0,
            torso_length: 100.0,
            neck_length: 100.0,
            straight_shoulders: 100.0,
            swan_neck: 100.0,
            bust_lift: 100.0,
            double_chin: 100.0,
            ai_sculpt: 100.0,
        };
        let (field, report) = sculpt_field(&fitted_spec(sliders), bounds(), CELL);
        assert!(report.skipped.is_empty(), "{:?}", report.skipped);
        let step = max_adjacent_step(&field);
        assert!(step < CELL, "adjacent nodes step {step:.2}px past the {CELL}px cell — the fitted stack folds");
    }

    /// Height and Leg Length stretch everything above the ankles: a feet position the pose only
    /// guessed must not move the frame at a strength nobody can see, dressed up as a success.
    #[test]
    fn guessed_feet_do_not_lengthen_the_legs() {
        let mut anchors = standing();
        for i in [27usize, 28] {
            anchors.pose[i][2] = 0.3; // the pose guessed the feet; the photo may not show them
        }
        let sliders = SculptSliders { height: 100.0, leg_length: 100.0, ..SculptSliders::default() };
        let (field, report) = sculpt_field(&SculptSpec { sliders, anchors, fit: Some(fitted()) }, bounds(), CELL);
        assert!(field.is_identity(), "a guessed foot must not stretch anything");
        let skipped: Vec<_> = report.skipped.iter().map(|(n, _)| *n).collect();
        assert!(skipped.contains(&"height") && skipped.contains(&"legs"), "{skipped:?}");
        assert!(report.skipped.iter().all(|(n, why)| !matches!(*n, "height" | "legs") || why.contains("feet")), "{report:?}");
    }

    #[test]
    fn hidden_shoulders_skip_their_rows_and_say_why() {
        let mut anchors = standing();
        for i in [11usize, 12] {
            anchors.pose[i][2] = 0.1;
        }
        let s = SculptSpec { sliders: SculptSliders { shoulders: 50.0, arms: 50.0, head_size: 50.0, ..SculptSliders::default() }, anchors, fit: None };
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
                straight_shoulders: 100.0 * sign,
                swan_neck: 100.0 * sign,
                bust_lift: 100.0 * sign,
                double_chin: 100.0 * sign,
                ai_sculpt: 100.0 * sign,
            };
            let (field, report) = build(sliders);
            assert!(report.skipped.is_empty(), "{:?}", report.skipped);
            let step = max_adjacent_step(&field);
            assert!(step < CELL, "adjacent nodes step {step:.2}px past the {CELL}px cell — the field folds");
        }
    }

    #[test]
    fn straight_shoulders_flatten_the_trapezius_slope() {
        // Above the shoulder line, on the slope: the contraction's inverse map points away
        // from the line (up there), which reads as the slope pulled down to it.
        let (field, _) = build(SculptSliders { straight_shoulders: 100.0, ..SculptSliders::default() });
        let slope = field.sample(280.0, 158.0); // 22 px above the shoulder line, right of centre
        assert!(slope[1] < -0.3, "the slope must pull toward the line (upward field): {slope:?}");
        let below = field.sample(280.0, 210.0); // below the line, inside the width
        assert!(below[1] > 0.3, "below the line the field points down to it: {below:?}");
        let far = field.sample(10.0, 180.0); // past the shoulder tip
        assert!(far[0].abs() < 0.05 && far[1].abs() < 0.05, "past the shoulder tip stays: {far:?}");
    }

    #[test]
    fn bust_lift_moves_the_band_up_rigidly() {
        // The inverse map points against the ride direction (down), which reads as lifted.
        let (field, _) = build(SculptSliders { bust_lift: 100.0, ..SculptSliders::default() });
        let in_band = field.sample(256.0, 280.0); // the torso axis inside the bust band
        assert!(in_band[1] > 1.0, "the band must ride: {in_band:?}");
        let outside = field.sample(256.0, 480.0); // well below the band
        assert!(outside[0].abs() < 0.05 && outside[1].abs() < 0.05, "below the band stays: {outside:?}");
    }

    #[test]
    fn the_composition_family_blends_into_its_parts() {
        let near = |a: f32, b: f32| (a - b).abs() < 0.01;
        // AI塑形 alone moves the slim family and leaves the rounder zones alone.
        let c = SculptSliders { ai_sculpt: 100.0, ..SculptSliders::default() }.composed();
        assert!(near(c.waist, 50.0) && near(c.slim_body, 60.0));
        assert!(near(c.hips, 0.0) && near(c.bust, 0.0), "the silhouette slider does not round the zones");
        // It stacks with a manual slider and clamps at the stop.
        let c = SculptSliders { ai_sculpt: 100.0, waist: 70.0, ..SculptSliders::default() }.composed();
        assert!(near(c.waist, 100.0));
        // 天鹅颈 is its trio; negative AI塑形 plumps.
        let c = SculptSliders { swan_neck: 50.0, ..SculptSliders::default() }.composed();
        assert!(near(c.neck, 30.0) && near(c.neck_length, 20.0) && near(c.straight_shoulders, 25.0));
        let c = SculptSliders { ai_sculpt: -100.0, ..SculptSliders::default() }.composed();
        assert!(near(c.waist, -50.0));
    }
}
