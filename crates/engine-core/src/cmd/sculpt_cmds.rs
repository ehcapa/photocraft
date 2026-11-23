//! Filter › Body Sculpt: whole-body shaping from the body-pose foundation — twenty-three
//! bidirectional shaping sliders (PixCake's 全身美型 panel in this fork's terms) whose warp
//! [`photocraft_algo::sculpt`] writes analytically over Liquify's own field, so the pixel
//! work, the selection limit and the smart-filter replay are all the distortion family's.
//!
//! The panel (and `portrait.analyze`) hands the command the anchors it analysed once; with
//! them Apply builds the field and warps — no portrait pipeline on the button. A script that
//! sends none falls back to one lightweight analysis: the person mask and the pose, not the
//! full Smart Portrait stages. The anchors it used embed into the params it stores: a smart
//! filter re-renders at composite time, when no session and no analysis exist, so the stored
//! params alone must rebuild the field. A linked layer mask rides the same field — slimming
//! pixels under a stale mask is a broken silhouette (#3083's rule, Puppet Warp's precedent).

use photocraft_algo::contour;
use photocraft_algo::liquify::{LiquifyField, apply_liquify, auto_cell};
use photocraft_algo::sculpt::{
    BodyFit, BodyRow, LIMB_SEGMENTS, LevelFit, LimbFit, POSE_COUNT, SYMMETRY_PAIRS, SculptAnchors, SculptHead, SculptReport, SculptSliders, SculptSpec,
    sculpt_field,
};
use photocraft_doc::LayerContent;
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::photo_cmds::Stopwatch;
use crate::{EngineError, Result, Session};

pub const SCULPT: &str = "filter.bodySculpt";

/// The refusal a run without usable anchors answers (`portrait.analyze` reports it too).
pub(crate) const NO_PERSON: &str = "no person found — Body Sculpt needs the body-pose landmarks";

/// The pose visibility a symmetry joint needs before it may be moved: below this the pose is
/// guessing, and the guess often lands on whatever is behind the joint (a raised arm's elbow on
/// the cheek, a cropped leg's knee on the jacket).
const SYMMETRY_SEEN: f32 = 0.5;

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: SCULPT.into(), msg: msg.into() }
}

/// One slider out of params: absent is zero (the dialog sends all twenty-three, a script the
/// few it means), a non-number is refused, and the range clamps as a backstop — the schema
/// gate has already refused out-of-range numbers by the time this runs.
fn slider(p: &Value, key: &str) -> Result<f32> {
    let raw = match p.get(key) {
        None => 0.0,
        Some(v) => v.as_f64().ok_or_else(|| bad(format!("`{key}` must be a number in -100..100")))?,
    };
    Ok(raw.clamp(-100.0, 100.0) as f32)
}

fn sliders(p: &Value) -> Result<SculptSliders> {
    Ok(SculptSliders {
        slim_upper: slider(p, "slimUpper")?,
        slim_full: slider(p, "slimFull")?,
        waist: slider(p, "waist")?,
        belly: slider(p, "belly")?,
        arms: slider(p, "arms")?,
        forearms: slider(p, "forearms")?,
        thighs: slider(p, "thighs")?,
        calves: slider(p, "calves")?,
        shoulders: slider(p, "shoulders")?,
        hips: slider(p, "hips")?,
        bust: slider(p, "bust")?,
        head_size: slider(p, "headSize")?,
        neck: slider(p, "neck")?,
        height: slider(p, "height")?,
        leg_length: slider(p, "legLength")?,
        waist_length: slider(p, "waistLength")?,
        torso_length: slider(p, "torsoLength")?,
        neck_length: slider(p, "neckLength")?,
        ai_sculpt: slider(p, "aiSculpt")?,
        straight_shoulders: slider(p, "straightShoulders")?,
        swan_neck: slider(p, "swanNeck")?,
        bust_lift: slider(p, "bustLift")?,
        double_chin: slider(p, "doubleChin")?,
    })
}

/// The field resolution, Liquify's own `meshSize` convention.
fn cell(p: &Value, canvas: Rect) -> Result<f64> {
    match p.get("meshSize") {
        None | Some(Value::Null) => Ok(auto_cell(canvas)),
        Some(v) => {
            let c = v.as_f64().ok_or_else(|| bad("`meshSize` is px per field node (1..64)"))?;
            if !(1.0..=64.0).contains(&c) {
                return Err(bad("`meshSize` must be 1..64"));
            }
            Ok(c)
        }
    }
}

/// The anchors as the stored params carry them (the smart-filter re-render path): sliders,
/// `anchors` and `meshSize`. `None` when the anchors are absent or malformed — a filter this
/// command never stored.
fn stored_spec(p: &Value, canvas: Rect) -> Option<(SculptSpec, f64)> {
    let anchors: SculptAnchors = serde_json::from_value(p.get("anchors")?.clone()).ok()?;
    Some((SculptSpec { sliders: sliders(p).ok()?, anchors, fit: fit_of(p) }, cell(p, canvas).ok()?))
}

/// The person-mask fit `portrait.analyze` computed, when the params carry one. A malformed
/// fit is ignored rather than trusted: the parts fall back to joint spacing, exactly as an
/// older stored filter does.
fn fit_of(p: &Value) -> Option<BodyFit> {
    match p.get("fit") {
        None | Some(Value::Null) => None,
        Some(v) => serde_json::from_value::<BodyFit>(v.clone()).ok().filter(BodyFit::is_valid),
    }
}

/// The user's placement lines, in torso units. `waistLine` and `bellyLine` are percentages of
/// the shoulder-mid → hip-mid span: ±100 moves the row ±0.30 and ±500 (±1.50, past the hip
/// joints into the thighs) is the stop; 0 leaves the measured or prior row. Their stop is far
/// wider than the shaping sliders' because a waistband or a belly can sit well below the
/// anatomical row — a low-slung waistband reads at the hip joints, a fuller belly below them —
/// and a line that stops short places the row where the user did not ask. Out-of-range values
/// are refused rather than clamped: a script that mistypes one should hear about it instead of
/// getting a silently different line.
fn line_offsets(p: &Value) -> Result<(f64, f64)> {
    let one = |key: &str| -> Result<f64> {
        match p.get(key) {
            None | Some(Value::Null) => Ok(0.0),
            Some(v) => {
                let v =
                    v.as_f64().ok_or_else(|| bad(format!("`{key}` is a percentage of the shoulder→hip-joint span (-500..500; 0 leaves the measured row)")))?;
                if !(-500.0..=500.0).contains(&v) {
                    return Err(bad(format!("`{key}` must be -500..500")));
                }
                Ok(v / 100.0 * 0.30)
            }
        }
    };
    Ok((one("waistLine")?, one("bellyLine")?))
}

/// Moves a fitted level onto the row a user line names. The widths come from the fit's own
/// silhouette rows — the run under the torso axis at that row — so a line moved to a garment's
/// waistband narrows the body where the band actually is, not where the prior guessed. `None`
/// when the line is zero or no stored row has a measurable run under the axis: the caller keeps
/// the measured row and reports the line approximate rather than pretending it landed.
fn line_level(fit: &BodyFit, anchors: &SculptAnchors, level: LevelFit, offset: f64) -> Option<LevelFit> {
    if offset == 0.0 || !offset.is_finite() {
        return None;
    }
    let mid = |i: usize, j: usize| {
        let (a, b) = (anchors.pose.get(i)?, anchors.pose.get(j)?);
        Some([(f64::from(a[0]) + f64::from(b[0])) * 0.5, (f64::from(a[1]) + f64::from(b[1])) * 0.5])
    };
    let (s, h) = (mid(11, 12)?, mid(23, 24)?);
    let ab = [h[0] - s[0], h[1] - s[1]];
    if ab[1].abs() <= 1e-6 || !ab.iter().all(|v| v.is_finite()) {
        return None;
    }
    // The clamp admits the thighs (t past 1.0): a low waistband or a fuller belly sits below
    // the hip joints, and stopping the line at the hips put it above where the user drew it.
    let target = (f64::from(level.t) + offset).clamp(0.02, 1.60);
    let target_y = s[1] + target * ab[1];
    // The stored rows are document pixels ordered by y, and t is the fraction of the way down
    // the shoulder→hip-joint line, so the nearest row in y is the nearest row in t. Rows whose
    // runs all lie away from the torso axis (a raised arm, a background figure) say nothing
    // about this line and are skipped.
    let mut best: Option<(f64, f64, f64, f64)> = None;
    for row in &fit.rows {
        let y = f64::from(row.y);
        let axis = s[0] + (y - s[1]) * ab[0] / ab[1];
        let Some(run) = row.runs.iter().find(|run| axis >= f64::from(run[0]) && axis <= f64::from(run[1])) else { continue };
        let (l, r) = (axis - f64::from(run[0]), f64::from(run[1]) - axis);
        if l < 2.0 || r < 2.0 {
            continue;
        }
        let d = (y - target_y).abs();
        if best.is_none_or(|b| d < b.0) {
            best = Some((d, (y - s[1]) / ab[1], l, r));
        }
    }
    let (_, t, l, r) = best?;
    if !t.is_finite() {
        return None;
    }
    Some(LevelFit { t: t as f32, left: l as f32, right: r as f32 })
}

/// Why a placement line did not land. Non-zero lines are the user's instruction, so a refusal
/// to honour one must be reported rather than swallowed (a line that silently does nothing is
/// the kind of partial success the operator must never be shown as done).
const LINE_UNPLACED_REASON: &str = "the placement line found no measurable silhouette row; the measured row is used";
const LINE_NO_FIT_REASON: &str = "there is no measured body fit to place the line on; the anatomical fallback is used";

/// Applies the placement lines to a fit, clearing the prior flag of each row the user placed:
/// a line drawn on a visible waistband is a measurement by eye, not a guess. The returned notes
/// name the lines that could not land.
fn apply_lines(fit: &mut BodyFit, anchors: &SculptAnchors, offsets: (f64, f64)) -> Vec<(&'static str, &'static str)> {
    let mut notes = Vec::new();
    if offsets.0 != 0.0 {
        match line_level(fit, anchors, fit.waist, offsets.0) {
            Some(level) => {
                fit.waist = level;
                fit.waist_prior = false;
            }
            None => notes.push(("waist", LINE_UNPLACED_REASON)),
        }
    }
    if offsets.1 != 0.0 {
        match line_level(fit, anchors, fit.belly, offsets.1) {
            Some(level) => {
                fit.belly = level;
                fit.belly_prior = false;
            }
            None => notes.push(("belly", LINE_UNPLACED_REASON)),
        }
    }
    notes
}

/// The notes a line needs when the run has no fit at all (a script sent anchors but no fit and
/// no mask was available): the part still acts from its anatomical fallback.
fn line_notes_without_fit(offsets: (f64, f64)) -> Vec<(&'static str, &'static str)> {
    let mut notes = Vec::new();
    if offsets.0 != 0.0 {
        notes.push(("waist", LINE_NO_FIT_REASON));
    }
    if offsets.1 != 0.0 {
        notes.push(("belly", LINE_NO_FIT_REASON));
    }
    notes
}

/// The params' own anchors — the panel's one analysis (`portrait.analyze`), or a script's.
/// `None` when absent: the lightweight analysis answers instead. Present but malformed is
/// refused by its shape rather than warping from garbage — the field builders index every
/// point, so nothing downstream may see a short or non-finite pose.
fn anchors_from_params(p: &Value) -> Result<Option<SculptAnchors>> {
    let Some(raw) = p.get("anchors") else { return Ok(None) };
    let anchors: SculptAnchors =
        serde_json::from_value(raw.clone()).map_err(|e| bad(format!("`anchors` must be the anchors `portrait.analyze` returns: {e}")))?;
    let finite = anchors.doc.iter().chain(&anchors.chin).chain(anchors.head.centre.iter()).all(|v| v.is_finite())
        && anchors.head.radius.is_finite()
        && anchors.pose.iter().flatten().all(|v| v.is_finite());
    if anchors.pose.len() != photocraft_algo::sculpt::POSE_COUNT || !finite {
        return Err(bad("`anchors` must carry 33 finite pose points [x, y, visibility], a finite head and chin, and the document size"));
    }
    Ok(Some(anchors))
}

/// The head and chin anchors: the face oval's own centroid and radius (grown to cover the
/// hair) and its lowest point; without a mesh run, proportions of the nose and shoulders
/// stand in — the head sliders are coarse enough that the estimate carries them.
fn head_anchors(a: &photocraft_portrait::Analysis) -> (SculptHead, [f32; 2]) {
    if a.landmarks.len() > 460 {
        let ring: Vec<[f64; 2]> = photocraft_portrait::face::FACE_OVAL.iter().map(|&i| [a.landmarks[i][0] as f64, a.landmarks[i][1] as f64]).collect();
        let n = ring.len() as f64;
        let (cx, cy) = ring.iter().fold((0.0, 0.0), |(x, y), p| (x + p[0] / n, y + p[1] / n));
        let radius = ring.iter().map(|p| (p[0] - cx).hypot(p[1] - cy)).sum::<f64>() / n * 1.15;
        let chin = a.landmarks[152];
        return (SculptHead { centre: [cx as f32, cy as f32], radius: radius as f32 }, [chin[0], chin[1]]);
    }
    let nose = a.pose.first().copied().unwrap_or_default();
    let (l, r) = (a.pose.get(11).copied().unwrap_or_default(), a.pose.get(12).copied().unwrap_or_default());
    let shoulder_mid = [(l[0] + r[0]) * 0.5, (l[1] + r[1]) * 0.5];
    let lerp = |t: f32| [nose[0] + (shoulder_mid[0] - nose[0]) * t, nose[1] + (shoulder_mid[1] - nose[1]) * t];
    let d = (shoulder_mid[0] - nose[0]).hypot(shoulder_mid[1] - nose[1]);
    (SculptHead { centre: lerp(-0.28), radius: 0.55 * d }, lerp(0.38))
}

/// The anchors of an analysis: the pose as `[x, y, visibility]`, the head and the chin.
/// `None` when the pose stage never answered — the caller's "no person" wording.
pub(crate) fn anchors_from_analysis(a: &photocraft_portrait::Analysis) -> Option<SculptAnchors> {
    if a.pose.len() != photocraft_algo::sculpt::POSE_COUNT {
        return None;
    }
    let pose = a.pose.iter().map(|q| [q[0], q[1], q[3]]).collect();
    let (head, chin) = head_anchors(a);
    Some(SculptAnchors { doc: [a.width as f32, a.height as f32], pose, head, chin })
}

/// The x of the pose's torso axis (shoulder mid → hip mid, extended to `y`) on a document row,
/// when the shoulders name a torso; `None` otherwise. The contour is seeded on this axis so the
/// subject is chosen by its landmarks rather than by whatever the mask happened to pick up
/// beside it.
fn torso_axis_x(anchors: &SculptAnchors, y: f64) -> Option<f64> {
    let mid = |i: usize, j: usize| {
        let (a, b) = (anchors.pose.get(i)?, anchors.pose.get(j)?);
        Some([(f64::from(a[0]) + f64::from(b[0])) * 0.5, (f64::from(a[1]) + f64::from(b[1])) * 0.5])
    };
    let (s, h) = (mid(11, 12)?, mid(23, 24)?);
    if !s.into_iter().chain(h).all(f64::is_finite) {
        return None;
    }
    let dy = h[1] - s[1];
    Some(if dy.abs() > 1.0 { s[0] + (y - s[1]) * (h[0] - s[0]) / dy } else { s[0] })
}

/// The person mask's silhouette as a closed polyline: the left and right rims of the subject's
/// own span in every sampled row, walked down one side and back up the other. The span is not
/// the row's outermost mask pixels: a person matte often carries a background figure, a crowd
/// or the sky on the same row, and taking the union rim made the contour jump hundreds of pixels
/// between rows — the low-pass then read the jump as a bump and smeared the whole neighbourhood
/// with a huge circular warp. The span is tracked row to row: each row takes every run that
/// overlaps the previous span, so legs between torso rows stay whole, while a disconnected
/// background figure beside the subject is left out. Rows that read empty — or edge-to-edge, a
/// subject tier that cannot separate this photo's background — contribute nothing, and a row
/// whose span widens implausibly fast is the mask fusing the subject with the background: the
/// walk stops there rather than following the fusion into the crowd, so the contour covers only
/// the part of the silhouette the mask can actually resolve.
pub(crate) fn person_contour(mask: &Surface, anchors: &SculptAnchors) -> Option<Vec<[f64; 2]>> {
    let b = mask.content_bounds();
    if b.width() < 8 || b.height() < 16 {
        return None;
    }
    let rows: usize = b.height() as usize;
    let step = (rows / 120).max(1);
    let full = b.x1 - b.x0;
    // Every sampled row's usable runs: at least a few pixels wide, and not the whole width.
    let mut ys: Vec<i32> = Vec::new();
    let mut runs: Vec<Vec<(i32, i32)>> = Vec::new();
    let mut y = b.y0;
    while y < b.y1 {
        let mut row: Vec<(i32, i32)> = Vec::new();
        let mut start: Option<i32> = None;
        for x in b.x0..b.x1 {
            if mask.sample_channel(x, y, 0) > 0.5 {
                start.get_or_insert(x);
            } else if let Some(s) = start.take()
                && x - s > 4
                && x - s < full * 9 / 10
            {
                row.push((s, x - 1));
            }
        }
        if let Some(s) = start
            && b.x1 - s > 4
            && b.x1 - s < full * 9 / 10
        {
            row.push((s, b.x1 - 1));
        }
        ys.push(y);
        runs.push(row);
        y += step as i32;
    }
    // The seed is the row nearest the content's vertical middle — the torso or hips on a
    // standing person — not the widest row anywhere: on a photo whose background fuses with the
    // subject, the widest row is the fusion, and starting there centred the whole contour on it.
    // Within that row the runs are taken around the pose's own torso axis, so a background
    // figure beside a standing subject is not the seed even when it is the widest thing there,
    // and two separated legs are both seeded.
    let mid = (b.y0 + b.y1) / 2;
    let seed = runs.iter().enumerate().filter(|(_, r)| !r.is_empty()).min_by_key(|&(k, _)| (ys[k] - mid).abs())?.0;
    let axis = torso_axis_x(anchors, f64::from(ys[seed])).unwrap_or(f64::from(b.x0 + full / 2));
    let shoulder_w = anchors.pose.get(11).zip(anchors.pose.get(12)).map_or(full as f32, |(a, b)| (a[0] - b[0]).hypot(a[1] - b[1])).max(8.0);
    // At the waist the torso, and the legs when they are apart, lie within about half a
    // shoulder width of the axis; a background figure that is its own run sits farther out.
    let window = f64::from(shoulder_w) * 0.5;
    let mut seeded: Vec<(i32, i32)> = runs[seed].iter().copied().filter(|&(l, r)| f64::from(r) >= axis - window && f64::from(l) <= axis + window).collect();
    if seeded.is_empty() {
        let near = |&&(l, r): &&(i32, i32)| ((f64::from(l) + f64::from(r)) * 0.5 - axis).abs();
        seeded.push(*runs[seed].iter().min_by(|a, b| near(a).total_cmp(&near(b)))?);
    }
    let widest = (seeded.iter().map(|&(l, _)| l).min()?, seeded.iter().map(|&(_, r)| r).max()?);
    let seed_w = f64::from(widest.1 - widest.0);
    let mut span: Vec<(i32, i32, i32)> = vec![(ys[seed], widest.0, widest.1)];
    let prev = (widest.0, widest.1);
    // Walk both ways from the seed; a row joins the contour when any of its runs overlaps the
    // previous span in x, and the span becomes the union of the overlapping runs. A span that
    // grows by more than a quarter of its width (plus a few pixels for a coarse row step) is the
    // mask running into the background, not the subject's own outline: the walk ends below it.
    let total_cap = seed_w * 3.0;
    for dir in [1isize, -1] {
        let mut here = prev;
        let mut k = seed as isize + dir;
        while (0..runs.len() as isize).contains(&k) {
            let row = &runs[k as usize];
            let joining: Vec<(i32, i32)> = row.iter().copied().filter(|&(l, r)| l <= here.1 && r >= here.0).collect();
            if joining.is_empty() {
                break;
            }
            let l = joining.iter().map(|r| r.0).min()?;
            let r = joining.iter().map(|r| r.1).max()?;
            let grow = f64::from(here.1 - here.0) * 0.25 + 8.0;
            if f64::from(r - l) > f64::from(here.1 - here.0) + grow || f64::from(r - l) > total_cap {
                break;
            }
            span.push((ys[k as usize], l, r));
            here = (l, r);
            k += dir;
        }
    }
    if span.len() < 8 {
        return None;
    }
    span.sort_by_key(|s| s.0);
    let mut left: Vec<[f64; 2]> = span.iter().map(|&(y, l, _)| [l as f64 + 0.5, y as f64 + 0.5]).collect();
    let mut right: Vec<[f64; 2]> = span.iter().rev().map(|&(y, _, r)| [r as f64 + 0.5, y as f64 + 0.5]).collect();
    left.append(&mut right);
    Some(left)
}

/// The lower abdomen's widest row in `[lo, hi]` and whether the outline confirms it as a bump:
/// the apex must be measurably wider than the profile at both ends of the range. `None` when
/// the range holds no rows. A monotonic slope puts the apex at an end, where it cannot be wider
/// than that end, so it reads as unconfirmed rather than as a measured belly.
fn belly_apex(profile: &[(f64, f64, f64)], lo: f64, hi: f64) -> Option<((f64, f64, f64), bool)> {
    let width_near = |t: f64| profile.iter().copied().min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs())).map_or(0.0, |p| p.1 + p.2);
    let rows: Vec<(f64, f64, f64)> = profile.iter().copied().filter(|&(t, _, _)| t >= lo && t <= hi).collect();
    let max = rows.iter().map(|p| p.1 + p.2).max_by(f64::total_cmp)?;
    let band: Vec<(f64, f64, f64)> = rows.iter().copied().filter(|p| p.1 + p.2 >= max * 0.98 - 1.0).collect();
    let apex = band.get(band.len() / 2).copied()?;
    let w = apex.1 + apex.2;
    let margin = (0.01 * w).max(2.0);
    Some((apex, w > width_near(lo) + margin && w > width_near(hi) + margin))
}

/// The person silhouette fitted to the body: the torso's waist, belly, hip and bust levels,
/// each limb's thickness at its bone, and the mask rows the field clips to. This is what makes
/// the slimming rows act on the visible outline instead of somewhere inside it: a part's
/// width is set to twice its distance to the silhouette, so the field's peak lands on the
/// outline, and the fit's `inside` flags let an occluded joint that still lies within the
/// person run at reduced strength instead of being skipped. Coordinates are document pixels;
/// the work runs on at most a 1600 px copy, which is plenty for a field whose cells are
/// several pixels.
pub fn body_fit(anchors: &SculptAnchors, mask: &Surface) -> Option<BodyFit> {
    if anchors.pose.len() != POSE_COUNT {
        return None;
    }
    let b = mask.content_bounds();
    if b.width() < 8 || b.height() < 16 {
        return None;
    }
    let scale = (b.width().max(b.height()) as usize).div_ceil(1600).max(1);
    let fw = (b.width() as usize).div_ceil(scale);
    let fh = (b.height() as usize).div_ceil(scale);
    let n = fw.checked_mul(fh)?;
    if n == 0 {
        return None;
    }
    let mut inside = vec![false; n];
    for y in 0..fh {
        let sy = b.y0 + (y * scale) as i32 + (scale / 2) as i32;
        for x in 0..fw {
            let sx = b.x0 + (x * scale) as i32 + (scale / 2) as i32;
            inside[y * fw + x] = mask.sample_channel(sx, sy, 0) > 0.5;
        }
    }
    if !inside.iter().any(|v| *v) {
        return None;
    }
    // Distance from every body pixel to the nearest background: at a bone, that is the limb's
    // half-thickness, and it survives an arm touching the torso (the nearest edge is the arm's).
    let outside: Vec<bool> = inside.iter().map(|v| !v).collect();
    let dist = photocraft_algo::selection::edt(&outside, fw, fh);
    if dist.len() != n {
        return None;
    }

    // The mask rows the field clips to, and the same rows in the reduced grid for the fit.
    let step = (fh / 160).max(1);
    let mut rows = Vec::with_capacity(160);
    let mut local: Vec<(usize, Vec<(usize, usize)>)> = Vec::with_capacity(160);
    let mut y = 0;
    while y < fh {
        let Some(row) = inside.get(y * fw..(y + 1) * fw) else { break };
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut x = 0;
        while x < fw {
            if !row.get(x).copied().unwrap_or(false) {
                x += 1;
                continue;
            }
            let x0 = x;
            while x < fw && row.get(x).copied().unwrap_or(false) {
                x += 1;
            }
            if x - x0 >= 2 {
                runs.push((x0, x - 1));
            }
        }
        if !runs.is_empty() {
            let doc_y = b.y0 + (y * scale) as i32;
            rows.push(BodyRow {
                y: doc_y as f32 + scale as f32 * 0.5,
                runs: runs.iter().map(|&(x0, x1)| [(b.x0 + (x0 * scale) as i32) as f32, (b.x0 + ((x1 + 1) * scale) as i32) as f32]).collect(),
            });
            local.push((y, runs));
        }
        y += step;
    }
    if rows.len() < 6 {
        return None;
    }

    let joints: Vec<[f64; 2]> =
        anchors.pose.iter().map(|q| [(f64::from(q[0]) - f64::from(b.x0)) / scale as f64, (f64::from(q[1]) - f64::from(b.y0)) / scale as f64]).collect();
    let pose = |i: usize| joints.get(i).copied().unwrap_or([0.0, 0.0]);
    let shoulder = [(pose(11)[0] + pose(12)[0]) * 0.5, (pose(11)[1] + pose(12)[1]) * 0.5];
    let hip = [(pose(23)[0] + pose(24)[0]) * 0.5, (pose(23)[1] + pose(24)[1]) * 0.5];
    let ab = [hip[0] - shoulder[0], hip[1] - shoulder[1]];
    let ab2 = ab[0] * ab[0] + ab[1] * ab[1];
    if ab2 < 1.0 {
        return None;
    }
    let axis_x = |yl: f64| if ab[1].abs() > 1e-6 { shoulder[0] + (yl - shoulder[1]) * ab[0] / ab[1] } else { shoulder[0] };
    let t_of = |yl: f64| {
        let x = axis_x(yl);
        ((x - shoulder[0]) * ab[0] + (yl - shoulder[1]) * ab[1]) / ab2
    };

    // The torso's own width profile: the run that holds the axis, at each sampled row.
    let mut profile: Vec<(f64, f64, f64)> = Vec::with_capacity(local.len());
    for &(yl, ref runs) in &local {
        let t = t_of(yl as f64);
        if !(-0.20..=1.25).contains(&t) {
            continue;
        }
        let ax = axis_x(yl as f64);
        // The run must show the subject's own edges on both sides: a run clipped by the mask's
        // content bounds measures distance to the frame (or to wherever the matte stops), not
        // the silhouette. A matte that fuses the background into the person is one full-width
        // run per row, and trusting it put the waist's envelope across the whole photo — the
        // slider read as warping the chest and the water, not the waist. Such rows carry no
        // width, so the fit as a whole has to answer `None` and the joints stand in.
        let bounded = |&&(x0, x1): &&(usize, usize)| x0 > 0 && x1 + 1 < fw;
        let Some(&(x0, x1)) = runs.iter().find(|run| bounded(run) && ax >= run.0 as f64 && ax <= run.1 as f64 + 1.0) else { continue };
        let (l, r) = (ax - x0 as f64, x1 as f64 + 1.0 - ax);
        if l >= 2.0 && r >= 2.0 {
            profile.push((t, l, r));
        }
    }
    if profile.len() < 6 {
        return None;
    }
    let smooth = |v: &[f64]| -> Vec<f64> {
        (0..v.len())
            .map(|i| {
                let (a, b) = (i.saturating_sub(2), (i + 3).min(v.len()));
                v.get(a..b).map_or(0.0, |s| s.iter().sum::<f64>() / (b - a).max(1) as f64)
            })
            .collect()
    };
    let ts: Vec<f64> = profile.iter().map(|p| p.0).collect();
    let ls = smooth(&profile.iter().map(|p| p.1).collect::<Vec<_>>());
    let rs = smooth(&profile.iter().map(|p| p.2).collect::<Vec<_>>());
    let profile: Vec<(f64, f64, f64)> = ts.iter().zip(&ls).zip(&rs).map(|((&t, &l), &r)| (t, l, r)).collect();
    // The narrowest row, taken as the middle of the rows within a hair of it: a flat waist
    // plateau should not put the level on its first row.
    let narrowest = |lo: f64, hi: f64| -> Option<(f64, f64, f64)> {
        let rows: Vec<(f64, f64, f64)> = profile.iter().copied().filter(|&(t, _, _)| t >= lo && t <= hi).collect();
        let min = rows.iter().map(|p| p.1 + p.2).min_by(f64::total_cmp)?;
        let band: Vec<(f64, f64, f64)> = rows.iter().copied().filter(|p| p.1 + p.2 <= min * 1.02 + 1.0).collect();
        band.get(band.len() / 2).copied()
    };
    let near = |t: f64| -> (f64, f64, f64) { profile.iter().copied().min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs())).unwrap_or((t, 24.0, 24.0)) };
    let width_at = |t: f64| {
        let p = near(t);
        p.1 + p.2
    };

    // The waist is a crease in the silhouette, not simply its narrowest row: a loose dress or
    // a straight-cut coat tapers all the way into its hem, so "narrowest" would put the waist
    // on the hips and the waist slider would squeeze the top of the skirt. The row counts
    // only when it is also measurably narrower than the torso at 35% and at 90% of the
    // shoulder→hip span. The bust is anatomical too, near a third of the way down: above
    // that the arms merge with the chest, so "widest above the waist" always returns the
    // armpit, not the bust. The hips are the joints' own row — a skirt's widest hem is not
    // the body.
    let hip = near(1.0);
    let crease = narrowest(0.40, 0.80).filter(|&(_, l, r)| {
        let w = l + r;
        let margin = (0.01 * w).max(2.0);
        w + margin < width_at(0.35) && w + margin < width_at(0.90)
    });
    // Without a crease the mask cannot say where the waist is, and the row it widens into is
    // not a substitute: a fuller body, a pregnancy and a low-slung garment all widen into the
    // hip row, but only the garment's cinch is there — on the other two that row is the belly
    // itself. The anatomical waist between the lower ribs and the hip joints, about 0.70 of
    // the shoulder→hip-joint span, is the body-type-stable prior; the panel's line control
    // moves it to a visible waistband, and the report names the prior so no one reads it as a
    // measurement. The older 0.54/0.62 middles sat on the ribs, which is what made the slider
    // read as squeezing the chest.
    const WAIST_PRIOR_T: f64 = 0.70;
    let mut waist = crease.unwrap_or_else(|| near(WAIST_PRIOR_T));
    // The waist row fell to a prior rather than to a measured crease: the report says so.
    let mut waist_prior = crease.is_none();
    let bust = near(0.30f64.min(waist.0 - 0.10));
    if bust.0 >= waist.0 {
        // No room above the waist for the bust: the waist measurement is suspect, so the
        // anatomical prior takes over and the bust keeps its third.
        waist = near(WAIST_PRIOR_T.max(bust.0 + 0.10));
        waist_prior = true;
    }
    waist.0 = waist.0.clamp(0.04, 1.02);

    // The belly is measured, not assumed: its centre is the lower abdomen's widest row below
    // the waist, where a fuller or pregnant belly's own silhouette peaks. When that row sits
    // at the edge of the range — a monotonic widening, or a low waistband with no room below
    // it — the outline shows no bump to confirm; the row is still the widest lower abdomen,
    // and the report says the position is approximate rather than calling it measured.
    let belly_lo = (waist.0 + 0.08).max(0.55);
    let belly_hi = (hip.0 - 0.04).min(1.15);
    let (belly, belly_prior) = if belly_hi > belly_lo + 0.02 {
        match belly_apex(&profile, belly_lo, belly_hi) {
            Some((apex, bump)) => (apex, !bump),
            None => (near(((waist.0 + hip.0) * 0.5).clamp(0.72, 0.95)), true),
        }
    } else {
        // No room below the waist (a low waistband): the belly sits above it.
        (near((waist.0 - 0.12).clamp(0.72, 0.95)), true)
    };
    let level = |p: (f64, f64, f64)| LevelFit { t: p.0 as f32, left: (p.1 * scale as f64) as f32, right: (p.2 * scale as f64) as f32 };

    let limbs = limb_fits(&joints, &inside, &dist, fw, fh, scale);

    let mut joint_inside = vec![false; POSE_COUNT];
    for (i, flag) in joint_inside.iter_mut().enumerate() {
        if let Some(q) = anchors.pose.get(i) {
            let x = (f64::from(q[0]) - f64::from(b.x0)) / scale as f64;
            let y = (f64::from(q[1]) - f64::from(b.y0)) / scale as f64;
            *flag = inside_near(&inside, fw, fh, x, y, 2);
        }
    }

    Some(BodyFit { rows, waist: level(waist), belly: level(belly), hip: level(hip), bust: level(bust), limbs, inside: joint_inside, waist_prior, belly_prior })
}

/// Each limb's fitted thickness and the fraction of its bone the mask covers, read on the
/// reduced grid: the median distance to the nearest background at samples along the bone is the
/// limb's half-thickness there, capped by the bone's own length so a torso next to the bone
/// cannot inflate an arm. A bone shorter than a few cells, or one whose samples all miss the
/// mask, contributes nothing and the field falls back to the joint spacing for that limb.
fn limb_fits(joints: &[[f64; 2]], inside: &[bool], dist: &[f32], fw: usize, fh: usize, scale: usize) -> [LimbFit; 8] {
    let mut limbs = [LimbFit::default(); LIMB_SEGMENTS.len()];
    for (idx, &(i, j)) in LIMB_SEGMENTS.iter().enumerate() {
        let a = joints.get(i).copied().unwrap_or([0.0, 0.0]);
        let c = joints.get(j).copied().unwrap_or([0.0, 0.0]);
        let len = (c[0] - a[0]).hypot(c[1] - a[1]);
        if len < 4.0 {
            continue;
        }
        let mut radii: Vec<f64> = Vec::new();
        let mut span: Vec<f64> = Vec::new();
        for k in 0..=20 {
            let t = 0.02 + 0.96 * f64::from(k) / 20.0;
            let x = a[0] + (c[0] - a[0]) * t;
            let y = a[1] + (c[1] - a[1]) * t;
            let (xi, yi) = (x.round() as isize, y.round() as isize);
            if xi < 0 || yi < 0 || xi as usize >= fw || yi as usize >= fh {
                continue;
            }
            let at = yi as usize * fw + xi as usize;
            if !inside.get(at).copied().unwrap_or(false) {
                continue;
            }
            span.push(t);
            if (0.25..=0.75).contains(&t)
                && let Some(&d) = dist.get(at)
                && d.is_finite()
                && d > 1.0
                && d < 1.0e8
            {
                radii.push(f64::from(d));
            }
        }
        let (Some(&t0), Some(&t1)) = (span.first(), span.last()) else { continue };
        if t1 - t0 < 0.10 {
            continue;
        }
        radii.sort_by(f64::total_cmp);
        let cap = match idx {
            0 | 2 => 0.30 * len,
            1 | 3 => 0.26 * len,
            4 | 5 => 0.36 * len,
            _ => 0.28 * len,
        };
        let floor = (0.10 * len).max(3.0).min(cap);
        let median = radii.get(radii.len() / 2).copied().unwrap_or(0.0);
        let radius = if median > 0.0 { median.clamp(floor, cap) } else { 0.0 };
        limbs[idx] = LimbFit { radius: (radius * scale as f64) as f32, t0: t0 as f32, t1: t1 as f32 };
    }
    limbs
}

/// Whether `(x, y)` or a small neighbourhood around it is inside `mask`; the tolerance keeps a
/// joint on the silhouette's own edge (a wrist, an ankle) from reading as outside it.
fn inside_near(mask: &[bool], w: usize, h: usize, x: f64, y: f64, r: isize) -> bool {
    let (cx, cy) = (x.round() as isize, y.round() as isize);
    for dy in -r..=r {
        for dx in -r..=r {
            let (px, py) = (cx + dx, cy + dy);
            if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h && mask.get(py as usize * w + px as usize).copied().unwrap_or(false) {
                return true;
            }
        }
    }
    false
}

/// The symmetry amount (对称液化·上半身, one-directional 0..100).
fn symmetry_of(p: &Value) -> f32 {
    p.get("symmetry").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, 100.0) as f32
}

/// The smoothing amount (身型流畅, one-directional 0..100).
fn smooth_of(p: &Value) -> f32 {
    p.get("smooth").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, 100.0) as f32
}

/// The stored contour as `[[x, y], …]` document pixels, when the params carry one.
fn contour_of(p: &Value) -> Option<Vec<[f64; 2]>> {
    serde_json::from_value::<Vec<[f64; 2]>>(p.get("contour")?.clone()).ok().filter(|c| c.len() >= 8)
}

/// Stamps the contour relaxation into a built field (the run path builds it from the
/// analysis, the replay path reads it back from the stored params — same stamp either way).
fn stamp_contour(field: &mut photocraft_algo::liquify::LiquifyField, pts: &[[f64; 2]], smooth: f32) -> usize {
    let len: f64 = pts.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum::<f64>()
        + (pts[0][0] - pts[pts.len() - 1][0]).hypot(pts[0][1] - pts[pts.len() - 1][1]);
    contour::stamp(field, pts, len / pts.len() as f64, f64::from(smooth))
}

/// The field every path builds: the slider warp, the symmetry pair alignment and the
/// contour relaxation. Apply, the smart-filter replay and the panel's live preview all call
/// it with the same values, so the field they show cannot drift from the one they commit.
fn field_with(spec: SculptSpec, contour: Option<&[[f64; 2]]>, symmetry: f32, smooth: f32, canvas: Rect, cell: f64) -> (LiquifyField, SculptReport) {
    let (mut field, mut report) = sculpt_field(&spec, canvas, cell);
    // 对称液化·上半身: the paired joints align toward each other's mirror about the
    // torso axis — pure geometry of the stored joints, replayed for free. Only a local touch
    // of asymmetry is aligned: a joint the pose guessed, or a pair in a genuinely different
    // pose (one arm raised), is left alone instead of dragging its whole falloff disc across
    // the photo. The report names the part approximate when only some pairs qualified.
    if symmetry > 0.0 {
        let pts: Vec<[f64; 2]> = spec.anchors.pose.iter().map(|q| [q[0] as f64, q[1] as f64]).collect();
        let vis: Vec<f32> = spec.anchors.pose.iter().map(|q| q[2]).collect();
        let seen = |i: usize| vis.get(i).copied().unwrap_or(0.0) >= SYMMETRY_SEEN;
        if !seen(11) || !seen(12) {
            report.skipped.push(("symmetry", "the pose does not see both shoulders"));
        } else {
            let shoulder_mid = [(pts[11][0] + pts[12][0]) * 0.5, (pts[11][1] + pts[12][1]) * 0.5];
            let reach = ((pts[11][0] - pts[12][0]).hypot(pts[11][1] - pts[12][1]) * 0.5).max(8.0);
            // The torso axis runs shoulder mid to hip mid when the pose sees the hips; when it
            // does not, the shoulder line's own downward normal is the torso's direction.
            let dir = if seen(23) && seen(24) {
                let hip_mid = [(pts[23][0] + pts[24][0]) * 0.5, (pts[23][1] + pts[24][1]) * 0.5];
                [hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]]
            } else {
                let u = [pts[11][0] - pts[12][0], pts[11][1] - pts[12][1]];
                let p = [-u[1], u[0]];
                if p[1] < 0.0 { [-p[0], -p[1]] } else { p }
            };
            let pairs = contour::alignable_pairs(&pts, &vis, SYMMETRY_PAIRS, shoulder_mid, dir, reach);
            if pairs.is_empty() {
                report.skipped.push(("symmetry", "the pose's paired joints are too far from symmetric to align"));
            } else {
                contour::align_pairs(&mut field, &pts, &pairs, shoulder_mid, dir, f64::from(symmetry), reach);
                if pairs.len() < SYMMETRY_PAIRS.len() {
                    report.approximate.push(("symmetry", "joints the pose missed or saw far from their mirror are left alone"));
                }
                report.applied.push("symmetry");
            }
        }
    }
    // 身型流畅: the silhouette's own contour, low-passed and stamped over the parts. The
    // contour rides the stored params like the anchors, so the re-render replays it with
    // no mask and no analysis.
    if smooth > 0.0 {
        match contour {
            Some(pts) => {
                let dropped = stamp_contour(&mut field, pts, smooth);
                if dropped > 0 {
                    report.approximate.push(("smooth", "parts of the subject outline merge into the background; only its own bumps are relaxed there"));
                }
                report.applied.push("smooth");
            }
            None => report.skipped.push(("smooth", "the subject mask has no contour to relax")),
        }
    }
    (field, report)
}

/// The field `params` alone describe — exactly the one Apply builds, from the anchors
/// `portrait.analyze` found and the contour it embedded. The panel's live preview calls it
/// with the params Apply will send, so what is shown is what is committed.
pub fn field_for(p: &Value, canvas: Rect) -> Result<(LiquifyField, SculptReport)> {
    let values = sliders(p)?;
    let anchors = anchors_from_params(p)?.ok_or_else(|| bad("`anchors` are required here — `portrait.analyze` supplies them"))?;
    let cell = cell(p, canvas)?;
    let offsets = line_offsets(p)?;
    let mut fit = fit_of(p);
    let line_notes = match fit.as_mut() {
        Some(fit) => apply_lines(fit, &anchors, offsets),
        None => line_notes_without_fit(offsets),
    };
    let spec = SculptSpec { sliders: values, anchors, fit };
    let (field, mut report) = field_with(spec, contour_of(p).as_deref(), symmetry_of(p), smooth_of(p), canvas, cell);
    report.approximate.extend(line_notes);
    Ok((field, report))
}

/// What a smart filter stores: the params the dialog sent plus the anchors they were resolved
/// against, the fitted rows and the contour, so the re-render rebuilds this exact field with no
/// analysis. The placement lines were baked into `fit` by the caller and are dropped here —
/// keeping them would move the rows a second time on every re-render.
fn stored_params(p: &Value, anchors: &SculptAnchors, fit: Option<&BodyFit>, contour: Option<&[[f64; 2]]>) -> Result<Value> {
    let mut stored = p.clone();
    if let Value::Object(m) = &mut stored {
        m.insert("anchors".into(), serde_json::to_value(anchors).map_err(|e| bad(format!("the anchors would not serialise: {e}")))?);
        if let Some(fit) = fit {
            m.insert("fit".into(), serde_json::to_value(fit).map_err(|e| bad(format!("the body fit would not serialise: {e}")))?);
        }
        if let Some(pts) = contour {
            m.insert("contour".into(), serde_json::to_value(pts).map_err(|e| bad(format!("the contour would not serialise: {e}")))?);
        }
        m.remove("waistLine");
        m.remove("bellyLine");
    }
    Ok(stored)
}

/// The two placement rows the panel draws, in document pixels: `(y, x0, x1)` for the waist
/// (solid) and the belly (dashed), after the user's line offsets. These are the levels the
/// field acts on — the same `apply_lines` the preview and Apply run — so the drawn line cannot
/// disagree with the warp. `None` when the params carry no usable fit to place a line on.
pub fn placement_lines(p: &Value) -> Option<[(f64, f64, f64); 2]> {
    let anchors = anchors_from_params(p).ok()??;
    let mut fit = fit_of(p)?;
    apply_lines(&mut fit, &anchors, line_offsets(p).ok()?);
    let mid = |i: usize, j: usize| {
        let (a, b) = (anchors.pose.get(i)?, anchors.pose.get(j)?);
        Some([(f64::from(a[0]) + f64::from(b[0])) * 0.5, (f64::from(a[1]) + f64::from(b[1])) * 0.5])
    };
    let (s, h) = (mid(11, 12)?, mid(23, 24)?);
    // t is the fraction along the straight shoulder-mid → hip-mid line, so the row's y and
    // axis x are one lerp; the fitted half-widths give the span the row's kernel acts over.
    let line = |level: LevelFit| {
        let t = f64::from(level.t);
        let (y, x) = (s[1] + t * (h[1] - s[1]), s[0] + t * (h[0] - s[0]));
        (y, x - f64::from(level.left), x + f64::from(level.right))
    };
    Some([line(fit.waist), line(fit.belly)])
}

/// The one lightweight analysis a run without anchors needs: the person mask and the
/// body-pose landmarks — not the Smart Portrait regions. Select Subject's precedent of a
/// synchronous, progress-less ML analysis (a detached job context whose progress goes
/// nowhere).
fn lightweight(s: &Session, all_layers: bool) -> Result<std::sync::Arc<photocraft_portrait::Analysis>> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let target = d.active_layer.filter(|l| !all_layers && d.doc.layer(*l).and_then(photocraft_doc::Layer::surface).is_some());
    let ctx = crate::jobs::JobCtx::new();
    crate::portrait_cmds::analyze_sculpt(&d.doc, target, all_layers, crate::portrait_cmds::SculptKind::Body, None, &ctx).map(|found| found.analysis)
}

fn sculpt(s: &mut Session, p: &Value) -> Result<Value> {
    let values = sliders(p)?;
    let offsets = line_offsets(p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let id = crate::commands::layer_param(s, p)?;
    let smooth = smooth_of(p);
    let symmetry = symmetry_of(p);
    if values == SculptSliders::default() && smooth <= 0.0 && symmetry <= 0.0 {
        return Ok(json!({"layer": id.0, "changed": false}));
    }
    let cell = cell(p, canvas)?;
    let all_layers = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let t0 = Stopwatch::start();
    // The anchors the panel analysed are the fast path: Apply is a field build and a warp.
    // Without them (a script, an action) one lightweight analysis answers the pose — never
    // the full Smart Portrait pipeline.
    let explicit = anchors_from_params(p)?;
    let mut fit = fit_of(p);
    let (anchors, mut analysis) = match explicit {
        Some(anchors) => (anchors, None),
        None => {
            let analysis = lightweight(s, all_layers)?;
            let anchors = anchors_from_analysis(&analysis).ok_or_else(|| EngineError::Other(NO_PERSON.into()))?;
            (anchors, Some(analysis))
        }
    };
    // 身型流畅: the silhouette contour needs the person mask. The params' own contour (what
    // `portrait.analyze` and the smart filter store) spares that ask; without one, the
    // lightweight analysis answers it, and a photo it cannot segment only skips the pass.
    let contour = if smooth > 0.0 {
        match contour_of(p) {
            Some(pts) => Some(pts),
            None => {
                if analysis.is_none() {
                    analysis = lightweight(s, all_layers).ok();
                }
                analysis.as_ref().and_then(|a| a.mask(photocraft_portrait::Region::Person)).and_then(|m| person_contour(m, &anchors))
            }
        }
    } else {
        None
    };
    // The fit is what turns joint spacing into the visible silhouette. The panel sends it with
    // the anchors; a script that sent anchors alone gets the joint-spacing fallback (no
    // analysis on Apply), and the analysis path above already has the mask to measure it.
    if fit.is_none()
        && let Some(a) = &analysis
        && let Some(mask) = a.mask(photocraft_portrait::Region::Person)
    {
        fit = body_fit(&anchors, mask);
    }
    // The user's placement lines move the fitted rows before the field is built; the fitted
    // widths follow the row they land on, so preview, Apply and the replay agree.
    let line_notes = match fit.as_mut() {
        Some(fit) => apply_lines(fit, &anchors, offsets),
        None => line_notes_without_fit(offsets),
    };
    let spec = SculptSpec { sliders: values, anchors: anchors.clone(), fit: fit.clone() };
    let (field, mut report) = field_with(spec, contour.as_deref(), symmetry, smooth, canvas, cell);
    report.approximate.extend(line_notes);
    let smart = matches!(s.active().and_then(|d| d.doc.layer(id)).map(|l| &l.content), Some(LayerContent::Smart(_)));
    if field.is_identity() && !smart {
        return Ok(json!({"layer": id.0, "changed": false, "parts": parts(&report)}));
    }
    let stored = stored_params(p, &anchors, fit.as_ref(), contour.as_deref())?;
    let r = crate::distort_cmds::run_on_layer(
        s,
        SCULPT,
        "Body Sculpt",
        &stored,
        false,
        &|surf, sel, _| {
            let out = apply_liquify(surf, &field);
            Ok(match sel {
                Some(sel) => crate::distort_cmds::mix_by_selection(surf, &out, sel)?,
                None => out,
            })
        },
        // A linked mask deforms on the same field (#3083): a slimmed body under an unslimmed
        // mask is a broken silhouette.
        Some(&|mask, _| Ok(apply_liquify(mask, &field))),
    )?;
    let mut r = r;
    r["ms"] = json!(t0.ms());
    r["maxDisplacement"] = json!(field.max_displacement());
    r["parts"] = parts(&report);
    Ok(r)
}

fn parts(report: &photocraft_algo::sculpt::SculptReport) -> Value {
    json!({
        "applied": report.applied,
        "approximate": report.approximate.iter().map(|(name, why)| json!({"part": name, "why": why})).collect::<Vec<_>>(),
        "skipped": report.skipped.iter().map(|(name, why)| json!({"part": name, "why": why})).collect::<Vec<_>>(),
    })
}

/// Smart-filter hook: re-applies a stored Body Sculpt to a rendered smart object.
pub(crate) fn apply_to_surface(params: &Value, surf: &Surface, canvas: Rect) -> Option<Surface> {
    let (spec, cell) = stored_spec(params, canvas)?;
    let (field, _) = field_with(spec, contour_of(params).as_deref(), symmetry_of(params), smooth_of(params), canvas, cell);
    Some(apply_liquify(surf, &field))
}

pub fn specs() -> Vec<CommandSpec> {
    const P: &str = r##"{"slimUpper":-100..100=0 (瘦身上身: the torso above the waist narrows as one coordinated envelope — shoulders, chest and waist together, easing out below the waist so the hips and legs stay still),"slimFull":-100..100=0 (瘦身全身: the whole body narrows in one coordinated pass — the torso through the hips, the legs and the arms each about their own bones, so the silhouette reads as a whole-body slim rather than a squeeze on the torso's sides),"waist":-100..100=0,"belly":-100..100=0,"arms":-100..100=0,"forearms":-100..100=0 (the lower arm from the elbow past the wrist through the palm and fingers; the pose has no hand landmarks, so the hand follows the forearm's direction and the report marks it approximate),"thighs":-100..100=0,"calves":-100..100=0,"shoulders":-100..100=0,"hips":-100..100=0,"bust":-100..100=0,"headSize":-100..100=0,"neck":-100..100=0,"height":-100..100=0,"legLength":-100..100=0,"waistLength":-100..100=0,"torsoLength":-100..100=0,"neckLength":-100..100=0,"aiSculpt":-100..100=0 (composition: a gentle whole-body slim layered over the shaping sliders),"swanNeck":-100..100=0 (composition: slim + lengthen + square the shoulders),"straightShoulders":-100..100=0,"bustLift":-100..100=0,"doubleChin":-100..100=0,"waistLine":-500..500=0 (placement: moves the fitted waist row by 30% of the shoulder→hip-joint span per 100, up to 150% either way at ±500 — past the hip joints into the thighs, positive downward; 0 leaves the measured or prior row. Set it on a visible waistband the outline's crease test missed; the row's own silhouette width follows),"bellyLine":-500..500=0 (placement: the same for the belly row, so a front-on belly the side outline cannot confirm still gets a line the user trusts),"symmetry":0..100=0 (对称液化·上半身: the paired joints align toward each other's mirror about the torso axis),"smooth":0..100=0 (身型流畅: the silhouette contour, low-passed and relaxed — bumps like a trapezius slope or an arm bulge flatten to their base; the contour is embedded from the person mask at apply time, like the anchors) (each slider's positive direction is its label's action — Waist narrows, Hips rounds, Leg Length lengthens; with `anchors` the analysis never runs on Apply),"anchors":{…}? (`portrait.analyze`'s document-pixel anchors — the panel supplies them so Apply is a field build and a warp; without them one lightweight pose analysis runs. Stored with the smart filter so re-renders replay exactly; never typed),"fit":{…}? (the body fit measured off the person mask — the fitted rows the slimming sliders and the placement lines act on; stored beside the anchors so re-renders replay exactly; never typed),"contour":[[x,y]…] (身型流畅's person contour as document-pixel points, embedded at apply time like the anchors; never typed),"meshSize":px? (field resolution, px per node),"sampleAllLayers":bool=false,"layer":id?} — a selection limits the effect; on a smart object it becomes a smart filter"##;
    vec![CommandSpec {
        id: SCULPT,
        label: "Body Sculpt…",
        menu: &["Filter"],
        shortcut: None,
        params: P,
        enabled: crate::filters::has_filterable_layer,
        journal: true,
        run: sculpt,
    }]
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::Session;
    use crate::color_cmds::ColorExt;

    /// A checkerboard the warp's resampling visibly changes (a flat fill would not).
    fn patterned(w: u32, h: u32) -> Session {
        let mut s = crate::session();
        s.execute("file.new", json!({"width": w, "height": h})).unwrap();
        s.edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let mut data = Vec::with_capacity((w * h) as usize * n);
            for y in 0..h {
                for x in 0..w {
                    let t = ((x + y) % 2) as f32;
                    data.extend_from_slice(&[t, 1.0 - t, 0.5][..n.min(3)]);
                    if n == 4 {
                        data.push(1.0);
                    }
                }
            }
            bg.write_region(Rect::new(0, 0, w as i32, h as i32), &data);
            Ok(())
        })
        .unwrap();
        s
    }

    /// A full 33-joint pose on a `w × h` document: shoulders, hips and ankles bracketing the
    /// torso the shaping sliders move.
    fn anchors_on(w: f32, h: f32) -> SculptAnchors {
        let mut pose = vec![[0.0f32; 3]; photocraft_algo::sculpt::POSE_COUNT];
        let (sx, sy) = (w / 512.0, h / 768.0);
        for &(i, x, y) in
            &[(0usize, 256.0, 92.0), (11, 204.0, 180.0), (12, 308.0, 180.0), (23, 222.0, 400.0), (24, 292.0, 400.0), (27, 230.0, 680.0), (28, 284.0, 680.0)]
        {
            pose[i] = [x * sx, y * sy, 1.0];
        }
        SculptAnchors { doc: [w, h], pose, head: SculptHead { centre: [w * 0.5, 88.0 * sy], radius: 52.0 * sx }, chin: [w * 0.5, 138.0 * sy] }
    }

    fn pixels(s: &Session, w: i32, h: i32) -> Vec<f32> {
        s.active().unwrap().doc.layers[0].surface().unwrap().read_region(Rect::new(0, 0, w, h))
    }

    /// A synthetic person mask on a 256×384 canvas: head, a torso that narrows at the waist
    /// and widens at the hips, two arms and two legs, each its own run. The anchors below sit
    /// on it, so [`body_fit`] has a known silhouette to measure.
    fn body_mask() -> Surface {
        let mut m = Surface::new(photocraft_color::PixelFormat::GRAY8);
        m.fill_rect(Rect::new(0, 0, 256, 384), &[0.0]);
        for (x0, x1, y0, y1) in [
            (110, 146, 40, 85),   // head
            (90, 166, 90, 110),   // shoulders
            (96, 160, 110, 150),  // chest
            (104, 152, 150, 180), // waist
            (94, 162, 180, 210),  // hips
            (104, 122, 210, 345), // left leg
            (134, 152, 210, 345), // right leg
            (72, 88, 90, 170),    // left arm
            (168, 184, 90, 170),  // right arm
        ] {
            m.fill_rect(Rect::new(x0, y0, x1, y1), &[1.0]);
        }
        m
    }

    fn body_anchors() -> SculptAnchors {
        let mut pose = vec![[0.0f32; 3]; photocraft_algo::sculpt::POSE_COUNT];
        for &(i, x, y) in &[
            (0usize, 10.0, 10.0), // the nose: off the mask, as a stray pose point
            (11, 80.0, 95.0),
            (12, 176.0, 95.0),
            (13, 80.0, 130.0),
            (14, 176.0, 130.0),
            (15, 80.0, 165.0),
            (16, 176.0, 165.0),
            (23, 112.0, 200.0),
            (24, 144.0, 200.0),
            (25, 114.0, 265.0),
            (26, 142.0, 265.0),
            (27, 115.0, 340.0),
            (28, 142.0, 340.0),
        ] {
            pose[i] = [x as f32, y as f32, 1.0];
        }
        SculptAnchors { doc: [256.0, 384.0], pose, head: SculptHead { centre: [128.0, 62.0], radius: 22.0 }, chin: [128.0, 86.0] }
    }

    #[test]
    fn body_fit_measures_the_mask_silhouette() {
        let fit = body_fit(&body_anchors(), &body_mask()).expect("the synthetic person fits");
        assert!(fit.is_valid(), "{fit:?}");
        // The torso levels land in order on the mask's own narrow and wide rows.
        assert!(fit.bust.t < fit.waist.t && fit.waist.t < fit.belly.t && fit.belly.t < fit.hip.t, "{fit:?}");
        assert!((fit.waist.t - 0.75).abs() < 0.15, "the waist row is the mask's narrowest: {:.2}", fit.waist.t);
        assert!((fit.waist.left - 24.0).abs() < 3.0, "the waist half-width is the mask's: {:.1}", fit.waist.left);
        assert!((fit.hip.left - 34.0).abs() < 3.0, "the hip half-width is the mask's: {:.1}", fit.hip.left);
        assert!((fit.hip.t - 1.0).abs() < 0.1, "the hips sit on the hip joints: {:.2}", fit.hip.t);
        // Each limb's radius is its drawn half-thickness, not its bone length.
        for (idx, want) in [(0usize, 8.0f32), (2, 8.0), (4, 9.0), (6, 8.0)] {
            let r = fit.limbs[idx].radius;
            assert!((r - want).abs() < 2.5, "limb {idx} radius {r:.1}, drew {want:.1}");
        }
        // Joints on the body are inside; a stray pose point off it is not.
        assert!(fit.inside[11] && fit.inside[23] && fit.inside[27], "{:?}", fit.inside);
        assert!(!fit.inside[0], "the nose off the mask must read outside");
        assert!(fit.rows.iter().any(|r| r.runs.len() >= 2), "the arms are their own runs");
    }

    /// The same person in a dress that narrows all the way to its hem: the silhouette has no
    /// waist crease, only a taper that ends in the hips.
    fn tapered_mask() -> Surface {
        let mut m = Surface::new(photocraft_color::PixelFormat::GRAY8);
        m.fill_rect(Rect::new(0, 0, 256, 384), &[0.0]);
        for (x0, x1, y0, y1) in [
            (110, 146, 40, 85),   // head
            (90, 166, 90, 110),   // shoulders
            (96, 160, 110, 150),  // chest
            (101, 155, 150, 180), // waist: narrower than the chest, but not a crease
            (104, 152, 180, 210), // hips: the narrowest row, where the skirt starts
            (104, 122, 210, 345), // left leg
            (134, 152, 210, 345), // right leg
            (72, 88, 90, 170),    // left arm
            (168, 184, 90, 170),  // right arm
        ] {
            m.fill_rect(Rect::new(x0, y0, x1, y1), &[1.0]);
        }
        m
    }

    /// The same person in a jacket whose mask shows no waist crease: the torso's own run
    /// widens down into the hips (the sleeves hang beside it as separate runs). The pose's
    /// hip joints are then the only visible hip line and the waistband sits at or just above
    /// them; the old fixed middle put the slider on the ribs.
    fn widening_mask() -> Surface {
        let mut m = Surface::new(photocraft_color::PixelFormat::GRAY8);
        m.fill_rect(Rect::new(0, 0, 256, 384), &[0.0]);
        for (x0, x1, y0, y1) in [
            (110, 146, 40, 85),   // head
            (64, 86, 90, 130),    // left sleeve, hanging wide of the torso
            (170, 192, 90, 130),  // right sleeve
            (66, 88, 130, 170),   // left sleeve, closer in
            (168, 190, 130, 170), // right sleeve
            (90, 166, 90, 110),   // shoulders
            (94, 162, 110, 150),  // chest
            (92, 164, 150, 180),  // waist: a little wider than the chest
            (88, 168, 180, 210),  // hips: the widest torso row
            (104, 122, 210, 345), // left leg
            (134, 152, 210, 345), // right leg
        ] {
            m.fill_rect(Rect::new(x0, y0, x1, y1), &[1.0]);
        }
        m
    }

    /// A silhouette that widens into the hip row has no crease to measure. It could be a fuller
    /// body, a pregnancy or a low-slung garment; on the first two the row it widens into is the
    /// belly, not a waist, so the hip row is not a safe stand-in. The anatomical waist is used
    /// and flagged, and the belly takes the widest lower-abdomen row — a monotonic widening has
    /// no bump to confirm, so that row is flagged too.
    #[test]
    fn a_silhouette_that_widens_into_the_hips_uses_the_anatomical_waist_and_flags_it() {
        let fit = body_fit(&body_anchors(), &widening_mask()).expect("the synthetic person fits");
        assert!(fit.is_valid(), "{fit:?}");
        assert!(fit.waist_prior, "the waist fell back to the anatomical prior, not a measured crease");
        assert!((0.62..=0.78).contains(&fit.waist.t), "the waist sits at the anatomical waist, not the hip line: {:.2}", fit.waist.t);
        assert!(fit.hip.t - fit.waist.t > 0.15, "waist {:.2} vs hip {:.2}", fit.waist.t, fit.hip.t);
        assert!(fit.belly_prior, "a monotonic widening gives no bump to confirm the belly");
        assert!(fit.belly.t > fit.waist.t + 0.08 && fit.belly.t < fit.hip.t, "the belly sits in the lower abdomen: {:.2}", fit.belly.t);
    }

    /// A dress that only ever tapers must not drag the waist down to its hem: the waist row
    /// stays well above the narrowest low row instead of following it into the hips.
    #[test]
    fn a_silhouette_that_only_tapers_keeps_the_waist_off_the_hem() {
        let fit = body_fit(&body_anchors(), &tapered_mask()).expect("the synthetic person fits");
        // The hips band begins where the mask is at its narrowest, at t = (180 - 95) / 105.
        let hem = (180.0 - 95.0) / (200.0 - 95.0);
        assert!(fit.waist.t < hem - 0.05, "the waist {:.2} followed the taper into the hem ({hem:.2})", fit.waist.t);
        assert!(fit.waist_prior && fit.belly_prior, "a taper confirms neither a waist nor a belly");
        assert!(fit.bust.t < fit.waist.t && fit.waist.t < fit.belly.t && fit.belly.t < fit.hip.t, "{fit:?}");
    }

    /// The same person with a lower-abdomen bump that the outline does confirm: the profile
    /// narrows again below the bump before the hip row, so the apex is measurably wider than
    /// both ends of the belly range.
    fn belied_mask() -> Surface {
        let mut m = Surface::new(photocraft_color::PixelFormat::GRAY8);
        m.fill_rect(Rect::new(0, 0, 256, 384), &[0.0]);
        for (x0, x1, y0, y1) in [
            (110, 146, 40, 85),   // head
            (90, 166, 90, 110),   // shoulders
            (94, 162, 110, 150),  // chest
            (99, 157, 150, 170),  // waist crease
            (97, 159, 170, 180),  // upper belly
            (94, 162, 180, 195),  // belly apex: wider than both ends of the lower abdomen
            (100, 156, 195, 210), // lower belly, narrowing again into the hip row
            (104, 122, 210, 345), // left leg
            (134, 152, 210, 345), // right leg
            (72, 88, 90, 170),    // left arm
            (168, 184, 90, 170),  // right arm
        ] {
            m.fill_rect(Rect::new(x0, y0, x1, y1), &[1.0]);
        }
        m
    }

    #[test]
    fn a_belly_the_outline_confirms_is_measured_at_its_apex() {
        let fit = body_fit(&body_anchors(), &belied_mask()).expect("the synthetic person fits");
        assert!(!fit.belly_prior, "the bump is wider than both ends of its range: it is measured");
        assert!((0.80..=0.93).contains(&fit.belly.t), "the belly row is the lower-abdomen apex: {:.2}", fit.belly.t);
        assert!(fit.belly.t > fit.waist.t, "belly {:.2} below waist {:.2}", fit.belly.t, fit.waist.t);
        assert!(fit.belly.left > fit.waist.left + 2.0, "the apex is wider than the crease: {:.1} vs {:.1}", fit.belly.left, fit.waist.left);
        assert!(!fit.waist_prior, "the crease is measured, not a prior");
    }

    /// The panel's line control moves a fitted row onto the row the user names, and the moved
    /// row keeps the silhouette's own widths; a line placed by eye is a measurement, so its
    /// prior flag clears. A zero line is the Reset: the measured fit is untouched.
    #[test]
    fn a_placement_line_moves_the_row_and_reads_the_width_there() {
        let anchors = body_anchors();
        let fit = body_fit(&anchors, &widening_mask()).expect("the synthetic person fits");
        let before = fit.waist;
        let mut moved = fit.clone();
        assert!(apply_lines(&mut moved, &anchors, (0.0, 0.0)).is_empty());
        assert_eq!(moved, fit, "zero is auto: nothing moves");
        let notes = apply_lines(&mut moved, &anchors, (0.30, -0.30));
        assert!(notes.is_empty(), "{notes:?}");
        assert!(moved.waist.t > before.t + 0.15, "waist line moved down: {:.2} -> {:.2}", before.t, moved.waist.t);
        assert!(!moved.waist_prior, "a placed waist is not the prior any more");
        assert!(moved.waist.left + moved.waist.right > before.left + before.right + 2.0, "the widths follow the row it landed on");
        assert!(moved.belly.t < fit.belly.t - 0.10, "belly line moved up: {:.2} -> {:.2}", fit.belly.t, moved.belly.t);
        assert!(!moved.belly_prior, "a placed belly is not the prior any more");
    }

    /// The stop is wide enough to cross the whole torso: the belly line at the ±300 stop
    /// reaches rows a ±100 stop could not, so a waistband far from the anatomical row is
    /// still placeable. The line lands on the nearest measurable row, not the exact target.
    #[test]
    fn a_placement_line_reaches_further_than_the_shaping_sliders() {
        let anchors = body_anchors();
        let fit = body_fit(&anchors, &widening_mask()).expect("the synthetic person fits");
        let moved = |value: i32| {
            let mut f = fit.clone();
            assert!(apply_lines(&mut f, &anchors, line_offsets(&json!({"bellyLine": value})).unwrap()).is_empty());
            f.belly.t
        };
        let near = moved(-100);
        let far = moved(-300);
        assert!(far < near - 0.2, "the wider stop reaches rows the old one could not: {near:.2} -> {far:.2}");
        assert!(far < fit.waist.t, "the far line is above the waist row: {far:.2} vs {:.2}", fit.waist.t);
    }

    /// The same person in a pencil skirt: the legs read as one run down to the hem, so the
    /// silhouette's measurable rows continue below the hip joints — where a low waistband or
    /// a fuller belly actually sits.
    fn skirted_mask() -> Surface {
        let mut m = Surface::new(photocraft_color::PixelFormat::GRAY8);
        m.fill_rect(Rect::new(0, 0, 256, 384), &[0.0]);
        for (x0, x1, y0, y1) in [
            (110, 146, 40, 85),  // head
            (90, 166, 90, 110),  // shoulders
            (94, 162, 110, 150), // chest
            (92, 164, 150, 180), // waist
            (88, 168, 180, 210), // hips
            (94, 162, 210, 345), // the skirt: one run through the legs to the hem
            (72, 88, 90, 170),   // left arm
            (168, 184, 90, 170), // right arm
        ] {
            m.fill_rect(Rect::new(x0, y0, x1, y1), &[1.0]);
        }
        m
    }

    /// A line has to reach below the hip joints: a low-slung waistband or a fuller belly sits
    /// past them, and the old stop both refused the offset beyond ±300 and clamped the row
    /// at t = 1.20 — the line stuck above where the user drew it. The wider stop admits the
    /// offset and the row lands in the skirt.
    #[test]
    fn a_placement_line_reaches_below_the_hip_joints() {
        let anchors = body_anchors();
        let fit = body_fit(&anchors, &skirted_mask()).expect("the synthetic person fits");
        let (waist, belly) = line_offsets(&json!({"waistLine": 500, "bellyLine": -500})).unwrap();
        assert!((waist - 1.50).abs() < 1e-9 && (belly + 1.50).abs() < 1e-9, "±500 is ±150% of the span: {waist}, {belly}");
        let mut moved = fit.clone();
        assert!(apply_lines(&mut moved, &anchors, line_offsets(&json!({"bellyLine": 300})).unwrap()).is_empty());
        assert!(moved.belly.t > 1.25, "the line lands below the hip joints, past the old 1.20 clamp: {:.2}", moved.belly.t);
        assert!(!moved.belly_prior, "a placed belly is not the prior any more");
    }

    #[test]
    fn placement_lines_refuse_values_outside_the_span() {
        assert!(line_offsets(&json!({"waistLine": 501})).is_err());
        assert!(line_offsets(&json!({"bellyLine": -501})).is_err());
        assert!(line_offsets(&json!({"waistLine": "low"})).is_err());
        assert_eq!(line_offsets(&json!({"waistLine": 100, "bellyLine": -100})).unwrap(), (0.30, -0.30));
        let (waist, belly) = line_offsets(&json!({"waistLine": 500, "bellyLine": -500})).unwrap();
        assert!((waist - 1.50).abs() < 1e-9 && (belly + 1.50).abs() < 1e-9, "±500 is ±150% of the span: {waist}, {belly}");
    }

    /// The panel draws the rows the field acts on: the moved waist line sits on the hip row and
    /// spans the fitted hip width, and the untouched belly line sits on the measured row.
    #[test]
    fn the_drawn_lines_are_the_rows_the_field_uses() {
        let anchors = body_anchors();
        let fit = body_fit(&anchors, &widening_mask()).expect("the synthetic person fits");
        let auto_belly_y = 95.0 + f64::from(fit.belly.t) * (200.0 - 95.0);
        let params = json!({"anchors": anchors, "fit": fit, "waistLine": 100});
        let [waist, belly] = placement_lines(&params).expect("the fit places both lines");
        assert!((waist.0 - 200.0).abs() < 6.0, "the waist line follows the moved row: {:.1}", waist.0);
        assert!((74.0..=86.0).contains(&(waist.2 - waist.1)), "the span is the hip row's width: {:.1}", waist.2 - waist.1);
        assert!((belly.0 - auto_belly_y).abs() < 0.5, "the belly line is on its measured row: {:.1} vs {auto_belly_y:.1}", belly.0);
        assert!(belly.1 < belly.2, "the belly line spans the silhouette");
    }

    /// A line without a fit has nowhere to land. The field still runs from the anatomical
    /// fallback, but the reply must say so rather than imply the user's line was honoured.
    #[test]
    fn a_line_without_a_fit_is_reported_not_silently_ignored() {
        let params = json!({"anchors": body_anchors(), "waist": 50, "waistLine": 40});
        let (_, report) = field_for(&params, Rect::new(0, 0, 256, 384)).expect("the field builds");
        assert!(
            report.approximate.iter().any(|(part, why)| *part == "waist" && why.contains("no measured body fit")),
            "the unplaced line is named approximate: {report:?}"
        );
    }

    /// A stored smart filter replays its params with no session and no analysis. The moved fit
    /// is stored, so the line offsets must not be kept as well: replaying both moved the rows a
    /// second time on every re-render.
    #[test]
    fn a_placement_line_is_baked_into_the_stored_fit_and_not_kept() {
        let anchors = body_anchors();
        let mut fit = body_fit(&anchors, &widening_mask()).expect("the synthetic person fits");
        let auto = fit.clone();
        apply_lines(&mut fit, &anchors, line_offsets(&json!({"waistLine": 100})).unwrap());
        let params = json!({"waist": 50.0, "waistLine": 100, "anchors": serde_json::to_value(&anchors).unwrap(), "fit": auto});
        let stored = stored_params(&params, &anchors, Some(&fit), None).expect("the params serialise");
        assert!(stored.get("waistLine").is_none(), "the offset is baked, not replayed: {stored}");
        assert_eq!(stored["fit"]["waist"], serde_json::to_value(fit.waist).unwrap(), "the moved row is what is stored");
        // The stored form rebuilds the same field as the params that produced it.
        let bounds = Rect::new(0, 0, 256, 384);
        let (direct, _) = field_for(&params, bounds).expect("the run's field");
        let (replay, _) = field_for(&stored, bounds).expect("the replayed field");
        assert_eq!(direct, replay, "a smart filter re-renders the committed warp");
    }

    /// The line reaches the field the panel previews and Apply commits, and the prior's
    /// approximate note goes away with it: a placed row must not still be reported as a guess.
    #[test]
    fn a_placement_line_reaches_the_field_and_clears_the_prior_report() {
        let anchors = body_anchors();
        let fit = body_fit(&anchors, &widening_mask()).expect("the synthetic person fits");
        let canvas = Rect::new(0, 0, 256, 384);
        let auto = json!({"anchors": anchors, "fit": fit, "waist": 50});
        let (before, report) = field_for(&auto, canvas).expect("the auto field builds");
        assert!(report.approximate.iter().any(|(part, why)| *part == "waist" && why.contains("anatomical waist")), "{report:?}");
        let mut placed = auto.clone();
        placed["waistLine"] = json!(100);
        let (after, report) = field_for(&placed, canvas).expect("the placed field builds");
        assert!(!report.approximate.iter().any(|(part, _)| *part == "waist"), "the placed waist is not approximate: {report:?}");
        // The field is an inverse map, so a contraction at the left rim pushes the sample
        // outward (negative x). The row moved down onto the hip line: that rim's outward pull
        // grows, and the old waist rim's pull shrinks.
        let (hip_rim, old_rim) = ([88.0, 200.0], [92.0, 168.5]);
        let (before_hip, after_hip) = (before.sample(hip_rim[0], hip_rim[1])[0], after.sample(hip_rim[0], hip_rim[1])[0]);
        assert!(after_hip < before_hip - 1.0, "the pull follows the line: {before_hip} -> {after_hip}");
        let (before_old, after_old) = (before.sample(old_rim[0], old_rim[1])[0], after.sample(old_rim[0], old_rim[1])[0]);
        assert!(after_old > before_old, "the old row's pull weakens: {before_old} -> {after_old}");
    }

    #[test]
    fn body_fit_refuses_a_blank_mask() {
        let mut blank = Surface::new(photocraft_color::PixelFormat::GRAY8);
        blank.fill_rect(Rect::new(0, 0, 64, 64), &[0.0]);
        assert!(body_fit(&body_anchors(), &blank).is_none());
    }

    /// A matte that fuses the background into the person is one full-width run per row: its
    /// edges are the frame's, not the body's. Trusting them put the waist's envelope across
    /// the whole photo. The fit has to answer `None` so the joints stand in instead.
    #[test]
    fn body_fit_refuses_a_matte_that_fills_the_frame() {
        let mut full = Surface::new(photocraft_color::PixelFormat::GRAY8);
        full.fill_rect(Rect::new(0, 0, 256, 384), &[1.0]);
        assert!(body_fit(&body_anchors(), &full).is_none());
    }

    /// The smooth contour is the subject's own rim, not the row's outermost mask pixel: a
    /// background figure beside the subject used to join the contour on every row it shared,
    /// and the low-pass then read the jump as a silhouette bump — a circular smear at the
    /// subject's waist, drawn from a person standing ten metres behind her.
    #[test]
    fn the_contour_follows_the_subject_and_leaves_a_background_figure_out() {
        let mut m = Surface::new(photocraft_color::PixelFormat::GRAY8);
        m.fill_rect(Rect::new(0, 0, 256, 384), &[0.0]);
        m.fill_rect(Rect::new(120, 40, 200, 340), &[1.0]); // the subject
        m.fill_rect(Rect::new(20, 100, 60, 300), &[1.0]); // a disconnected background figure
        let contour = person_contour(&m, &body_anchors()).expect("the subject has a contour");
        let min_x = contour.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        assert!(min_x > 110.0, "the contour must stay on the subject, got x={min_x}");
    }

    /// The mask can also fuse the subject with the background into one run (a blurred sky or
    /// crowd touching the shoulders). The contour must stop where the fusion begins instead of
    /// following it across the frame: a row whose span jumps to the background is not the
    /// subject's outline, and the low-pass would turn the jump into a huge circular smear.
    #[test]
    fn a_mask_fused_with_the_background_stops_the_contour() {
        let mut m = Surface::new(photocraft_color::PixelFormat::GRAY8);
        m.fill_rect(Rect::new(0, 0, 256, 384), &[0.0]);
        m.fill_rect(Rect::new(120, 40, 200, 340), &[1.0]); // the subject
        m.fill_rect(Rect::new(0, 60, 250, 100), &[1.0]); // a background band touching the subject
        let contour = person_contour(&m, &body_anchors()).expect("the subject has a contour");
        let min_x = contour.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        assert!(min_x > 110.0, "the contour followed the fused band into the background, got x={min_x}");
        let max_x = contour.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
        assert!(max_x < 210.0, "the contour followed the fused band into the background, got x={max_x}");
    }

    /// 对称液化 at 100 is a touch of asymmetry, not a pose transplant: a wrist raised to the
    /// head must not be dragged down to the other wrist's mirror through a disc of the photo.
    /// The pairs the pose saw in a plausible pose still align; the report says some were left.
    #[test]
    fn the_symmetry_pass_leaves_a_raised_arm_alone() {
        let mut anchors = body_anchors();
        anchors.pose[15] = [80.0, 60.0, 1.0]; // the left wrist up by the head
        let fit = body_fit(&anchors, &body_mask()).expect("the synthetic person fits");
        let bounds = Rect::new(0, 0, 256, 384);
        let (field, report) =
            field_with(SculptSpec { sliders: SculptSliders::default(), anchors, fit: Some(fit) }, None, 100.0, 0.0, bounds, auto_cell(bounds));
        assert!(field.max_displacement() < 5.0, "the raised wrist must not teleport: {:.1}px", field.max_displacement());
        assert!(report.applied.contains(&"symmetry"), "{report:?}");
        assert!(report.approximate.iter().any(|(n, _)| *n == "symmetry"), "the skipped pair must be named: {report:?}");
    }

    #[test]
    fn a_stored_fit_replays_the_same_field() {
        let anchors = body_anchors();
        let fit = body_fit(&anchors, &body_mask()).expect("the synthetic person fits");
        let bounds = Rect::new(0, 0, 256, 384);
        let params = json!({
            "waist": 100.0,
            "anchors": serde_json::to_value(&anchors).unwrap(),
            "fit": serde_json::to_value(&fit).unwrap(),
        });
        let (spec, _) = stored_spec(&params, bounds).expect("the stored form");
        assert_eq!(spec.fit.as_ref(), Some(&fit), "the fit survives the params round-trip");
        let (f1, _) = field_for(&params, bounds).expect("the field");
        let (f2, _) = sculpt_field(&spec, bounds, auto_cell(bounds));
        assert_eq!(f1, f2, "field_for and the stored replay must build the same field");
        assert!(!f1.is_identity());
    }

    #[test]
    fn explicit_anchors_and_a_fit_apply_in_one_step() {
        let mut s = patterned(256, 384);
        let before = pixels(&s, 256, 384);
        let steps = s.active().unwrap().history.past_len();
        let anchors = body_anchors();
        let fit = body_fit(&anchors, &body_mask()).expect("the synthetic person fits");
        let r =
            s.execute(SCULPT, json!({"waist": 100.0, "anchors": serde_json::to_value(&anchors).unwrap(), "fit": serde_json::to_value(&fit).unwrap()})).unwrap();
        assert_eq!(r["changed"], json!(true));
        assert_ne!(before, pixels(&s, 256, 384), "the fitted warp must move pixels");
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "one history step");
        // The engine reports the body's parts in the reply, not as a silent success.
        assert!(r["parts"]["applied"].as_array().is_some_and(|a| !a.is_empty()), "{r}");
    }

    /// The panel's fast path: the params carry `portrait.analyze`'s anchors, so Apply builds
    /// the field and warps — the fallback analysis, which would fail on this document, never
    /// runs.
    #[test]
    fn explicit_anchors_apply_without_a_person_analysis() {
        let mut s = patterned(256, 384);
        let before = pixels(&s, 256, 384);
        let steps = s.active().unwrap().history.past_len();
        let anchors = anchors_on(256.0, 384.0);
        let r = s.execute(SCULPT, json!({"waist": -60.0, "anchors": serde_json::to_value(&anchors).unwrap()})).unwrap();
        assert_eq!(r["changed"], json!(true));
        assert!(r["maxDisplacement"].as_f64().unwrap_or_default() > 0.0);
        assert_ne!(before, pixels(&s, 256, 384), "the warp must move pixels");
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "one history step");
    }

    /// Malformed anchors are refused by their shape, not warped from garbage: the field
    /// builders index every joint, so a short pose must never reach them.
    #[test]
    fn malformed_anchors_are_refused_not_analysed() {
        let mut s = patterned(64, 64);
        let steps = s.active().unwrap().history.past_len();
        let mut anchors = serde_json::to_value(anchors_on(64.0, 64.0)).unwrap();
        anchors["pose"] = serde_json::Value::Array(vec![json!([0.0, 0.0, 1.0]); 3]);
        let err = s.execute(SCULPT, json!({"waist": 40.0, "anchors": anchors})).unwrap_err().to_string();
        assert!(err.contains("33"), "{err}");
        assert_eq!(s.active().unwrap().history.past_len(), steps, "a refused run edits nothing");
    }

    #[test]
    fn zero_sliders_change_nothing_without_an_analysis() {
        let mut s = crate::session();
        s.execute("file.new", json!({"width": 96, "height": 128, "background": "white"})).unwrap();
        let r = s.execute(SCULPT, json!({})).unwrap();
        assert_eq!(r, json!({"layer": s.active().unwrap().active_layer.unwrap().0, "changed": false}));
    }

    #[test]
    fn a_personless_photo_reports_no_person() {
        let mut s = crate::session();
        s.execute("file.new", json!({"width": 96, "height": 128, "background": "white"})).unwrap();
        let err = s.execute(SCULPT, json!({"waist": 40})).unwrap_err().to_string();
        assert!(err.contains("no person"), "{err}");
        // A failed run leaves no history step and no edit.
        assert!(s.execute("edit.undo", json!({})).is_err());
    }

    #[test]
    fn sliders_clamp_and_bad_ones_are_refused() {
        let mut s = crate::session();
        s.execute("file.new", json!({"width": 32, "height": 32, "background": "white"})).unwrap();
        // A mistyped 250 is an error naming the range (the schema gate), not a silently
        // clamped warp; a non-number is refused the same way.
        let err = s.execute(SCULPT, json!({"waist": 250.0})).unwrap_err().to_string();
        assert!(err.contains("-100..100"), "{err}");
        assert!(s.execute(SCULPT, json!({"waist": "slim"})).is_err());
    }

    #[test]
    fn stored_params_replay_the_same_field() {
        // A stored filter's params, anchors included, build one field — the smart filter's
        // re-render is a pure function of them, with no session or analysis to ask.
        let mut pose = vec![[0.0f32; 3]; photocraft_algo::sculpt::POSE_COUNT];
        for &(i, x, y) in
            &[(0usize, 256.0, 92.0), (11, 204.0, 180.0), (12, 308.0, 180.0), (23, 222.0, 400.0), (24, 292.0, 400.0), (27, 230.0, 680.0), (28, 284.0, 680.0)]
        {
            pose[i] = [x as f32, y as f32, 1.0];
        }
        let anchors = SculptAnchors { doc: [512.0, 768.0], pose, head: SculptHead { centre: [256.0, 88.0], radius: 52.0 }, chin: [256.0, 138.0] };
        let params = json!({"waist": -40.0, "legLength": 50.0, "anchors": serde_json::to_value(&anchors).unwrap(), "meshSize": 4});
        let bounds = Rect::new(0, 0, 512, 768);
        let (spec, _) = stored_spec(&params, bounds).unwrap();
        assert_eq!(spec.sliders.waist, -40.0);
        assert_eq!(spec.anchors, anchors, "the anchors survive the params round-trip");
        let f1 = sculpt_field(&spec, bounds, 4.0).0;
        let f2 = sculpt_field(&stored_spec(&params, bounds).unwrap().0, bounds, 4.0).0;
        assert!(!f1.is_identity());
        assert_eq!(f1, f2, "the same params must rebuild the same field");
        // The live preview and Apply share this builder, so all three must agree.
        assert_eq!(f1, field_for(&params, bounds).unwrap().0, "field_for and the stored replay must build the same field");
        // Without anchors it is not this command's stored form.
        assert!(stored_spec(&json!({"waist": -40.0}), bounds).is_none());
    }

    /// The mask row's covered extent, `(first x, last x)` above half coverage — `None` when
    /// the row reads full-width or nearly empty: a subject matte that cannot separate this
    /// photo's background (the u2netp tier's known wall-on-white failure; the ort tier
    /// exists for it) has no rim to measure, and the gauge says so instead of lying.
    fn row_rim(mask: &Surface, y: i32, w: i32) -> Option<(i32, i32)> {
        let mut first: Option<i32> = None;
        let mut last = 0i32;
        for x in 0..w {
            if mask.sample_channel(x, y, 0) > 0.5 {
                first.get_or_insert(x);
                last = x;
            }
        }
        let l = first?;
        ((last - l) >= w / 10 && (last - l) < w * 9 / 10).then_some((l, last))
    }

    /// The covered run (above half coverage) that holds `xc` in a mask row, when that run is
    /// neither full-width nor nearly empty. The head row can be several runs — the subject's
    /// hands sit beside it — and the gauge wants the run the head centre is in.
    fn row_run(mask: &Surface, y: i32, xc: i32, w: i32) -> Option<(i32, i32)> {
        let mut start: Option<i32> = None;
        for x in 0..w {
            if mask.sample_channel(x, y, 0) > 0.5 {
                start.get_or_insert(x);
            } else if let Some(s) = start.take()
                && s <= xc
                && xc < x
            {
                return ((x - s) >= w / 10 && (x - s) < w * 9 / 10).then_some((s, x - 1));
            }
        }
        start.and_then(|s| ((w - s) >= w / 10 && (w - s) < w * 9 / 10).then_some((s, w - 1)))
    }

    /// The narrowing between two rims, when both rows have one.
    fn narrowing(before: Option<(i32, i32)>, after: Option<(i32, i32)>) -> Option<i32> {
        Some(after?.1 - after?.0 - (before?.1 - before?.0))
    }

    /// The fraction of a horizontal band whose pixels moved by more than a visible step
    /// between two surfaces — the gauge that needs no mask: the warp's own footprint.
    fn band_changed(before: &Surface, after: &Surface, y0: i32, y1: i32, w: i32) -> f64 {
        let region = Rect::new(0, y0, w, y1.max(y0 + 1));
        let (a, b) = (before.read_region(region), after.read_region(region));
        let n = before.format().channels().min(3);
        let step = 4.0 / 255.0;
        let ch = a.chunks_exact(n).zip(b.chunks_exact(n)).filter(|(x, y)| (0..n).any(|c| (x[c] - y[c]).abs() > step)).count();
        ch as f64 / a.len() as f64 * n as f64
    }

    /// The real-photo acceptance and calibration instrument: `PHOTOCRAFT_SCULPT_PHOTO=<image>`
    /// imports it, analyses the person exactly as the command does and prints the per-part
    /// table for every slider alone at +100 (applied or skipped and why, the field's reach) —
    /// then measures what a warp does to the Person mask's own silhouette: the waist row
    /// narrows with the waist slider and roughly in proportion, the head row narrows with
    /// head size, and a smart-object run stores the anchors and replays bit-identically
    /// through the real re-render path. The flattened result exports for eyeballing
    /// (`PHOTOCRAFT_SCULPT_OUT`, default `/tmp/sculpt_out.png`). Skipped without the env
    /// var, like the portrait acceptance: CI has neither the weights nor the photos.
    #[test]
    fn real_photo_sculpt_diagnostic() {
        let Some(photo) = std::env::var_os("PHOTOCRAFT_SCULPT_PHOTO") else {
            eprintln!("skipping: PHOTOCRAFT_SCULPT_PHOTO unset");
            return;
        };
        let run = || -> Result<()> {
            let mut s = crate::session();
            let name = std::path::Path::new(&photo).file_name().and_then(|n| n.to_str()).unwrap_or("photo").to_string();
            let imported =
                photocraft_io::import_file(&name, std::path::Path::new(&photo), &photocraft_io::ImportOptions::default(), &photocraft_raster::Interrupt::NONE)
                    .map_err(|e| EngineError::Other(format!("import: {e}")))?;
            s.open_document(imported.document, None);
            let (canvas, analysis, photo) = {
                let d = s.active().ok_or(EngineError::NoDocument)?;
                let target = d.active_layer.filter(|l| d.doc.layer(*l).and_then(photocraft_doc::Layer::surface).is_some());
                let ctx = crate::jobs::JobCtx::new();
                let analysis = crate::portrait_cmds::analyze_person(&d.doc, target, false, None, &ctx)?;
                let photo =
                    target.and_then(|l| d.doc.layer(l)).and_then(|l| l.surface()).cloned().ok_or_else(|| EngineError::Other("no pixel layer".into()))?;
                (d.doc.bounds(), analysis, photo)
            };
            let anchors = anchors_from_analysis(&analysis).ok_or_else(|| EngineError::Other("no pose landmarks in the photo".into()))?;
            let person = analysis.mask(photocraft_portrait::Region::Person).ok_or_else(|| EngineError::Other("no person mask".into()))?.clone();
            let fit = body_fit(&anchors, &person);
            if let Some(f) = &fit {
                eprintln!(
                    "fit: waist t={:.2} l/r={:.0}/{:.0}, belly t={:.2}, hip t={:.2}, bust t={:.2}",
                    f.waist.t, f.waist.left, f.waist.right, f.belly.t, f.hip.t, f.bust.t
                );
                if std::env::var_os("PHOTOCRAFT_BODY_FIT_DUMP").is_some() {
                    let vis: Vec<String> = (11..=28).map(|i| format!("{i}:{:.2}/{}", anchors.pose[i][2], if f.inside[i] { "in" } else { "out" })).collect();
                    eprintln!("  pose visibility (joint:score/silhouette): {}", vis.join(" "));
                    for (idx, limb) in f.limbs.iter().enumerate() {
                        let (i, j) = LIMB_SEGMENTS[idx];
                        eprintln!("  limb {i}->{j}: radius {:.0}, covered t {:.2}..{:.2}", limb.radius, limb.t0, limb.t1);
                    }
                    let mid = |a: usize, b: usize| [(anchors.pose[a][0] + anchors.pose[b][0]) * 0.5, (anchors.pose[a][1] + anchors.pose[b][1]) * 0.5];
                    let (sh, hp) = (mid(11, 12), mid(23, 24));
                    for row in &f.rows {
                        let t = (row.y - sh[1]) / (hp[1] - sh[1]);
                        if !(-0.10..=1.35).contains(&t) {
                            continue;
                        }
                        let ax = sh[0] + (row.y - sh[1]) * (hp[0] - sh[0]) / (hp[1] - sh[1]);
                        if let Some(run) = row.runs.iter().find(|r| ax >= r[0] && ax <= r[1]) {
                            eprintln!("  y={:.0} t={t:.2} l={:.0} r={:.0}", row.y, ax - run[0], run[1] - ax);
                        } else {
                            eprintln!("  y={:.0} t={t:.2} (axis not in a run: {:?})", row.y, row.runs);
                        }
                    }
                }
            } else {
                eprintln!("fit: none (the mask has no measurable silhouette)");
            }
            let cell = auto_cell(canvas);

            // The calibration table: every slider alone at +100.
            type Single = (&'static str, fn(f32) -> SculptSliders);
            let singles: Vec<Single> = vec![
                ("slimUpper", |v| SculptSliders { slim_upper: v, ..SculptSliders::default() }),
                ("slimFull", |v| SculptSliders { slim_full: v, ..SculptSliders::default() }),
                ("waist", |v| SculptSliders { waist: v, ..SculptSliders::default() }),
                ("belly", |v| SculptSliders { belly: v, ..SculptSliders::default() }),
                ("arms", |v| SculptSliders { arms: v, ..SculptSliders::default() }),
                ("forearms", |v| SculptSliders { forearms: v, ..SculptSliders::default() }),
                ("thighs", |v| SculptSliders { thighs: v, ..SculptSliders::default() }),
                ("calves", |v| SculptSliders { calves: v, ..SculptSliders::default() }),
                ("shoulders", |v| SculptSliders { shoulders: v, ..SculptSliders::default() }),
                ("hips", |v| SculptSliders { hips: v, ..SculptSliders::default() }),
                ("bust", |v| SculptSliders { bust: v, ..SculptSliders::default() }),
                ("headSize", |v| SculptSliders { head_size: v, ..SculptSliders::default() }),
                ("neck", |v| SculptSliders { neck: v, ..SculptSliders::default() }),
                ("height", |v| SculptSliders { height: v, ..SculptSliders::default() }),
                ("legLength", |v| SculptSliders { leg_length: v, ..SculptSliders::default() }),
                ("waistLength", |v| SculptSliders { waist_length: v, ..SculptSliders::default() }),
                ("torsoLength", |v| SculptSliders { torso_length: v, ..SculptSliders::default() }),
                ("neckLength", |v| SculptSliders { neck_length: v, ..SculptSliders::default() }),
                ("aiSculpt", |v| SculptSliders { ai_sculpt: v, ..SculptSliders::default() }),
                ("straightShoulders", |v| SculptSliders { straight_shoulders: v, ..SculptSliders::default() }),
                ("swanNeck", |v| SculptSliders { swan_neck: v, ..SculptSliders::default() }),
                ("bustLift", |v| SculptSliders { bust_lift: v, ..SculptSliders::default() }),
                ("doubleChin", |v| SculptSliders { double_chin: v, ..SculptSliders::default() }),
            ];
            eprintln!("slider        rows applied/skipped, max displacement (px)");
            for (name, make) in &singles {
                let (field, report) = sculpt_field(&SculptSpec { sliders: make(100.0), anchors: anchors.clone(), fit: fit.clone() }, canvas, cell);
                let skipped = report.skipped.iter().map(|(n, w)| format!("{n}: {w}")).collect::<Vec<_>>().join("; ");
                eprintln!("{name:12}  {:>2} applied, {skipped}, max {:.1}px", report.applied.len(), field.max_displacement());
            }

            // The silhouette gauge: the waist row narrows with the slider, in proportion;
            // the head row answers head size. Both only where the mask has a rim to read.
            // The waist row is the field's own measured level, not a fixed fraction: the
            // gauge must read where the curve actually sits.
            let mid_y = |a: usize, b: usize| ((anchors.pose[a][1] + anchors.pose[b][1]) * 0.5) as i32;
            let (sh_y, hip_y) = (mid_y(11, 12) as f32, mid_y(23, 24) as f32);
            let at = |t: f32| (sh_y + (hip_y - sh_y) * t) as i32;
            // With no fit the field's own fallback row is the anatomical waist (0.70) with the
            // belly just below it, so the gauges read where the field acts rather than at the
            // anatomical middle.
            let waist_y = fit.as_ref().map_or(at(0.70), |f| at(f.waist.t));
            let waist_band = fit.as_ref().map_or((at(0.44), at(0.96)), |f| (at(f.waist.t - 0.085), at(f.waist.t + 0.085)));
            let head_y = anchors.head.centre[1] as i32;
            let head_x = anchors.head.centre[0] as i32;
            let w = canvas.width() as i32;
            let rim = |m: &Surface, y: i32| row_rim(m, y, w);
            let waist0 = rim(&person, waist_y);
            let mut rims = vec![waist0];
            for v in [25.0f32, 50.0, 100.0] {
                let (field, _) = sculpt_field(
                    &SculptSpec { sliders: SculptSliders { waist: v, ..SculptSliders::default() }, anchors: anchors.clone(), fit: fit.clone() },
                    canvas,
                    cell,
                );
                rims.push(rim(&apply_liquify(&person, &field), waist_y));
            }
            eprintln!("waist row {waist_y}: rims at 0/25/50/100 = {rims:?}");
            if let Some(step100) = narrowing(rims[0], rims[3]).filter(|_| rims[0].is_some()) {
                assert!(step100 < 0, "waist at 100 must narrow the rim: {rims:?}");
                assert!(step100 <= -2, "waist at 100 must narrow measurably: {rims:?}");
                assert!(narrowing(rims[0], rims[1]).is_none_or(|s| s >= step100), "narrowing must grow with the slider: {rims:?}");
            } else {
                eprintln!("waist row reads without a rim — this photo's subject mask cannot gauge the waist; the field itself is covered by the algo tests");
            }
            let (head_field, _) = sculpt_field(
                &SculptSpec { sliders: SculptSliders { head_size: 100.0, ..SculptSliders::default() }, anchors: anchors.clone(), fit: fit.clone() },
                canvas,
                cell,
            );
            let run_at = |m: &Surface, y: i32| row_run(m, y, head_x, w);
            let head1 = run_at(&apply_liquify(&person, &head_field), head_y);
            let head0 = run_at(&person, head_y);
            eprintln!("head row {head_y}: {head0:?} → {head1:?}");
            if let Some(step) = narrowing(head0, head1).filter(|_| head0.is_some()) {
                assert!(step < 0, "head size must shrink the head row: {step}");
            } else {
                eprintln!("head row reads without a rim; skipping the head gauge");
            }

            // The footprint gauge, mask-free: waist at 100 moves pixels in the waist band
            // and only there — the head and legs bands must stay put.
            let (waist_field, _) = sculpt_field(
                &SculptSpec { sliders: SculptSliders { waist: 100.0, ..SculptSliders::default() }, anchors: anchors.clone(), fit: fit.clone() },
                canvas,
                cell,
            );
            let warped = apply_liquify(&photo, &waist_field);
            let (head_band, waist_band, legs_band) = (
                band_changed(&photo, &warped, head_y - anchors.head.radius as i32, head_y + anchors.head.radius as i32, w),
                band_changed(&photo, &warped, waist_band.0, waist_band.1, w),
                // Below the waist band's own reach: the fallback row sits at the anatomical
                // waist, so the legs start past the band rather than past the hip joints.
                band_changed(&photo, &warped, waist_band.1 + 40, canvas.height() as i32, w),
            );
            eprintln!("waist-100 footprint: head {head_band:.3}, waist {waist_band:.3}, legs {legs_band:.3}");
            assert!(waist_band > 0.01, "waist at 100 must move the waist band: {waist_band:.3}");
            assert!(
                head_band < waist_band / 4.0 && legs_band < waist_band / 4.0,
                "the warp must stay in its band: head {head_band:.3} legs {legs_band:.3} vs waist {waist_band:.3}"
            );

            // A realistic slimming stack, exported for the eyeball pass: the smart-filter
            // export below proves replay, this one shows what a user sees.
            if let Some(path) = std::env::var_os("PHOTOCRAFT_SCULPT_SLIM_OUT") {
                let stack = json!({"slimFull": 40, "waist": 70, "belly": 60, "arms": 60, "thighs": 60, "calves": 50, "hips": 40, "bust": 30});
                let r = s.execute(SCULPT, stack)?;
                assert_eq!(r["changed"], json!(true), "{r}");
                s.execute("layer.quickExportAsPng", json!({"path": path.to_string_lossy().into_owned()}))?;
                eprintln!("exported slimming stack: {}", path.to_string_lossy());
                s.execute("edit.undo", json!({}))?;
            }

            // A smart-object run stores the anchors and replays through the real path.
            s.execute("select.all", json!({}))?;
            s.execute("layer.smartObjects.convertToSmartObject", json!({}))?;
            let r = s.execute(SCULPT, json!({"waist": -40.0}))?;
            assert_eq!(r["changed"], json!(true), "{r}");
            let (params, source) = {
                let d = s.active().ok_or(EngineError::NoDocument)?;
                let id = d.active_layer.ok_or(EngineError::Other("no active layer".into()))?;
                let photocraft_doc::LayerContent::Smart(sm) = &d.doc.layer(id).ok_or(EngineError::NoLayer(id))?.content else { panic!("just converted") };
                assert_eq!(sm.smart_filters.len(), 1);
                (sm.smart_filters[0].params.clone(), sm.cache.clone().ok_or_else(|| EngineError::Other("no rendered source".into()))?)
            };
            assert!(params.get("anchors").is_some(), "the anchors must ride the stored params");
            let a = crate::filters::apply_filter_to_surface(SCULPT, &params, &source, source.content_bounds(), None, canvas)?
                .ok_or_else(|| EngineError::Other("the re-render path refused the stored filter".into()))?;
            let b = crate::filters::apply_filter_to_surface(SCULPT, &params, &source, source.content_bounds(), None, canvas)?
                .ok_or_else(|| EngineError::Other("the re-render path refused the stored filter".into()))?;
            assert_eq!(a.read_region(a.content_bounds()), b.read_region(b.content_bounds()), "re-render must be bit-identical");

            // Flatten and export for the eyeball pass.
            s.execute("layer.flattenImage", json!({}))?;
            let out = std::env::var_os("PHOTOCRAFT_SCULPT_OUT").map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "/tmp/sculpt_out.png".into());
            s.execute("layer.quickExportAsPng", json!({"path": out}))?;
            eprintln!("exported: {out}");
            Ok(())
        };
        if let Err(e) = run() {
            panic!("sculpt diagnostic failed: {e}");
        }
    }
}
