//! The panel's runtime: the one analysis at open, Apply with its anchors, and the live proxy
//! preview — what it shows, what it hides and where it refuses.

use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT as _, Queryable};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use super::{Live, display_doc};
use crate::PhotocraftApp;
use crate::sculpt_ui::{BODY, FACE, Panel};

/// A document whose pixels move visibly under a warp, so an unchanged preview is not just a
/// uniform image.
fn patterned(w: u32, h: u32) -> photocraft_engine::Session {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": w, "height": h})).unwrap();
    s.edit("paint", |doc, _| {
        let bg = doc.layers[0].surface_mut().unwrap();
        let n = bg.channels();
        let mut data = Vec::with_capacity((w * h) as usize * n);
        for y in 0..h {
            for x in 0..w {
                // Blocks, not single pixels: the proxy downsamples, and a one-pixel checker
                // averages to flat grey at a scale above 1 — a warp of a flat image moves nothing.
                let t = (((x / 8) + (y / 8)) % 2) as f32;
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

/// A one-pixel checker: a proxy that downsamples by two averages it to flat grey, while one
/// pixel per document pixel keeps every block.
fn fine_checker(w: u32, h: u32) -> photocraft_engine::Session {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": w, "height": h})).unwrap();
    s.edit("paint", |doc, _| {
        let bg = doc.layers[0].surface_mut().unwrap();
        let n = bg.channels();
        let mut data = Vec::with_capacity((w * h) as usize * n);
        for y in 0..h {
            for x in 0..w {
                let t = ((x + y) % 2) as f32;
                data.extend_from_slice(&[t, t, t][..n.min(3)]);
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

/// `face_anchors` on a `side`-pixel document, the 512-pixel layout scaled uniformly.
fn scaled_face_anchors(side: f32) -> Value {
    let k = f64::from(side) / 512.0;
    let mut a = face_anchors();
    if let Some(pts) = a["pts"].as_array_mut() {
        for p in pts {
            let x = p[0].as_f64().unwrap_or(256.0) * k;
            let y = p[1].as_f64().unwrap_or(240.0) * k;
            *p = json!([x, y, 0.0]);
        }
    }
    a["doc"] = json!([side, side]);
    a
}

/// The red channel's spread: how much detail an image still carries.
fn contrast(image: &egui::ColorImage) -> u8 {
    let (mut lo, mut hi) = (255u8, 0u8);
    for p in &image.pixels {
        lo = lo.min(p.r());
        hi = hi.max(p.r());
    }
    hi - lo
}

/// A blank document: no person and no face for an analysis to find.
fn blank(w: u32, h: u32) -> photocraft_engine::Session {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": w, "height": h, "background": "white"})).unwrap();
    s
}

fn app(session: photocraft_engine::Session) -> Box<PhotocraftApp> {
    PhotocraftApp::new(session, crate::Services::default())
}

/// A full 33-joint pose on a `w × h` document (the engine tests' shape): shoulders, hips and
/// ankles bracketing the torso the shaping sliders move.
fn body_anchors(w: f32, h: f32) -> Value {
    let mut pose = vec![json!([0.0, 0.0, 0.0]); 33];
    let (sx, sy) = (w / 512.0, h / 768.0);
    for &(i, x, y) in
        &[(0usize, 256.0, 92.0), (11, 204.0, 180.0), (12, 308.0, 180.0), (23, 222.0, 400.0), (24, 292.0, 400.0), (27, 230.0, 680.0), (28, 284.0, 680.0)]
    {
        pose[i] = json!([x * sx, y * sy, 1.0]);
    }
    json!({"doc": [w, h], "pose": pose, "head": {"centre": [w * 0.5, 88.0 * sy], "radius": 52.0 * sx}, "chin": [w * 0.5, 138.0 * sy]})
}

/// The 478-point mesh the face tests' sliders shape.
fn face_anchors() -> Value {
    let mut pts = vec![json!([256.0, 240.0, 0.0]); 478];
    for &(i, x, y) in &[(10usize, 256.0, 96.0), (152, 256.0, 392.0), (454, 376.0, 248.0), (234, 136.0, 248.0)] {
        pts[i] = json!([x, y, 0.0]);
    }
    json!({"doc": [512.0, 512.0], "pts": pts})
}

fn panel(command: &'static str) -> Panel {
    Panel { command, preset: "none".into(), ..Panel::default() }
}

fn harness(session: photocraft_engine::Session, size: egui::Vec2) -> Harness<'static, Box<PhotocraftApp>> {
    let app = app(session);
    let h =
        Harness::builder().with_context(app.ctx().clone()).with_size(size).build_ui_state(|ui, app: &mut Box<PhotocraftApp>| crate::dialogs::run(app, ui), app);
    PhotocraftApp::setup_context(&h.ctx, crate::theme::AppTheme::pro());
    h
}

/// A document with no person keeps the analysis's refusal inline: the panel opens with the
/// translated note, Apply disabled, no alert, and no history step.
#[test]
fn opening_a_body_panel_without_a_person_keeps_the_error_inline() {
    let mut app = app(blank(64, 64));
    let steps = app.session.active().unwrap().history.past_len();
    let mut live = Live::open(&mut app, BODY);
    assert!(live.busy(), "the analysis starts in the background, not on the open call");
    live.analysis_now(&mut app);
    assert!(!live.busy());
    let notice = live.notice_text().expect("the refusal is shown inline");
    assert!(notice.contains("No person"), "{notice}");
    assert!(!live.ready(), "Apply has no anchors to send");
    assert!(!app.ui.status_error, "the panel must not alert");
    assert_eq!(app.session.active().unwrap().history.past_len(), steps);
}

/// Face Sculpt needs the face mesh only: a document without a face reports exactly that, never
/// that no person was found — the person stage is not part of Face Sculpt.
#[test]
fn opening_a_face_panel_without_a_face_keeps_the_error_inline() {
    let mut app = app(blank(64, 64));
    let mut live = Live::open(&mut app, FACE);
    live.analysis_now(&mut app);
    let notice = live.notice_text().expect("the refusal is shown inline");
    assert!(notice.contains("No face"), "{notice}");
    assert!(!notice.contains("person"), "Face Sculpt does not need the person stage: {notice}");
    assert!(!live.ready());
    assert!(!app.ui.status_error);
}

/// Without anchors Apply is greyed out, and even a direct apply call edits nothing.
#[test]
fn apply_is_disabled_until_anchors_exist() {
    let mut h = harness(blank(64, 64), egui::vec2(700.0, 700.0));
    crate::sculpt_ui::open(h.state_mut(), BODY);
    h.run_steps(3);
    assert!(h.get_by_label("Apply").accesskit_node().is_disabled(), "Apply is greyed out without anchors");
    let steps = h.state().session.active().unwrap().history.past_len();
    let mut panel = h.state().sculpt.clone().expect("panel open");
    super::apply(h.state_mut(), &mut panel);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps, "a refused Apply edits nothing");
    assert!(h.state().sculpt.is_some(), "the panel stays open");
}

/// The panel's fast path through the UI: one click is one history step, the panel stays open,
/// and the preview is gone so the canvas shows the committed warp.
#[test]
fn apply_with_anchors_is_one_step_and_keeps_the_panel_open() {
    let mut h = harness(patterned(256, 384), egui::vec2(700.0, 900.0));
    h.state_mut().sculpt = Some(panel(BODY));
    let live = Live::test(h.state_mut(), BODY, body_anchors(256.0, 384.0));
    h.state_mut().sculpt_live = Some(live);
    h.state_mut().sculpt.as_mut().expect("panel open").values.insert("waist".into(), -60.0);
    h.run_steps(2);
    let steps = h.state().session.active().unwrap().history.past_len();
    h.get_by_label("Apply").click();
    h.run_steps(3);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1, "one history step");
    assert!(h.state().sculpt.is_some(), "Apply keeps the panel open");
    assert!(h.state().sculpt_live.as_ref().is_some_and(|live| !live.has_preview()), "the preview gives way to the commit");
}

/// Applying again without touching a slider is no second warp: the committed values are the
/// baseline, so the request is empty and the engine reports `changed: false`. Before the
/// baseline, the same press re-applied every slider over the committed pixels and dragged the
/// face out of shape.
#[test]
fn a_second_apply_without_a_change_edits_nothing() {
    let mut h = harness(patterned(256, 384), egui::vec2(700.0, 900.0));
    h.state_mut().sculpt = Some(panel(BODY));
    let live = Live::test(h.state_mut(), BODY, body_anchors(256.0, 384.0));
    h.state_mut().sculpt_live = Some(live);
    h.state_mut().sculpt.as_mut().expect("panel open").values.insert("waist".into(), -60.0);
    h.run_steps(2);
    h.get_by_label("Apply").click();
    h.run_steps(3);
    let steps = h.state().session.active().unwrap().history.past_len();
    h.get_by_label("Apply").click();
    h.run_steps(3);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps, "an unchanged second Apply is no edit");
}

/// Height and Leg Length stretch from the ankles; a pose that only guessed the feet moves the
/// photo by a few invisible pixels. The panel says so before Apply instead of leaving the
/// sliders looking dead.
#[test]
fn the_length_sliders_warn_when_the_pose_did_not_see_the_feet() {
    let mut app = app(blank(256, 384));
    let seen = Live::test(&mut app, BODY, body_anchors(256.0, 384.0));
    assert_eq!(seen.warning_text(), None, "feet in view raise no warning");
    let mut anchors = body_anchors(256.0, 384.0);
    for i in [27usize, 28] {
        anchors["pose"][i][2] = json!(0.3);
    }
    let guessed = Live::test(&mut app, BODY, anchors);
    assert!(guessed.warning_text().is_some_and(|t| t.contains("feet")), "a guessed foot is said before Apply");
}

/// An Apply whose reply leaves a region out, or handles one approximately, says so inline. A
/// clean reply leaves the status line alone.
#[test]
fn an_apply_reports_regions_it_left_out_or_approximated() {
    let skipped = json!({"parts": {"applied": ["waist"], "approximate": [], "skipped": [{"part": "right upper arm", "why": "outside"}]}});
    assert!(super::parts_notice(&skipped).is_some_and(|n| n.contains("left out")), "a skipped part is said");
    let approximate = json!({"parts": {"applied": ["thigh"], "approximate": [{"part": "left thigh", "why": "inferred"}], "skipped": []}});
    assert!(super::parts_notice(&approximate).is_some_and(|n| n.contains("approximately")), "an approximate part is said");
    let clean = json!({"parts": {"applied": ["waist"], "approximate": [], "skipped": []}});
    assert!(super::parts_notice(&clean).is_none(), "a clean Apply adds no notice");
    assert!(super::parts_notice(&json!({"changed": true})).is_none(), "a reply without parts adds none");
}

/// The measured body fit must ride with the anchors into every param set the panel sends —
/// preview and Apply share `merged_params`, so one insertion covers both. Without a fit the
/// params stay as they were (Face, or a mask with no measurable silhouette).
#[test]
fn the_body_fit_rides_with_the_anchors_into_the_params() {
    let mut app = app(patterned(64, 64));
    let mut live = Live::test(&mut app, BODY, body_anchors(64.0, 64.0));
    assert!(live.merged_params(&json!({"waist": 40.0})).get("fit").is_none(), "no fit, no fit param");
    let fit = json!({
        "rows": [{"y": 10.0, "runs": [[4.0, 20.0]]}, {"y": 30.0, "runs": [[6.0, 18.0]]}],
        "waist": {"t": 0.5, "left": 8.0, "right": 8.0},
        "belly": {"t": 0.7, "left": 9.0, "right": 9.0},
        "hip": {"t": 1.0, "left": 10.0, "right": 10.0},
        "bust": {"t": 0.3, "left": 7.0, "right": 7.0},
        "limbs": [
            {"radius": 4.0, "t0": 0.1, "t1": 0.9}, {"radius": 4.0, "t0": 0.1, "t1": 0.9},
            {"radius": 4.0, "t0": 0.1, "t1": 0.9}, {"radius": 4.0, "t0": 0.1, "t1": 0.9},
            {"radius": 4.0, "t0": 0.1, "t1": 0.9}, {"radius": 4.0, "t0": 0.1, "t1": 0.9},
            {"radius": 4.0, "t0": 0.1, "t1": 0.9}, {"radius": 4.0, "t0": 0.1, "t1": 0.9},
        ],
        "inside": vec![true; photocraft_algo::sculpt::POSE_COUNT],
    });
    live.fit = Some(fit.clone());
    let merged = live.merged_params(&json!({"waist": 40.0}));
    assert_eq!(merged.get("fit"), Some(&fit), "the fit must ride with the anchors");
    assert_eq!(merged.get("anchors"), Some(&live.anchors), "the anchors still ride too");
}

/// Apply's warp is already on the canvas: unchanged values must not start a preview that would
/// draw the warp a second time over the committed pixels.
#[test]
fn an_applied_panel_does_not_preview_the_same_values_again() {
    let mut app = app(patterned(256, 384));
    let mut p = panel(BODY);
    p.values.insert("waist".into(), -60.0);
    let mut live = Live::test(&mut app, BODY, body_anchors(256.0, 384.0));
    assert!(live.render_now(&mut app, &p).is_some(), "a frame is drawn");
    app.sculpt_live = Some(live);
    let steps = app.session.active().unwrap().history.past_len();
    super::apply(&mut app, &mut p);
    assert_eq!(app.session.active().unwrap().history.past_len(), steps + 1, "Apply added the document's one step");
    let mut live = app.sculpt_live.take().expect("Apply keeps the runtime");
    assert!(!live.has_preview(), "Apply drops the shown frame");
    let generation = live.generation;
    let ctx = app.ctx().clone();
    live.tick(&mut app, &p, &ctx);
    assert_eq!(live.generation, generation, "the applied values are not requested again");
    assert!(!live.has_preview(), "no second warp is drawn over the commit");
    app.sculpt_live = Some(live);
}

/// Apply leaves the panel's anchors where the warp put the pixels: the next slider move
/// previews over the features the pixels now show, not over their pre-warp positions. The
/// engine's dot-marker test holds the reply to the pixels; here the panel must adopt it.
#[test]
fn apply_adopts_the_warped_anchors_for_the_next_preview() {
    let mut app = app(patterned(512, 512));
    let before = face_anchors();
    let mut p = panel(FACE);
    p.values.insert("faceSize".into(), 100.0);
    let live = Live::test(&mut app, FACE, before.clone());
    app.sculpt_live = Some(live);
    super::apply(&mut app, &mut p);
    let live = app.sculpt_live.as_ref().expect("Apply keeps the runtime");
    assert!(live.ready(), "the adopted anchors still enable the next Apply");
    assert_ne!(live.anchors, before, "the reply's anchors replace the analysed ones");
    let old = before["pts"].as_array().expect("pts");
    let new = live.anchors["pts"].as_array().expect("pts");
    let travel = old
        .iter()
        .zip(new)
        .map(|(a, b)| {
            (a[0].as_f64().unwrap_or_default() - b[0].as_f64().unwrap_or_default()).hypot(a[1].as_f64().unwrap_or_default() - b[1].as_f64().unwrap_or_default())
        })
        .fold(0.0f64, f64::max);
    assert!(travel > 1.0, "the adopted mesh must follow a visible move: {travel}");
    let merged = live.merged_params(&p.params());
    assert_eq!(merged.get("anchors"), Some(&live.anchors), "the next preview and Apply carry the adopted anchors");
}

/// Apply bakes the panel's values into the pixels: the next request carries only the change,
/// so a tweak after Apply moves that slider alone instead of stacking the whole set again —
/// which turned a fully adjusted face into a deformity. The preset bundle was applied too,
/// so its chip clears.
#[test]
fn after_an_apply_the_next_request_is_only_the_change() {
    let mut app = app(patterned(512, 512));
    let mut p = panel(FACE);
    p.values.insert("jaw".into(), 40.0);
    p.values.insert("eyeSize".into(), 20.0);
    p.values.insert("presetStrength".into(), 60.0);
    p.preset = "classic".into();
    app.sculpt_live = Some(Live::test(&mut app, FACE, face_anchors()));
    super::apply(&mut app, &mut p);
    let live = app.sculpt_live.as_ref().expect("Apply keeps the runtime");
    assert_eq!(p.preset, "none", "the applied preset bundle is baked, so its chip clears");
    assert_eq!(live.request_params(&p.params()), json!({}), "untouched sliders request nothing");
    p.values.insert("jaw".into(), 45.0);
    let delta = live.request_params(&p.params());
    assert_eq!(delta, json!({"jaw": 5.0}), "only the moved slider is requested: {}", p.params());
}

/// A paired row is compared per side, because the engine's side overrides are absolute: a
/// pair that moved together is sent as one base difference, a side that moved alone as its
/// two sides — the form that makes each side land where the panel shows it.
#[test]
fn a_paired_row_requests_the_change_per_side() {
    let mut app = app(patterned(512, 512));
    let mut live = Live::test(&mut app, FACE, face_anchors());
    // The last Apply baked eyeSize 40 on both sides.
    live.applied = Some((json!({"eyeSize": 40.0}), 1));
    let mut p = panel(FACE);
    p.values.insert("eyeSize".into(), 40.0);
    assert_eq!(live.request_params(&p.params()), json!({}), "an unchanged pair requests nothing");
    // Unlinked, the panel shows both sides at 0: both must come back down by 40.
    p.locks.insert("eyeSize".into(), false);
    assert_eq!(live.request_params(&p.params()), json!({"eyeSize": -40.0}), "a pair that moved together stays one base value");
    // A side moved alone goes as the two sides; a base difference would have moved both.
    let mut one_side = panel(FACE);
    one_side.locks.insert("eyeSize".into(), false);
    one_side.values.insert("eyeSizeLeft".into(), 30.0);
    one_side.values.insert("eyeSizeRight".into(), 50.0);
    assert_eq!(live.request_params(&one_side.params()), json!({"eyeSizeLeft": -10.0, "eyeSizeRight": 10.0}));
}

/// Undoing the committed warp puts the pixels back: the sliders must describe the whole
/// request again, not a difference from a warp that is gone.
#[test]
fn an_undo_of_the_apply_makes_the_sliders_the_whole_request_again() {
    let mut app = app(patterned(512, 512));
    let mut p = panel(FACE);
    p.values.insert("jaw".into(), 40.0);
    app.sculpt_live = Some(Live::test(&mut app, FACE, face_anchors()));
    super::apply(&mut app, &mut p);
    let mut live = app.sculpt_live.take().expect("Apply keeps the runtime");
    assert_eq!(live.request_params(&p.params()), json!({}), "the commit is the baseline");
    assert!(app.session.undo(), "the committed warp is undone");
    let ctx = app.ctx().clone();
    live.tick(&mut app, &p, &ctx);
    let whole = live.request_params(&p.params());
    assert_eq!(whole.get("jaw").and_then(Value::as_f64), Some(40.0), "the slider is the whole request again: {whole}");
    app.sculpt_live = Some(live);
}

/// Clear History forgets the rows but not the image: the warp the last Apply baked is still
/// in the pixels, so the baseline must survive it. (Matching history by position would read
/// the emptied list as an undo and stack the whole slider set on the next pass.)
#[test]
fn a_clear_history_keeps_the_committed_baseline() {
    let mut app = app(patterned(512, 512));
    let mut p = panel(FACE);
    p.values.insert("jaw".into(), 40.0);
    app.sculpt_live = Some(Live::test(&mut app, FACE, face_anchors()));
    super::apply(&mut app, &mut p);
    let mut live = app.sculpt_live.take().expect("Apply keeps the runtime");
    let before = app.session.active().unwrap().history.past_len();
    app.session.execute("history.clear", json!({})).expect("the history can be cleared");
    assert!(app.session.active().unwrap().history.past_len() < before, "the rows are gone");
    let ctx = app.ctx().clone();
    live.tick(&mut app, &p, &ctx);
    assert_eq!(live.request_params(&p.params()), json!({}), "the committed warp still stands, so nothing is re-requested");
    app.sculpt_live = Some(live);
}

/// The canvas shows the document without the target layer while the proxy is drawn over it;
/// the session's document is untouched, and closing the runtime brings the layer back.
#[test]
fn the_preview_hides_only_the_bound_layer_and_clears_with_it() {
    let mut session = patterned(64, 64);
    session.execute("layer.new.layer", json!({})).unwrap();
    let target = session.active().unwrap().active_layer.expect("the new layer");
    let below = session.active().unwrap().doc.layers[0].id;
    let mut app = app(session);
    let index = app.session.active_index().expect("active document");
    let p = panel(BODY);
    let mut live = Live::test(&mut app, BODY, body_anchors(64.0, 64.0));
    assert!(live.render_now(&mut app, &p).is_some());
    app.sculpt_live = Some(live);
    let (shown, _) = display_doc(&app, index).expect("the canvas shows the preview document");
    assert!(!shown.layer(target).expect("target").visible, "the target is hidden behind its proxy");
    assert!(shown.layer(below).expect("the layer below").visible, "other layers stay");
    assert!(app.session.active().unwrap().doc.layer(target).expect("target").visible, "the session document is not edited");
    app.sculpt_live = None;
    assert!(display_doc(&app, index).is_none(), "no runtime, no preview document");
}

/// The proxy covers the whole document within the side cap, derives from the layer's own
/// pixels at zero sliders, and moves pixels when a slider asks.
#[test]
fn the_proxy_covers_the_document_within_the_side_cap() {
    let mut app = app(patterned(1800, 1200));
    let mut p = panel(BODY);
    let mut live = Live::test(&mut app, BODY, body_anchors(1800.0, 1200.0));
    let identity = live.render_now(&mut app, &p).expect("a frame");
    let proxy = live.proxy.as_ref().expect("a cached proxy");
    assert!(proxy.proxy.w <= 1600 && proxy.proxy.h <= 1600, "the long side is capped: {}×{}", proxy.proxy.w, proxy.proxy.h);
    assert!(proxy.proxy.w as f64 * proxy.proxy.scale >= 1800.0, "the proxy covers the document width");
    assert!(proxy.proxy.h as f64 * proxy.proxy.scale >= 1200.0, "the proxy covers the document height");
    assert_eq!(proxy.proxy.bounds, app.session.active().unwrap().doc.bounds());
    // Zero sliders displace nothing: the frame is the proxy's own pixels.
    assert_eq!(identity.size, [proxy.proxy.w, proxy.proxy.h]);
    for (i, (px, source)) in identity.pixels.iter().zip(&proxy.proxy.px).enumerate() {
        let got = px.to_array();
        for k in 0..4 {
            let want = (source[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            assert!(got[k].abs_diff(want) <= 1, "proxy pixel {i} channel {k}: {got:?} vs {source:?}");
        }
    }
    p.values.insert("waist".into(), -60.0);
    let warped = live.render_now(&mut app, &p).expect("a frame");
    assert!(
        warped.pixels.iter().zip(&identity.pixels).any(|(a, b)| a.to_array().iter().zip(b.to_array()).any(|(x, y)| x.abs_diff(y) > 1)),
        "a slider warps the proxy"
    );
}

/// The face panel previews through the face field: its own sliders move the mesh's pixels.
#[test]
fn the_face_panel_previews_through_the_face_field() {
    let mut app = app(patterned(512, 512));
    let mut p = panel(FACE);
    let mut live = Live::test(&mut app, FACE, face_anchors());
    let identity = live.render_now(&mut app, &p).expect("a frame");
    p.values.insert("eyeSize".into(), -25.0);
    p.values.insert("faceSlim".into(), 40.0);
    let warped = live.render_now(&mut app, &p).expect("a frame");
    assert!(
        warped.pixels.iter().zip(&identity.pixels).any(|(a, b)| a.to_array().iter().zip(b.to_array()).any(|(x, y)| x.abs_diff(y) > 1)),
        "the face field warps the proxy"
    );
}

/// Face Sculpt draws the face's own crop over the whole-document proxy: on a 2000-pixel photo
/// the proxy has one pixel per two and a one-pixel checker dissolves to flat grey, while the
/// crop at one pixel per document pixel keeps it — the edge a moved feature is judged by. The
/// crop's quad covers the mesh and stops well inside the document.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn the_face_crop_previews_near_one_to_one_over_the_face() {
    let mut app = app(fine_checker(2000, 2000));
    let anchors = scaled_face_anchors(2000.0);
    let mut live = Live::test(&mut app, FACE, anchors.clone());
    let p = panel(FACE);
    let job = live.job(&mut app, &p.params()).expect("a face preview job");
    let frame = match super::worker::render(&job, &photocraft_engine::jobs::JobCtx::new()).expect("a frame") {
        super::worker::Output::Frame(frame) => frame,
        super::worker::Output::Analysis(_) => panic!("a preview job renders a frame"),
    };
    let crop = frame.crop.expect("Face Sculpt previews the face's own crop");
    let base = contrast(&frame.image);
    let fine = contrast(&crop.image);
    assert!(fine > 32, "the crop keeps the detail: contrast {fine}");
    assert!(fine > 8 * base.max(1), "the crop is sharper than the whole-document proxy: {fine} vs {base}");
    for (i, point) in anchors["pts"].as_array().expect("pts").iter().enumerate() {
        let (x, y) = (point[0].as_f64().unwrap_or(0.0), point[1].as_f64().unwrap_or(0.0));
        assert!(
            x >= f64::from(crop.bounds.x0) && x <= f64::from(crop.bounds.x1) && y >= f64::from(crop.bounds.y0) && y <= f64::from(crop.bounds.y1),
            "landmark {i} at ({x}, {y}) outside the crop {:?}",
            crop.bounds
        );
    }
    assert!(crop.bounds.width() < 2000 && crop.bounds.height() < 2000, "the crop is the face, not the whole document: {:?}", crop.bounds);
}

/// A translucent target previews through the whole-document proxy only: two stacked quads
/// would composite the layer's pixels twice and draw it darker than the commit.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn a_translucent_layer_previews_without_the_crop() {
    let mut session = fine_checker(2000, 2000);
    session.execute("layer.setProps", json!({"opacity": 0.5})).unwrap();
    let mut app = app(session);
    let mut live = Live::test(&mut app, FACE, scaled_face_anchors(2000.0));
    let p = panel(FACE);
    let job = live.job(&mut app, &p.params()).expect("a face preview job");
    assert!(job.crop.is_none(), "a half-opacity layer must not stack two semi-transparent quads");
}

/// A slider change after the anchors are in previews through the worker: `tick` starts the
/// job, the frame lands on a later tick, and the canvas gets a preview to draw. Both the
/// analysis path and the preview path share that one worker, so this guards the second half.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn a_slider_change_previews_through_the_worker() {
    let mut app = app(patterned(256, 384));
    let mut p = panel(BODY);
    p.values.insert("waist".into(), -60.0);
    let mut live = Live::test(&mut app, BODY, body_anchors(256.0, 384.0));
    let ctx = app.ctx().clone();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !live.has_preview() {
        live.tick(&mut app, &p, &ctx);
        assert!(std::time::Instant::now() < deadline, "the worker's frame never published");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// The analysis lands on a later tick than the panel opened, and the first slider change after
/// it previews — the panel was already on screen, at values the open call never saw.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn a_slider_change_after_the_analysis_lands_previews() {
    let mut app = app(patterned(256, 384));
    let doc = app.session.active().unwrap().doc.id;
    let layer = app.session.active().unwrap().active_layer.expect("a layer");
    let revision = app.session.active().unwrap().revision;
    let mut live = Live::test(&mut app, BODY, Value::Null);
    live.anchors = Value::Null;
    live.sampled = false;
    live.busy = true;
    live.analysis_pending = Some((doc, layer, revision));
    let anchors = body_anchors(256.0, 384.0);
    live.worker
        .start(live.serial, app.ctx().clone(), move |_| {
            Ok(super::worker::Output::Analysis(super::worker::Analysis { anchors, fit: None, contour: None, quality: None, layer: Some(layer.0) }))
        })
        .unwrap();
    let ctx = app.ctx().clone();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while live.busy() {
        live.tick(&mut app, &panel(BODY), &ctx);
        assert!(std::time::Instant::now() < deadline, "the analysis never landed");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(live.ready(), "the landed anchors enable Apply");
    let mut p = panel(BODY);
    p.values.insert("waist".into(), -60.0);
    while !live.has_preview() {
        live.tick(&mut app, &p, &ctx);
        assert!(std::time::Instant::now() < deadline, "the first slider change never previewed");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// A whole-document face reading is the fallback the sliders are least honest on. The panel
/// names it, and its selection box becomes the next analysis's crop: a face that filled the
/// photo is the case the whole-document mesh's small-face failure makes worth fixing.
#[test]
fn a_whole_document_face_reading_warns_and_reanalyse_uses_the_selection_box() {
    let mut h = harness(blank(64, 64), egui::vec2(700.0, 800.0));
    h.state_mut().sculpt = Some(panel(FACE));
    let mut live = Live::test(h.state_mut(), FACE, face_anchors());
    live.quality = Some(json!({"source": "wholeDocument", "turnDeg": 0.0}));
    h.state_mut().sculpt_live = Some(live);
    h.run_steps(2);
    assert!(h.query_by_label_contains("No face box").is_some(), "the untrusted fallback is named");
    assert!(h.query_by_label("Use selection as face box").is_some_and(|b| b.accesskit_node().is_disabled()), "no selection, no box to use");

    h.state_mut().session.execute("select.rect", json!({"x": 10, "y": 12, "width": 40, "height": 48})).unwrap();
    h.run_steps(2);
    h.get_by_label("Use selection as face box").click();
    h.run_steps(1);
    assert_eq!(h.state().sculpt_live.as_ref().and_then(Live::roi), Some(Rect::new(10, 12, 50, 60)), "the selection bounds become the face box");
    assert!(!h.state().sculpt_live.as_ref().expect("runtime").ready(), "the old anchors do not apply to the new box");
}

/// The eye line and the nose are the mesh's own pose reading: a turned head is where the
/// front-view sliders stop being honest, so the panel warns on it and not on a small roll.
#[test]
fn a_turned_face_reading_warns_and_a_level_one_does_not() {
    let mut h = harness(blank(64, 64), egui::vec2(700.0, 800.0));
    h.state_mut().sculpt = Some(panel(FACE));
    let mut live = Live::test(h.state_mut(), FACE, face_anchors());
    live.quality = Some(json!({"source": "detected", "turnDeg": 32.0, "rollDeg": 2.0}));
    h.state_mut().sculpt_live = Some(live);
    h.run_steps(2);
    assert!(h.query_by_label_contains("face is turned").is_some(), "a profile is named");

    let mut live = h.state_mut().sculpt_live.take().expect("runtime");
    live.quality = Some(json!({"source": "detected", "turnDeg": 4.0, "rollDeg": 2.0}));
    h.state_mut().sculpt_live = Some(live);
    h.run_steps(2);
    assert!(h.query_by_label_contains("face is turned").is_none(), "a slightly turned face is not a warning");
}

/// Another active layer has its own pose or face: the runtime rebinds to it, drops the old
/// frame, and the panel's next Apply targets the new layer.
#[test]
fn switching_layers_rebinds_and_drops_the_old_preview() {
    let mut session = patterned(64, 64);
    session.execute("layer.new.layer", json!({})).unwrap();
    let below = session.active().unwrap().doc.layers[0].id;
    let mut app = app(session);
    let p = panel(BODY);
    let mut live = Live::test(&mut app, BODY, body_anchors(64.0, 64.0));
    assert!(live.render_now(&mut app, &p).is_some());
    app.sculpt_live = Some(live);
    app.session.execute("layer.select", json!({"layer": below.0})).unwrap();
    let mut live = app.sculpt_live.take().expect("runtime");
    let ctx = app.ctx().clone();
    live.tick(&mut app, &p, &ctx);
    assert_eq!(live.layer, Some(below), "the runtime follows the active layer");
    assert!(!live.has_preview(), "the old layer's frame is gone");
}
