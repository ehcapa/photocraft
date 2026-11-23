//! The panel's height and its params: the actions stay on screen where the rows do not
//! fit, the window can shrink to its chrome, and Apply never sends a preset nobody chose.

use super::*;
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT as _, Queryable};
use serde_json::json;

/// A document to edit and the panel drawn through the app's dialog pass, on a screen no
/// larger than a laptop's.
fn harness(size: egui::Vec2) -> Harness<'static, Box<PhotocraftApp>> {
    let mut session = photocraft_engine::Session::new();
    session.execute("file.new", json!({"width": 64, "height": 64, "depth": 8})).unwrap();
    let app = PhotocraftApp::new(session, crate::Services::default());
    let h =
        Harness::builder().with_context(app.ctx().clone()).with_size(size).build_ui_state(|ui, app: &mut Box<PhotocraftApp>| crate::dialogs::run(app, ui), app);
    PhotocraftApp::setup_context(&h.ctx, crate::theme::AppTheme::pro());
    h
}

/// The face's Eyes section (ten paired rows, three sliders each once unlinked) is the panel's
/// tallest; on a short screen it must scroll inside a capped window, leaving Apply and Close
/// on screen. Before the cap the content-sized window ran past the screen and the buttons
/// were clipped below it, unreachable.
#[test]
fn the_actions_stay_on_screen_where_the_rows_do_not_fit() {
    let mut h = harness(egui::vec2(560.0, 420.0));
    crate::sculpt_ui::open(h.state_mut(), FACE);
    h.state_mut().sculpt.as_mut().expect("panel open").section = 1;
    h.run_steps(6);
    let screen = h.ctx.content_rect();
    for label in ["Apply", "Close"] {
        let rect = h.get_by_label(label).rect();
        assert!(screen.contains_rect(rect), "{label} at {rect:?} escapes the {screen:?} screen");
    }
    // The rows are what scrolls: the section's first row is laid out inside the panel, above
    // the actions, and the window as a whole stays inside the screen.
    let first = h.get_by_role_and_label(egui::accesskit::Role::Label, "Eye Size").rect();
    assert!(screen.contains_rect(first), "the first eye row is on screen: {first:?}");
    assert!(first.max.y <= h.get_by_label("Apply").rect().min.y, "the rows sit above the actions");
}

/// The window's floor is its chrome, not its content: a taller section scrolls, so shrinking
/// past the rows' natural height must stick instead of springing back to it.
#[test]
fn a_shrunk_window_stays_shrunk() {
    let mut h = harness(egui::vec2(560.0, 700.0));
    crate::sculpt_ui::open(h.state_mut(), FACE);
    h.run_steps(6);
    let id = egui::Id::unique("photocraft.sculpt");
    let rect = h.ctx.memory(|m| m.area_rect(id)).expect("window area");
    let corner = rect.right_bottom() - egui::vec2(4.0, 4.0);
    h.drag_at(corner);
    h.run_steps(1);
    let target = corner - egui::vec2(0.0, 200.0);
    h.hover_at(target);
    h.run_steps(1);
    h.drop_at(target);
    h.run_steps(4);
    let after = h.ctx.memory(|m| m.area_rect(id)).expect("window area");
    assert!(after.height() < rect.height() - 100.0, "the shrink sticks: {rect:?} -> {after:?}");
}

/// The rows' sliders carry their visible label as their accessible name, so a screen reader
/// announces "Waist, slider, 0" instead of an unnamed slider; `labelled_by` is the egui pattern
/// for a text-less slider beside its label.
#[test]
fn each_slider_is_named_after_its_row_label() {
    let mut h = harness(egui::vec2(560.0, 700.0));
    crate::sculpt_ui::open(h.state_mut(), BODY);
    h.run_steps(3);
    let named: Vec<String> =
        h.query_all_by_role(egui::accesskit::Role::Slider).filter_map(|node| node.accesskit_node().labelled_by().find_map(|label| label.value())).collect();
    for row in ["Slim Upper Body", "Slim Full Body", "Waist", "Belly"] {
        assert!(named.iter().any(|name| name == row), "a slider is named {row}: {named:?}");
    }
}

/// The Placement section is where a garment's waistband, or a front-on belly the side outline
/// cannot show, is placed by eye: both rows exist as named sliders, and a moved line is carried
/// in the command params. Zero is auto — an untouched panel sends neither key, so the measured
/// or anatomical row stands.
#[test]
fn the_placement_rows_carry_the_line_offsets() {
    let mut h = harness(egui::vec2(560.0, 700.0));
    crate::sculpt_ui::open(h.state_mut(), BODY);
    h.state_mut().sculpt.as_mut().expect("panel open").section = BODY_SECTIONS.len() - 1;
    h.run_steps(3);
    let named: Vec<String> =
        h.query_all_by_role(egui::accesskit::Role::Slider).filter_map(|node| node.accesskit_node().labelled_by().find_map(|label| label.value())).collect();
    for row in ["Waist Line", "Belly Line"] {
        assert!(named.iter().any(|name| name == row), "a slider is named {row}: {named:?}");
    }
    // The placement rows take the wider stop: their slider reaches beyond the shaping sliders'
    // ±100 and past the hip joints, so a waistband or a belly below the anatomical row is
    // still reachable.
    let (lo, hi) = {
        let node = h
            .query_all_by_role(egui::accesskit::Role::Slider)
            .find(|node| node.accesskit_node().labelled_by().any(|label| label.value().as_deref() == Some("Waist Line")))
            .expect("the waist line slider");
        let a11y = node.accesskit_node();
        (a11y.min_numeric_value(), a11y.max_numeric_value())
    };
    assert_eq!((lo, hi), (Some(-500.0), Some(500.0)), "the placement line reaches past the hips, not only ±100");
    let panel = h.state_mut().sculpt.as_mut().expect("panel open");
    assert!(panel.params().get("waistLine").is_none() && panel.params().get("bellyLine").is_none(), "zero is auto and sends nothing: {}", panel.params());
    panel.set("waistLine", 40.0);
    panel.set("bellyLine", -30.0);
    assert_eq!(panel.params()["waistLine"], 40.0, "the moved waist line is sent");
    assert_eq!(panel.params()["bellyLine"], -30.0, "the moved belly line is sent");
}

/// The analysis's status line appears inline and is a polite live region: it changes while
/// focus is elsewhere, so a screen reader must announce the refusal — and the progress before
/// it — rather than leave Apply disabled without a word.
#[test]
fn the_analysis_status_is_a_live_region() {
    let mut h = harness(egui::vec2(560.0, 700.0));
    crate::sculpt_ui::open(h.state_mut(), BODY);
    {
        let app = h.state_mut();
        let mut live = app.sculpt_live.take().expect("runtime");
        live.analysis_now(app);
        app.sculpt_live = Some(live);
    }
    h.run_steps(3);
    let status = h.get_by_label_contains("No person found");
    assert_eq!(status.accesskit_node().live(), egui::accesskit::Live::Polite, "the status line is announced");
}

/// Opening the face panel leaves its preset row untouched; Apply must send no preset at all.
/// The panel used to start on the empty string — which is not `none` — so Apply sent it and
/// the engine answered `unknown preset ""`.
#[test]
fn an_untouched_face_panel_sends_no_preset() {
    let mut session = photocraft_engine::Session::new();
    session.execute("file.new", json!({"width": 64, "height": 64, "depth": 8})).unwrap();
    let mut app = PhotocraftApp::new(session, crate::Services::default());
    crate::sculpt_ui::open(&mut app, FACE);
    let panel = app.sculpt.as_ref().expect("panel open");
    assert_eq!(panel.preset, "none");
    assert!(panel.params().get("preset").is_none(), "an untouched panel sends no preset: {}", panel.params());

    // And a panel built without `open` still must not pass the empty string on.
    let unset = Panel { command: FACE, ..Panel::default() };
    assert!(unset.params().get("preset").is_none(), "an empty preset is no preset: {}", unset.params());

    let chosen = Panel { command: FACE, preset: "classic".into(), ..Panel::default() };
    assert_eq!(chosen.params()["preset"], "classic");
}
