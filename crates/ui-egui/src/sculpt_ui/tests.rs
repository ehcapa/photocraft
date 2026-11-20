//! The panel's height cap: the actions stay on screen where the rows do not fit.

use super::*;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
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
    let first = h.get_by_label("Eye Size").rect();
    assert!(screen.contains_rect(first), "the first eye row is on screen: {first:?}");
    assert!(first.max.y <= h.get_by_label("Apply").rect().min.y, "the rows sit above the actions");
}
