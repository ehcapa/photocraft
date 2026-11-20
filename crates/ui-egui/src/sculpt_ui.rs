//! The Body/Face Sculpt panel — the bespoke grouped face of the two sculpt commands:
//! sections as one tab row, the active section's sliders in a two-column grid, the preset
//! row above them (face), a chain link on every two-sided feature (linked moves the pair,
//! unlinked edits each side), and an Apply that runs the engine command and keeps the panel
//! open for the next pass. The generic schema dialog served both commands until here; this
//! is the pixel-parity surface over the same engine params.
//!
//! The window is capped to the screen and the rows scroll, so the panel fits a laptop: an
//! earlier left-hand section column held the window open at the height of all its buttons
//! and the content-sized window ran past the screen bottom, putting Apply out of reach.

use std::collections::BTreeMap;

use serde_json::json;

use crate::PhotocraftApp;

pub const BODY: &str = "filter.bodySculpt";
pub const FACE: &str = "filter.faceSculpt";

/// One slider row: the params key and its label (the `tl!` key).
struct Row {
    key: &'static str,
    label: &'static str,
}

/// One section: its name, its rows, and the base keys that carry a Left/Right lock.
struct Section {
    name: &'static str,
    rows: &'static [Row],
    pairs: &'static [&'static str],
}

const fn r(key: &'static str, label: &'static str) -> Row {
    Row { key, label }
}

const BODY_SECTIONS: &[Section] = &[
    Section {
        name: "Shaping",
        rows: &[
            r("slimBody", "Slim Body"),
            r("waist", "Waist"),
            r("belly", "Belly"),
            r("arms", "Arms"),
            r("thighs", "Thighs"),
            r("calves", "Calves"),
            r("shoulders", "Shoulder Width"),
            r("hips", "Hips"),
            r("bust", "Bust"),
            r("headSize", "Head Size"),
            r("neck", "Neck"),
            r("straightShoulders", "Straight Shoulders"),
            r("doubleChin", "Double Chin"),
        ],
        pairs: &[],
    },
    Section { name: "Composition", rows: &[r("aiSculpt", "AI Shaping"), r("swanNeck", "Swan Neck")], pairs: &[] },
    Section {
        name: "Lengths",
        rows: &[
            r("height", "Height"),
            r("legLength", "Leg Length"),
            r("waistLength", "Waist Length"),
            r("torsoLength", "Torso Length"),
            r("neckLength", "Neck Length"),
            r("bustLift", "Bust Lift"),
        ],
        pairs: &[],
    },
    Section { name: "Refine", rows: &[r("smooth", "Silhouette Flow"), r("symmetry", "Symmetry")], pairs: &[] },
];

const FACE_SECTIONS: &[Section] = &[
    Section {
        name: "Face",
        rows: &[
            r("faceSlim", "Slim Face"),
            r("temples", "Temples"),
            r("cheekbones", "Cheekbones"),
            r("jaw", "Jaw"),
            r("faceSize", "Face Size"),
            r("vFace", "V Face"),
            r("faceWidth", "Face Width"),
            r("hairline", "Hairline"),
            r("foreheadWidth", "Forehead Width"),
            r("foreheadHeight", "Forehead Height"),
            r("philtrum", "Philtrum"),
            r("midFace", "Mid Face"),
            r("lowerFace", "Lower Face"),
            r("chinLength", "Chin Length"),
            r("chinPoint", "Pointed Chin"),
            r("chinRetract", "Retract Chin"),
        ],
        pairs: &["temples", "cheekbones"],
    },
    Section {
        name: "Eyes",
        rows: &[
            r("eyeSize", "Eye Size"),
            r("pupilSize", "Pupil Size"),
            r("eyeHeight", "Eye Height"),
            r("eyeWidth", "Eye Width"),
            r("eyeSpacing", "Eye Spacing"),
            r("innerCorner", "Inner Corner"),
            r("outerCorner", "Outer Corner"),
            r("lowerLid", "Lower Lid"),
            r("eyeTilt", "Eye Tilt"),
            r("eyePosition", "Eye Position"),
        ],
        pairs: &["eyeSize", "pupilSize", "eyeHeight", "eyeWidth", "eyeSpacing", "innerCorner", "outerCorner", "lowerLid", "eyeTilt", "eyePosition"],
    },
    Section {
        name: "Nose",
        rows: &[
            r("noseSize", "Nose Size"),
            r("noseHeight", "Nose Height"),
            r("noseShift", "Nose Shift"),
            r("noseBridge", "Nose Bridge"),
            r("noseWings", "Nose Wings"),
            r("noseTip", "Nose Tip"),
        ],
        pairs: &["noseWings"],
    },
    Section {
        name: "Mouth",
        rows: &[
            r("mouthSize", "Mouth Size"),
            r("mouthWidth", "Mouth Width"),
            r("mouthHeight", "Mouth Height"),
            r("mouthShift", "Mouth Shift"),
            r("gumSmile", "Gum Smile"),
            r("lipTilt", "Lip Tilt"),
            r("upperLip", "Upper Lip"),
            r("lowerLip", "Lower Lip"),
        ],
        pairs: &["mouthWidth", "lipTilt"],
    },
    Section {
        name: "Brows",
        rows: &[
            r("browHead", "Brow Head"),
            r("browTail", "Brow Tail"),
            r("browThickness", "Brow Thickness"),
            r("browSpacing", "Brow Spacing"),
            r("browTilt", "Brow Tilt"),
            r("browShape", "Brow Shape"),
            r("browHeight", "Brow Height"),
        ],
        pairs: &["browHead", "browTail", "browThickness", "browSpacing", "browTilt", "browShape", "browHeight"],
    },
    Section {
        name: "Refine",
        rows: &[
            r("poseYaw", "Turn Yaw"),
            r("posePitch", "Turn Pitch"),
            r("poseRoll", "Turn Roll"),
            r("symmetry", "Feature Symmetry"),
            r("smooth", "Face Flow"),
        ],
        pairs: &[],
    },
];

/// The presets the face panel's top row carries (the engine's own names).
const PRESETS: &[(&str, &str)] = &[
    ("classic", "Classic"),
    ("youthful", "Youthful"),
    ("heroic", "Heroic"),
    ("juvenile", "Juvenile"),
    ("round", "Round"),
    ("long", "Long"),
    ("square", "Square"),
];

/// The open panel: which command, every slider's value, every lock's state, the section on
/// show. Plain data on the app (not UiState — the control channel does not drive it).
#[derive(Clone, Debug, Default)]
pub struct Panel {
    pub command: &'static str,
    pub values: BTreeMap<String, f32>,
    pub locks: BTreeMap<String, bool>,
    pub section: usize,
    pub preset: String,
}

impl Panel {
    fn sections(&self) -> &'static [Section] {
        if self.command == FACE { FACE_SECTIONS } else { BODY_SECTIONS }
    }

    fn value(&self, key: &str) -> f32 {
        self.values.get(key).copied().unwrap_or(0.0)
    }

    fn set(&mut self, key: &str, v: f32) {
        self.values.insert(key.to_owned(), v);
    }

    /// The params of the current state: every value, the pairs resolved through their
    /// locks (linked sends the base, unlinked sends the two sides), the face's preset.
    fn params(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        let sections = self.sections();
        let pairs: Vec<&str> = sections.iter().flat_map(|s| s.pairs.iter().copied()).collect();
        for (k, v) in &self.values {
            if *v == 0.0 {
                continue;
            }
            m.insert(k.clone(), json!(v));
        }
        for base in pairs {
            let linked = *self.locks.get(base).unwrap_or(&true);
            let side = |s: &str| self.value(&format!("{base}{s}"));
            if !linked && (side("Left") != 0.0 || side("Right") != 0.0 || self.value(base) != 0.0) {
                m.remove(base);
                m.insert(format!("{base}Left"), json!(side("Left")));
                m.insert(format!("{base}Right"), json!(side("Right")));
            }
        }
        // No preset is `none`; an unset panel must not send the empty string the engine
        // refuses (`unknown preset ""`).
        if self.command == FACE && !matches!(self.preset.as_str(), "" | "none") {
            m.insert("preset".into(), json!(self.preset));
            m.insert("presetStrength".into(), json!(self.value("presetStrength")));
        }
        serde_json::Value::Object(m)
    }
}

/// The id the menu-routing contract wants: this panel is not a hosted dialog, so none.
pub fn dialog_id() -> craft::ui::dialog::DialogId {
    craft::ui::dialog::DialogId::NULL
}

/// Opens the panel for `command` (fresh values, every lock linked, no preset chosen).
pub fn open(app: &mut PhotocraftApp, command: &'static str) {
    app.sculpt = Some(Panel { command, preset: "none".into(), ..Panel::default() });
}

/// Draws the open panel. Kept open across Apply — the point of a panel.
pub fn windows(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(panel) = app.sculpt.clone() else { return };
    let title = if panel.command == FACE { tl!("Face Sculpt") } else { tl!("Body Sculpt") };
    let mut close = false;
    let mut run: Option<serde_json::Value> = None;
    let mut panel = panel;
    // The tallest sections (the face's sixteen rows, or the ten eye pairs at three sliders
    // each once unlinked) outgrow a laptop's usable height. Cap the window to the screen and
    // let the rows scroll, so Apply and Close can never be pushed below its bottom edge.
    let max_height = (ctx.content_rect().height() - 48.0).max(240.0);
    egui::Window::new(title).id(egui::Id::unique("photocraft.sculpt")).min_width(360.0).default_size([440.0, 420.0]).max_height(max_height).show(ctx, |ui| {
        let sections = panel.sections();
        let mut apply = false;
        let mut reset = false;
        // Bottom-up: the actions take the window's bottom edge first and the body fills what
        // is left — so a section taller than the window scrolls instead of hiding them.
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            ui.horizontal(|ui| {
                if ui.button(tl!("Apply")).clicked() {
                    apply = true;
                }
                if ui.button(tl!("Close")).clicked() {
                    close = true;
                }
                if ui.button(tl!("Reset")).clicked() {
                    reset = true;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak(tl!("Applies and stays open"));
                });
            });
            ui.separator();
            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                // Sections as one wrapped tab row. A left-hand column held the window open at
                // the height of all its buttons — the floor the shrink grip bounced back to —
                // and a tab row costs a line instead.
                ui.horizontal_wrapped(|ui| {
                    for (i, s) in sections.iter().enumerate() {
                        if ui.selectable_label(panel.section == i, tl!(s.name)).clicked() {
                            panel.section = i;
                        }
                    }
                });
                ui.separator();
                // The preset row (face only): seven one-clicks, the chosen one held. Wrapped, so
                // a narrow window takes two lines rather than widening past the screen.
                if panel.command == FACE {
                    ui.horizontal_wrapped(|ui| {
                        ui.strong(tl!("Preset"));
                        for (id, label) in PRESETS {
                            if ui.selectable_label(panel.preset == *id, tl!(label)).clicked() {
                                panel.preset = if panel.preset == *id { "none".into() } else { (*id).into() };
                            }
                        }
                    });
                    // Strength belongs to a chosen preset; without one the row would only
                    // add height (and a slider that does nothing).
                    if !matches!(panel.preset.as_str(), "" | "none") {
                        let mut strength = panel.value("presetStrength");
                        if strength == 0.0 {
                            strength = 60.0; // the dialog's own default, shown rather than sent
                        }
                        ui.horizontal(|ui| {
                            ui.label(tl!("Preset Strength"));
                            ui.add(egui::Slider::new(&mut strength, 0.0..=100.0));
                        });
                        panel.set("presetStrength", strength);
                    }
                    ui.separator();
                }
                // The remaining height, and no shrinking to the rows: past it they
                // scroll (a ScrollArea that auto-shrinks would grow with them).
                egui::ScrollArea::vertical().auto_shrink([false, false]).min_scrolled_height(24.0).max_height(ui.available_height().max(1.0)).show(ui, |ui| {
                    let section = &sections[panel.section.min(sections.len() - 1)];
                    #[derive(Clone, Copy)]
                    struct Edit {
                        key: &'static str,
                        base: f32,
                        sides: Option<(f32, f32)>,
                    }
                    let mut edits: Vec<Edit> = Vec::new();
                    // Two columns — the link and label, then the slider — so the sliders line
                    // up and every row costs one line; an open pair adds its L and R rows
                    // under its own. The slider takes the width the labels leave.
                    ui.spacing_mut().slider_width = (ui.available_width() - 180.0).clamp(80.0, 220.0);
                    egui::Grid::new("photocraft.sculpt.rows").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
                        for row in section.rows {
                            let paired = section.pairs.contains(&row.key);
                            let linked = *panel.locks.get(row.key).unwrap_or(&true);
                            if paired {
                                ui.horizontal(|ui| {
                                    if ui.small_button(if linked { "🔗" } else { "⛓" }).clicked() {
                                        panel.locks.insert(row.key.to_owned(), !linked);
                                    }
                                    ui.label(tl!(row.label));
                                });
                                let mut base = panel.value(row.key);
                                ui.add(egui::Slider::new(&mut base, -100.0..=100.0));
                                ui.end_row();
                                if linked {
                                    edits.push(Edit { key: row.key, base, sides: None });
                                } else {
                                    let mut l = panel.value(&format!("{}Left", row.key));
                                    let mut rt = panel.value(&format!("{}Right", row.key));
                                    ui.label("L");
                                    ui.add(egui::Slider::new(&mut l, -100.0..=100.0));
                                    ui.end_row();
                                    ui.label("R");
                                    ui.add(egui::Slider::new(&mut rt, -100.0..=100.0));
                                    ui.end_row();
                                    panel.set(row.key, base);
                                    edits.push(Edit { key: row.key, base, sides: Some((l, rt)) });
                                }
                            } else {
                                let mut v = panel.value(row.key);
                                ui.label(tl!(row.label));
                                ui.add(egui::Slider::new(&mut v, -100.0..=100.0));
                                ui.end_row();
                                edits.push(Edit { key: row.key, base: v, sides: None });
                            }
                        }
                    });
                    for e in edits {
                        panel.set(e.key, e.base);
                        if let Some((l, rt)) = e.sides {
                            panel.set(&format!("{}Left", e.key), l);
                            panel.set(&format!("{}Right", e.key), rt);
                        }
                    }
                });
            });
        });
        if reset {
            let section = &sections[panel.section.min(sections.len() - 1)];
            for row in section.rows {
                panel.set(row.key, 0.0);
                panel.set(&format!("{}Left", row.key), 0.0);
                panel.set(&format!("{}Right", row.key), 0.0);
            }
        }
        if apply {
            run = Some(panel.params());
        }
    });
    app.sculpt = (!close).then_some(panel);
    if let Some(params) = run {
        let command = app.sculpt.as_ref().map(|p| p.command).unwrap_or(BODY);
        if let Err(e) = app.run(command, params) {
            crate::dialogs::alerts::report_error(app, e);
        }
    }
}

#[cfg(test)]
mod tests;
