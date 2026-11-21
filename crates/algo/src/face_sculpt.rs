//! Filter › Face Sculpt: the face panel — PixCake's 面部重塑 in the same terms as the body
//! panel ([`crate::sculpt`]), and the architecture promise kept: a second parts table over
//! the same four primitives ([`crate::sculpt::Shape`], written by the shared
//! [`crate::sculpt::write_shape`]), anchored on FaceMesh's 478 landmarks instead of
//! BlazePose's 33 joints. No new machinery.
//!
//! Five groups, each slider −100..100 with its positive direction the label's action
//! (PixCake's own right side): the face outline (sixteen), the eyes (ten, per-side rows),
//! the nose (six), the mouth (eight) and the brows (seven, per-side rows). The presets,
//! the symmetry mirror, the fake-rotation 3D塑颜 and the AI smoothness pass are a later
//! wave — they are not landmark warps or not warps at all.
//!
//! The face's own scale is small: an eye is thirty pixels where a waist is three hundred,
//! so the gains here are relatives of the feature they shape (an eye's width, the philtrum
//! gap), and the fold guard of the body table applies unchanged — the parts are smaller,
//! not the rules.

use crate::liquify::LiquifyField;
use crate::sculpt::{Shape, write_shape};
use photocraft_geom::Rect;
use serde::{Deserialize, Serialize};

/// The landmarks the face table reads (restated: this crate never depends on the ml one).
/// FaceMesh answers all 478 or none, so the frame's gate is a degenerate check, not a
/// per-joint visibility.
pub const MESH_COUNT: usize = 478;

/// The forty-seven face sliders. Grouped in declaration order — the dialog reads the
/// params string, the table reads these.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FaceSliders {
    // face outline
    pub face_slim: f32,
    pub temples: f32,
    pub cheekbones: f32,
    pub jaw: f32,
    pub face_size: f32,
    pub v_face: f32,
    pub face_width: f32,
    pub hairline: f32,
    pub forehead_width: f32,
    pub forehead_height: f32,
    pub philtrum: f32,
    pub mid_face: f32,
    pub lower_face: f32,
    pub chin_length: f32,
    pub chin_point: f32,
    pub chin_retract: f32,
    // eyes
    pub eye_size: f32,
    pub pupil_size: f32,
    pub eye_height: f32,
    pub eye_width: f32,
    pub eye_spacing: f32,
    pub inner_corner: f32,
    pub outer_corner: f32,
    pub lower_lid: f32,
    pub eye_tilt: f32,
    pub eye_position: f32,
    // nose
    pub nose_size: f32,
    pub nose_height: f32,
    pub nose_shift: f32,
    pub nose_bridge: f32,
    pub nose_wings: f32,
    pub nose_tip: f32,
    // mouth
    pub mouth_size: f32,
    pub mouth_width: f32,
    pub mouth_height: f32,
    pub mouth_shift: f32,
    pub gum_smile: f32,
    pub lip_tilt: f32,
    pub upper_lip: f32,
    pub lower_lip: f32,
    // brows
    pub brow_head: f32,
    pub brow_tail: f32,
    pub brow_thickness: f32,
    pub brow_spacing: f32,
    pub brow_tilt: f32,
    pub brow_shape: f32,
    pub brow_height: f32,
    // the preset family (PixCake's 脸型预设): a named bundle of the sliders above, scaled
    // by `preset_strength`, blended in [`FaceSliders::composed`] before any row runs.
    pub preset: FacePreset,
    pub preset_strength: f32,
    /// Per-side overrides for the two-sided features: `None` follows the base slider —
    /// PixCake's lock stays locked until a side is set. The left/right rows below read
    /// their own side's override.
    pub sides: FaceSides,
}

/// The per-side overrides (`None` = follow the base slider).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FaceSides {
    pub temples: [Option<f32>; 2],
    pub cheekbones: [Option<f32>; 2],
    pub eye_size: [Option<f32>; 2],
    pub pupil_size: [Option<f32>; 2],
    pub eye_height: [Option<f32>; 2],
    pub eye_width: [Option<f32>; 2],
    pub eye_spacing: [Option<f32>; 2],
    pub inner_corner: [Option<f32>; 2],
    pub outer_corner: [Option<f32>; 2],
    pub lower_lid: [Option<f32>; 2],
    pub eye_tilt: [Option<f32>; 2],
    pub eye_position: [Option<f32>; 2],
    pub brow_head: [Option<f32>; 2],
    pub brow_tail: [Option<f32>; 2],
    pub brow_thickness: [Option<f32>; 2],
    pub brow_spacing: [Option<f32>; 2],
    pub brow_tilt: [Option<f32>; 2],
    pub brow_shape: [Option<f32>; 2],
    pub brow_height: [Option<f32>; 2],
    pub mouth_width: [Option<f32>; 2],
    pub lip_tilt: [Option<f32>; 2],
}

/// The side an override slot addresses: 0 = the person's left (image right), 1 = right.
pub const SIDE_L: usize = 0;
pub const SIDE_R: usize = 1;

/// The seven face-type presets, PixCake's own list: each is the adjustment its face shape
/// wants (their wiki's wording), as a bundle of this panel's sliders at strength 100.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FacePreset {
    #[default]
    None,
    /// 经典: the universal retouch — a gentle slim with slightly larger eyes.
    Classic,
    /// 幼态: rounder, lower-set features; bigger eyes, a shorter mid face.
    Youthful,
    /// 英气: a narrower face with a stronger jaw line.
    Heroic,
    /// 少年: a smoother contour, the least intrusive.
    Juvenile,
    /// 圆脸适用: narrow and taper — the V line, a pointer chin.
    Round,
    /// 长脸适用: shorten the middle and lower thirds.
    Long,
    /// 方脸适用: soften the jaw, lengthen the chin.
    Square,
}

impl FacePreset {
    /// The wire names, as the params and the dialog carry them.
    pub fn parse(name: &str) -> Option<FacePreset> {
        Some(match name {
            "none" => FacePreset::None,
            "classic" => FacePreset::Classic,
            "youthful" => FacePreset::Youthful,
            "heroic" => FacePreset::Heroic,
            "juvenile" => FacePreset::Juvenile,
            "round" => FacePreset::Round,
            "long" => FacePreset::Long,
            "square" => FacePreset::Square,
            _ => return None,
        })
    }

    /// The bundle at strength 100 as a slider delta: only the fields a preset touches are
    /// set, everything else stays zero (and out of `composed`'s blend).
    fn delta(self) -> FaceSliders {
        let d = |fields: &[(&str, f32)]| {
            let mut s = FaceSliders::default();
            for &(name, v) in fields {
                match name {
                    "faceSlim" => s.face_slim = v,
                    "eyeSize" => s.eye_size = v,
                    "faceWidth" => s.face_width = v,
                    "midFace" => s.mid_face = v,
                    "lowerFace" => s.lower_face = v,
                    "chinLength" => s.chin_length = v,
                    "jaw" => s.jaw = v,
                    "vFace" => s.v_face = v,
                    "chinPoint" => s.chin_point = v,
                    "browTilt" => s.brow_tilt = v,
                    "hairline" => s.hairline = v,
                    _ => {}
                }
            }
            s
        };
        match self {
            FacePreset::None => FaceSliders::default(),
            FacePreset::Classic => d(&[("faceSlim", 25.0), ("eyeSize", 15.0)]),
            FacePreset::Youthful => d(&[("eyeSize", 20.0), ("faceWidth", 10.0), ("midFace", -15.0), ("chinLength", -10.0)]),
            FacePreset::Heroic => d(&[("faceSlim", 35.0), ("eyeSize", 8.0), ("jaw", 20.0), ("browTilt", 15.0)]),
            FacePreset::Juvenile => d(&[("faceSlim", 18.0), ("eyeSize", 10.0), ("midFace", -8.0), ("chinLength", -6.0)]),
            FacePreset::Round => d(&[("faceSlim", 40.0), ("vFace", 35.0), ("chinPoint", 25.0), ("eyeSize", 12.0)]),
            FacePreset::Long => d(&[("midFace", -30.0), ("lowerFace", -25.0), ("hairline", -20.0), ("faceSlim", 15.0)]),
            FacePreset::Square => d(&[("jaw", 40.0), ("chinLength", 20.0), ("faceSlim", 25.0)]),
        }
    }
}

impl FaceSliders {
    /// The sliders after the preset blends in: the bundle's values, scaled by
    /// `preset_strength` (0..100), stack with the manual sliders and clamp at the stops —
    /// the body panel's AI塑形 rule, so a stored filter's raw sliders compose to the same
    /// field bit for bit.
    pub fn composed(self) -> Self {
        let k = self.preset_strength.clamp(0.0, 100.0) / 100.0;
        let blend = |own: f32, extra: f32| (own + extra * k).clamp(-100.0, 100.0);
        let d = self.preset.delta();
        Self {
            face_slim: blend(self.face_slim, d.face_slim),
            eye_size: blend(self.eye_size, d.eye_size),
            face_width: blend(self.face_width, d.face_width),
            mid_face: blend(self.mid_face, d.mid_face),
            lower_face: blend(self.lower_face, d.lower_face),
            chin_length: blend(self.chin_length, d.chin_length),
            jaw: blend(self.jaw, d.jaw),
            v_face: blend(self.v_face, d.v_face),
            chin_point: blend(self.chin_point, d.chin_point),
            brow_tilt: blend(self.brow_tilt, d.brow_tilt),
            hairline: blend(self.hairline, d.hairline),
            ..self
        }
    }
}

/// The anchors a stored face field needs: the 478 mesh points (`x, y` in document pixels)
/// and the document size they belong to. Embedded in the smart filter's params at apply
/// time, exactly like the body's.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FaceAnchors {
    pub doc: [f32; 2],
    /// The 478 mesh points as `x, y, z` document pixels — z is FaceMesh's own depth,
    /// kept because 3D塑颜's rotation reads it; everything else ignores it.
    pub pts: Vec<[f32; 3]>,
}

/// The sliders plus the anchors they shape.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceSpec {
    pub sliders: FaceSliders,
    pub anchors: FaceAnchors,
}

/// The face's geometric quality, as the panel reports it: the face's extent in the photo
/// and how far the mesh is turned or rolled. It is not a confidence — FaceMesh has none —
/// but a turned face is where the front-view sliders stop being honest, so naming it is
/// better than letting the user read a wrong warp as a model failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceQuality {
    pub face_width: f32,
    pub face_height: f32,
    /// The eye line's angle in degrees. A small value is an upright head.
    pub roll_deg: f32,
    /// The nose's offset from the face's mid-axis, as the angle that offset would project
    /// under a turn. A proxy for yaw, not a measured one: the mesh's own depth is too weak
    /// to trust as a pose angle, and the slider range is a few degrees anyway.
    pub turn_deg: f32,
}

impl FaceAnchors {
    /// The quality of a usable 478-point mesh, `None` when the anchors are not one.
    pub fn quality(&self) -> Option<FaceQuality> {
        if self.pts.len() != MESH_COUNT || self.pts.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        let p = |i: usize| [self.pts[i][0] as f64, self.pts[i][1] as f64];
        let face_width = (p(mesh::FACE_SIDE_L)[0] - p(mesh::FACE_SIDE_R)[0]).abs();
        let face_height = (p(mesh::FOREHEAD_TOP)[1] - p(mesh::CHIN)[1]).abs();
        if face_width < 8.0 || face_height < 8.0 {
            return None;
        }
        // The eye line: the outer corners are the most stable pair the mesh names.
        let (left, right) = (p(mesh::EYE_OUTER_L), p(mesh::EYE_OUTER_R));
        let roll = (left[1] - right[1]).atan2(left[0] - right[0]).to_degrees();
        // The nose's offset from the mid-point of the face's sides, along the side axis.
        let (side_l, side_r) = (p(mesh::FACE_SIDE_L), p(mesh::FACE_SIDE_R));
        let axis = [side_l[0] - side_r[0], side_l[1] - side_r[1]];
        let axis_len = axis[0].hypot(axis[1]);
        let mid = [(side_l[0] + side_r[0]) * 0.5, (side_l[1] + side_r[1]) * 0.5];
        let nose = p(mesh::NOSE_TIP);
        let offset = if axis_len < 1e-9 { 0.0 } else { ((nose[0] - mid[0]) * axis[0] + (nose[1] - mid[1]) * axis[1]) / axis_len };
        let turn = ((2.0 * offset / face_width).clamp(-1.0, 1.0)).asin().to_degrees();
        Some(FaceQuality { face_width: face_width as f32, face_height: face_height as f32, roll_deg: roll as f32, turn_deg: turn as f32 })
    }
}

/// What a face build did — the body table's semantics: the rows that asked and ran, and
/// the rows that asked but could not (a mesh that never answered), with the reason.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FaceReport {
    pub applied: Vec<&'static str>,
    pub skipped: Vec<(&'static str, &'static str)>,
}

// ---------- the frame ----------

/// MediaPipe FaceMesh indices the frame and the table read. Left is the person's left
/// (image right); the table addresses sides by index.
mod mesh {
    pub const FOREHEAD_TOP: usize = 10;
    pub const CHIN: usize = 152;
    pub const FACE_SIDE_L: usize = 454;
    pub const FACE_SIDE_R: usize = 234;
    pub const TEMPLE_L: usize = 251;
    pub const TEMPLE_R: usize = 21;
    pub const CHEEKBONE_L: usize = 457;
    pub const CHEEKBONE_R: usize = 227;
    /// The eye rings (the portrait crate's own canonical lists, restated).
    pub const EYE_RING_L: &[usize] = &[263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
    pub const EYE_RING_R: &[usize] = &[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
    /// MediaPipe's FACE_OVAL ring: the face's own outline, the field's clip.
    pub const FACE_OVAL: &[usize] = &[
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162,
        21, 54, 103, 67, 109,
    ];
    /// The outer lip ring: the mouth's opening, protected from the face-outline parts.
    pub const LIPS_RING: &[usize] = &[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
    pub const EYE_OUTER_L: usize = 263;
    pub const EYE_OUTER_R: usize = 33;
    pub const EYE_INNER_L: usize = 362;
    pub const EYE_INNER_R: usize = 133;
    pub const EYE_LOW_L: usize = 374;
    pub const EYE_LOW_R: usize = 145;
    pub const IRIS_L: usize = 468;
    pub const IRIS_R: usize = 473;
    pub const BROW_OUT_L: usize = 336;
    pub const BROW_MID_L: usize = 334;
    pub const BROW_IN_L: usize = 300;
    pub const BROW_INNER_L: usize = 285;
    pub const BROW_OUT_R: usize = 70;
    pub const BROW_MID_R: usize = 105;
    pub const BROW_IN_R: usize = 107;
    pub const BROW_INNER_R: usize = 55;
    pub const NOSE_BRIDGE_TOP: usize = 168;
    pub const NOSE_TIP: usize = 1;
    pub const NOSE_BASE: usize = 2;
    pub const ALAR_L: usize = 358;
    pub const ALAR_R: usize = 129;
    pub const MOUTH_CORNER_L: usize = 291;
    pub const MOUTH_CORNER_R: usize = 61;
    pub const UPPER_LIP_TOP: usize = 0;
    pub const UPPER_LIP_LOW: usize = 13;
    pub const LOWER_LIP_TOP: usize = 14;
    pub const LOWER_LIP_LOW: usize = 17;
}

/// The symmetry pairs (left index, right index) 对称液化 aligns: the eye corners and
/// rims, the brows, the mouth corners, the nose wings, the cheekbones, the temples, the
/// jaw — every paired feature the mesh names.
pub const SYMMETRY_PAIRS: &[(usize, usize)] = &[
    (263, 33),
    (362, 133),
    (374, 145),
    (336, 70),
    (334, 105),
    (300, 107),
    (285, 55),
    (296, 63),
    (293, 66),
    (291, 61),
    (358, 129),
    (457, 227),
    (251, 21),
    (397, 172),
];

/// The face's derived geometry: named anchors and the feature scales the table's widths
/// and gains hang on, resolved once per build from the mesh.
struct FaceFrame {
    p: Vec<[f64; 2]>,
    face_c: [f64; 2],
    face_w: f64,
    face_h: f64,
}

#[inline]
fn v(add: [f64; 2], k: f64) -> [f64; 2] {
    [add[0] * k, add[1] * k]
}

#[inline]
fn down(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [b[0] - a[0], b[1] - a[1]]
}

#[inline]
fn len_of(d: [f64; 2]) -> f64 {
    d[0].hypot(d[1])
}

impl FaceFrame {
    /// `None` when the mesh is absent or degenerate (FaceMesh's no-face fallback parks
    /// every point at the image centre, which this catches by the face's width).
    fn new(a: &FaceAnchors) -> Option<FaceFrame> {
        if a.pts.len() != MESH_COUNT || !a.doc.iter().all(|v| v.is_finite()) || a.pts.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        let p: Vec<[f64; 2]> = a.pts.iter().map(|q| [q[0] as f64, q[1] as f64]).collect();
        let at = |i: usize| p[i];
        let face_w = at(mesh::FACE_SIDE_L)[0] - at(mesh::FACE_SIDE_R)[0];
        if face_w < 8.0 {
            return None;
        }
        let face_h = at(mesh::FOREHEAD_TOP)[1] - at(mesh::CHIN)[1];
        let face_c = [(at(mesh::FACE_SIDE_L)[0] + at(mesh::FACE_SIDE_R)[0]) * 0.5, (at(mesh::FOREHEAD_TOP)[1] + at(mesh::CHIN)[1]) * 0.5];
        Some(FaceFrame { p, face_c, face_w: face_w.abs(), face_h: face_h.abs() })
    }

    fn pt(&self, i: usize) -> [f64; 2] {
        self.p[i]
    }

    /// A landmark ring as points, for the field's mask polygons.
    fn ring(&self, ids: &[usize]) -> Vec<[f64; 2]> {
        ids.iter().map(|&i| self.pt(i)).collect()
    }

    fn mid(&self, a: usize, b: usize) -> [f64; 2] {
        [(self.pt(a)[0] + self.pt(b)[0]) * 0.5, (self.pt(a)[1] + self.pt(b)[1]) * 0.5]
    }

    /// The unit vector pointing up the face (forehead away from chin).
    fn up(&self) -> [f64; 2] {
        let d = down(self.pt(mesh::CHIN), self.pt(mesh::FOREHEAD_TOP));
        let n = len_of(d);
        if n < 1e-9 { [0.0, -1.0] } else { v(d, 1.0 / n) }
    }

    fn along_up(&self, from: usize, t: f64) -> [f64; 2] {
        let d = down(self.pt(from), self.pt(mesh::FOREHEAD_TOP));
        [self.pt(from)[0] + d[0] * t, self.pt(from)[1] + d[1] * t]
    }

    /// The same, toward the chin (the forehead's own bands hang downward from the top).
    fn along_down(&self, from: usize, t: f64) -> [f64; 2] {
        let d = down(self.pt(from), self.pt(mesh::CHIN));
        [self.pt(from)[0] + d[0] * t, self.pt(from)[1] + d[1] * t]
    }

    /// The unit vector pointing to the person's left (image right).
    fn left(&self) -> [f64; 2] {
        let d = down(self.pt(mesh::FACE_SIDE_R), self.pt(mesh::FACE_SIDE_L));
        let n = len_of(d);
        if n < 1e-9 { [1.0, 0.0] } else { v(d, 1.0 / n) }
    }

    // feature anchors, resolved from their landmark clusters
    fn eye_c(&self, ring: &[usize]) -> [f64; 2] {
        let n = ring.len() as f64;
        ring.iter().fold([0.0, 0.0], |a, &i| [a[0] + self.pt(i)[0] / n, a[1] + self.pt(i)[1] / n])
    }

    fn eye_w(&self) -> f64 {
        (self.pt(mesh::EYE_OUTER_L)[0] - self.pt(mesh::EYE_INNER_L)[0]).abs().max(6.0)
    }

    fn brow_line(&self, side: f64) -> Option<Shape> {
        let (o, i) = if side > 0.0 { (mesh::BROW_OUT_L, mesh::BROW_IN_L) } else { (mesh::BROW_OUT_R, mesh::BROW_IN_R) };
        let l = len_of(down(self.pt(o), self.pt(i)));
        (l >= 4.0).then_some(Shape::Contract { a: self.pt(o), b: self.pt(i), band: (0.0, 1.0), width: (l * 0.35).max(6.0), gain: -0.42 })
    }

    fn nose_c(&self) -> [f64; 2] {
        self.mid(mesh::NOSE_TIP, mesh::NOSE_BASE)
    }

    fn nose_w(&self) -> f64 {
        (self.pt(mesh::ALAR_L)[0] - self.pt(mesh::ALAR_R)[0]).abs().max(6.0)
    }

    fn mouth_c(&self) -> [f64; 2] {
        self.mid(mesh::UPPER_LIP_LOW, mesh::LOWER_LIP_TOP)
    }

    fn mouth_w(&self) -> f64 {
        (self.pt(mesh::MOUTH_CORNER_L)[0] - self.pt(mesh::MOUTH_CORNER_R)[0]).abs().max(6.0)
    }
}

/// A convex hull (monotone chain, counter-clockwise) of a landmark ring. The rings are
/// near-convex and their index order is not always the boundary order, so the hull is the
/// safe polygon for a membership test. `None` for fewer than three distinct points.
fn hull(mut pts: Vec<[f64; 2]>) -> Option<Vec<[f64; 2]>> {
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    pts.dedup_by(|a, b| a[0] == b[0] && a[1] == b[1]);
    if pts.len() < 3 {
        return None;
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let build = |src: &[[f64; 2]]| {
        let mut out: Vec<[f64; 2]> = Vec::with_capacity(src.len());
        for &p in src {
            while out.len() >= 2 && cross(out[out.len() - 2], out[out.len() - 1], p) <= 0.0 {
                out.pop();
            }
            out.push(p);
        }
        out
    };
    let lower = build(&pts);
    pts.reverse();
    let upper = build(&pts);
    let mut ring = lower;
    ring.pop();
    let mut tail = upper;
    tail.pop();
    ring.extend(tail);
    (ring.len() >= 3).then_some(ring)
}

/// Squared distance from `p` to the segment `a..b`.
fn seg_dist2(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = down(a, b);
    let d2 = ab[0] * ab[0] + ab[1] * ab[1];
    if d2 <= 1e-18 {
        let d = down(a, p);
        return d[0] * d[0] + d[1] * d[1];
    }
    let t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / d2;
    let t = t.clamp(0.0, 1.0);
    let q = [a[0] + ab[0] * t, a[1] + ab[1] * t];
    let d = down(q, p);
    d[0] * d[0] + d[1] * d[1]
}

/// The signed distance from `p` to a counter-clockwise convex polygon: negative inside,
/// positive outside, absolute value the distance to the nearest edge.
fn signed_distance(p: [f64; 2], poly: &[[f64; 2]]) -> f64 {
    let mut d2 = f64::MAX;
    let mut inside = true;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let side = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        if side < 0.0 {
            inside = false;
        }
        d2 = d2.min(seg_dist2(p, a, b));
    }
    let d = d2.sqrt();
    if inside { -d } else { d }
}

/// The face field's soft bounds, computed over the face's own node window so a large
/// document does not pay a canvas-sized mask: `clip` is the FACE_OVAL hull (1 inside, 0
/// outside, a feather across the outline), `eye` and `lip` are 0 inside the eye rings and
/// the lip ring and 1 outside. The outline parts must not move the eyes or the mouth; an
/// eye or mouth part is exempt from its own feature's protection.
struct FaceMask {
    i0: usize,
    j0: usize,
    w: usize,
    h: usize,
    clip: Vec<f32>,
    eye: Vec<f32>,
    lip: Vec<f32>,
}

impl FaceMask {
    fn new(frame: &FaceFrame, field: &LiquifyField) -> Option<FaceMask> {
        let oval = hull(frame.ring(mesh::FACE_OVAL))?;
        let eye_l = hull(frame.ring(mesh::EYE_RING_L));
        let eye_r = hull(frame.ring(mesh::EYE_RING_R));
        let lip = hull(frame.ring(mesh::LIPS_RING));
        let (min_x, max_x) = oval.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
        let (min_y, max_y) = oval.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| (lo.min(p[1]), hi.max(p[1])));
        // The clip's outward feather is the furthest the mask is nonzero outside the ring.
        let inward = (frame.face_w * 0.02).max(2.0);
        let outward = (frame.face_w * 0.05).clamp(4.0, 40.0);
        let (bx, by) = (f64::from(field.bounds.x0) + 0.5, f64::from(field.bounds.y0) + 0.5);
        let node_x = |x: f64| ((x - bx) / field.cell).floor();
        let node_y = |y: f64| ((y - by) / field.cell).floor();
        let i0 = node_x(min_x - outward).max(0.0) as usize;
        let j0 = node_y(min_y - outward).max(0.0) as usize;
        let i1 = node_x(max_x + outward).min(field.w as f64 - 1.0).max(i0 as f64) as usize;
        let j1 = node_y(max_y + outward).min(field.h as f64 - 1.0).max(j0 as f64) as usize;
        let (w, h) = (i1 - i0 + 1, j1 - j0 + 1);
        let mut mask = FaceMask { i0, j0, w, h, clip: vec![0.0; w * h], eye: vec![1.0; w * h], lip: vec![1.0; w * h] };
        // The rings' whole interiors stay still and the weight ramps to 1 outside them: an
        // eye ring is only a few pixels tall, so a ramp centred on its outline (as the face
        // clip's is) would leave the pupil partly free to ride the outline sliders. The
        // ramp is long enough that its steepest slope keeps the around-the-face parts
        // within the fold guard when every slider is at full.
        let feather = (frame.face_w * 0.07).clamp(6.0, 32.0);
        for j in j0..=j1 {
            for i in i0..=i1 {
                let p = field.node_pos(i, j);
                let idx = (j - j0) * w + (i - i0);
                mask.clip[idx] = (1.0 - crate::sculpt::smoothstep(-inward, outward, signed_distance(p, &oval))).clamp(0.0, 1.0) as f32;
                let mut eye = 1.0f64;
                for ring in [&eye_l, &eye_r].into_iter().flatten() {
                    eye = eye.min(crate::sculpt::smoothstep(0.0, feather, signed_distance(p, ring)));
                }
                mask.eye[idx] = eye as f32;
                if let Some(ring) = &lip {
                    mask.lip[idx] = crate::sculpt::smoothstep(0.0, feather, signed_distance(p, ring)) as f32;
                }
            }
        }
        Some(mask)
    }

    /// The weight closure one part writes through. Outside the window the face clip is 0,
    /// which is also every face part's answer there.
    fn weights(&self, protect_eyes: bool, protect_lips: bool) -> impl Fn(usize, usize) -> f32 + '_ {
        move |i, j| {
            if i < self.i0 || j < self.j0 {
                return 0.0;
            }
            let (x, y) = (i - self.i0, j - self.j0);
            if x >= self.w || y >= self.h {
                return 0.0;
            }
            let idx = y * self.w + x;
            let mut weight = self.clip[idx];
            if protect_eyes {
                weight *= self.eye[idx];
            }
            if protect_lips {
                weight *= self.lip[idx];
            }
            weight
        }
    }
}

/// Whether a part must leave the eye or lip interior alone. Eye parts own the eyes, mouth
/// parts own the lips; everything else — the outline, the nose, the brows — protects both.
fn protects(name: &str) -> (bool, bool) {
    let eye = name.contains("eye") || name.contains("pupil") || name.contains("lid");
    let mouth = name.contains("mouth") || name.contains("lip") || name.contains("gum") || name.contains("philtrum");
    (!eye, !mouth)
}

/// One row of the face table — the body table's Part, over the face frame.
struct FacePart {
    name: &'static str,
    slider: fn(&FaceSliders) -> f32,
    shape: fn(&FaceFrame) -> Option<Shape>,
}

/// The parts. Table order = build order (determinism, as the body table's).
const FACE_PARTS: &[FacePart] = &[
    // ---------- the outline ----------
    FacePart {
        name: "face slim",
        slider: |s| s.face_slim,
        shape: |f| f.contract_v(f.along_up(mesh::CHIN, 0.30), f.pt(mesh::CHIN), (0.0, 1.0), f.face_w * 0.62, 0.27),
    },
    FacePart {
        name: "left temple",
        slider: |s| s.sides.temples[SIDE_L].unwrap_or(s.temples),
        shape: |f| f.radial(f.pt(mesh::TEMPLE_L), f.face_w * 0.14, 0.42),
    },
    FacePart {
        name: "right temple",
        slider: |s| s.sides.temples[SIDE_R].unwrap_or(s.temples),
        shape: |f| f.radial(f.pt(mesh::TEMPLE_R), f.face_w * 0.14, 0.06),
    },
    FacePart {
        name: "left cheekbone",
        slider: |s| s.sides.cheekbones[SIDE_L].unwrap_or(s.cheekbones),
        shape: |f| f.radial(f.pt(mesh::CHEEKBONE_L), f.face_w * 0.13, 0.64),
    },
    FacePart {
        name: "right cheekbone",
        slider: |s| s.sides.cheekbones[SIDE_R].unwrap_or(s.cheekbones),
        shape: |f| f.radial(f.pt(mesh::CHEEKBONE_R), f.face_w * 0.13, 0.08),
    },
    FacePart { name: "jaw", slider: |s| s.jaw, shape: |f| f.contract_v(f.along_up(mesh::CHIN, 0.55), f.pt(mesh::CHIN), (0.0, 1.0), f.face_w * 0.50, 0.34) },
    FacePart { name: "face size", slider: |s| s.face_size, shape: |f| f.radial(f.face_c, f.face_h * 0.62, 0.26) },
    FacePart {
        name: "v face",
        slider: |s| s.v_face,
        shape: |f| f.contract_v(f.along_up(mesh::CHIN, 0.40), f.pt(mesh::CHIN), (0.0, 1.0), f.face_w * 0.55, 0.27),
    },
    // Width-family convention (PixCake's own right side): positive widens, so the gain
    // expands.
    FacePart {
        name: "face width",
        slider: |s| s.face_width,
        shape: |f| f.contract_v(f.pt(mesh::FOREHEAD_TOP), f.pt(mesh::CHIN), (0.05, 0.95), f.face_w * 0.70, -0.24),
    },
    FacePart {
        name: "hairline",
        slider: |s| s.hairline,
        shape: |f| f.shift_band(f.pt(mesh::FOREHEAD_TOP), f.along_down(mesh::FOREHEAD_TOP, 0.45), v(f.up(), 1.0), (0.1, 0.9), f.face_w * 0.55, 0.075),
    },
    FacePart {
        name: "forehead width",
        slider: |s| s.forehead_width,
        shape: |f| Some(Shape::Contract { a: f.pt(mesh::TEMPLE_L), b: f.pt(mesh::TEMPLE_R), band: (0.0, 1.0), width: f.face_h * 0.25, gain: -0.38 }),
    },
    FacePart {
        name: "forehead height",
        slider: |s| s.forehead_height,
        shape: |f| f.shift_band(f.pt(mesh::BROW_INNER_L), f.along_up(mesh::FOREHEAD_TOP, 0.5), v(f.up(), -1.0), (0.1, 0.9), f.face_w * 0.55, 0.08),
    },
    FacePart {
        name: "philtrum",
        slider: |s| s.philtrum,
        shape: |f| f.shift_band(f.pt(mesh::NOSE_BASE), f.pt(mesh::UPPER_LIP_TOP), v(f.up(), 1.0), (0.0, 1.0), f.face_w * 0.30, 0.25),
    },
    FacePart {
        name: "mid face",
        slider: |s| s.mid_face,
        shape: |f| Some(Shape::Ramp { a: f.pt(mesh::CHIN), b: f.pt(mesh::NOSE_BASE), band: (0.1, 0.9), width: f.face_w * 0.60, gain: 0.08 }),
    },
    FacePart {
        name: "lower face",
        slider: |s| s.lower_face,
        shape: |f| Some(Shape::Ramp { a: f.pt(mesh::CHIN), b: f.pt(mesh::NOSE_BASE), band: (0.25, 0.95), width: f.face_w * 0.55, gain: 0.08 }),
    },
    FacePart {
        name: "chin length",
        slider: |s| s.chin_length,
        shape: |f| Some(Shape::Ramp { a: f.pt(mesh::LOWER_LIP_LOW), b: f.pt(mesh::CHIN), band: (0.1, 0.9), width: f.face_w * 0.40, gain: 0.15 }),
    },
    FacePart {
        name: "chin point",
        slider: |s| s.chin_point,
        shape: |f| f.contract_v(f.pt(mesh::LOWER_LIP_LOW), f.pt(mesh::CHIN), (0.0, 1.0), f.face_w * 0.28, 0.50),
    },
    FacePart {
        name: "chin retract",
        slider: |s| s.chin_retract,
        shape: |f| f.shift_band(f.pt(mesh::MOUTH_CORNER_L), f.pt(mesh::CHIN), v(f.up(), 1.0), (0.3, 1.0), f.face_w * 0.45, 0.10),
    },
    // ---------- the eyes (per-side rows share their slider) ----------
    FacePart {
        name: "left eye size",
        slider: |s| s.sides.eye_size[SIDE_L].unwrap_or(s.eye_size),
        shape: |f| f.radial(f.eye_c(mesh::EYE_RING_L), f.eye_w() * 1.15, -0.36),
    },
    FacePart {
        name: "right eye size",
        slider: |s| s.sides.eye_size[SIDE_R].unwrap_or(s.eye_size),
        shape: |f| f.radial(f.eye_c(mesh::EYE_RING_R), f.eye_w() * 1.15, -0.12),
    },
    FacePart {
        name: "left pupil",
        slider: |s| s.sides.pupil_size[SIDE_L].unwrap_or(s.pupil_size),
        shape: |f| f.radial(f.pt(mesh::IRIS_L), f.eye_w() * 0.45, -0.25),
    },
    FacePart {
        name: "right pupil",
        slider: |s| s.sides.pupil_size[SIDE_R].unwrap_or(s.pupil_size),
        shape: |f| f.radial(f.pt(mesh::IRIS_R), f.eye_w() * 0.45, -0.10),
    },
    FacePart {
        name: "left eye height",
        slider: |s| s.sides.eye_height[SIDE_L].unwrap_or(s.eye_height),
        shape: |f| Some(Shape::Contract { a: f.pt(mesh::EYE_INNER_L), b: f.pt(mesh::EYE_OUTER_L), band: (0.0, 1.0), width: f.eye_w() * 0.80, gain: -0.30 }),
    },
    FacePart {
        name: "right eye height",
        slider: |s| s.sides.eye_height[SIDE_R].unwrap_or(s.eye_height),
        shape: |f| Some(Shape::Contract { a: f.pt(mesh::EYE_INNER_R), b: f.pt(mesh::EYE_OUTER_R), band: (0.0, 1.0), width: f.eye_w() * 0.80, gain: -0.10 }),
    },
    FacePart {
        name: "left eye width",
        slider: |s| s.sides.eye_width[SIDE_L].unwrap_or(s.eye_width),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_L), f.pt(mesh::EYE_OUTER_L), v(f.left(), 1.0), (0.0, 1.0), f.eye_w() * 1.4, 0.30),
    },
    FacePart {
        name: "right eye width",
        slider: |s| s.sides.eye_width[SIDE_R].unwrap_or(s.eye_width),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_R), f.pt(mesh::EYE_OUTER_R), v(f.left(), -1.0), (0.0, 1.0), f.eye_w() * 1.4, 0.30),
    },
    FacePart {
        name: "left eye spacing",
        slider: |s| s.sides.eye_spacing[SIDE_L].unwrap_or(s.eye_spacing),
        shape: |f| f.shift_band(f.pt(mesh::EYE_INNER_L), f.eye_c(mesh::EYE_RING_L), v(f.left(), 1.0), (0.0, 1.0), f.eye_w() * 1.6, 0.30),
    },
    FacePart {
        name: "right eye spacing",
        slider: |s| s.sides.eye_spacing[SIDE_R].unwrap_or(s.eye_spacing),
        shape: |f| f.shift_band(f.pt(mesh::EYE_INNER_R), f.eye_c(mesh::EYE_RING_R), v(f.left(), -1.0), (0.0, 1.0), f.eye_w() * 1.6, 0.30),
    },
    FacePart {
        name: "left inner corner",
        slider: |s| s.sides.inner_corner[SIDE_L].unwrap_or(s.inner_corner),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_L), f.pt(mesh::EYE_INNER_L), v(f.left(), -1.0), (0.0, 1.0), f.eye_w() * 0.9, 0.27),
    },
    FacePart {
        name: "right inner corner",
        slider: |s| s.sides.inner_corner[SIDE_R].unwrap_or(s.inner_corner),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_R), f.pt(mesh::EYE_INNER_R), v(f.left(), 1.0), (0.0, 1.0), f.eye_w() * 0.9, 0.50),
    },
    FacePart {
        name: "left outer corner",
        slider: |s| s.sides.outer_corner[SIDE_L].unwrap_or(s.outer_corner),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_L), f.pt(mesh::EYE_OUTER_L), v(f.left(), 1.0), (0.0, 1.0), f.eye_w() * 0.9, 0.50),
    },
    FacePart {
        name: "right outer corner",
        slider: |s| s.sides.outer_corner[SIDE_R].unwrap_or(s.outer_corner),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_R), f.pt(mesh::EYE_OUTER_R), v(f.left(), -1.0), (0.0, 1.0), f.eye_w() * 0.9, 0.50),
    },
    FacePart {
        name: "left lower lid",
        slider: |s| s.sides.lower_lid[SIDE_L].unwrap_or(s.lower_lid),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_L), f.pt(mesh::EYE_LOW_L), v(f.up(), -1.0), (0.0, 1.0), f.eye_w() * 1.2, 0.40),
    },
    FacePart {
        name: "right lower lid",
        slider: |s| s.sides.lower_lid[SIDE_R].unwrap_or(s.lower_lid),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_R), f.pt(mesh::EYE_LOW_R), v(f.up(), -1.0), (0.0, 1.0), f.eye_w() * 1.2, 0.40),
    },
    FacePart {
        name: "left eye tilt",
        slider: |s| s.sides.eye_tilt[SIDE_L].unwrap_or(s.eye_tilt),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_L), f.pt(mesh::EYE_OUTER_L), v(f.up(), 1.0), (0.0, 1.0), f.eye_w() * 1.0, 0.30),
    },
    FacePart {
        name: "right eye tilt",
        slider: |s| s.sides.eye_tilt[SIDE_R].unwrap_or(s.eye_tilt),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_R), f.pt(mesh::EYE_OUTER_R), v(f.up(), 1.0), (0.0, 1.0), f.eye_w() * 1.0, 0.30),
    },
    FacePart {
        name: "left eye position",
        slider: |s| s.sides.eye_position[SIDE_L].unwrap_or(s.eye_position),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_L), f.pt(mesh::EYE_OUTER_L), v(f.up(), 1.0), (0.2, 0.8), f.eye_w() * 1.6, 0.30),
    },
    FacePart {
        name: "right eye position",
        slider: |s| s.sides.eye_position[SIDE_R].unwrap_or(s.eye_position),
        shape: |f| f.shift_band(f.eye_c(mesh::EYE_RING_R), f.pt(mesh::EYE_OUTER_R), v(f.up(), 1.0), (0.2, 0.8), f.eye_w() * 1.6, 0.30),
    },
    // ---------- the nose ----------
    FacePart { name: "nose size", slider: |s| s.nose_size, shape: |f| f.radial(f.nose_c(), f.nose_w() * 1.30, 0.26) },
    FacePart {
        name: "nose height",
        slider: |s| s.nose_height,
        shape: |f| f.shift_band(f.pt(mesh::NOSE_BASE), f.pt(mesh::NOSE_BRIDGE_TOP), v(f.up(), 1.0), (0.1, 0.9), f.nose_w() * 2.0, 0.13),
    },
    FacePart {
        name: "nose shift",
        slider: |s| s.nose_shift,
        shape: |f| f.shift_band(f.pt(mesh::NOSE_BASE), f.pt(mesh::NOSE_BRIDGE_TOP), v(f.left(), 1.0), (0.1, 0.9), f.nose_w() * 2.0, 0.13),
    },
    FacePart {
        name: "nose bridge",
        slider: |s| s.nose_bridge,
        shape: |f| Some(Shape::Contract { a: f.pt(mesh::NOSE_BRIDGE_TOP), b: f.pt(mesh::NOSE_BASE), band: (0.0, 1.0), width: f.nose_w() * 1.60, gain: -0.10 }),
    },
    FacePart { name: "left nose wing", slider: |s| s.nose_wings, shape: |f| f.radial(f.pt(mesh::ALAR_L), f.nose_w() * 0.50, -0.26) },
    FacePart { name: "right nose wing", slider: |s| s.nose_wings, shape: |f| f.radial(f.pt(mesh::ALAR_R), f.nose_w() * 0.50, -0.08) },
    FacePart { name: "nose tip", slider: |s| s.nose_tip, shape: |f| f.radial(f.pt(mesh::NOSE_TIP), f.nose_w() * 0.60, 0.28) },
    // ---------- the mouth ----------
    FacePart { name: "mouth size", slider: |s| s.mouth_size, shape: |f| f.radial(f.mouth_c(), f.mouth_w() * 0.65, -0.30) },
    FacePart {
        name: "left mouth width",
        slider: |s| s.sides.mouth_width[SIDE_L].unwrap_or(s.mouth_width),
        shape: |f| f.radial(f.pt(mesh::MOUTH_CORNER_L), f.mouth_w() * 0.25, -0.35),
    },
    FacePart {
        name: "right mouth width",
        slider: |s| s.sides.mouth_width[SIDE_R].unwrap_or(s.mouth_width),
        shape: |f| f.radial(f.pt(mesh::MOUTH_CORNER_R), f.mouth_w() * 0.25, -0.10),
    },
    FacePart {
        name: "mouth height",
        slider: |s| s.mouth_height,
        shape: |f| f.shift_band(f.pt(mesh::MOUTH_CORNER_L), f.pt(mesh::LOWER_LIP_LOW), v(f.up(), -1.0), (0.1, 0.9), f.mouth_w() * 0.60, 0.18),
    },
    FacePart {
        name: "mouth shift",
        slider: |s| s.mouth_shift,
        shape: |f| f.shift_band(f.pt(mesh::MOUTH_CORNER_L), f.pt(mesh::LOWER_LIP_LOW), v(f.left(), 1.0), (0.1, 0.9), f.mouth_w() * 0.60, 0.18),
    },
    FacePart {
        name: "gum smile",
        slider: |s| s.gum_smile,
        shape: |f| f.shift_band(f.pt(mesh::NOSE_BASE), f.pt(mesh::UPPER_LIP_TOP), v(f.up(), -1.0), (0.0, 1.0), f.mouth_w() * 0.55, 0.30),
    },
    FacePart {
        name: "left lip tilt",
        slider: |s| s.sides.lip_tilt[SIDE_L].unwrap_or(s.lip_tilt),
        shape: |f| f.shift_band(f.mouth_c(), f.pt(mesh::MOUTH_CORNER_L), v(f.up(), -1.0), (0.0, 1.0), f.mouth_w() * 0.30, 0.16),
    },
    FacePart {
        name: "right lip tilt",
        slider: |s| s.sides.lip_tilt[SIDE_R].unwrap_or(s.lip_tilt),
        shape: |f| f.shift_band(f.mouth_c(), f.pt(mesh::MOUTH_CORNER_R), v(f.up(), 1.0), (0.0, 1.0), f.mouth_w() * 0.30, 0.30),
    },
    FacePart { name: "upper lip", slider: |s| s.upper_lip, shape: |f| f.radial(f.mid(mesh::UPPER_LIP_TOP, mesh::UPPER_LIP_LOW), f.mouth_w() * 0.22, -0.50) },
    FacePart { name: "lower lip", slider: |s| s.lower_lip, shape: |f| f.radial(f.mid(mesh::LOWER_LIP_LOW, mesh::LOWER_LIP_TOP), f.mouth_w() * 0.22, -0.15) },
    // ---------- the brows (per-side rows) ----------
    FacePart {
        name: "left brow head",
        slider: |s| s.sides.brow_head[SIDE_L].unwrap_or(s.brow_head),
        shape: |f| f.shift_band(f.pt(mesh::BROW_IN_L), f.pt(mesh::BROW_MID_L), v(f.left(), -1.0), (0.0, 1.0), f.face_w * 0.09, 0.16),
    },
    FacePart {
        name: "right brow head",
        slider: |s| s.sides.brow_head[SIDE_R].unwrap_or(s.brow_head),
        shape: |f| f.shift_band(f.pt(mesh::BROW_IN_R), f.pt(mesh::BROW_MID_R), v(f.left(), 1.0), (0.0, 1.0), f.face_w * 0.09, 0.17),
    },
    FacePart {
        name: "left brow tail",
        slider: |s| s.sides.brow_tail[SIDE_L].unwrap_or(s.brow_tail),
        shape: |f| f.shift_band(f.pt(mesh::BROW_MID_L), f.pt(mesh::BROW_OUT_L), v(f.left(), 1.0), (0.0, 1.0), f.face_w * 0.09, 0.30),
    },
    FacePart {
        name: "right brow tail",
        slider: |s| s.sides.brow_tail[SIDE_R].unwrap_or(s.brow_tail),
        shape: |f| f.shift_band(f.pt(mesh::BROW_MID_R), f.pt(mesh::BROW_OUT_R), v(f.left(), -1.0), (0.0, 1.0), f.face_w * 0.09, 0.30),
    },
    FacePart { name: "left brow thickness", slider: |s| s.sides.brow_thickness[SIDE_L].unwrap_or(s.brow_thickness), shape: |f| f.brow_line(1.0) },
    FacePart { name: "right brow thickness", slider: |s| s.sides.brow_thickness[SIDE_R].unwrap_or(s.brow_thickness), shape: |f| f.brow_line(-1.0) },
    FacePart {
        name: "left brow spacing",
        slider: |s| s.sides.brow_spacing[SIDE_L].unwrap_or(s.brow_spacing),
        shape: |f| f.shift_band(f.pt(mesh::BROW_INNER_L), f.pt(mesh::BROW_IN_L), v(f.left(), -1.0), (0.0, 1.0), f.face_w * 0.09, 0.13),
    },
    FacePart {
        name: "right brow spacing",
        slider: |s| s.sides.brow_spacing[SIDE_R].unwrap_or(s.brow_spacing),
        shape: |f| f.shift_band(f.pt(mesh::BROW_INNER_R), f.pt(mesh::BROW_IN_R), v(f.left(), 1.0), (0.0, 1.0), f.face_w * 0.09, 0.25),
    },
    FacePart {
        name: "left brow tilt",
        slider: |s| s.sides.brow_tilt[SIDE_L].unwrap_or(s.brow_tilt),
        shape: |f| f.shift_band(f.pt(mesh::BROW_MID_L), f.pt(mesh::BROW_OUT_L), v(f.up(), -1.0), (0.0, 1.0), f.face_w * 0.09, 0.30),
    },
    FacePart {
        name: "right brow tilt",
        slider: |s| s.sides.brow_tilt[SIDE_R].unwrap_or(s.brow_tilt),
        shape: |f| f.shift_band(f.pt(mesh::BROW_MID_R), f.pt(mesh::BROW_OUT_R), v(f.up(), -1.0), (0.0, 1.0), f.face_w * 0.09, 0.30),
    },
    FacePart {
        name: "left brow shape",
        slider: |s| s.sides.brow_shape[SIDE_L].unwrap_or(s.brow_shape),
        shape: |f| f.shift_band(f.pt(mesh::BROW_IN_L), f.pt(mesh::BROW_OUT_L), v(f.up(), 1.0), (0.3, 0.7), f.face_w * 0.08, 0.15),
    },
    FacePart {
        name: "right brow shape",
        slider: |s| s.sides.brow_shape[SIDE_R].unwrap_or(s.brow_shape),
        shape: |f| f.shift_band(f.pt(mesh::BROW_IN_R), f.pt(mesh::BROW_OUT_R), v(f.up(), 1.0), (0.3, 0.7), f.face_w * 0.08, 0.25),
    },
    FacePart {
        name: "left brow height",
        slider: |s| s.sides.brow_height[SIDE_L].unwrap_or(s.brow_height),
        shape: |f| f.shift_band(f.pt(mesh::BROW_IN_L), f.pt(mesh::BROW_OUT_L), v(f.up(), 1.0), (0.1, 0.9), f.face_w * 0.10, 0.15),
    },
    FacePart {
        name: "right brow height",
        slider: |s| s.sides.brow_height[SIDE_R].unwrap_or(s.brow_height),
        shape: |f| f.shift_band(f.pt(mesh::BROW_IN_R), f.pt(mesh::BROW_OUT_R), v(f.up(), 1.0), (0.1, 0.9), f.face_w * 0.10, 0.25),
    },
];

impl FaceFrame {
    /// The contract primitive's constructor, restated for the face (the body frame's own
    /// is private to its module).
    fn contract_v(&self, a: [f64; 2], b: [f64; 2], band: (f64, f64), width: f64, gain: f64) -> Option<Shape> {
        (len_of(down(b, a)) >= 4.0 && width >= 4.0 && band.1 > band.0).then_some(Shape::Contract { a, b, band, width, gain })
    }

    fn radial(&self, c: [f64; 2], radius: f64, gain: f64) -> Option<Shape> {
        (radius >= 4.0).then_some(Shape::Radial { c, radius, gain })
    }

    fn shift_band(&self, a: [f64; 2], b: [f64; 2], dir: [f64; 2], band: (f64, f64), width: f64, gain: f64) -> Option<Shape> {
        (len_of(down(b, a)) >= 2.0 && len_of(dir) > 1e-9 && width >= 4.0 && band.1 > band.0).then_some(Shape::Shift { a, b, dir, band, width, gain })
    }
}

/// The face field of `spec` over `bounds`. Pure and deterministic like the body's; the
/// length family's fold budget applies to the two face ramps scaled to the face.
pub fn face_field(spec: &FaceSpec, bounds: Rect, cell: f64) -> (LiquifyField, FaceReport) {
    let mut field = LiquifyField::new(bounds, cell);
    let mut report = FaceReport::default();
    let Some(frame) = FaceFrame::new(&spec.anchors) else { return (field, report) };
    // The face's own bounds: every part is clipped to the FACE_OVAL hull, and the outline,
    // nose and brow parts cannot move the eyes or the mouth — the field used to run to the
    // canvas edge, dragging hair and background with the cheek.
    let mask = FaceMask::new(&frame, &field);
    let mut asked: Vec<(&'static FacePart, f64, Shape)> = Vec::new();
    let sliders = spec.sliders.composed();
    for part in FACE_PARTS {
        let slider = (part.slider)(&sliders);
        if slider.abs() < 0.5 {
            continue;
        }
        let Some(shape) = (part.shape)(&frame) else {
            report.skipped.push((part.name, "the face landmarks are too close to name an axis"));
            continue;
        };
        asked.push((part, f64::from(slider / 100.0).clamp(-1.0, 1.0), shape));
    }
    // The face's fold budgets: the shaping families (contract and radial — the outline
    // sliders are alternatives by design, exactly the length family's case) and the
    // ride families (shift and ramp) each share a pool. A face is a hundred pixels where
    // a body is a thousand and its feathers are short; solo sliders sit well inside
    // either pool, and stacked, the family scales down together, not apart.
    let contract_pool = 0.35 * frame.face_w;
    let ride_pool = 0.10 * frame.face_w;
    let contract_sum: f64 = asked
        .iter()
        .map(|&(_, k, ref s)| match s {
            Shape::Contract { width, gain, .. } => gain.abs() * k.abs() * width,
            Shape::Radial { radius, gain, .. } => gain.abs() * k.abs() * radius,
            _ => 0.0,
        })
        .sum();
    let ride_sum: f64 = asked
        .iter()
        .map(|&(_, k, ref s)| match s {
            Shape::Ramp { a, b, gain, .. } | Shape::Shift { a, b, gain, .. } => gain.abs() * k.abs() * len_of(down(*b, *a)),
            _ => 0.0,
        })
        .sum();
    let (c_scale, r_scale) = (
        if contract_sum > contract_pool && contract_sum > 0.0 { contract_pool / contract_sum } else { 1.0 },
        if ride_sum > ride_pool && ride_sum > 0.0 { ride_pool / ride_sum } else { 1.0 },
    );
    for (part, k, shape) in asked {
        let k = match shape {
            Shape::Contract { .. } | Shape::Radial { .. } => k * c_scale,
            Shape::Ramp { .. } | Shape::Shift { .. } => k * r_scale,
        };
        match &mask {
            Some(mask) => {
                let (protect_eyes, protect_lips) = protects(part.name);
                let weight = mask.weights(protect_eyes, protect_lips);
                write_shape(&mut field, &shape, k, Some(&weight));
            }
            None => write_shape(&mut field, &shape, k, None),
        }
        report.applied.push(part.name);
    }
    (field, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CELL: f64 = 2.0;

    fn bounds() -> Rect {
        Rect::new(0, 0, 512, 512)
    }

    /// A canonical face: the indices the table reads, placed on the standard front-view
    /// layout (a 240 px-wide face about (256, 240)); the rest of the 478 park at the
    /// centre, as FaceMesh's own no-face fallback does — the frame only reads named ones.
    fn face() -> FaceAnchors {
        let mut pts = vec![[256.0f32, 240.0f32, 0.0f32]; MESH_COUNT];
        // The mask rings first: the face's own outline (an ellipse through forehead and
        // chin) and the lip ring, so the clip and the protection have their polygons. The
        // named anchors below keep their exact layout over the rings'.
        let ring = |ids: &[usize], cx: f64, cy: f64, rx: f64, ry: f64, pts: &mut Vec<[f32; 3]>| {
            for (k, &i) in ids.iter().enumerate() {
                let t = -std::f64::consts::FRAC_PI_2 + std::f64::consts::TAU * k as f64 / ids.len() as f64;
                pts[i] = [(cx + rx * t.cos()) as f32, (cy + ry * t.sin()) as f32, 0.0];
            }
        };
        ring(mesh::FACE_OVAL, 256.0, 244.0, 120.0, 148.0, &mut pts);
        ring(mesh::LIPS_RING, 256.0, 343.0, 40.0, 13.0, &mut pts);
        let mut set = |i: usize, x: f64, y: f64| pts[i] = [x as f32, y as f32, 0.0];
        set(mesh::FOREHEAD_TOP, 256.0, 96.0);
        set(mesh::CHIN, 256.0, 392.0);
        set(mesh::FACE_SIDE_L, 376.0, 248.0);
        set(mesh::FACE_SIDE_R, 136.0, 248.0);
        set(mesh::TEMPLE_L, 362.0, 180.0);
        set(mesh::TEMPLE_R, 150.0, 180.0);
        set(mesh::CHEEKBONE_L, 332.0, 268.0);
        set(mesh::CHEEKBONE_R, 180.0, 268.0);
        // left eye (person's left, image right): centre (306, 232), width 30
        for (k, &i) in mesh::EYE_RING_L.iter().enumerate() {
            let t = k as f64 / mesh::EYE_RING_L.len() as f64;
            set(i, 306.0 + 15.0 * (t * std::f64::consts::TAU).cos(), 232.0 + 8.0 * (t * std::f64::consts::TAU).sin());
        }
        set(mesh::EYE_OUTER_L, 321.0, 232.0);
        set(mesh::EYE_INNER_L, 291.0, 232.0);
        set(mesh::EYE_LOW_L, 306.0, 240.0);
        set(mesh::IRIS_L, 306.0, 232.0);
        for (k, &i) in mesh::EYE_RING_R.iter().enumerate() {
            let t = k as f64 / mesh::EYE_RING_R.len() as f64;
            set(i, 206.0 + 15.0 * (t * std::f64::consts::TAU).cos(), 232.0 + 8.0 * (t * std::f64::consts::TAU).sin());
        }
        set(mesh::EYE_OUTER_R, 191.0, 232.0);
        set(mesh::EYE_INNER_R, 221.0, 232.0);
        set(mesh::EYE_LOW_R, 206.0, 240.0);
        set(mesh::IRIS_R, 206.0, 232.0);
        // brows
        set(mesh::BROW_OUT_L, 344.0, 198.0);
        set(mesh::BROW_MID_L, 312.0, 186.0);
        set(mesh::BROW_IN_L, 284.0, 192.0);
        set(mesh::BROW_INNER_L, 272.0, 196.0);
        set(mesh::BROW_OUT_R, 168.0, 198.0);
        set(mesh::BROW_MID_R, 200.0, 186.0);
        set(mesh::BROW_IN_R, 228.0, 192.0);
        set(mesh::BROW_INNER_R, 240.0, 196.0);
        // nose
        set(mesh::NOSE_BRIDGE_TOP, 256.0, 222.0);
        set(mesh::NOSE_TIP, 256.0, 306.0);
        set(mesh::NOSE_BASE, 256.0, 314.0);
        set(mesh::ALAR_L, 280.0, 308.0);
        set(mesh::ALAR_R, 232.0, 308.0);
        // mouth
        set(mesh::MOUTH_CORNER_L, 296.0, 342.0);
        set(mesh::MOUTH_CORNER_R, 216.0, 342.0);
        set(mesh::UPPER_LIP_TOP, 256.0, 330.0);
        set(mesh::UPPER_LIP_LOW, 256.0, 338.0);
        set(mesh::LOWER_LIP_TOP, 256.0, 348.0);
        set(mesh::LOWER_LIP_LOW, 256.0, 356.0);
        FaceAnchors { doc: [512.0, 512.0], pts }
    }

    fn build(sliders: FaceSliders) -> (LiquifyField, FaceReport) {
        face_field(&FaceSpec { sliders, anchors: face() }, bounds(), CELL)
    }

    #[test]
    fn the_presets_blend_their_bundles_into_the_manual_sliders() {
        let near = |a: f32, b: f32| (a - b).abs() < 0.01;
        // 经典 at 100: slim 25, eyes +15; the untouched zones stay.
        let c = FaceSliders { preset: FacePreset::Classic, preset_strength: 100.0, ..Default::default() }.composed();
        assert!(near(c.face_slim, 25.0) && near(c.eye_size, 15.0));
        assert!(near(c.jaw, 0.0) && near(c.mouth_size, 0.0));
        // Strength scales: 50 halves the bundle; manual values stack and clamp.
        let c = FaceSliders { preset: FacePreset::Classic, preset_strength: 50.0, face_slim: 20.0, ..Default::default() }.composed();
        assert!(near(c.face_slim, 32.5) && near(c.eye_size, 7.5));
        let c = FaceSliders { preset: FacePreset::Classic, preset_strength: 100.0, face_slim: 90.0, ..Default::default() }.composed();
        assert!(near(c.face_slim, 100.0), "the stop clamps");
        // 长脸 shortens the middle third; 方脸 works the jaw.
        let c = FaceSliders { preset: FacePreset::Long, preset_strength: 100.0, ..Default::default() }.composed();
        assert!(near(c.mid_face, -30.0) && near(c.lower_face, -25.0));
        let c = FaceSliders { preset: FacePreset::Square, preset_strength: 100.0, ..Default::default() }.composed();
        assert!(near(c.jaw, 40.0) && near(c.chin_length, 20.0));
        // A preset alone moves the field; zero strength does not.
        let (f, _) = build(FaceSliders { preset: FacePreset::Round, preset_strength: 100.0, ..Default::default() });
        assert!(!f.is_identity());
        let (f, _) = build(FaceSliders { preset: FacePreset::Round, preset_strength: 0.0, ..Default::default() });
        assert!(f.is_identity());
    }

    #[test]
    fn a_side_override_takes_only_its_own_row() {
        // The base slider moves both temples; the left override answers only the left row.
        let mut sliders = FaceSliders { temples: 60.0, ..Default::default() };
        let (both, _) = build(sliders);
        sliders.sides.temples[SIDE_L] = Some(-20.0);
        let (split, _) = build(sliders);
        let at = |f: &LiquifyField, x: f64, y: f64| {
            let d = f.sample(x, y);
            d[0].hypot(d[1])
        };
        let left = |f: &LiquifyField| at(f, 362.0, 180.0);
        let right = |f: &LiquifyField| at(f, 150.0, 180.0);
        let (bl, br) = (left(&both), right(&both));
        assert!((bl - br).abs() < 0.2, "linked sides match: {bl:?} {br:?}");
        let (sl, sr) = (left(&split), right(&split));
        assert!(sl < bl * 0.6, "the overridden side reads its own value: {sl:?} vs {bl:?}");
        assert!((sr - br).abs() < 0.2, "the other side follows the base: {sr:?} vs {br:?}");
    }

    #[test]
    fn symmetry_levels_the_uneven_eyes() {
        use crate::contour::align_pairs;
        let mut pts = vec![[256.0f64, 240.0f64]; MESH_COUNT];
        // Both eyes at 232, but the left one sits 8 px lower — the pair's asymmetry.
        pts[263] = [321.0, 240.0];
        pts[33] = [191.0, 232.0];
        let (l, r) = (pts[263], pts[33]);
        let axis = [256.0, 240.0];
        let mut field = LiquifyField::new(bounds(), CELL);
        align_pairs(&mut field, &pts, &[(263, 33)], axis, [0.0, -1.0], 100.0, 40.0);
        let dl = field.sample(l[0], l[1]);
        let dr = field.sample(r[0], r[1]);
        // The lower eye moves up, the higher one moves down: both dipoles oppose the
        // 8 px gap, closing it at strength 100 (4 px of correction each, inverse-mapped).
        assert!(dl[1] > 2.0, "the low eye reads lifted: {dl:?}");
        assert!(dr[1] < -2.0, "the high eye reads lowered: {dr:?}");
        // A straight run: the perfectly mirrored pair corrects nothing.
        pts[263] = [321.0, 240.0];
        pts[33] = [191.0, 240.0];
        let mut flat = LiquifyField::new(bounds(), CELL);
        align_pairs(&mut flat, &pts, &[(263, 33)], axis, [0.0, -1.0], 100.0, 40.0);
        assert!(flat.is_identity(), "a symmetric pair needs no correction");
    }

    #[test]
    fn zero_sliders_build_the_identity_field() {
        let (field, report) = build(FaceSliders::default());
        assert!(field.is_identity());
        assert!(report.applied.is_empty() && report.skipped.is_empty());
    }

    #[test]
    fn the_same_spec_builds_the_same_field_twice() {
        let sliders = FaceSliders { face_slim: 40.0, eye_size: -30.0, chin_length: 50.0, ..FaceSliders::default() };
        let (a, ra) = build(sliders);
        let (b, rb) = build(sliders);
        assert_eq!(a, b);
        assert_eq!(ra, rb);
    }

    #[test]
    fn every_slider_alone_moves_the_field() {
        let all: Vec<(&str, FaceSliders)> = vec![
            ("face_slim", FaceSliders { face_slim: 100.0, ..Default::default() }),
            ("temples", FaceSliders { temples: 100.0, ..Default::default() }),
            ("cheekbones", FaceSliders { cheekbones: 100.0, ..Default::default() }),
            ("jaw", FaceSliders { jaw: 100.0, ..Default::default() }),
            ("face_size", FaceSliders { face_size: 100.0, ..Default::default() }),
            ("v_face", FaceSliders { v_face: 100.0, ..Default::default() }),
            ("face_width", FaceSliders { face_width: 100.0, ..Default::default() }),
            ("hairline", FaceSliders { hairline: 100.0, ..Default::default() }),
            ("forehead_width", FaceSliders { forehead_width: 100.0, ..Default::default() }),
            ("forehead_height", FaceSliders { forehead_height: 100.0, ..Default::default() }),
            ("philtrum", FaceSliders { philtrum: 100.0, ..Default::default() }),
            ("mid_face", FaceSliders { mid_face: 100.0, ..Default::default() }),
            ("lower_face", FaceSliders { lower_face: 100.0, ..Default::default() }),
            ("chin_length", FaceSliders { chin_length: 100.0, ..Default::default() }),
            ("chin_point", FaceSliders { chin_point: 100.0, ..Default::default() }),
            ("chin_retract", FaceSliders { chin_retract: 100.0, ..Default::default() }),
            ("eye_size", FaceSliders { eye_size: 100.0, ..Default::default() }),
            ("pupil_size", FaceSliders { pupil_size: 100.0, ..Default::default() }),
            ("eye_height", FaceSliders { eye_height: 100.0, ..Default::default() }),
            ("eye_width", FaceSliders { eye_width: 100.0, ..Default::default() }),
            ("eye_spacing", FaceSliders { eye_spacing: 100.0, ..Default::default() }),
            ("inner_corner", FaceSliders { inner_corner: 100.0, ..Default::default() }),
            ("outer_corner", FaceSliders { outer_corner: 100.0, ..Default::default() }),
            ("lower_lid", FaceSliders { lower_lid: 100.0, ..Default::default() }),
            ("eye_tilt", FaceSliders { eye_tilt: 100.0, ..Default::default() }),
            ("eye_position", FaceSliders { eye_position: 100.0, ..Default::default() }),
            ("nose_size", FaceSliders { nose_size: 100.0, ..Default::default() }),
            ("nose_height", FaceSliders { nose_height: 100.0, ..Default::default() }),
            ("nose_shift", FaceSliders { nose_shift: 100.0, ..Default::default() }),
            ("nose_bridge", FaceSliders { nose_bridge: 100.0, ..Default::default() }),
            ("nose_wings", FaceSliders { nose_wings: 100.0, ..Default::default() }),
            ("nose_tip", FaceSliders { nose_tip: 100.0, ..Default::default() }),
            ("mouth_size", FaceSliders { mouth_size: 100.0, ..Default::default() }),
            ("mouth_width", FaceSliders { mouth_width: 100.0, ..Default::default() }),
            ("mouth_height", FaceSliders { mouth_height: 100.0, ..Default::default() }),
            ("mouth_shift", FaceSliders { mouth_shift: 100.0, ..Default::default() }),
            ("gum_smile", FaceSliders { gum_smile: 100.0, ..Default::default() }),
            ("lip_tilt", FaceSliders { lip_tilt: 100.0, ..Default::default() }),
            ("upper_lip", FaceSliders { upper_lip: 100.0, ..Default::default() }),
            ("lower_lip", FaceSliders { lower_lip: 100.0, ..Default::default() }),
            ("brow_head", FaceSliders { brow_head: 100.0, ..Default::default() }),
            ("brow_tail", FaceSliders { brow_tail: 100.0, ..Default::default() }),
            ("brow_thickness", FaceSliders { brow_thickness: 100.0, ..Default::default() }),
            ("brow_spacing", FaceSliders { brow_spacing: 100.0, ..Default::default() }),
            ("brow_tilt", FaceSliders { brow_tilt: 100.0, ..Default::default() }),
            ("brow_shape", FaceSliders { brow_shape: 100.0, ..Default::default() }),
            ("brow_height", FaceSliders { brow_height: 100.0, ..Default::default() }),
        ];
        for (name, sliders) in all {
            let (field, report) = build(sliders);
            assert!(!field.is_identity(), "{name} left the field untouched");
            assert!(report.skipped.is_empty(), "{name} skipped {:?}", report.skipped);
        }
    }

    #[test]
    fn the_face_slims_toward_its_axis_and_the_corners_stay() {
        let (field, _) = build(FaceSliders { face_slim: 100.0, ..Default::default() });
        let right_rim = field.sample(330.0, 340.0); // inside the slim band and the face, to the axis's right
        assert!(right_rim[0] > 0.2, "the rim field must point outward (reads pulled in): {right_rim:?}");
        let corner = field.sample(10.0, 10.0);
        assert!(corner[0].abs() < 1e-6 && corner[1].abs() < 1e-6, "the corner stays: {corner:?}");
    }

    /// The field belongs to the face: a whole-face part must not reach the hair or the
    /// background outside the FACE_OVAL ring, and it must leave the eyes and the mouth
    /// still for the parts that own them. Before the clip, `face_size` pulled on the whole
    /// canvas and on the eyes and lips as if they were cheek.
    #[test]
    fn the_outline_parts_stop_at_the_face_and_leave_the_features_still() {
        let (field, _) = build(FaceSliders { face_size: 100.0, ..Default::default() });
        let at = |x: f64, y: f64| {
            let d = field.sample(x, y);
            d[0].hypot(d[1])
        };
        for (x, y) in [(40.0, 40.0), (480.0, 480.0), (256.0, 470.0), (40.0, 244.0), (470.0, 244.0)] {
            assert!(at(x, y) < 0.5, "outside the face stays still at ({x}, {y}): {}", at(x, y));
        }
        // The centres and points just inside each ring's outline: the eye ring is only
        // sixteen pixels tall on this face, and a protection ramp centred on the outline
        // left its inner edge free to ride the outline sliders.
        for (x, y) in [(306.0, 232.0), (206.0, 232.0), (256.0, 343.0), (306.0, 238.0), (206.0, 226.0), (256.0, 333.0)] {
            assert!(at(x, y) < 0.5, "the protected feature at ({x}, {y}) must not ride the outline: {}", at(x, y));
        }
        assert!(at(330.0, 340.0) > 1.0, "the cheek inside the face still shapes: {}", at(330.0, 340.0));
    }

    /// The protection is per feature, not a blanket freeze: the mouth rows still move the
    /// lips they name.
    #[test]
    fn the_mouth_parts_still_move_the_lips_they_own() {
        let (field, _) = build(FaceSliders { mouth_size: 100.0, ..Default::default() });
        let corner = field.sample(296.0, 342.0);
        assert!(corner[0].hypot(corner[1]) > 0.5, "the mouth corner moves: {corner:?}");
    }

    #[test]
    fn eye_size_expands_about_the_eye_centres() {
        let (field, _) = build(FaceSliders { eye_size: 100.0, ..Default::default() });
        let rim = field.sample(321.0, 232.0); // the left eye's outer corner
        assert!(rim[0] < -0.2, "enlarging points the field inward (bloat): {rim:?}");
        let bridge = field.sample(256.0, 226.0); // between the eyes
        let far = field.sample(60.0, 60.0);
        assert!(far[0].abs() < 1e-6 && far[1].abs() < 1e-6, "the corner stays: {far:?}");
        let _ = bridge;
    }

    #[test]
    fn philtrum_lifts_the_lips_band() {
        let (field, _) = build(FaceSliders { philtrum: 100.0, ..Default::default() });
        let lip = field.sample(256.0, 322.0); // inside the philtrum band
        assert!(lip[1] > 0.5, "shortening points the field down (the lip reads lifted): {lip:?}");
    }

    /// The fold guard, the body table's own: the face's cell is finer, the parts smaller.
    #[test]
    fn every_slider_at_full_superposes_smoothly() {
        let sliders = FaceSliders {
            face_slim: 100.0,
            temples: 100.0,
            cheekbones: 100.0,
            jaw: 100.0,
            face_size: 100.0,
            v_face: 100.0,
            face_width: 100.0,
            hairline: 100.0,
            forehead_width: 100.0,
            forehead_height: 100.0,
            philtrum: 100.0,
            mid_face: 100.0,
            lower_face: 100.0,
            chin_length: 100.0,
            chin_point: 100.0,
            chin_retract: 100.0,
            eye_size: 100.0,
            pupil_size: 100.0,
            eye_height: 100.0,
            eye_width: 100.0,
            eye_spacing: 100.0,
            inner_corner: 100.0,
            outer_corner: 100.0,
            lower_lid: 100.0,
            eye_tilt: 100.0,
            eye_position: 100.0,
            nose_size: 100.0,
            nose_height: 100.0,
            nose_shift: 100.0,
            nose_bridge: 100.0,
            nose_wings: 100.0,
            nose_tip: 100.0,
            mouth_size: 100.0,
            mouth_width: 100.0,
            mouth_height: 100.0,
            mouth_shift: 100.0,
            gum_smile: 100.0,
            lip_tilt: 100.0,
            upper_lip: 100.0,
            lower_lip: 100.0,
            brow_head: 100.0,
            brow_tail: 100.0,
            brow_thickness: 100.0,
            brow_spacing: 100.0,
            brow_tilt: 100.0,
            brow_shape: 100.0,
            brow_height: 100.0,
            preset: FacePreset::None,
            preset_strength: 0.0,
            sides: FaceSides::default(),
        };
        let (field, report) = build(sliders);
        assert!(report.skipped.is_empty(), "{:?}", report.skipped);
        let mut worst = (0.0f32, 0, 0);
        for j in 0..field.h {
            for i in 0..field.w {
                let d = field.d[j * field.w + i];
                if i + 1 < field.w {
                    let r = field.d[j * field.w + i + 1];
                    let step = (d[0] - r[0]).abs().max((d[1] - r[1]).abs());
                    if step > worst.0 {
                        worst = (step, i, j);
                    }
                }
                if j + 1 < field.h {
                    let dn = field.d[(j + 1) * field.w + i];
                    let step = (d[0] - dn[0]).abs().max((d[1] - dn[1]).abs());
                    if step > worst.0 {
                        worst = (step, i, j);
                    }
                }
            }
        }
        let (worst, i, j) = worst;
        let p = field.node_pos(i, j);
        assert!((worst as f64) < CELL, "adjacent nodes step {worst:.2}px past the {CELL}px cell at ({}, {}) — the field folds", p[0], p[1]);
    }

    #[test]
    fn the_quality_reads_an_upright_face_as_straight_on() {
        let q = face().quality().expect("a full mesh has a quality");
        assert!((q.face_width - 240.0).abs() < 0.5, "the face's width: {}", q.face_width);
        assert!((q.face_height - 296.0).abs() < 0.5, "the face's height: {}", q.face_height);
        assert!(q.roll_deg.abs() < 0.01, "the eye line is level: {}", q.roll_deg);
        assert!(q.turn_deg.abs() < 0.01, "the nose is on the axis: {}", q.turn_deg);
    }

    /// A turned head moves the nose off the face's mid-axis; a tilted one lowers an eye. The
    /// panel warns on both, so the reading must follow the landmarks rather than sit at zero.
    #[test]
    fn the_quality_follows_a_turned_and_a_tilted_head() {
        let mut anchors = face();
        anchors.pts[mesh::NOSE_TIP][0] += 24.0;
        let q = anchors.quality().expect("a full mesh has a quality");
        assert!((q.turn_deg - 11.5).abs() < 1.0, "a 24 px nose offset is the turn the panel reads: {}", q.turn_deg);

        let mut anchors = face();
        anchors.pts[mesh::EYE_OUTER_L][1] += 10.0;
        anchors.pts[mesh::EYE_OUTER_R][1] -= 10.0;
        let q = anchors.quality().expect("a full mesh has a quality");
        assert!((q.roll_deg - 8.75).abs() < 0.5, "a 20 px eye-line rise over 130 px is the roll: {}", q.roll_deg);

        let mut short = face();
        short.pts.truncate(MESH_COUNT - 1);
        assert!(short.quality().is_none(), "a truncated mesh is not a face");
    }
}
