//! The portrait pipeline's recipe: which stages run and the user-tunable values they carry.
//!
//! The stages form a DAG (docs/fork/2026-11-18-portrait-pipeline.md), not a chain: analysis
//! stages (Cutout, Parsing, Landmarks) cache region metadata the execution stages consume in
//! parallel branches, and the grade runs last over the composited result. [`StageId`] declares
//! each node's regions so a later per-stage cache can invalidate exactly what a change touches.
//!
//! A recipe is engine-validated input, not an interchange format: every field carries
//! `#[serde(default)]` so a stored recipe opens as defaults change by change, but an unknown
//! stage id fails loudly rather than being silently dropped.

use serde::{Deserialize, Serialize};

/// A semantic region a stage consumes or produces: the per-stage cache invalidation key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Region {
    Person,
    Skin,
    Hair,
    Face,
    Eyes,
    Lips,
    Clothes,
    Background,
    Whole,
}

/// One node of the portrait DAG. Parsing and Landmarks are typed stubs: their regions and
/// dependencies are declared, no analysis populates them yet (`!implemented()`), and the
/// plan builder leaves their layers out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StageId {
    Cutout,
    Parsing,
    Landmarks,
    Enhance,
    Skin,
    HairColor,
    Inpaint,
    Grade,
}

impl StageId {
    /// The regions this stage reads; empty for the analysis roots.
    pub fn consumes(self) -> &'static [Region] {
        match self {
            StageId::Cutout | StageId::Parsing | StageId::Landmarks => &[],
            // The face GAN regenerates skin texture, so skin work runs after it, never before.
            StageId::Enhance => &[Region::Face],
            StageId::Skin => &[Region::Skin],
            StageId::HairColor => &[Region::Hair],
            StageId::Inpaint => &[Region::Clothes, Region::Background],
            StageId::Grade => &[Region::Whole],
        }
    }

    /// The regions this stage's analysis yields; empty for the execution stages, whose output
    /// is layers rather than metadata.
    pub fn produces(self) -> &'static [Region] {
        match self {
            StageId::Cutout => &[Region::Person],
            StageId::Parsing => &[Region::Skin, Region::Hair, Region::Face, Region::Eyes, Region::Lips, Region::Clothes, Region::Background],
            StageId::Landmarks => &[],
            StageId::Enhance | StageId::Skin | StageId::HairColor | StageId::Inpaint | StageId::Grade => &[],
        }
    }

    /// Whether this build can run the stage. Only the person cutout is wired today; the rest
    /// stay declared-but-off until their models land on the ladder.
    pub fn implemented(self) -> bool {
        matches!(self, StageId::Cutout)
    }
}

/// The Hue/Saturation values the person-masked adjustment layer carries (Photoshop's ranges).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HueSat {
    pub hue: f32,
    pub saturation: f32,
    pub lightness: f32,
}

impl Default for HueSat {
    fn default() -> Self {
        HueSat { hue: 0.0, saturation: 0.0, lightness: 0.0 }
    }
}

/// What the user asked the pipeline to do. The defaults describe the vertical slice the
/// skeleton ships: analyze the person, then one Hue/Saturation layer masked to it inside a
/// named group. `hair` opts into a second Hue/Saturation layer masked to the hair region —
/// it only materializes when the parsing analysis produced one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Recipe {
    pub stages: Vec<StageId>,
    pub group_name: String,
    pub hue_saturation: HueSat,
    pub hair: Option<HueSat>,
}

impl Default for Recipe {
    fn default() -> Self {
        Recipe { stages: vec![StageId::Cutout], group_name: "Portrait".into(), hue_saturation: HueSat::default(), hair: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_the_person_slice() {
        let r = Recipe::default();
        assert_eq!(r.stages, [StageId::Cutout]);
        assert_eq!(r.group_name, "Portrait");
        assert_eq!(r.hue_saturation, HueSat { hue: 0.0, saturation: 0.0, lightness: 0.0 });
        assert!(StageId::Cutout.implemented());
        assert!(
            [StageId::Parsing, StageId::Landmarks, StageId::Enhance, StageId::Skin, StageId::HairColor, StageId::Inpaint, StageId::Grade]
                .iter()
                .all(|s| !s.implemented())
        );
    }

    #[test]
    fn recipe_round_trips_and_defaults_missing_fields() {
        let full = Recipe {
            stages: vec![StageId::Cutout, StageId::HairColor],
            group_name: "Studio".into(),
            hue_saturation: HueSat { hue: 30.0, saturation: -20.0, lightness: 5.0 },
            hair: Some(HueSat { hue: -15.0, saturation: 40.0, lightness: 0.0 }),
        };
        let json = serde_json::to_string(&full).unwrap();
        assert_eq!(serde_json::from_str::<Recipe>(&json).unwrap(), full);
        // An empty object opens as the defaults, field by field.
        let sparse: Recipe = serde_json::from_str("{}").unwrap();
        assert_eq!(sparse.stages, Recipe::default().stages);
        assert_eq!(sparse.group_name, Recipe::default().group_name);
        // An unknown stage id fails loudly: a recipe is validated input, not best-effort data.
        assert!(serde_json::from_str::<Recipe>(&json.replace("HairColor", "Tomorrow")).is_err());
    }

    #[test]
    fn every_region_some_stage_produces() {
        // Each region is named as some stage's output, so no region can be orphaned by typo.
        let all = [Region::Person, Region::Skin, Region::Hair, Region::Face, Region::Eyes, Region::Lips, Region::Clothes, Region::Background, Region::Whole];
        for r in all {
            assert!(
                [StageId::Cutout, StageId::Parsing, StageId::Landmarks, StageId::Enhance, StageId::Skin, StageId::HairColor, StageId::Inpaint, StageId::Grade]
                    .iter()
                    .any(|s| s.produces().contains(&r) || s.consumes().contains(&r)),
                "{r:?} is declared by no stage"
            );
        }
    }
}
