//! The plan: what `portrait.create` materializes. Standard layers only — the whole design
//! point (docs/fork/2026-11-18-portrait-pipeline.md) is that the pipeline's output is a group
//! of ordinary Photoshop-shaped layers the user keeps tuning, never a new layer kind.

use serde_json::Value;

use crate::analysis::Analysis;
use crate::recipe::{HueSat, Recipe, Region};

/// One layer of the plan. `kind` is the engine's adjustment wire name ("hueSaturation", …)
/// and `params` its JSON, so the engine materializes through the same `from_params` path a
/// `layer.newAdjustmentLayer` command takes. Masks stay in the [`Analysis`], referenced by
/// region: the plan is pure data, cheap to rebuild.
#[derive(Clone, Debug, PartialEq)]
pub enum PlannedLayer {
    Group {
        name: String,
        children: Vec<PlannedLayer>,
    },
    Adjustment {
        name: String,
        kind: String,
        params: Value,
        mask: Option<Region>,
    },
    /// A rasterized stage result (face enhance, inpainting); the placeholder the generative
    /// stages will land in. The default plan emits none.
    Raster {
        name: String,
        mask: Option<Region>,
    },
}

/// The layer tree the recipe asks for, given what the analysis found.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerPlan {
    pub group_name: String,
    pub layers: Vec<PlannedLayer>,
}

/// Recipe + analysis → layer tree. An unimplemented stage contributes nothing, and a stage
/// whose region the analysis lacks contributes nothing: the plan is what can actually run.
pub fn plan(recipe: &Recipe, analysis: &Analysis) -> LayerPlan {
    let mut layers = Vec::new();
    if analysis.mask(Region::Person).is_some() {
        let HueSat { hue, saturation, lightness } = recipe.hue_saturation;
        layers.push(PlannedLayer::Adjustment {
            name: "Hue/Saturation".into(),
            kind: "hueSaturation".into(),
            params: serde_json::json!({ "hue": hue, "saturation": saturation, "lightness": lightness }),
            mask: Some(Region::Person),
        });
    }
    // The hair recolor rides on the parsing analysis: no hair region, no layer.
    if let Some(HueSat { hue, saturation, lightness }) = recipe.hair
        && analysis.mask(Region::Hair).is_some()
    {
        layers.push(PlannedLayer::Adjustment {
            name: "Hair Color".into(),
            kind: "hueSaturation".into(),
            params: serde_json::json!({ "hue": hue, "saturation": saturation, "lightness": lightness }),
            mask: Some(Region::Hair),
        });
    }
    LayerPlan { group_name: recipe.group_name.clone(), layers }
}

#[cfg(test)]
mod tests {
    use photocraft_doc::PixelFormat;
    use photocraft_raster::Surface;

    use super::*;
    use crate::recipe::StageId;

    fn analysis_with_person(person: bool) -> Analysis {
        let mut a = Analysis::new(16, 16);
        if person {
            a.set(Region::Person, Surface::with_default(PixelFormat::GRAY8, &[1.0]));
        }
        a
    }

    #[test]
    fn default_recipe_plans_one_person_masked_hue_layer() {
        let recipe = Recipe::default();
        let planned = plan(&recipe, &analysis_with_person(true));
        assert_eq!(planned.group_name, "Portrait");
        assert_eq!(
            planned.layers,
            [PlannedLayer::Adjustment {
                name: "Hue/Saturation".into(),
                kind: "hueSaturation".into(),
                params: serde_json::json!({ "hue": 0.0, "saturation": 0.0, "lightness": 0.0 }),
                mask: Some(Region::Person),
            }]
        );
    }

    #[test]
    fn hue_values_carry_through_to_the_planned_params() {
        let recipe = Recipe {
            stages: vec![StageId::Cutout],
            group_name: "Studio".into(),
            hue_saturation: HueSat { hue: 30.0, saturation: -20.0, lightness: 5.0 },
            hair: None,
        };
        let planned = plan(&recipe, &analysis_with_person(true));
        let PlannedLayer::Adjustment { name, kind, params, .. } = &planned.layers[0] else { panic!("not an adjustment") };
        assert_eq!((name.as_str(), kind.as_str()), ("Hue/Saturation", "hueSaturation"));
        assert_eq!((params["hue"].as_f64(), params["saturation"].as_f64(), params["lightness"].as_f64()), (Some(30.0), Some(-20.0), Some(5.0)));
    }

    #[test]
    fn no_person_analysis_plans_no_layers() {
        let planned = plan(&Recipe::default(), &analysis_with_person(false));
        assert!(planned.layers.is_empty());
    }

    #[test]
    fn hair_layer_needs_both_the_ask_and_the_hair_region() {
        let mut recipe = Recipe::default();
        recipe.hair = Some(HueSat { hue: -15.0, saturation: 40.0, lightness: 1.0 });
        // Asked but not analyzed: nothing.
        assert_eq!(plan(&recipe, &analysis_with_person(true)).layers.len(), 1);
        // Analyzed but not asked: nothing (the default recipe).
        let mut analysis = analysis_with_person(true);
        analysis.set(Region::Hair, Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[1.0]));
        assert_eq!(plan(&Recipe::default(), &analysis).layers.len(), 1);
        // Both: the hair layer lands second, masked to Hair.
        let planned = plan(&recipe, &analysis);
        assert_eq!(planned.layers.len(), 2);
        let PlannedLayer::Adjustment { name, kind, params, mask } = &planned.layers[1] else { panic!("not an adjustment") };
        assert_eq!((name.as_str(), kind.as_str(), *mask), ("Hair Color", "hueSaturation", Some(Region::Hair)));
        assert_eq!((params["hue"].as_f64(), params["saturation"].as_f64(), params["lightness"].as_f64()), (Some(-15.0), Some(40.0), Some(1.0)));
    }
}
