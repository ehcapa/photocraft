//! The plan: what `portrait.create` materializes. Standard layers only — the whole design
//! point (docs/fork/2026-11-18-portrait-pipeline.md) is that the pipeline's output is a group
//! of ordinary Photoshop-shaped layers the user keeps tuning, never a new layer kind.

use serde_json::Value;

use crate::analysis::Analysis;
use crate::recipe::{Grade, HueSat, Recipe, Region};

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
    // The restoration sits between the colour layers and the smoothing: the GAN redraws
    // skin texture, so anything later must see it (the engine computes the smoothing from
    // the enhanced composite when both are asked for). Needs the face region the landmark
    // hull provides.
    if recipe.enhance && analysis.mask(Region::Face).is_some() {
        layers.push(PlannedLayer::Raster { name: "Face Enhance".into(), mask: Some(Region::Face) });
    }
    // The smoothed skin rides on the parsing analysis like the hair layer does. The
    // raster's pixels are the engine's to fill (a Surface-Blur of the composite); the plan
    // carries the ask and the mask.
    if recipe.skin.is_some() && analysis.mask(Region::Skin).is_some() {
        layers.push(PlannedLayer::Raster { name: "Skin Smoothing".into(), mask: Some(Region::Skin) });
    }
    // The facial features: the eye rings brighten and the lip ring takes a Colourize —
    // both ride the landmark rings (model-free), landing under the grade.
    if let Some(amount) = recipe.eyes
        && analysis.mask(Region::Eyes).is_some()
    {
        layers.push(PlannedLayer::Adjustment {
            name: "Eye Brighten".into(),
            kind: "brightnessContrast".into(),
            params: serde_json::json!({ "brightness": amount }),
            mask: Some(Region::Eyes),
        });
    }
    if let Some(HueSat { hue, saturation, lightness }) = recipe.lips
        && analysis.mask(Region::Lips).is_some()
    {
        layers.push(PlannedLayer::Adjustment {
            name: "Lip Color".into(),
            kind: "hueSaturation".into(),
            params: serde_json::json!({ "hue": hue, "saturation": saturation, "lightness": lightness, "colorize": true }),
            mask: Some(Region::Lips),
        });
    }
    // The makeup pair rides the landmark geometry like the features do.
    if let Some(HueSat { hue, saturation, lightness }) = recipe.blush
        && analysis.mask(Region::Cheeks).is_some()
    {
        layers.push(PlannedLayer::Adjustment {
            name: "Blush".into(),
            kind: "hueSaturation".into(),
            params: serde_json::json!({ "hue": hue, "saturation": saturation, "lightness": lightness, "colorize": true }),
            mask: Some(Region::Cheeks),
        });
    }
    if let Some(HueSat { hue, saturation, lightness }) = recipe.eyeshadow
        && analysis.mask(Region::Eyelids).is_some()
    {
        layers.push(PlannedLayer::Adjustment {
            name: "Eyeshadow".into(),
            kind: "hueSaturation".into(),
            params: serde_json::json!({ "hue": hue, "saturation": saturation, "lightness": lightness, "colorize": true }),
            mask: Some(Region::Eyelids),
        });
    }
    // The colour match runs last (topmost): the Curves points carry the transfer the
    // engine computed from the reference, masked to the person like the rest of the
    // portrait's work.
    if let Some(grade) = recipe.grade
        && analysis.mask(Region::Person).is_some()
    {
        let curves = |g: &Grade, c: usize| {
            let (slope, offset) = (g.slope[c] as f64, g.offset[c] as f64);
            serde_json::json!([[0, (offset * 255.0).round().clamp(0.0, 255.0)], [255, ((slope + offset) * 255.0).round().clamp(0.0, 255.0)]])
        };
        layers.push(PlannedLayer::Adjustment {
            name: "Color Match".into(),
            kind: "curves".into(),
            params: serde_json::json!({ "red": curves(&grade, 0), "green": curves(&grade, 1), "blue": curves(&grade, 2) }),
            mask: Some(Region::Person),
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
            skin: None,
            grade: None,
            enhance: false,
            lips: None,
            eyes: None,
            blush: None,
            eyeshadow: None,
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

    #[test]
    fn skin_raster_needs_both_the_ask_and_the_skin_region() {
        let mut recipe = Recipe::default();
        recipe.skin = Some(crate::Skin::default());
        // Asked but not analyzed: nothing.
        assert_eq!(plan(&recipe, &analysis_with_person(true)).layers.len(), 1);
        // Analyzed but not asked: nothing.
        let mut analysis = analysis_with_person(true);
        analysis.set(Region::Skin, Surface::with_default(photocraft_doc::PixelFormat::GRAY8, &[1.0]));
        assert_eq!(plan(&Recipe::default(), &analysis).layers.len(), 1);
        // Both: the raster lands masked to Skin.
        let planned = plan(&recipe, &analysis);
        assert_eq!(planned.layers[1], PlannedLayer::Raster { name: "Skin Smoothing".into(), mask: Some(Region::Skin) });
    }

    #[test]
    fn grade_lands_a_person_masked_curves_layer_last() {
        let mut recipe = Recipe::default();
        recipe.grade = Some(Grade { slope: [1.2, 1.0, 0.8], offset: [-0.05, 0.0, 0.1] });
        let planned = plan(&recipe, &analysis_with_person(true));
        let PlannedLayer::Adjustment { name, kind, params, mask } = &planned.layers[1] else { panic!("not an adjustment") };
        assert_eq!((name.as_str(), kind.as_str(), *mask), ("Color Match", "curves", Some(Region::Person)));
        // The transfer in curve points: [0, offset*255] and [255, (slope+offset)*255],
        // clamped into Curves' 0..255 domain (an out-of-range point is refused, not moved).
        let pt = |v: f64| (v * 255.0).round().clamp(0.0, 255.0);
        assert_eq!(params["red"], serde_json::json!([[0, pt(-0.05)], [255, pt(1.15)]]));
        assert_eq!(params["green"], serde_json::json!([[0, pt(0.0)], [255, pt(1.0)]]));
        assert_eq!(params["blue"], serde_json::json!([[0, pt(0.1)], [255, pt(0.9)]]));
        // No person analysis, no grade layer.
        assert_eq!(plan(&recipe, &analysis_with_person(false)).layers.len(), 0);
    }
}
