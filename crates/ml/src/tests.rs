use crate::{AVAILABLE, MODEL_SIDE, PARSING_CLASSES, parsing_probs, subject_mask};

#[test]
fn size_mismatch_is_none() {
    let px = vec![[0.5f32; 3]; 90];
    // 90 samples against 10 × 10, and empty dimensions: guards, not inference.
    assert!(subject_mask(&px, 10, 10).is_none());
    assert!(subject_mask(&px, 0, 10).is_none());
    assert!(subject_mask(&[], 0, 0).is_none());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn disc_on_contrast_yields_a_wellformed_mask() {
    // A red disc on a blue field is the plain case the network was trained for; the assert is
    // directional (disc above field) rather than absolute, to stay robust to weight details.
    let (w, h) = (240, 180);
    let px: Vec<[f32; 3]> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let inside = (x as f32 - 120.0).powi(2) + (y as f32 - 90.0).powi(2) < 40.0f32.powi(2);
            if inside { [0.9, 0.2, 0.1] } else { [0.1, 0.2, 0.4] }
        })
        .collect();
    let Some(m) = subject_mask(&px, w, h) else { panic!("no mask on a plain disc scene") };
    assert_eq!((m.w, m.h), (w, h));
    assert!(m.px.iter().all(|v| (0.0..=1.0).contains(v)));
    let at = |x: usize, y: usize| m.px[y * w + x];
    let disc = at(120, 90);
    let field = at(8, 8);
    assert!(disc > field + 0.3, "disc {disc} should stand out over field {field}");
}

#[test]
#[ignore = "timing only"]
fn timing() {
    if !AVAILABLE {
        return;
    }
    let (w, h) = (320, 240);
    let px: Vec<[f32; 3]> = (0..w * h).map(|i| [(i % w) as f32 / w as f32, (i / w) as f32 / h as f32, 0.5]).collect();
    let t = std::time::Instant::now();
    let m = subject_mask(&px, w, h);
    eprintln!("first call (parse + plan + inference): {:?}", t.elapsed());
    assert!(m.is_some());
    let t = std::time::Instant::now();
    let _ = subject_mask(&px, w, h);
    eprintln!("second call (inference only): {:?}", t.elapsed());
}

/// `U2NETP_REFERENCE=<dir>` holds `input.png` and `mask.png`, the same image and the mask
/// `rembg i` produced from it; our mask must binarise to ≥ 0.9 IoU with rembg's.
#[test]
#[ignore = "needs U2NETP_REFERENCE with input.png and mask.png from rembg"]
fn matches_rembg() {
    let Some(dir) = std::env::var_os("U2NETP_REFERENCE") else {
        eprintln!("skipping: U2NETP_REFERENCE unset (an input.png + rembg's mask.png)");
        return;
    };
    let load_luma = |name: &str| -> Option<(Vec<f32>, usize, usize)> {
        let img = image::open(std::path::Path::new(&dir).join(name)).ok()?.to_luma8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let px = img.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
        Some((px, w, h))
    };
    let img = match image::open(std::path::Path::new(&dir).join("input.png")) {
        Ok(img) => img.to_rgb8(),
        Err(_) => {
            eprintln!("skipping: no readable input.png in {}", dir.display());
            return;
        }
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb: Vec<[f32; 3]> = img.pixels().map(|p| [p.0[0] as f32 / 255.0, p.0[1] as f32 / 255.0, p.0[2] as f32 / 255.0]).collect();
    let (want, ww, wh) = match load_luma("mask.png") {
        Some(v) => v,
        None => {
            eprintln!("skipping: no readable mask.png in {}", dir.display());
            return;
        }
    };
    assert_eq!((w, h), (ww, wh), "input and mask differ in size");
    let Some(got) = subject_mask(&rgb, w, h) else { panic!("no mask on a real photo") };
    let mut inter = 0u64;
    let mut union = 0u64;
    for i in 0..w * h {
        let (a, b) = (got.px[i] >= 0.5, want[i] >= 0.5);
        inter += (a && b) as u64;
        union += (a || b) as u64;
    }
    let iou = inter as f64 / union.max(1) as f64;
    eprintln!("rembg parity: IoU {iou:.4} over {w}×{h}, input side {MODEL_SIDE}");
    assert!(iou > 0.9, "IoU {iou} below 0.9 against rembg");
}

/// `parsing_probs` smoke: sizes, softmax sums, and non-degenerate classes — but only when a
/// model path is handed in (the weights are not in the tree). Without it the None path is
/// the behaviour under test, which the plain run covers everywhere else.
#[test]
fn parsing_probs_smoke_when_model_provided() {
    let Some(path) = std::env::var_os("PHOTOCRAFT_PORTRAIT_PARSING") else {
        assert!(parsing_probs(&[], 0, 0).is_none());
        eprintln!("skipping: PHOTOCRAFT_PORTRAIT_PARSING unset");
        return;
    };
    let _ = path;
    // A synthetic 64×64 face-ish blob: any content exercises the graph; the assertions are
    // about the output's shape, not its semantics.
    let (w, h) = (64, 64);
    let px: Vec<[f32; 3]> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let inside = (x as f32 - 32.0).powi(2) + (y as f32 - 32.0).powi(2) < 20.0 * 20.0;
            [if inside { 0.8 } else { 0.3 }; 3]
        })
        .collect();
    let probs = parsing_probs(&px, w, h).expect("model provided but no probs");
    assert_eq!((probs.w, probs.h), (w, h));
    assert_eq!(probs.planes.len(), PARSING_CLASSES);
    for i in 0..w * h {
        let sum: f32 = probs.planes.iter().map(|p| p[i]).sum();
        assert!((sum - 1.0).abs() < 1e-3, "softmax sums to {sum}");
    }
    // Argmax variety: more than one class wins somewhere on a two-tone image.
    let mut winners = std::collections::HashSet::new();
    for i in 0..w * h {
        let (mut best, mut bc) = (f32::MIN, 0);
        for (c, p) in probs.planes.iter().enumerate() {
            if p[i] > best {
                (best, bc) = (p[i], c);
            }
        }
        winners.insert(bc);
    }
    assert!(winners.len() >= 2, "one class everywhere: {winners:?}");
}
