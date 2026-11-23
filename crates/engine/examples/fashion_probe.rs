//! A diagnostic: the fashion-tier candidate's contract and class semantics, answered with
//! anchor windows instead of eyeballing (docs/fork/decisions.md 2026-11-20 names SCHP
//! ATR-18 as the next body-parsing tier; its arm/leg geometric classes are the known trap).
//!
//! For a real photo it loads `$PHOTOCRAFT_MODELS_DIR/schp-atr.onnx` directly through tract,
//! prints the graph's input/output facts and a 512² timing, then runs the photo under each
//! candidate input contract (raw 0-255 NCHW; 0..1; 0..1 ImageNet-normalized) and reports
//! per-class dominance inside anchored windows — the hair band above the mesh forehead, the
//! chin band below it, the eyes/lips rings, the person-bbox fraction bands (torso, lower,
//! shoes, arms) and the free `PHOTOCRAFT_REGION_PROBE=[x,y,w,h]` window. The contract whose
//! windows answer their semantics owns the tier; a class must dominate its window to enter
//! the region table. The value range of the raw output is printed too — softmax baked in or
//! not is a measured fact, not an assumption (body.rs's two wrong readings are the cautionary
//! precedent).
//!
//! Run: PHOTOCRAFT_PORTRAIT_PHOTO=<image> PHOTOCRAFT_MODELS_DIR=<dir>
//!      cargo run -p photocraft-engine --example fashion_probe

use tract_onnx::prelude::*;

const SIDE: usize = 512;
const CLASSES: usize = 18;
const ATR: [&str; CLASSES] = [
    "background",
    "hat",
    "hair",
    "sunglasses",
    "upper-clothes",
    "skirt",
    "pants",
    "dress",
    "belt",
    "left-shoe",
    "right-shoe",
    "face",
    "left-leg",
    "right-leg",
    "left-arm",
    "right-arm",
    "bag",
    "scarf",
];
const LEFT_EYE_RING: &[usize] = &[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const RIGHT_EYE_RING: &[usize] = &[263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
const LIPS_RING: &[usize] = &[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
const RINGS: [(&str, &[usize]); 3] = [("left-eye", LEFT_EYE_RING), ("right-eye", RIGHT_EYE_RING), ("lips", LIPS_RING)];

/// A bilinear stretch of `px` onto `s × s`, the triangle filter's diagnostic stand-in.
fn stretch(px: &[([f32; 3], usize, usize)], w: usize, h: usize, s: usize) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; s * s];
    for y in 0..s {
        let fy = (y as f32 + 0.5) * h as f32 / s as f32 - 0.5;
        let y0 = fy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(h - 1);
        let wy = fy - y0 as f32;
        for x in 0..s {
            let fx = (x as f32 + 0.5) * w as f32 / s as f32 - 0.5;
            let x0 = fx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(w - 1);
            let wx = fx - x0 as f32;
            let p = |xi: usize, yi: usize| px[yi * w + xi].0;
            for (c, o) in out[y * s + x].iter_mut().enumerate() {
                let a = p(x0, y0)[c] * (1.0 - wx) + p(x1, y0)[c] * wx;
                let b = p(x0, y1)[c] * (1.0 - wx) + p(x1, y1)[c] * wx;
                *o = a * (1.0 - wy) + b * wy;
            }
        }
    }
    out
}

/// One probe window in person-relative or pixel terms, evaluated at the output resolution.
struct Window {
    name: &'static str,
    rect: (usize, usize, usize, usize), // x0, y0, x1, y1 exclusive, in output pixels
}

/// One contract's result: its label and the per-class probability planes at (ow, oh).
struct Contract {
    label: String,
    planes: Vec<Vec<f32>>,
    ow: usize,
    oh: usize,
}

/// Even-point-in-polygon test for the face rings.
fn inside(poly: &[(f32, f32)], x: f32, y: f32) -> bool {
    let mut hit = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            hit = !hit;
        }
        j = i;
    }
    hit
}

/// The top-4 classes by summed mass, as "name N%" entries.
fn rank(sums: &[f64; CLASSES], area: usize) -> String {
    let mut ranked: Vec<(usize, f64)> = sums.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.iter().take(4).map(|(c, s)| format!("{} {:.0}%", ATR[*c], 100.0 * s / area as f64)).collect::<Vec<_>>().join(", ")
}

/// Softmax over the class axis per spatial position (harmless if already probabilities).
fn softmax_planes(raw: &[f32], stride: usize) -> Vec<f32> {
    let mut softmaxed = vec![0.0f32; raw.len()];
    for i in 0..stride {
        let mut max = f32::NEG_INFINITY;
        for c in 0..CLASSES {
            max = max.max(raw[c * stride + i]);
        }
        let mut sum = 0.0;
        for c in 0..CLASSES {
            softmaxed[c * stride + i] = (raw[c * stride + i] - max).exp();
            sum += softmaxed[c * stride + i];
        }
        if sum > 0.0 {
            for c in 0..CLASSES {
                softmaxed[c * stride + i] /= sum;
            }
        }
    }
    softmaxed
}

/// Run every contract through the plan and lay out its per-class planes; a contract whose
/// output this probe cannot read is reported and skipped.
fn run_contracts(
    run: impl Fn(&[f32]) -> Result<(Vec<usize>, Vec<f32>, std::time::Duration), String>,
    contracts: &[(&str, Vec<f32>)],
) -> Result<Vec<Contract>, String> {
    let mut all = Vec::new();
    for (label, nchw) in contracts {
        let (shape, raw, took) = run(nchw).map_err(|e| format!("run ({label}): {e}"))?;
        let (lo, hi) = raw.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)));
        println!("contract {label}: out {shape:?}, raw range {lo:.2}..{hi:.2}, {took:?}");
        if raw.len() % CLASSES != 0 {
            println!("  class count {CLASSES} does not divide the output — skipped");
            continue;
        }
        // Find the class axis (the dim of size CLASSES) and lay out per-class planes.
        let Some(axis) = shape.iter().position(|d| *d == CLASSES) else {
            println!("  no axis of size {CLASSES} — skipped");
            continue;
        };
        let spatial: Vec<usize> = shape.iter().enumerate().filter(|(i, _)| *i != axis && shape[*i] != 1).map(|(_, d)| *d).collect();
        let (ow, oh) = (spatial[0], spatial[1]);
        let stride = raw.len() / CLASSES;
        let softmaxed = softmax_planes(&raw, stride);
        let planes: Vec<Vec<f32>> = (0..CLASSES).map(|c| softmaxed[c * stride..(c + 1) * stride].to_vec()).collect();
        all.push(Contract { label: label.to_string(), planes, ow, oh });
    }
    Ok(all)
}

/// The windows' per-class mass at this contract's output resolution. The crop maps
/// document → output the same way the blob table maps back.
fn report_windows(windows: &[Window], planes: &[Vec<f32>], ow: usize, oh: usize, (cx0, cy0, cw, chh): (usize, usize, usize, usize)) {
    let fx = |x: usize| x.saturating_sub(cx0) * ow / cw.max(1);
    let fy = |y: usize| y.saturating_sub(cy0) * oh / chh.max(1);
    let mut sums_windows: Vec<(&str, usize, [f64; CLASSES])> = Vec::new();
    for win in windows {
        let (x0, y0, x1, y1) = (fx(win.rect.0), fy(win.rect.1), fx(win.rect.2).max(fx(win.rect.0)), fy(win.rect.3).max(fy(win.rect.1)));
        let (x1, y1) = (x1.min(ow), y1.min(oh));
        let area = (x1 - x0) * (y1 - y0);
        if area == 0 {
            println!("{:10} empty", win.name);
            continue;
        }
        let mut sums = [0.0f64; CLASSES];
        for y in y0..y1 {
            for x in x0..x1 {
                for (c, plane) in planes.iter().enumerate() {
                    sums[c] += plane[y * ow + x] as f64;
                }
            }
        }
        sums_windows.push((win.name, area, sums));
    }
    for (name, area, sums) in &sums_windows {
        println!("{name:10} ({area} px): {}", rank(sums, *area));
    }
}

/// The face rings' class readout at this contract's resolution; `to_out` maps a landmark
/// index to output coordinates.
fn report_rings(planes: &[Vec<f32>], ow: usize, oh: usize, to_out: impl Fn(usize) -> (f32, f32)) {
    for (name, idx) in RINGS {
        let poly: Vec<(f32, f32)> = idx.iter().map(|&i| to_out(i)).collect();
        let (mut px0, mut py0, mut px1, mut py1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (x, y) in &poly {
            px0 = px0.min(*x);
            px1 = px1.max(*x);
            py0 = py0.min(*y);
            py1 = py1.max(*y);
        }
        let (bx0, bx1, by0, by1) =
            (px0.floor().max(0.0) as usize, (px1.ceil() as usize).min(ow), py0.floor().max(0.0) as usize, (py1.ceil() as usize).min(oh));
        let mut n = 0usize;
        let mut sums = [0.0f64; CLASSES];
        for y in by0..by1 {
            for x in bx0..bx1 {
                if inside(&poly, x as f32 + 0.5, y as f32 + 0.5) {
                    n += 1;
                    for (c, plane) in planes.iter().enumerate() {
                        sums[c] += plane[y * ow + x] as f64;
                    }
                }
            }
        }
        if n == 0 {
            println!("{name:10} empty");
            continue;
        }
        println!("{name:10} ({n} px): {}", rank(&sums, n));
    }
}

/// Per-class mass and blob bbox (document px): the coordinate-free semantic readout —
/// the scarf blob should sit at the neck, shoes at the bottom, hair at the head.
fn report_mass(planes: &[Vec<f32>], ow: usize, oh: usize, (cx0, cy0, cw, chh): (usize, usize, usize, usize)) {
    println!("per-class >0.5 mass (bbox in document px):");
    for (c, plane) in planes.iter().enumerate() {
        let mut n = 0usize;
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        for (i, v) in plane.iter().enumerate() {
            if *v > 0.5 {
                n += 1;
                let (x, y) = (i % ow, i / ow);
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
        if n > 20 {
            println!(
                "  {:12} {:5.1}%  at {},{} +{}×{}",
                ATR[c],
                100.0 * n as f64 / (ow * oh) as f64,
                cx0 + x0 * cw / ow,
                cy0 + y0 * chh / oh,
                (x1 - x0) * cw / ow,
                (y1 - y0) * chh / oh
            );
        }
    }
}

fn main() -> Result<(), String> {
    let photo = std::env::var_os("PHOTOCRAFT_PORTRAIT_PHOTO").ok_or("PHOTOCRAFT_PORTRAIT_PHOTO unset")?;
    let dir = std::env::var_os("PHOTOCRAFT_MODELS_DIR").ok_or("PHOTOCRAFT_MODELS_DIR unset")?;
    let path = std::path::Path::new(&dir).join("schp-atr.onnx");
    if !path.is_file() {
        return Err(format!("no schp-atr.onnx in {}", dir.to_string_lossy()));
    }

    let name = std::path::Path::new(&photo).file_name().and_then(|n| n.to_str()).unwrap_or("photo").to_string();
    let imported =
        photocraft_io::import_file(&name, std::path::Path::new(&photo), &photocraft_io::ImportOptions::default(), &photocraft_raster::Interrupt::NONE)
            .map_err(|e| format!("import: {e}"))?;
    let doc = imported.document;
    let canvas = doc.bounds();
    let (w, h) = (doc.size.width as usize, doc.size.height as usize);
    let composite = photocraft_compose::composite_to_surface(&doc, canvas, photocraft_doc::PixelFormat::RGBA8, None);
    let rgba = composite.try_read_region(canvas).map_err(|e| format!("read: {e}"))?;
    let ch = rgba.len() / (w * h);
    let rgb: Vec<([f32; 3], usize, usize)> = rgba.chunks(ch).enumerate().map(|(i, c)| ([c[0], c[1], c[2]], i % w, i / w)).collect();

    // The anchors: the person matte's bounds for the body bands, FaceMesh for the face.
    let subject = photocraft_ml::subject_mask(&rgb.iter().map(|(p, _, _)| *p).collect::<Vec<_>>(), w, h).ok_or("no subject mask")?;
    let (mut bx0, mut by0, mut bx1, mut by1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for (i, v) in subject.px.iter().enumerate() {
        if *v > 0.5 {
            let (x, y) = (i % subject.w, i / subject.w);
            bx0 = bx0.min(x);
            by0 = by0.min(y);
            bx1 = bx1.max(x + 1);
            by1 = by1.max(y + 1);
        }
    }
    if bx0 == usize::MAX {
        return Err("no person in the subject mask".into());
    }
    let (bw, bh) = (bx1 - bx0, by1 - by0);
    // The pipeline's own crop (body_regions): the person bbox grown by BODY_MARGIN = 1.25,
    // clamped to the document — the probe measures the model on the crop it will see.
    let margin = 1.25f32;
    let (cw, chh) = ((bw as f32 * margin) as usize, (bh as f32 * margin) as usize);
    let (cx0, cy0) = (bx0.saturating_sub((cw - bw) / 2), by0.saturating_sub((chh - bh) / 2));
    let (cx1, cy1) = ((cx0 + cw).min(w), (cy0 + chh).min(h));
    let (cx0, cy0) = (cx1.saturating_sub(cw).min(cx0), cy1.saturating_sub(chh).min(cy0));
    let (cw, chh) = (cx1 - cx0, cy1 - cy0);
    println!("image {w}×{h}, person bbox {bx0},{by0} +{bw}×{bh}, crop {cx0},{cy0} +{cw}×{chh}");

    let pts = photocraft_ml::face_landmarks(&rgb.iter().map(|(p, _, _)| *p).collect::<Vec<_>>(), w, h).ok_or("no landmarks")?;
    let forehead_y = pts[10].y;
    let chin_y = pts[152].y;
    let face_cx = pts[1].x;
    let eye_span = (pts[263].x - pts[33].x).abs().max(1.0);

    // The plan. The export's input may be dynamic (detect.rs's SCRFD precedent): pin the
    // 512² NCHW input fact before into_typed, else the conv analysis fails.
    let proto = tract_onnx::onnx().model_for_path(&path).map_err(|e| format!("onnx load: {e}"))?;
    println!("declared inputs: {} output: {}", proto.inputs.len(), proto.outputs.len());
    let proto = proto.with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, [1usize, 3, SIDE, SIDE])).map_err(|e| format!("with_input_fact: {e}"))?;
    let typed = proto.into_typed().map_err(|e| format!("into_typed: {e}"))?;
    let typed = typed.clone().into_optimized().unwrap_or(typed);
    let plan = typed.into_runnable().map_err(|e| format!("into_runnable: {e}"))?;
    println!("input  fact: {:?}", plan.input_fact(0).map_err(|e| e.to_string())?);
    println!("output fact: {:?}", plan.output_fact(0).map_err(|e| e.to_string())?);

    let crop_px: Vec<([f32; 3], usize, usize)> =
        rgb.iter().filter(|(_, x, y)| *x >= cx0 && *x < cx1 && *y >= cy0 && *y < cy1).map(|(p, x, y)| (*p, *x - cx0, *y - cy0)).collect();
    let scaled = stretch(&crop_px, cw, chh, SIDE);
    // The three candidate contracts, NCHW f32.
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut contracts: Vec<(&str, Vec<f32>)> = Vec::new();
    for (label, scale, normalize) in [("raw-0-255", 255.0, false), ("unit-0-1", 1.0, false), ("imagenet", 1.0, true)] {
        let mut nchw = vec![0.0f32; 3 * SIDE * SIDE];
        for (i, p) in scaled.iter().enumerate() {
            for c in 0..3 {
                let v = if normalize { (p[c] - mean[c]) / std[c] } else { p[c] * scale };
                nchw[c * SIDE * SIDE + i] = v;
            }
        }
        contracts.push((label, nchw));
    }
    let run = |nchw: &[f32]| -> Result<(Vec<usize>, Vec<f32>, std::time::Duration), String> {
        let tensor = Tensor::from_shape(&[1, 3, SIDE, SIDE], nchw).map_err(|e| e.to_string())?;
        let began = std::time::Instant::now();
        let outputs = plan.run(tvec![tensor.into()]).map_err(|e| e.to_string())?;
        let took = began.elapsed();
        let view = outputs.into_iter().next().ok_or("no output")?;
        let shape: Vec<usize> = view.shape().to_vec();
        let raw: Vec<f32> = view.to_plain_array_view::<f32>().map_err(|e| e.to_string())?.iter().copied().collect();
        Ok((shape, raw, took))
    };
    let all = run_contracts(run, &contracts)?;
    if all.is_empty() {
        return Err("no contract produced planes".into());
    }

    // The windows, defined in document pixels and mapped per contract's output resolution.
    // The face bands follow class_probe's anchoring (face-relative, not person-bbox-relative:
    // a half-body portrait puts the face high in the matte, so bbox-fraction bands land off).
    let band = |x0: usize, y0: usize, x1: usize, y1: usize| (x0.min(w), y0.min(h), x1.min(w).max(x0.min(w)), y1.min(h).max(y0.min(h)));
    let hair_y0 = (forehead_y - eye_span * 0.9).max(0.0) as usize;
    let hair_y1 = (forehead_y - eye_span * 0.05).max(0.0) as usize;
    let chin_y0 = (chin_y + eye_span * 0.08) as usize;
    let chin_y1 = (chin_y + eye_span * 0.8) as usize;
    let face_w = |_lo: f32, hi: f32| band(((face_cx - eye_span * hi).max(0.0)) as usize, 0, ((face_cx + eye_span * hi).min(w as f32)) as usize, h);
    let mut windows: Vec<Window> = vec![
        Window {
            name: "hair-band",
            rect: {
                let (x0, _, x1, _) = face_w(0.0, 1.4);
                (x0, hair_y0, x1, hair_y1)
            },
        },
        Window {
            name: "chin-band",
            rect: {
                let (x0, _, x1, _) = face_w(0.0, 1.4);
                (x0, chin_y0, x1, chin_y1)
            },
        },
        Window { name: "torso-band", rect: band(bx0 + bw * 15 / 100, chin_y as usize, bx1 - bw * 15 / 100, by0 + bh * 55 / 100) },
        Window { name: "lower-band", rect: band(bx0 + bw * 20 / 100, by0 + bh * 60 / 100, bx1 - bw * 20 / 100, by0 + bh * 85 / 100) },
        Window { name: "shoes-band", rect: band(bx0 + bw * 20 / 100, by0 + bh * 88 / 100, bx1 - bw * 20 / 100, by1) },
        Window { name: "left-arm", rect: band(bx0, by0 + bh * 35 / 100, bx0 + bw * 20 / 100, by0 + bh * 65 / 100) },
        Window { name: "right-arm", rect: band(bx1 - bw * 20 / 100, by0 + bh * 35 / 100, bx1, by0 + bh * 65 / 100) },
    ];
    if let Ok(free) = std::env::var("PHOTOCRAFT_REGION_PROBE") {
        let nums: Vec<usize> = free.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if nums.len() == 4 {
            windows.push(Window { name: "free-probe", rect: band(nums[0], nums[1], nums[0] + nums[2], nums[1] + nums[3]) });
        }
    }

    for Contract { label, planes, ow, oh } in &all {
        println!("== contract {label}, probing at output {ow}×{oh} ==");
        report_windows(&windows, planes, *ow, *oh, (cx0, cy0, cw, chh));
        // The face rings at this contract's resolution.
        let sx = *ow as f32 / cw.max(1) as f32;
        let sy = *oh as f32 / chh.max(1) as f32;
        let to_out = |i: usize| ((pts[i].x - cx0 as f32) * sx, (pts[i].y - cy0 as f32) * sy);
        report_rings(planes, *ow, *oh, to_out);
        report_mass(planes, *ow, *oh, (cx0, cy0, cw, chh));
    }
    let _ = (forehead_y, face_cx);
    Ok(())
}
