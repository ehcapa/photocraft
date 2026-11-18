//! A diagnostic: which CelebAMask-HQ class index is which region, answered with
//! landmark anchors instead of eyeballing. For a real portrait it runs the parsing model
//! and FaceMesh over the composite (both resample internally), then reports each class's
//! argmax dominance inside landmark-anchored probes — the eyes, the mouth, the nose, the
//! band above the forehead (hair), the band below the chin (clothes). The class that
//! dominates a probe owns its semantics; docs/fork's mapping follows this table.
//!
//! Run: PHOTOCRAFT_PORTRAIT_PHOTO=<image> PHOTOCRAFT_MODELS_DIR=<dir>
//!      cargo run -p photocraft-engine --example class_probe

fn main() -> Result<(), String> {
    let photo = std::env::var_os("PHOTOCRAFT_PORTRAIT_PHOTO").ok_or("PHOTOCRAFT_PORTRAIT_PHOTO unset")?;
    let name = std::path::Path::new(&photo).file_name().and_then(|n| n.to_str()).unwrap_or("photo").to_string();
    let imported =
        photocraft_io::import_file(&name, std::path::Path::new(&photo), &photocraft_io::ImportOptions::default(), &photocraft_raster::Interrupt::NONE)
            .map_err(|e| format!("import: {e}"))?;
    let doc = imported.document;
    let canvas = doc.bounds();
    let (w, h) = (doc.size.width as usize, doc.size.height as usize);

    let composite = photocraft_compose::composite_to_surface(&doc, canvas, photocraft_color::PixelFormat::RGBA8, None);
    let rgba = composite.try_read_region(canvas).map_err(|e| format!("read: {e}"))?;
    let rgb: Vec<[f32; 3]> = rgba.chunks(4).map(|c| [c[0], c[1], c[2]]).collect();

    let pts = photocraft_ml::face_landmarks(&rgb, w, h).ok_or("no landmarks")?;
    let probs = photocraft_ml::parsing_probs(&rgb, w, h).ok_or("no parsing model (PHOTOCRAFT_MODELS_DIR?)")?;
    let (pw, ph) = (probs.w, probs.h);
    // Both outputs are in their own coordinate space; scale factors between them.
    let sx = pw as f32 / w as f32;
    let sy = ph as f32 / h as f32;

    let left_eye = (pts[33].x * sx, pts[33].y * sy);
    let right_eye = (pts[263].x * sx, pts[263].y * sy);
    let mouth = ((pts[61].x + pts[291].x) * 0.5 * sx, (pts[61].y + pts[291].y) * 0.5 * sy);
    let nose = (pts[1].x * sx, pts[1].y * sy);
    let forehead_y = pts[10].y * sy;
    let chin_y = pts[152].y * sy;
    let eye_span = (right_eye.0 - left_eye.0).abs();
    // Probe sizes against the model's effective resolution (~512²), not the upsampled
    // output: tight discs and bands, else they probe into the background.
    let face_cx = mouth.0;

    // The decisive method: each class's >0.5 mask scored by coverage of the landmark
    // RING polygons (eyes, lips) and hull bands (hair above the oval, clothes below the
    // chin) — exact contours, immune to probe sizing. The class with the highest coverage
    // of a shape owns that shape's semantics.
    const LEFT_EYE_RING: &[usize] = &[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
    const RIGHT_EYE_RING: &[usize] = &[263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
    const LIPS_RING: &[usize] = &[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
    let ring = |idx: &[usize]| -> Vec<(f32, f32)> { idx.iter().map(|&i| (pts[i].x * sx, pts[i].y * sy)).collect() };
    let shapes: [(&str, Vec<(f32, f32)>); 3] = [("left-eye", ring(LEFT_EYE_RING)), ("right-eye", ring(RIGHT_EYE_RING)), ("lips", ring(LIPS_RING))];
    let inside = |poly: &[(f32, f32)], x: f32, y: f32| -> bool {
        let mut hit = false;
        let n = poly.len();
        let mut i = 0;
        let mut j = n - 1;
        while i < n {
            let (xi, yi) = poly[i];
            let (xj, yj) = poly[j];
            if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                hit = !hit;
            }
            j = i;
            i += 1;
        }
        hit
    };
    // Per class: coverage of each polygon (share of the polygon's pixels the class holds)
    // and of the two bands.
    let mut coverage = [[0u64; 19]; 5];
    let mut shape_px = [0u64; 5];
    for (si, (_, poly)) in shapes.iter().enumerate() {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (x, y) in poly {
            x0 = x0.min(*x);
            x1 = x1.max(*x);
            y0 = y0.min(*y);
            y1 = y1.max(*y);
        }
        let (bx0, bx1, by0, by1) = (x0.floor().max(0.0) as usize, (x1.ceil() as usize).min(pw), y0.floor().max(0.0) as usize, (y1.ceil() as usize).min(ph));
        for y in by0..by1 {
            for x in bx0..bx1 {
                if inside(poly, x as f32 + 0.5, y as f32 + 0.5) {
                    shape_px[si] += 1;
                    for (c, plane) in probs.planes.iter().enumerate() {
                        if plane[y * pw + x] > 0.5 {
                            coverage[si][c] += 1;
                        }
                    }
                }
            }
        }
    }
    // Bands: hair = above the oval top, clothes = below the chin, face-wide.
    for y in 0..ph {
        let yf = y as f32 + 0.5;
        let band = if yf < forehead_y - eye_span * 0.05 && yf > forehead_y - eye_span * 0.9 && (yf - 0.0) >= 0.0 {
            3
        } else if yf > chin_y + eye_span * 0.08 && yf < chin_y + eye_span * 0.8 {
            4
        } else {
            usize::MAX
        };
        if band == usize::MAX {
            continue;
        }
        for x in 0..pw {
            if (x as f32 + 0.5 - face_cx).abs() < eye_span * 1.4 {
                shape_px[band] += 1;
                for (c, plane) in probs.planes.iter().enumerate() {
                    if plane[y * pw + x] > 0.5 {
                        coverage[band][c] += 1;
                    }
                }
            }
        }
    }
    let names = ["left-eye", "right-eye", "lips", "hair-band", "cloth-band"];
    println!("image {w}×{h}, eye span {eye_span:.0} prob px");
    let mut mass = [0u64; photocraft_ml::PARSING_CLASSES];
    for (c, plane) in probs.planes.iter().enumerate() {
        mass[c] = plane.iter().filter(|v| **v > 0.5).count() as u64;
    }
    println!("per-class >0.5 coverage:");
    for (c, m) in mass.iter().enumerate() {
        if *m > 500 {
            println!("  class {c:2}: {:.1}%", 100.0 * *m as f64 / (pw * ph) as f64);
        }
    }
    for (si, name) in names.iter().enumerate() {
        let mut ranked: Vec<(usize, u64)> = coverage[si].iter().enumerate().filter(|(_, v)| **v > 0).map(|(c, v)| (c, *v)).collect();
        ranked.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
        let top: Vec<String> = ranked.iter().take(3).map(|(c, v)| format!("class {c}: {:.0}%", 100.0 * *v as f64 / shape_px[si].max(1) as f64)).collect();
        println!("{name:10} ({:>5} px): {}", shape_px[si], top.join(", "));
    }
    let _ = (left_eye, right_eye, mouth, nose, forehead_y);
    Ok(())
}
