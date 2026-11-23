//! A diagnostic for Smart Portrait real-photo acceptance: runs `portrait.create` over a
//! photo, then dumps every child of the Portrait group — a solo composite (only that child
//! visible), the mask as PGM, coverage/bbox — plus the base and full composites. When a
//! retouch looks wrong, this names the guilty layer in one run instead of guessing.
//!
//! Run: PHOTOCRAFT_MODELS_DIR=<dir> [ORT_DYLIB_PATH=<lib>]
//!      cargo run -p photocraft-engine --example portrait_diag [--features ort] \
//!          -- <photo> [params JSON, default: the dialog's recipe]

use serde_json::{Value, json};
use std::path::Path;

use photocraft_doc::{Document, Layer, LayerContent, LayerMask};
use photocraft_engine::Session;

fn write_ppm(path: &str, w: u32, h: u32, rgb: &[u8]) -> Result<(), String> {
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(rgb);
    std::fs::write(path, out).map_err(|e| format!("write {path}: {e}"))
}

fn composite_ppm(s: &Session, path: &str) -> Result<(), String> {
    let st = s.active().ok_or("no document")?;
    let doc = &st.doc;
    let bounds = doc.bounds();
    let surf = photocraft_compose::composite_to_surface(doc, bounds, photocraft_doc::PixelFormat::RGBA8, None);
    let (w, h) = (bounds.width(), bounds.height());
    let px = surf.try_read_region(bounds).map_err(|e| e.to_string())?;
    let ch = px.len() / ((w * h) as usize);
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for pixel in px.chunks_exact(ch) {
        buf.extend(pixel.iter().take(3).map(|v| (v * 255.0).clamp(0.0, 255.0) as u8));
    }
    write_ppm(path, w, h, &buf)
}

/// The child's mask as PGM + coverage: masks are document-sized, read over the doc bounds.
fn mask_pgm(layer: &Layer, bounds: photocraft_geom::Rect, path: &str) -> Result<(f64, String), String> {
    let Some(raw) = layer.mask.as_ref() else { return Ok((-1.0, "NO MASK".into())) };
    let m = LayerMask { surface: raw.surface.clone(), ..LayerMask::reveal_all() };
    let (w, h) = (bounds.width(), bounds.height());
    let mut buf = Vec::with_capacity((w * h) as usize);
    let mut cov = 0u64;
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    for y in 0..h {
        for x in 0..w {
            let v = m.value(x as i32, y as i32);
            buf.push((v * 255.0).clamp(0.0, 255.0) as u8);
            if v >= 0.5 {
                cov += 1;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    let bbox = if cov > 0 { format!("bbox x{x0}..{x1} y{y0}..{y1}") } else { "empty".into() };
    let mut out = format!("P5\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(&buf);
    std::fs::write(path, out).map_err(|e| format!("write {path}: {e}"))?;
    Ok((cov as f64 / (w * h) as f64 * 100.0, bbox))
}

fn safe(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

/// Composite with every OTHER child of the Portrait group hidden, then undo.
fn solo(s: &mut Session, keep: &str, out: &str) -> Result<(), String> {
    s.edit("diag solo", |doc: &mut Document, _| {
        for layer in doc.layers.iter_mut() {
            if let LayerContent::Group(g) = &mut layer.content {
                for child in g.children.iter_mut() {
                    child.visible = child.name == keep;
                }
            }
        }
        Ok(())
    })
    .map_err(|e| format!("solo edit: {e}"))?;
    let shown = composite_ppm(s, out);
    s.execute("edit.undo", json!({})).map_err(|e| format!("undo solo: {e}"))?;
    shown
}

fn main() -> Result<(), String> {
    let photo = std::env::args().nth(1).ok_or("usage: portrait_diag <photo> [params json]")?;
    let params: Value = std::env::args().nth(2).map(|p| serde_json::from_str(&p).map_err(|e| format!("params json: {e}"))).transpose()?.unwrap_or_else(|| {
        json!({
            "skinRadius": 10.0, "skinThreshold": 18.0,
            "lipHue": 330.0, "lipSaturation": 30.0,
            "eyeBrighten": 12.0,
            "blushHue": 350.0, "blushSaturation": 20.0,
            "shadowHue": 30.0, "shadowSaturation": 25.0
        })
    });
    eprintln!("params: {params}");
    std::fs::create_dir_all("/tmp/diag").map_err(|e| format!("create /tmp/diag: {e}"))?;

    let mut s = Session::new();
    let name = Path::new(&photo).file_name().and_then(|n| n.to_str()).unwrap_or("photo").to_string();
    let imported = photocraft_io::import_file(&name, Path::new(&photo), &photocraft_io::ImportOptions::default(), &photocraft_raster::Interrupt::NONE)
        .map_err(|e| format!("import: {e}"))?;
    s.open_document(imported.document, None);
    composite_ppm(&s, "/tmp/diag/00_base.ppm")?;

    let t = std::time::Instant::now();
    let r = s.execute("portrait.create", params.clone()).map_err(|e| format!("portrait.create: {e}"))?;
    eprintln!("portrait.create: {:.1}s -> {r}", t.elapsed().as_secs_f32());
    composite_ppm(&s, "/tmp/diag/99_full.ppm")?;

    let st = s.active().ok_or("no document after the run")?;
    let doc = &st.doc;
    let bounds = doc.bounds();
    let mut names: Vec<String> = Vec::new();
    let mut i = 0u32;
    for layer in doc.layers.iter() {
        let LayerContent::Group(g) = &layer.content else { continue };
        println!("group '{}' opacity={} children={}", layer.name, layer.opacity, g.children.len());
        for child in &g.children {
            i += 1;
            println!("{i:02} [{}] '{}' visible={} blend={:?} opacity={}", child.content.kind_name(), child.name, child.visible, child.blend, child.opacity);
            let (cov, bbox) = mask_pgm(child, bounds, &format!("/tmp/diag/{i:02}_{}.pgm", safe(&child.name)))?;
            println!("    mask coverage {cov:.1}% {bbox}");
            names.push(child.name.clone());
        }
    }
    for (i, n) in names.iter().enumerate() {
        solo(&mut s, n, &format!("/tmp/diag/{:02}_{}_solo.ppm", i + 1, safe(n)))?;
    }
    eprintln!("done: /tmp/diag/");
    Ok(())
}
