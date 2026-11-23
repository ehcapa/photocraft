//! A diagnostic: the pose-landmark candidate's contract and the bbox-square crop
//! convention (docs/fork/2026-11-20 — the foundation layer's spike). The MediaPipe
//! pipeline rotates its detector's ROI before the landmark model; v1 skips the rotation
//! (portrait photos are upright), so this probe answers whether the axis-aligned square
//! about the person matte's bounds is good enough: it prints every output tensor's shape,
//! then the 33 body landmarks mapped back to document pixels with a shoulder/hip
//! plausibility summary.
//!
//! Run: PHOTOCRAFT_PORTRAIT_PHOTO=<image> PHOTOCRAFT_MODELS_DIR=<dir>
//!      cargo run -p photocraft-engine --example pose_probe

use tract_onnx::prelude::*;

const SIDE: usize = 256;
const COUNT: usize = 39;
const STRIDE: usize = 5;

fn main() -> Result<(), String> {
    let photo = std::env::var_os("PHOTOCRAFT_PORTRAIT_PHOTO").ok_or("PHOTOCRAFT_PORTRAIT_PHOTO unset")?;
    let dir = std::env::var_os("PHOTOCRAFT_MODELS_DIR").ok_or("PHOTOCRAFT_MODELS_DIR unset")?;
    let path = std::path::Path::new(&dir).join("poselandmarks.onnx");
    if !path.is_file() {
        return Err(format!("no poselandmarks.onnx in {}", dir.to_string_lossy()));
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
    let rgb: Vec<[f32; 3]> = rgba.chunks(ch).map(|c| [c[0], c[1], c[2]]).collect();

    // The person matte's bounds → the axis-aligned square about its centre, clamped.
    let subject = photocraft_ml::subject_mask(&rgb, w, h).ok_or("no subject mask")?;
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
    let side = (bx1 - bx0).max(by1 - by0);
    let (cx, cy) = ((bx0 + bx1) as i64 / 2, (by0 + by1) as i64 / 2);
    let (mut sx0, mut sy0) = ((cx - side as i64 / 2).max(0) as usize, (cy - side as i64 / 2).max(0) as usize);
    let (sx1, sy1) = ((sx0 + side).min(w), (sy0 + side).min(h));
    sx0 = sx1.saturating_sub(side).min(sx1);
    sy0 = sy1.saturating_sub(side).min(sy1);
    let (sw, sh) = (sx1 - sx0, sy1 - sy0);
    println!("image {w}×{h}, person {bx0},{by0}+{}×{}, square crop {sx0},{sy0}+{sw}×{sh}", bx1 - bx0, by1 - by0);

    // The plan.
    let proto = tract_onnx::onnx().model_for_path(&path).map_err(|e| format!("onnx load: {e}"))?;
    let proto = proto.with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, [1usize, SIDE, SIDE, 3])).map_err(|e| format!("with_input_fact: {e}"))?;
    let typed = proto.into_typed().map_err(|e| format!("into_typed: {e}"))?;
    let typed = typed.clone().into_optimized().unwrap_or(typed);
    println!("{} outputs:", typed.outputs.len());
    let plan = typed.into_runnable().map_err(|e| format!("into_runnable: {e}"))?;

    // Stretch the square crop onto 256², NHWC 0..1.
    let mut input = vec![0.0f32; 3 * SIDE * SIDE];
    for y in 0..SIDE {
        let sy = (y as f32 + 0.5) * sh as f32 / SIDE as f32 - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        for x in 0..SIDE {
            let sxf = (x as f32 + 0.5) * sw as f32 / SIDE as f32 - 0.5;
            let x0 = sxf.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let (wx, wy) = (sxf - x0 as f32, sy - y0 as f32);
            for c in 0..3 {
                let p = |xi: usize, yi: usize| rgb[(sy0 + yi) * w + sx0 + xi][c];
                let a = p(x0, y0) * (1.0 - wx) + p(x1, y0) * wx;
                let b = p(x0, y1) * (1.0 - wx) + p(x1, y1) * wx;
                input[(y * SIDE + x) * 3 + c] = a * (1.0 - wy) + b * wy;
            }
        }
    }
    let tensor = Tensor::from_shape(&[1, SIDE, SIDE, 3], &input).map_err(|e| e.to_string())?;
    let began = std::time::Instant::now();
    let outputs = plan.run(tvec![tensor.into()]).map_err(|e| format!("run: {e}"))?;
    println!("run: {:?}", began.elapsed());

    // The screen-landmarks output: the [1,195] tensor.
    let mut screen: Option<Vec<f32>> = None;
    for (i, out) in outputs.iter().enumerate() {
        let shape: Vec<usize> = out.shape().to_vec();
        println!("out[{i}] {shape:?}");
        if out.len() == COUNT * STRIDE {
            screen = Some(out.to_plain_array_view::<f32>().map_err(|e| e.to_string())?.as_slice().ok_or("non-contiguous output")?.to_vec());
        }
    }
    let Some(screen) = screen else {
        return Err("no [1,195] screen-landmarks output".into());
    };
    // 33 body landmarks back to document pixels.
    let names = [
        "nose", "leye_i", "leye", "leye_o", "reye_i", "reye", "reye_o", "lear", "rear", "lmouth", "rmouth", "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
        "lpink", "rpink", "lindx", "rindx", "lthmb", "rthmb", "lhip", "rhip", "lknee", "rknee", "lankl", "rankl", "lheel", "rheel", "lfoot", "rfoot",
    ];
    let mut rows = Vec::new();
    for i in 0..33 {
        let (x, y, v) = (screen[i * STRIDE], screen[i * STRIDE + 1], screen[i * STRIDE + 3]);
        let dx = sx0 as f32 + x * sw as f32 / SIDE as f32;
        let dy = sy0 as f32 + y * sh as f32 / SIDE as f32;
        rows.push((i, names[i], dx, dy, v));
    }
    for (i, n, x, y, v) in &rows {
        println!("  {i:2} {n:6} x={x:7.1} y={y:7.1} vis={v:.2}");
    }
    let shoulder = (rows[11].3 + rows[12].3) / 2.0;
    let hip = (rows[23].3 + rows[24].3) / 2.0;
    let span = (rows[11].2 - rows[12].2).abs();
    let vis: f32 = rows.iter().map(|r| r.4).sum::<f32>() / rows.len() as f32;
    println!("plausibility: shoulder y {shoulder:.0} above hip y {hip:.0}: {}, shoulder span {span:.0}px, mean visibility {vis:.2}", shoulder < hip);
    Ok(())
}
