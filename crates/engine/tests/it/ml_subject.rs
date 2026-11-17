//! Select Subject through the learned model, end to end (integration tests build the engine
//! without `cfg(test)`, so the model is on wherever `photocraft-ml` is available; a build
//! without it takes the classical path and the test still holds).

use serde_json::json;

fn disc_session() -> photocraft_engine::Session {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 200, "height": 150})).expect("file.new");
    s.edit("paint", |doc, _| {
        let (w, h) = (200usize, 150usize);
        let bg = doc.layers[0].surface_mut().expect("surface");
        let n = bg.channels();
        let mut data = Vec::with_capacity(w * h * n);
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 + 0.5 - 100.0, y as f32 + 0.5 - 75.0);
                let inside = dx * dx + dy * dy <= 40.0 * 40.0;
                let c: [f32; 3] = if inside { [0.85, 0.2, 0.15] } else { [0.25, 0.45, 0.7] };
                data.extend_from_slice(&c[..n.min(3)]);
                if n == 4 {
                    data.push(1.0);
                }
            }
        }
        bg.write_region(photocraft_geom::Rect::new(0, 0, w as i32, h as i32), &data);
        Ok(())
    })
    .expect("paint");
    s
}

#[test]
fn select_subject_with_the_model_or_classically_selects_the_disc() {
    let mut s = disc_session();
    let out = s.execute("select.subject", json!({})).expect("select.subject runs");
    assert_eq!(out["selected"], json!(true), "a plain disc is found either way: {out}");
}

#[test]
fn remove_background_with_the_model_or_classically_masks_the_disc() {
    let mut s = disc_session();
    let out = s.execute("layer.removeBackground", json!({})).expect("removeBackground runs");
    // The disc: centre (100, 75), radius 40 → the mask's bounds are its box.
    assert_eq!(out["bounds"], json!([60, 35, 80, 80]), "{out}");
}
