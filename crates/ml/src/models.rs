//! Runtime-loaded model files — the weights past the 50MB commit line (docs/fork's
//! models manifest). No network: the user places the file, this finds it and checks it.
//!
//! Resolution is `PHOTOCRAFT_MODELS_DIR/<canonical filename>` — the one mechanism; an
//! experiment with another export is a symlink or copy under the canonical name. Without
//! the variable (or the file), the stage that wanted the model silently does without. A
//! blake3 fingerprint is checked when the manifest records one: a mismatch warns and
//! proceeds — a broken download fails in the loader anyway.

/// One runtime model: its canonical filename and the manifest's blake3 of it.
pub struct ModelFile {
    pub canonical: &'static str,
    pub blake3: Option<&'static str>,
}

/// The BiSeNet face-parsing export (docs/fork/2026-11-18-portrait-pipeline.md).
pub const FACEPARSER: ModelFile = ModelFile { canonical: "faceparser.onnx", blake3: Some("c1f14e23404535fe2b685dfab09abdb9067b823da32110de4cdb12344d0a8b8e") };

/// The ort-tier matting model (feature `ort`): FeyNoBg/BiRefNet, the hair-edge rung the
/// tract tier cannot run interactively. fp16 export; see docs/fork/models.md.
pub const FEYNOBG: ModelFile = ModelFile { canonical: "feynobg.onnx", blake3: Some("c9634e838aabe6a620f67b7b69a68508a040d9d0afefb02231296bdf7885fcef") };

/// The face detector (SCRFD det_10g): insightface's buffalo_l detection model as hosted by
/// immich-app — the automatic face box of `detect`, the drawn `roi`'s automatic sibling.
pub const FACEDETECTOR: ModelFile = ModelFile { canonical: "facedetector.onnx", blake3: Some("fa3e5d8c62722a7b7122dc185245236589dcf0b3b5c9c5561cc742ce356fba56") };

/// The whole-body six-class segmentation (MediaPipe selfie multiclass, tflite→ONNX
/// conversion): body-skin/face-skin/clothes/hair over the whole document — the body
/// complement of the face-crop parsing stage. Apache-2.0, the one clean-license body
/// parser; see docs/fork/models.md.
pub const SELFIE: ModelFile = ModelFile { canonical: "selfiemulticlass.onnx", blake3: Some("847b94b7812e47cd0daa22986bd1a7fe9fdf846026c38a1922746b78003c3e31") };

/// The face-enhancement model's slot — version-free on purpose: the GFPGAN v1.x line
/// shares the 512² / 0..1 / RGB interface, so another version dropped into
/// `gfpgan.onnx` runs too (a fingerprint mismatch warns, never blocks; register a kept
/// version's hash here and in docs/fork/models.md). The hash below is v1.4's.
pub const GFPGAN: ModelFile = ModelFile { canonical: "gfpgan.onnx", blake3: Some("42a4f8e1a701cd6d56b2087ab00de1f76f4f7c1fe015556fa496d19b37a25d08") };

/// The pure resolution: `dir` joined with the canonical name; a directory that is not
/// there is the same as none.
pub fn resolve(dir: Option<std::ffi::OsString>, model: &ModelFile) -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(dir?).join(model.canonical);
    path.is_file().then_some(path)
}

/// [`resolve`] over the environment: `PHOTOCRAFT_MODELS_DIR`.
pub fn locate(model: &ModelFile) -> Option<std::path::PathBuf> {
    resolve(std::env::var_os("PHOTOCRAFT_MODELS_DIR"), model)
}

/// Whether `path`'s contents fingerprint to what the manifest recorded. `true` when the
/// model has no recorded fingerprint yet (a new file the manifest has not caught up with).
/// A mismatch warns on stderr and answers `false`; the caller proceeds regardless.
pub fn matches(path: &std::path::Path, model: &ModelFile) -> bool {
    model.blake3.is_none_or(|want| matches_hash(path, want))
}

/// [`matches`] against an explicit fingerprint: equal, or a warned mismatch.
fn matches_hash(path: &std::path::Path, want: &str) -> bool {
    let Some(got) = fingerprint(path) else { return false };
    if got == want {
        return true;
    }
    eprintln!("photocraft-ml: {path_display} does not match the recorded fingerprint ({got} ≠ {want}); loading it anyway", path_display = path.display());
    false
}

/// The blake3 of `path`'s contents, streamed; `None` when it cannot be read.
fn fingerprint(path: &std::path::Path) -> Option<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            return Some(hasher.finalize().to_hex().to_string());
        }
        hasher.update(&buf[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dir_and_its_canonical_name_or_nothing() {
        let m = ModelFile { canonical: "m.onnx", blake3: None };
        let dir = std::env::temp_dir();
        let placed = dir.join("m.onnx");
        std::fs::write(&placed, b"x").unwrap();
        assert_eq!(resolve(Some(dir.clone().into_os_string()), &m).as_deref(), Some(placed.as_path()));
        assert!(resolve(Some(dir.join("missing").into_os_string()), &m).is_none(), "a dir without the file is no model");
        assert!(resolve(None, &m).is_none());
        let _ = std::fs::remove_file(&placed);
    }

    #[test]
    fn a_fingerprint_match_and_a_mismatch() {
        let path = std::env::temp_dir().join("photocraft-ml-fingerprint-test.bin");
        std::fs::write(&path, b"weights").unwrap();
        let recorded = fingerprint(&path).unwrap();
        assert!(matches_hash(&path, &recorded));
        assert!(!matches_hash(&path, "0000000000000000000000000000000000000000000000000000000000000000"));
        let _ = std::fs::remove_file(&path);
    }
}
