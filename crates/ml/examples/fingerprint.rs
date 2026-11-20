//! The intake step of docs/fork/models.md ("下载 → blake3 指纹 → 加 ModelFile 常量 + 本表
//! 一行"): print a model file's streamed blake3, the same fingerprint `models::matches`
//! checks.
//!
//! Run: cargo run -p photocraft-ml --example fingerprint -- <file>

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: fingerprint <model-file>");
        return;
    };
    match photocraft_ml::models::fingerprint_of(std::path::Path::new(&path)) {
        Some(hash) => println!("{hash}"),
        None => eprintln!("could not read {path}"),
    }
}
