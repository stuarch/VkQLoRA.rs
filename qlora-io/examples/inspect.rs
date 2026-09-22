//! Inspect an .npz file: `cargo run --example inspect -- file.npz`.

use qlora_io::read_npz;

fn main() {
    let path = std::env::args().nth(1).expect("usage: inspect <file.npz>");
    let bytes = std::fs::read(&path).expect("read file");
    let arrays = read_npz(&bytes).expect("parse npz");
    println!("{}: {} arrays", path, arrays.len());
    for a in &arrays {
        let min = a.data.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        let max = a.data.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        println!(
            "  {:32} shape={:?} min={min:.4} max={max:.4}",
            a.name, a.shape
        );
    }
}
