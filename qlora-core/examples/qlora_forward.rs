//! Minimal end-to-end example: quantize a base weight, attach LoRA, run forward.
//!
//! Run with: `cargo run --example qlora_forward` (inside `qlora-core/`).

use qlora_core::{LoraAdapter, QloraLinear, QuantConfig};

fn main() {
    let (out_dim, in_dim, r) = (16usize, 32usize, 4usize);

    // Fake "pretrained" weight: smooth values in [-1, 1].
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32).sin() * 0.8)
        .collect();

    let cfg = QuantConfig::default();
    let probe = qlora_core::QuantizedTensor::quantize(&w, out_dim, in_dim, &cfg).unwrap();
    println!(
        "weight: {} fp32 bytes -> {} quantized bytes ({:.2}x)",
        w.len() * 4,
        probe.storage_bytes(),
        w.len() as f64 * 4.0 / probe.storage_bytes() as f64
    );

    // Fake adapter.
    let a: Vec<f32> = (0..r * in_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();
    let b: Vec<f32> = (0..out_dim * r)
        .map(|i| (i as f32 * 0.02).cos() * 0.1)
        .collect();
    let adapter = LoraAdapter::new(&a, &b, in_dim, out_dim, r, 8.0).unwrap();

    let layer = QloraLinear::new(&w, out_dim, in_dim, &cfg, Some(adapter)).unwrap();

    // Batch of 2 tokens.
    let x: Vec<f32> = (0..2 * in_dim).map(|i| (i as f32 * 0.05).sin()).collect();
    let y = layer.forward(&x, 2).unwrap();
    println!("output shape: (2, {out_dim}), y[0..4] = {:?}", &y[..4]);
}
