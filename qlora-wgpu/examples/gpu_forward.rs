//! GPU QLoRA forward with automatic CPU fallback.
//!
//! Run with: `cargo run --example gpu_forward` (inside `qlora-wgpu/`).
//! Needs network once to fetch `wgpu` from crates.io.

use qlora_core::{LoraAdapter, QloraLinear, QuantConfig};
use qlora_wgpu::{gpu_qlora_forward, GpuContext};

fn main() {
    let (out_dim, in_dim, r) = (16usize, 32usize, 4usize);
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32).sin() * 0.8)
        .collect();
    let a: Vec<f32> = (0..r * in_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();
    let b: Vec<f32> = (0..out_dim * r)
        .map(|i| (i as f32 * 0.02).cos() * 0.1)
        .collect();
    let adapter = LoraAdapter::new(&a, &b, in_dim, out_dim, r, 8.0).unwrap();
    let layer =
        QloraLinear::new(&w, out_dim, in_dim, &QuantConfig::default(), Some(adapter)).unwrap();
    let x: Vec<f32> = (0..2 * in_dim).map(|i| (i as f32 * 0.05).sin()).collect();

    let ctx = GpuContext::try_new_blocking().unwrap();
    match &ctx {
        Some(_) => println!("backend: GPU (WGPU)"),
        None => println!("backend: CPU fallback (no adapter)"),
    }
    let y = gpu_qlora_forward(ctx.as_ref(), &layer, &x, 2).unwrap();
    println!("output shape: (2, {out_dim}), y[0..4] = {:?}", &y[..4]);
}
