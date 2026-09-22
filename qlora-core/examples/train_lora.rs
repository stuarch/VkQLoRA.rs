//! Train a LoRA adapter with paged 8-bit Adam on synthetic data.
//!
//! Run with: `cargo run --example train_lora` (inside `qlora-core/`).
//! Each step: forward -> MSE loss -> backward -> Adam update.

use qlora_core::{Adam8bit, AdamConfig, LoraAdapter, QloraLinear, QuantConfig};

fn main() {
    let (out_dim, in_dim, r, batch) = (8usize, 16usize, 4usize, 4usize);
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 37) % 17) as f32 / 17.0 - 1.0)
        .collect();
    let x: Vec<f32> = (0..batch * in_dim)
        .map(|i| ((i * 3) % 11) as f32 / 11.0 - 0.5)
        .collect();
    let target = vec![0.0f32; batch * out_dim];

    let mut params = vec![vec![0.01f32; r * in_dim], vec![0.01f32; out_dim * r]];
    let mut opt = Adam8bit::new(AdamConfig {
        lr: 0.05,
        max_resident_pages: 2,
        ..Default::default()
    })
    .unwrap();

    for step in 0..=20 {
        let adapter = LoraAdapter::new(&params[0], &params[1], in_dim, out_dim, r, 8.0).unwrap();
        let layer =
            QloraLinear::new(&w, out_dim, in_dim, &QuantConfig::default(), Some(adapter)).unwrap();
        let y = layer.forward(&x, batch).unwrap();
        let loss: f32 = y
            .iter()
            .zip(target.iter())
            .map(|(u, v)| (u - v).powi(2))
            .sum::<f32>()
            / y.len() as f32;
        if step % 5 == 0 {
            println!("step {step:3}: loss = {loss:.6}");
        }
        let dy: Vec<f32> = y
            .iter()
            .zip(target.iter())
            .map(|(u, v)| 2.0 * (u - v) / y.len() as f32)
            .collect();
        let g = layer.backward(&x, &dy, batch).unwrap();
        let grads = vec![g.grad_a.unwrap(), g.grad_b.unwrap()];
        opt.step(&mut params, &grads).unwrap();
    }
    println!("adam state: {} bytes", opt.state_bytes());
}
