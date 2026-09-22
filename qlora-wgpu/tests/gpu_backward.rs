//! GPU backward + tiled-GEMM parity.
//!
//! Sizes are deliberately NOT multiples of 16 (batch 3, in 20, out 18,
//! r 5) so the tile-boundary guards in `shaders/gemm.wgsl` are exercised.
//! Without a GPU the CPU-fallback path runs instead (still green).

use qlora_core::{lora, LoraAdapter, QloraLinear, QuantConfig};
use qlora_wgpu::{
    gpu_matmul, gpu_matmul_trans_a, gpu_matmul_trans_b, gpu_qlora_backward, gpu_qlora_forward,
    GpuContext,
};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn gpu_tiled_gemms_match_cpu_on_odd_sizes() {
    let ctx = GpuContext::try_new_blocking().expect("context init must not fail");
    let ctx = ctx.as_ref();
    println!("gpu available: {}", ctx.is_some());

    // (m, k, n) = (18, 20, 33): nothing is a multiple of 16.
    let (m, k, n) = (18usize, 20usize, 33usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 11) % 13) as f32 / 13.0 - 0.5)
        .collect();
    let b_nn: Vec<f32> = (0..k * n)
        .map(|i| ((i * 7) % 11) as f32 / 11.0 - 0.5)
        .collect();
    let b_nt: Vec<f32> = (0..n * k)
        .map(|i| ((i * 5) % 9) as f32 / 9.0 - 0.5)
        .collect();
    let b_tn: Vec<f32> = (0..m * n)
        .map(|i| ((i * 3) % 7) as f32 / 7.0 - 0.5)
        .collect();

    let g = gpu_matmul(ctx, &a, &b_nn, m, k, n).unwrap();
    let c = lora::matmul(&a, &b_nn, m, k, n);
    assert!(max_abs_diff(&g, &c) < 1e-3, "mm_nn");

    let g = gpu_matmul_trans_b(ctx, &a, &b_nt, m, k, n).unwrap();
    let c = lora::matmul_trans_b(&a, &b_nt, m, k, n);
    assert!(max_abs_diff(&g, &c) < 1e-3, "mm_nt");

    let g = gpu_matmul_trans_a(ctx, &a, &b_tn, m, k, n).unwrap();
    let c = lora::matmul_trans_a(&a, &b_tn, m, k, n);
    assert!(max_abs_diff(&g, &c) < 1e-3, "mm_tn");
}

#[test]
fn gpu_backward_matches_cpu_on_odd_sizes() {
    let ctx = GpuContext::try_new_blocking().expect("context init must not fail");
    let ctx = ctx.as_ref();
    println!("gpu available: {}", ctx.is_some());

    let (out_dim, in_dim, r, batch) = (18usize, 20usize, 5usize, 3usize);
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 37) % 17) as f32 / 17.0 - 1.0)
        .collect();
    let a: Vec<f32> = (0..r * in_dim)
        .map(|i| ((i * 13) % 7) as f32 / 7.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..out_dim * r)
        .map(|i| ((i * 29) % 5) as f32 / 5.0 - 0.5)
        .collect();
    let layer = QloraLinear::new(
        &w,
        out_dim,
        in_dim,
        &QuantConfig::default(),
        Some(LoraAdapter::new(&a, &b, in_dim, out_dim, r, 8.0).unwrap()),
    )
    .unwrap();
    let x: Vec<f32> = (0..batch * in_dim)
        .map(|i| ((i * 3) % 11) as f32 / 11.0 - 0.5)
        .collect();
    let dy: Vec<f32> = (0..batch * out_dim)
        .map(|i| ((i * 7) % 13) as f32 / 13.0 - 0.5)
        .collect();

    // Forward parity on odd sizes too (tiled mm_nt).
    let y_gpu = gpu_qlora_forward(ctx, &layer, &x, batch).unwrap();
    let y_cpu = layer.forward(&x, batch).unwrap();
    assert!(max_abs_diff(&y_gpu, &y_cpu) < 1e-3, "forward");

    let g_gpu = gpu_qlora_backward(ctx, &layer, &x, &dy, batch).unwrap();
    let g_cpu = layer.backward(&x, &dy, batch).unwrap();
    assert!(max_abs_diff(&g_gpu.grad_x, &g_cpu.grad_x) < 1e-2, "grad_x");
    let (ga_g, ga_c) = (g_gpu.grad_a.unwrap(), g_cpu.grad_a.unwrap());
    let (gb_g, gb_c) = (g_gpu.grad_b.unwrap(), g_cpu.grad_b.unwrap());
    assert!(max_abs_diff(&ga_g, &ga_c) < 1e-2, "grad_a");
    assert!(max_abs_diff(&gb_g, &gb_c) < 1e-2, "grad_b");
}
