//! GPU/CPU parity: when a GPU is present the kernels must agree with the
//! CPU reference; without one the fallback path is exercised instead, so
//! this test passes on headless machines too.

use qlora_core::{LoraAdapter, QloraLinear, QuantConfig};
use qlora_wgpu::{gpu_dequantize, gpu_matmul_trans_b, gpu_qlora_forward, GpuContext};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn gpu_matches_cpu_reference() {
    let ctx = GpuContext::try_new_blocking().expect("context init must not fail");
    let ctx = ctx.as_ref(); // None on headless machines -> CPU fallback path.
    println!("gpu available: {}", ctx.is_some());

    let (out_dim, in_dim, r) = (8usize, 32usize, 4usize);
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 11) % 13) as f32 / 13.0 - 0.5)
        .collect();
    let cfg = QuantConfig::default();
    let layer = QloraLinear::new(
        &w,
        out_dim,
        in_dim,
        &cfg,
        Some(
            LoraAdapter::new(
                &vec![0.05; r * in_dim],
                &vec![0.07; out_dim * r],
                in_dim,
                out_dim,
                r,
                8.0,
            )
            .unwrap(),
        ),
    )
    .unwrap();

    // Dequant parity.
    let dq_gpu = gpu_dequantize(ctx, layer.weight()).unwrap();
    let dq_cpu = layer.weight().dequantize();
    assert!(max_abs_diff(&dq_gpu, &dq_cpu) < 1e-5);

    // GEMM parity against the CPU reference (same loop order by design).
    let a = vec![0.25f32; 2 * in_dim];
    let g = gpu_matmul_trans_b(ctx, &a, &dq_cpu, 2, in_dim, out_dim).unwrap();
    let c = qlora_core::lora::matmul_trans_b(&a, &dq_cpu, 2, in_dim, out_dim);
    assert!(max_abs_diff(&g, &c) < 1e-4);

    // Fused forward parity.
    let y_gpu = gpu_qlora_forward(ctx, &layer, &a, 2).unwrap();
    let y_cpu = layer.forward(&a, 2).unwrap();
    assert!(max_abs_diff(&y_gpu, &y_cpu) < 1e-3);
}
