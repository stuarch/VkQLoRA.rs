//! `qlora-wgpu`: WGPU compute backend for QLoRA operations.
//!
//! The GPU path mirrors [`qlora_core::QloraLinear`] with compute kernels
//! (see `shaders/`):
//!
//! 1. `dequant_nf4`: packed NF4 codes + fp32 block scales -> fp32 weights.
//! 2. `gemm` (one module, three entry points, all 16x16 tiled with shared
//!    workgroup memory): `mm_nn` (`C = A·B`), `mm_nt` (`C = A·B^T`),
//!    `mm_tn` (`C = A^T·B`).
//!
//! A fused QLoRA forward is therefore `dequant` + up to three GEMM
//! dispatches (base, `X·A^T`, `·B^T`; the `alpha/r` scaling is folded into
//! the uploaded `A` matrix, so no extra kernel is needed). The backward
//! pass reuses the same kernels in transposed/plain combinations, with the
//! upstream gradient pre-scaled on the CPU.
//!
//! [`GpuContext::try_new_blocking`] returns `Ok(None)` when no GPU adapter
//! is available (headless CI, no drivers, ...). All entry points fall back
//! to the CPU reference in that case, so code using this crate keeps
//! working everywhere.
//!
//! Requires network once to fetch `wgpu`/`pollster`/`bytemuck` from
//! crates.io; afterwards `cargo build --offline` works.

pub mod context;
pub mod kernels;

pub use context::GpuContext;
pub use kernels::{
    gpu_dequantize, gpu_matmul, gpu_matmul_trans_a, gpu_matmul_trans_b, gpu_qlora_backward,
    gpu_qlora_forward,
};
