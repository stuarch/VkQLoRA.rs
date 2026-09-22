//! `qlora-core`: dependency-free CPU reference implementation of QLoRA ops.
//!
//! QLoRA (Dettmers et al., 2023) = frozen 4-bit quantized base weights
//! (NF4, block-wise, with double quantization) + trainable low-rank
//! adapters (LoRA). A fused forward computes:
//!
//! ```text
//! Y = X · dequant(W_q)^T + (alpha / r) · (X · A^T) · B^T
//! ```
//!
//! with `X: (m, k)`, `W_q: (n, k)` quantized, `A: (r, k)`, `B: (n, r)`.
//!
//! This crate is intentionally dependency-free so it always builds,
//! including offline. The GPU backend lives in `qlora-wgpu` and the
//! Python bindings in `qlora-python`; both reuse these exact types and
//! layouts.
//!
//! # Example
//!
//! ```
//! use qlora_core::{QloraLinear, QuantConfig, LoraAdapter};
//!
//! // 4x8 base weight, rank-2 adapter.
//! let w: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
//! let a: Vec<f32> = vec![0.1; 2 * 8];
//! let b: Vec<f32> = vec![0.2; 4 * 2];
//! let layer = QloraLinear::new(
//!     &w, 4, 8, &QuantConfig::default(),
//!     Some(LoraAdapter::new(&a, &b, 8, 4, 2, 16.0).unwrap()),
//! )
//! .unwrap();
//! let x = vec![1.0f32; 1 * 8];
//! let y = layer.forward(&x, 1).unwrap();
//! assert_eq!(y.len(), 4);
//! ```

pub mod error;
pub mod lora;
pub mod optim;
pub mod qlora;
pub mod quant;

pub use error::QloraError;
pub use lora::LoraAdapter;
pub use optim::{Adam8bit, AdamConfig, AdamStats};
pub use qlora::{QloraGrads, QloraLinear};
pub use quant::{QuantConfig, QuantizedTensor};
