//! `qlora-model`: Llama-architecture forward pass (SmolLM-135M).
//!
//! Loads `.safetensors` weights via `qlora-io` and runs a plain fp32 CPU
//! forward: embedding gather, 30x (RMSNorm, GQA attention with RoPE,
//! SwiGLU MLP, residuals), final norm, tied LM head.
//!
//! Conventions (matching HuggingFace `transformers` Llama, the parity
//! oracle): weights stored row-major `(out, in)`; RoPE uses `rotate_half`
//! with `inv_freq[i] = theta^(-2i/d)`; attention scale `1/sqrt(head_dim)`;
//! no biases anywhere; embeddings tied to the LM head.
//!
//! # Example
//!
//! ```no_run
//! use qlora_model::{AdapterConfig, LlamaConfig, LlamaModel};
//!
//! let bytes = std::fs::read("models/SmolLM-135M/model.safetensors").unwrap();
//! let model = LlamaModel::load(&bytes, &LlamaConfig::smollm_135m(), &AdapterConfig::none()).unwrap();
//! let logits = model.forward(&[1, 2, 3]);
//! assert_eq!(logits.len(), 3 * 49152);
//! ```

pub mod llama;

pub use llama::{
    softmax_cross_entropy, AdapterConfig, AdapterGrad, LayerWeights, LlamaConfig, LlamaModel, Proj,
    ProjKind,
};
