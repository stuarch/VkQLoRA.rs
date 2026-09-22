//! Finetune SmolLM-135M LoRA adapters (Q/V) on one prompt, CPU.
//!
//! Run with: `cargo run --release --example train_smollm` (inside
//! `qlora-model/`; debug works too but is ~10x slower).
//! Each step: forward -> causal-LM cross-entropy -> backward -> paged
//! 8-bit Adam. Prints loss per step; it should go down.

use qlora_core::{Adam8bit, AdamConfig};
use qlora_model::{softmax_cross_entropy, AdapterConfig, LlamaConfig, LlamaModel, ProjKind};
use qlora_tokenizer::Tokenizer;
use std::collections::HashMap;

fn main() {
    let steps = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5usize);
    let lr: f32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.005);

    let tok_bytes = std::fs::read("../models/SmolLM-135M/tokenizer.json").unwrap();
    let tok = Tokenizer::from_json(&tok_bytes).unwrap();
    let text = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>";
    let ids = tok.encode(text).unwrap();
    println!("tokens: {} {:?}", ids.len(), &ids[..8.min(ids.len())]);
    let input = &ids[..ids.len() - 1];
    let targets = &ids[1..];

    let weights = std::fs::read("../models/SmolLM-135M/model.safetensors").unwrap();
    let t0 = std::time::Instant::now();
    let mut model = LlamaModel::load(
        &weights,
        &LlamaConfig::smollm_135m(),
        &AdapterConfig::qv(8, 16.0),
    )
    .unwrap();
    println!("load+quantize: {:?}", t0.elapsed());

    // Stable optimizer order: layers reversed (matches backward push), Q then V.
    let order: Vec<(usize, ProjKind)> = (0..model.cfg.layers)
        .rev()
        .flat_map(|l| [(l, ProjKind::Q), (l, ProjKind::V)])
        .collect();
    let mut opt = Adam8bit::new(AdamConfig {
        lr,
        max_resident_pages: 8,
        ..Default::default()
    })
    .unwrap();

    for step in 0..steps {
        let t0 = std::time::Instant::now();
        let logits = model.forward(input);
        let (loss, dlogits) = softmax_cross_entropy(&logits, targets, input.len(), model.cfg.vocab);
        let grads = model.backward(input, &dlogits);
        let gmap: HashMap<(usize, ProjKind), (Vec<f32>, Vec<f32>)> = grads
            .into_iter()
            .map(|g| ((g.layer, g.kind), (g.grad_a, g.grad_b)))
            .collect();
        // Flatten params as [A, B, A, B, ...] in `order`.
        let mut params: Vec<Vec<f32>> = Vec::with_capacity(order.len() * 2);
        let mut gs: Vec<Vec<f32>> = Vec::with_capacity(order.len() * 2);
        for (l, k) in &order {
            let (a, b) = model.adapter_ab(*l, *k).unwrap();
            let (ga, gb) = &gmap[&(*l, *k)];
            params.push(a);
            params.push(b);
            gs.push(ga.clone());
            gs.push(gb.clone());
        }
        opt.step(&mut params, &gs).unwrap();
        for (i, (l, k)) in order.iter().enumerate() {
            model
                .set_adapter_weights(*l, *k, &params[2 * i], &params[2 * i + 1])
                .unwrap();
        }
        println!("step {step}: loss = {loss:.4} ({:?})", t0.elapsed());
    }
    println!("adam state: {} bytes", opt.state_bytes());
}
