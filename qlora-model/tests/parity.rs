//! SmolLM-135M forward parity against `transformers` (fp32 CPU).
//!
//! Reference: `last_logits.npy` = `AutoModelForCausalLM(...,
//! torch_dtype=float32)([[1, 4093, 198, 26843]]).logits[0, -1]`, i.e. the
//! last row for "<|im_start|>user\nHi". Takes ~6s in debug (30 layers).

use qlora_io::read_npy;
use qlora_model::{LlamaConfig, LlamaModel};
use std::path::PathBuf;

fn root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p
}

#[test]
fn forward_matches_transformers_last_row() {
    let bytes = std::fs::read(root().join("models/SmolLM-135M/model.safetensors")).unwrap();
    let model =
        LlamaModel::load(&bytes, &LlamaConfig::smollm_135m(), &AdapterConfig::none()).unwrap();
    let ids = [1u32, 4093, 198, 26843];
    let logits = model.forward(&ids);

    let expected = read_npy(
        &std::fs::read(root().join("qlora-model/tests/fixtures/last_logits.npy")).unwrap(),
    )
    .unwrap();
    assert_eq!(expected.shape, vec![49152]);
    let got = &logits[3 * 49152..4 * 49152];
    assert_eq!(got.len(), expected.data.len());
    // Different matmul order than BLAS: compare with tolerance + exact top-1.
    let max_err = got
        .iter()
        .zip(expected.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 5e-4, "max abs err {max_err}");
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    };
    assert_eq!(argmax(got), 28);
    assert_eq!(argmax(&expected.data), 28);
}
