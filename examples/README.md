# examples

Runnable examples (the root deliberately has no cargo workspace, so Rust
examples live in each crate's `examples/`; this directory holds the
cross-crate entry points).

## Full-model finetune (Rust, SmolLM-135M)

[`finetune_smollm.sh`](finetune_smollm.sh) — wraps the `train_smollm`
example from `qlora-model`: r=8 LoRA on Q/V projections of all 30 layers,
paged 8-bit Adam, 30 steps with loss 8.07 → 0.58 (release, ~2 minutes).
Checks for the `models/` weights and downloads them via the guix shell
if missing.

```sh
./examples/finetune_smollm.sh [steps] [lr]   # defaults: 30 0.005
```

The implementation is
[`qlora-model/examples/train_smollm.rs`](../qlora-model/examples/train_smollm.rs).

## Single-layer QLoRA training (Python)

[`train_qlora_layer.py`](train_qlora_layer.py) — uses only the public API
(`quantize_nf4`, `qlora_linear_forward/backward`, `PagedAdam`):
a frozen NF4 base plus a trainable adapter fitting a rank-4 task delta,
200 steps with loss 0.67 → ~0. Build the module with maturin first
(see `qlora-python` instructions):

```sh
qlora-python/.venv/bin/python examples/train_qlora_layer.py [--use-gpu]
```

## Per-crate examples (also reachable via `cargo run -p <crate> --example <name>` from the root workspace)

* `qlora-model/examples/train_smollm.rs` — what the sh script above wraps
* `qlora-core/examples/` — `train_lora` (single-layer LoRA regression,
  20 CPU steps), `qlora_forward` (single-layer forward)
* `qlora-io/examples/` — `inspect` (reads torch-exported `.npz`),
  `inspect_st` (reads `.safetensors` and counts parameters)
