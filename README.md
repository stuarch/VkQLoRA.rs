# VkQLoRA.rs

A QLoRA library in Rust + WGPU (Vulkan), callable from both Rust and Python.
Reproducible build environment via Guix (`guix.scm`).

QLoRA (Dettmers et al., 2023) = frozen 4-bit quantized base weights
(NF4, block-wise, double quantization) + trainable low-rank adapters (LoRA).
Fused forward:

```text
Y = X · dequant(W_q)^T + (alpha / r) · (X · A^T) · B^T
```

## Project structure

Six crates under one cargo workspace (root `Cargo.toml`, virtual manifest).
Each crate also builds standalone from its own directory, so the
zero-dependency `qlora-core` always builds offline:

| Directory | Description | Dependencies |
|---|---|---|
| `qlora-core/` | CPU reference: NF4 quantize/dequantize, LoRA, fused `QloraLinear` forward+backward, paged 8-bit Adam | zero deps |
| `qlora-wgpu/` | WGPU backend: `dequant_nf4` + tiled `gemm` (`mm_nn`/`mm_nt`/`mm_tn`, 16×16 workgroup tiling), forward+backward, automatic CPU fallback without a GPU | `wgpu 22`, `pollster`, `bytemuck` |
| `qlora-python/` | PyO3 bindings (module `qlora_wgpu`): quantize, forward, backward, `PagedAdam`, built with maturin | `pyo3 0.22` |
| `qlora-io/` | Zero-dependency `.npy`/`.npz`/`.safetensors` reader/writer (pure Rust, never touches pickle) | zero deps |
| `qlora-tokenizer/` | BPE from `tokenizer.json` (SmolLM GPT-2-style recipe, id-for-id match with HF transformers) | `fancy-regex` |
| `qlora-model/` | Llama forward＋backward (fp32 CPU), Q/V adapter mounting, SmolLM-135M parity | core/io |
| `models/` | Local weight cache (not versioned): SmolLM-135M safetensors＋tokenizer | — |
| `tools/` | `.pt` boundary bridges: `pt2npz.py` (in), `npz2pt.py` (out, back to the PEFT ecosystem) | torch (from the shell) |
| `examples/` | Runnable example index: `finetune_smollm.sh` (full-model finetune), `train_qlora_layer.py` (Python single-layer training) | — |
| `guix.scm` | Guix dev environment: Rust, Python, maturin, Vulkan/Mesa, PyTorch | — |

All matrices are row-major; see
[qlora-core/src/quant.rs](qlora-core/src/quant.rs) for the NF4 codebook and
packing layout and
[qlora-wgpu/src/kernels.rs](qlora-wgpu/src/kernels.rs) for the GPU upload format.

## Building with Guix

```sh
# Enter the dev environment (rustc/cargo, python, maturin, vulkan-loader, mesa…)
guix shell -f guix.scm

# Inside the shell — whole workspace from the root:
cargo test --offline                           # everything at once
cargo test -p qlora-core                       # or one crate
# Per-crate standalone still works too:
cd qlora-core && cargo test --offline          # zero deps, works offline
cd ../qlora-wgpu && cargo test                 # needs network for wgpu (first time)
cd ../qlora-python && maturin develop          # needs a venv first (next section)
```

TLS note: if cargo inside the shell cannot reach crates.io, run

```sh
export SSL_CERT_FILE="$GUIX_ENVIRONMENT/etc/ssl/certs/ca-certificates.crt"
```

## Rust usage

```rust
use qlora_core::{LoraAdapter, QloraLinear, QuantConfig};

let layer = QloraLinear::new(
    &w_fp32, out_dim, in_dim,
    &QuantConfig::default(),               // block 64 + double quant
    Some(LoraAdapter::new(&a, &b, in_dim, out_dim, r, alpha)?),
)?;
let y = layer.forward(&x, batch)?;         // x: (batch, in_dim)
```

GPU version (`qlora-wgpu`, same semantics; falls back to CPU with no adapter):

```rust
let ctx = GpuContext::try_new_blocking()?; // Ok(None) = no GPU
let y = gpu_qlora_forward(ctx.as_ref(), &layer, &x, batch)?;
```

## Python usage

```sh
cd qlora-python
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop
python -m pytest tests/
```

```python
import qlora_wgpu
q = qlora_wgpu.quantize_nf4(weights, rows, cols)   # NF4 + double quant
y = qlora_wgpu.qlora_linear_forward(
    x, batch, q,
    lora_a=a, lora_b=b, lora_alpha=8.0,
    use_gpu=True,    # automatic CPU fallback without a GPU
)

# training: backward + paged 8-bit Adam
g = qlora_wgpu.qlora_linear_backward(x, dy, batch, q, a, b, 8.0)
opt = qlora_wgpu.PagedAdam(lr=1e-3, max_resident_pages=4)
a, b = opt.step([a, b], [g.grad_a, g.grad_b])
```

## SmolLM-135M finetune (end-to-end smoke)

```sh
# fetch weights first (once; models/ is not versioned)
guix shell -f guix.scm -- python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceTB/SmolLM-135M', local_dir='models/SmolLM-135M',
                  allow_patterns=['*.safetensors','*.json'])"

# tokenizer id-for-id alignment with transformers
# (40 sentences incl. Chinese/emoji/code; expectations embedded in source)
cd qlora-tokenizer && cargo test --offline --test parity
# forward parity with torch (23 parameter tensors, max_err < 3e-5;
# reference file lives in tests/fixtures)
cd ../qlora-model && cargo test --offline --test parity
# both of the above need models/ downloaded first (not versioned, see top
# of this section)
# backward unit test (finite differences) + finetune smoke: Q/V adapters (r=8),
# 30 steps, loss 8.07 → 0.58 (release, ~2 minutes, needs models/)
cargo test --offline --test backward
cargo run --release --offline --example train_smollm -- 30
```

Runnable cross-crate examples live in [`examples/`](examples/README.md):
`finetune_smollm.sh` (the finetune above as a one-shot script) and
`train_qlora_layer.py` (Python single-layer QLoRA training,
200 steps, loss 0.67 → ~0).

## Verification status (measured 2026-09-22)

* `qlora-core`: **24 tests** (unit 6 + optim 4 + qlora 6 + quant 7 +
  doctest 1) all green, `cargo clippy` zero warnings, `cargo fmt --check`
  clean, runs offline; `examples/train_lora` 20 steps, loss 0.257 → ~0.05.
* `qlora-wgpu`: builds, clippy/fmt clean; **3 tests** pass on the **real GPU
  path** (`gpu available: true`) on lavapipe (software Vulkan):
  tiled GEMM entries match CPU on non-multiple-of-16 sizes,
  fused forward/backward parity; automatic fallback without a GPU (also green).
  Command: `VK_ICD_FILENAMES=.../lvp_icd.x86_64.json cargo test`
* `qlora-python`: `maturin develop` works, `pytest tests/` **5 passed**
  (incl. backward finite differences, end-to-end Adam training with falling
  loss); `use_gpu=True` path differs from CPU by exactly 0.
* `guix.scm`: `guix shell -f guix.scm` provides python 3.12, cargo, maturin,
  pytest, python-pytorch 2.10 and Vulkan ICDs (incl. lavapipe), verified.
* `qlora-io`: **16 tests** (unit 6 + numpy golden 9 + doctest 1) all green,
  clippy/fmt clean; both directions proven interoperable against real
  numpy 2.4.6.
* `tools/`: `.pt → .npz → .pt` round-trips with zero error inside the shell
  (fp16/fp32/int64, `state_dict` wrappers); the Rust `inspect` example
  reads torch-exported `.npz`.
* `qlora-tokenizer`: `tests/parity.rs` uses transformers as the oracle,
  40 sentences match id-for-id (incl. Chinese/emoji/code).
* `qlora-model`: `tests/parity.rs` passes forward parity against the torch
  reference (23 parameter tensors, max_err < 3e-5); `tests/backward.rs`
  checks the backward pass with finite differences;
  `examples/train_smollm` trains Q/V adapters on SmolLM-135M for 30 steps
  with falling loss (see previous section; fp32 CPU + 8-bit Adam,
  zero-initialized adapters).

## Scope (honest version)

* Done: full QLoRA forward (NF4/DQ quantization, LoRA, fused linear) +
  **backpropagation** (frozen base, adapters only; `QloraLinear::backward` /
  `gpu_qlora_backward`, both verified with finite differences) +
  **paged 8-bit Adam** (block-wise INT8 state, `v` in sqrt-domain;
  paging bit-identical to fully-resident; checked against an f64 reference
  plus a quadratic convergence test) +
  **GEMM tiling** (16×16 workgroup tiling, all three transpose combos,
  non-multiple-of-16 sizes covered by parity tests).
* Design differences: the sqrt-domain quantization of `v` is a simplified
  version (bitsandbytes uses a dynamic-exponent map); CPU paging is a
  streaming-window scheme, mathematically exactly equal to fully-resident
  (asserted).
* Not done: distributed training, more aggressive kernel optimization
  (double-buffering, vec4 access); `qlora-model` finetuning is currently an
  fp32 CPU validation path — wiring up the NF4 base + WGPU forward/backward
  is the next step (single-layer `QloraLinear` is fully verified).
