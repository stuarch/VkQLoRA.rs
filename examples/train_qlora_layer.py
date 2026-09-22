#!/usr/bin/env python3
"""Single-layer QLoRA training example (Python + `qlora_wgpu`).

Task: a frozen NF4 base matrix W0 (random, frozen after quantization) plus
trainable LoRA adapters (A: small random init, B: zero init) fitting a
low-rank delta stacked on the base output: Y = X @ W0' + X @ (U @ V).
The training loop uses only two public APIs:

* `qlora_linear_forward` / `qlora_linear_backward` (forward + backward)
* `PagedAdam.step` (8-bit paged Adam, updates A and B only)

Setup (once):

```sh
cd qlora-python
python -m venv .venv && source .venv/bin/activate
pip install maturin && maturin develop
```

Run (from the repo root):

```sh
qlora-python/.venv/bin/python examples/train_qlora_layer.py --steps 200
```

Expected: loss falls from ~0.6 to below ~0.01 (the delta is rank-r, so the
adapter can express it exactly). `--use-gpu` selects the WGPU backend
(automatic CPU fallback without a GPU).
"""

import argparse
import random

import qlora_wgpu


def matvec_rows(x, mat, rows, cols):
    """x: (batch, cols), mat: (rows, cols) -> (batch, rows); row-major lists."""
    batch = len(x) // cols
    return [
        sum(x[b * cols + k] * mat[o * cols + k] for k in range(cols))
        for b in range(batch)
        for o in range(rows)
    ]


def main():
    ap = argparse.ArgumentParser(description="Single-layer QLoRA training example")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    in_dim, out_dim, rank, batch = 64, 32, 4, 16

    # Frozen base (never touched after quantization). The task is to learn a
    # small low-rank delta stacked on the base output — like real finetuning:
    # base capabilities stay, the adapter learns only the difference.
    w0 = [rng.uniform(-1, 1) for _ in range(out_dim * in_dim)]
    q = qlora_wgpu.quantize_nf4(w0, out_dim, in_dim)
    base = q.dequantize()
    u = [rng.uniform(-0.5, 0.5) for _ in range(out_dim * rank)]
    v = [rng.uniform(-0.5, 0.5) for _ in range(rank * in_dim)]
    delta = [
        sum(u[o * rank + i] * v[i * in_dim + k] for i in range(rank))
        for o in range(out_dim)
        for k in range(in_dim)
    ]
    x = [rng.uniform(-1, 1) for _ in range(batch * in_dim)]
    y_base = matvec_rows(x, base, out_dim, in_dim)
    y_delta = matvec_rows(x, delta, out_dim, in_dim)
    target = [b + d for b, d in zip(y_base, y_delta)]
    # Standard LoRA init: small random A, all-zero B (adapter outputs zero at start).
    a = [rng.uniform(-0.1, 0.1) for _ in range(rank * in_dim)]
    b = [0.0] * (out_dim * rank)
    alpha = float(rank)
    opt = qlora_wgpu.PagedAdam(lr=args.lr, max_resident_pages=4)

    n = float(batch * out_dim)
    for step in range(args.steps):
        y = qlora_wgpu.qlora_linear_forward(
            x, batch, q, a, b, alpha, use_gpu=args.use_gpu
        )
        loss = sum((yi - ti) ** 2 for yi, ti in zip(y, target)) / n
        dy = [2.0 * (yi - ti) / n for yi, ti in zip(y, target)]
        g = qlora_wgpu.qlora_linear_backward(
            x, dy, batch, q, a, b, alpha, use_gpu=args.use_gpu
        )
        a, b = opt.step([a, b], [g.grad_a, g.grad_b])
        if step % 20 == 0 or step == args.steps - 1:
            print(f"step {step}: loss = {loss:.6f}", flush=True)

    print(f"adam state: {opt.state_bytes()} bytes")
    assert loss < 0.01, f"loss did not converge: {loss}"
    print("converged: adapter fit the rank-4 delta on top of NF4 base")


if __name__ == "__main__":
    main()
