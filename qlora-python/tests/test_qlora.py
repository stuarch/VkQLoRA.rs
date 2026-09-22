"""Python-side tests for qlora_wgpu (needs `maturin develop` first).

Run with: `python -m pytest tests/` (inside `qlora-python/`, venv active).
"""

import qlora_wgpu


def test_quantize_roundtrip():
    rows, cols = 8, 32
    w = [((i * 11) % 13) / 13.0 - 0.5 for i in range(rows * cols)]
    q = qlora_wgpu.quantize_nf4(w, rows, cols)
    assert q.rows == rows and q.cols == cols
    assert len(q.codes_list) == (rows * cols + 1) // 2
    back = q.dequantize()
    assert len(back) == rows * cols
    mse = sum((b - o) ** 2 for b, o in zip(back, w)) / len(w)
    assert mse < 0.01, mse


def test_qlora_forward_parity_with_plain_linear():
    out_dim, in_dim, r, batch = 4, 16, 2, 2
    w = [((i * 37) % 17) / 8.0 - 1.0 for i in range(out_dim * in_dim)]
    a = [0.05] * (r * in_dim)
    b = [0.07] * (out_dim * r)
    x = [0.5] * (batch * in_dim)

    q = qlora_wgpu.quantize_nf4(w, out_dim, in_dim)
    y = qlora_wgpu.qlora_linear_forward(x, batch, q, a, b, 8.0)

    # unfused fp32 reference
    scale = 8.0 / r
    expected = []
    for bi in range(batch):
        for oi in range(out_dim):
            base = sum(
                x[bi * in_dim + k] * w[oi * in_dim + k] for k in range(in_dim)
            )
            adapt = scale * sum(
                sum(x[bi * in_dim + k] * a[ri * in_dim + k] for k in range(in_dim))
                * b[oi * r + ri]
                for ri in range(r)
            )
            expected.append(base + adapt)
    assert len(y) == batch * out_dim
    assert max(abs(u - v) for u, v in zip(y, expected)) < 0.2


def test_no_adapter_is_quantized_linear():
    out_dim, in_dim = 4, 16
    w = [0.1] * (out_dim * in_dim)
    q = qlora_wgpu.quantize_nf4(w, out_dim, in_dim)
    y = qlora_wgpu.qlora_linear_forward([1.0] * in_dim, 1, q)
    assert len(y) == out_dim


def test_backward_matches_finite_differences():
    out_dim, in_dim, r, batch = 3, 4, 2, 2
    w = [((i * 37) % 11) / 11.0 - 0.5 for i in range(out_dim * in_dim)]
    a = [((i * 13) % 7) / 7.0 - 0.5 for i in range(r * in_dim)]
    b = [((i * 29) % 5) / 5.0 - 0.5 for i in range(out_dim * r)]
    x = [0.5, -0.25, 0.75, 1.0, 0.0, 0.25, -0.5, 1.5]
    dy = [1.0] * (batch * out_dim)
    q = qlora_wgpu.quantize_nf4(w, out_dim, in_dim)

    g = qlora_wgpu.qlora_linear_backward(x, dy, batch, q, a, b, 8.0)
    assert len(g.grad_x) == batch * in_dim
    assert len(g.grad_a) == r * in_dim
    assert len(g.grad_b) == out_dim * r

    def loss(x_, a_, b_):
        return sum(qlora_wgpu.qlora_linear_forward(x_, batch, q, a_, b_, 8.0))

    def fd(buf, idx):
        eps = 1e-3
        bufs = [list(x), list(a), list(b)]
        bufs[buf][idx] += eps
        hi = loss(*bufs)
        bufs[buf][idx] -= 2 * eps
        lo = loss(*bufs)
        return (hi - lo) / (2 * eps)

    for i, got in enumerate(g.grad_x):
        assert abs(got - fd(0, i)) < 2e-2, (i, got)
    for i, got in enumerate(g.grad_a):
        assert abs(got - fd(1, i)) < 2e-2, (i, got)
    for i, got in enumerate(g.grad_b):
        assert abs(got - fd(2, i)) < 2e-2, (i, got)


def test_paged_adam_trains_adapter_downhill():
    # One full step: forward -> backward -> Adam must reduce sum-of-squares.
    out_dim, in_dim, r, batch = 4, 8, 2, 2
    w = [((i * 37) % 17) / 8.0 - 1.0 for i in range(out_dim * in_dim)]
    a = [0.05] * (r * in_dim)
    b = [0.07] * (out_dim * r)
    x = [0.5] * (batch * in_dim)
    target = [0.0] * (batch * out_dim)
    q = qlora_wgpu.quantize_nf4(w, out_dim, in_dim)
    opt = qlora_wgpu.PagedAdam(lr=0.05, max_resident_pages=1)

    def mse(a_, b_):
        y = qlora_wgpu.qlora_linear_forward(x, batch, q, a_, b_, 8.0)
        return sum((u - v) ** 2 for u, v in zip(y, target)) / len(y)

    before = mse(a, b)
    for _ in range(5):
        y = qlora_wgpu.qlora_linear_forward(x, batch, q, a, b, 8.0)
        dy = [2 * (u - v) / len(y) for u, v in zip(y, target)]
        g = qlora_wgpu.qlora_linear_backward(x, dy, batch, q, a, b, 8.0)
        a, b = opt.step([a, b], [g.grad_a, g.grad_b])
    after = mse(a, b)
    assert after < before, (before, after)
    # 8-bit states stay compact: 2 params x (r*in + out*r) int8 + scales.
    assert opt.state_bytes() < 8 * (len(a) + len(b))
