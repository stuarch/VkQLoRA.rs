//! Fused QLoRA forward: parity with unfused FP32 math (up to quant noise).

use qlora_core::{LoraAdapter, QloraLinear, QuantConfig};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn qlora_forward_matches_unfused_reference() {
    // out=4, in=8, r=2, alpha=8 -> scale 4.
    let (out_dim, in_dim, r) = (4usize, 8usize, 2usize);
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 37) % 17) as f32 / 8.0 - 1.0)
        .collect();
    let a: Vec<f32> = (0..r * in_dim)
        .map(|i| ((i * 13) % 7) as f32 / 7.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..out_dim * r)
        .map(|i| ((i * 29) % 5) as f32 / 5.0 - 0.5)
        .collect();
    let x = vec![0.5f32; 2 * in_dim]; // batch = 2

    let adapter = LoraAdapter::new(&a, &b, in_dim, out_dim, r, 8.0).unwrap();
    let layer =
        QloraLinear::new(&w, out_dim, in_dim, &QuantConfig::default(), Some(adapter)).unwrap();
    let y = layer.forward(&x, 2).unwrap();

    // Unfused FP32 reference: X·W^T + 4·((X·A^T)·B^T).
    let mut xat = vec![0.0f32; 2 * r];
    for bi in 0..2 {
        for ri in 0..r {
            let mut acc = 0.0f32;
            for k in 0..in_dim {
                acc += x[bi * in_dim + k] * a[ri * in_dim + k];
            }
            xat[bi * r + ri] = acc;
        }
    }
    let mut dxb = vec![0.0f32; 2 * out_dim];
    for bi in 0..2 {
        for oi in 0..out_dim {
            let mut acc = 0.0f32;
            for ri in 0..r {
                acc += xat[bi * r + ri] * b[oi * r + ri];
            }
            dxb[bi * out_dim + oi] = 4.0 * acc;
        }
    }
    let mut base = vec![0.0f32; 2 * out_dim];
    for bi in 0..2 {
        for oi in 0..out_dim {
            let mut acc = 0.0f32;
            for k in 0..in_dim {
                acc += x[bi * in_dim + k] * w[oi * in_dim + k];
            }
            base[bi * out_dim + oi] = acc;
        }
    }
    let expected: Vec<f32> = base.iter().zip(dxb.iter()).map(|(u, v)| u + v).collect();

    let max_err = max_abs_diff(&y, &expected);
    assert!(max_err < 0.2, "max_err={max_err}");
}

#[test]
fn qlora_without_adapter_is_quantized_linear() {
    let (out_dim, in_dim) = (8, 32);
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 11) % 13) as f32 / 13.0 - 0.5)
        .collect();
    let layer = QloraLinear::new(&w, out_dim, in_dim, &QuantConfig::default(), None).unwrap();
    let x = vec![1.0f32; in_dim];
    let y = layer.forward(&x, 1).unwrap();
    assert_eq!(y.len(), out_dim);
    // Output must be close to the exact FP32 linear (quant noise only).
    let exact: Vec<f32> = (0..out_dim)
        .map(|oi| (0..in_dim).map(|k| w[oi * in_dim + k]).sum::<f32>())
        .collect();
    let max_err = max_abs_diff(&y, &exact);
    assert!(max_err < 0.3, "max_err={max_err}");
}

#[test]
fn qlora_backward_matches_finite_differences() {
    // L = sum(Y); central differences on every input, A and B element.
    let (out_dim, in_dim, r, batch) = (3usize, 4usize, 2usize, 2usize);
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 37) % 11) as f32 / 11.0 - 0.5)
        .collect();
    let a: Vec<f32> = (0..r * in_dim)
        .map(|i| ((i * 13) % 7) as f32 / 7.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..out_dim * r)
        .map(|i| ((i * 29) % 5) as f32 / 5.0 - 0.5)
        .collect();
    let x = vec![0.5f32, -0.25, 0.75, 1.0, 0.0, 0.25, -0.5, 1.5];
    assert_eq!(x.len(), batch * in_dim);

    let loss = |x: &[f32], a: &[f32], b: &[f32]| -> f32 {
        let adapter = LoraAdapter::new(a, b, in_dim, out_dim, r, 8.0).unwrap();
        let layer =
            QloraLinear::new(&w, out_dim, in_dim, &QuantConfig::default(), Some(adapter)).unwrap();
        layer.forward(x, batch).unwrap().iter().sum()
    };
    // Central difference of `loss` w.r.t. element `idx` of buffer `buf`
    // (0 = x, 1 = a, 2 = b).
    let fd = |x: &[f32], a: &[f32], b: &[f32], idx: usize, buf: usize| -> f32 {
        let eps = 1e-3f32;
        let perturbed = |sign: f32| -> f32 {
            let mut xp = x.to_vec();
            let mut ap = a.to_vec();
            let mut bp = b.to_vec();
            match buf {
                0 => xp[idx] += sign * eps,
                1 => ap[idx] += sign * eps,
                _ => bp[idx] += sign * eps,
            }
            loss(&xp, &ap, &bp)
        };
        (perturbed(1.0) - perturbed(-1.0)) / (2.0 * eps)
    };

    let dy = vec![1.0f32; batch * out_dim];
    let adapter = LoraAdapter::new(&a, &b, in_dim, out_dim, r, 8.0).unwrap();
    let layer =
        QloraLinear::new(&w, out_dim, in_dim, &QuantConfig::default(), Some(adapter)).unwrap();
    let g = layer.backward(&x, &dy, batch).unwrap();
    let ga = g.grad_a.unwrap();
    let gb = g.grad_b.unwrap();

    for (i, &got) in g.grad_x.iter().enumerate() {
        let want = fd(&x, &a, &b, i, 0);
        assert!((got - want).abs() < 2e-2, "grad_x[{i}]: {got} vs {want}");
    }
    for (i, &got) in ga.iter().enumerate() {
        let want = fd(&x, &a, &b, i, 1);
        assert!((got - want).abs() < 2e-2, "grad_a[{i}]: {got} vs {want}");
    }
    for (i, &got) in gb.iter().enumerate() {
        let want = fd(&x, &a, &b, i, 2);
        assert!((got - want).abs() < 2e-2, "grad_b[{i}]: {got} vs {want}");
    }
}

#[test]
fn qlora_backward_without_adapter_gives_grad_x_only() {
    let (out_dim, in_dim) = (4, 8);
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 11) % 13) as f32 / 13.0 - 0.5)
        .collect();
    let layer = QloraLinear::new(&w, out_dim, in_dim, &QuantConfig::default(), None).unwrap();
    let x = vec![0.5f32; in_dim];
    let dy = vec![1.0f32; out_dim];
    let g = layer.backward(&x, &dy, 1).unwrap();
    assert!(g.grad_a.is_none() && g.grad_b.is_none());
    assert_eq!(g.grad_x.len(), in_dim);
    // dX = W rows summed (dy = ones): each grad_x[k] = sum_o dequant(W)[o][k].
    let wd = layer.weight().dequantize();
    for k in 0..in_dim {
        let want: f32 = (0..out_dim).map(|o| wd[o * in_dim + k]).sum();
        assert!((g.grad_x[k] - want).abs() < 1e-6);
    }
    assert!(layer.backward(&x[..7], &dy, 1).is_err());
    assert!(layer.backward(&x, &dy[..3], 1).is_err());
}

#[test]
fn adapter_set_weights_updates_forward() {
    let (out_dim, in_dim, r) = (4usize, 8usize, 2usize);
    let w = vec![0.1f32; out_dim * in_dim];
    let a = vec![0.1f32; r * in_dim];
    let b = vec![0.0f32; out_dim * r]; // zero-init: adapter contributes nothing
    let mut layer = QloraLinear::new(
        &w,
        out_dim,
        in_dim,
        &QuantConfig::default(),
        Some(LoraAdapter::new(&a, &b, in_dim, out_dim, r, 8.0).unwrap()),
    )
    .unwrap();
    let x = vec![1.0f32; in_dim];
    let y0 = layer.forward(&x, 1).unwrap();
    // Turn the adapter on: forward must change, and stay consistent.
    let b2 = vec![0.5f32; out_dim * r];
    layer.adapter_mut().unwrap().set_weights(&a, &b2).unwrap();
    let y1 = layer.forward(&x, 1).unwrap();
    assert!(y0 != y1);
    assert!(layer
        .adapter_mut()
        .unwrap()
        .set_weights(&[1.0], &b2)
        .is_err());
}

#[test]
fn qlora_rejects_dimension_mismatches() {
    let w = vec![0.1f32; 4 * 8];
    let bad_adapter = LoraAdapter::new(&[0.1; 2 * 8], &[0.1; 5 * 2], 8, 5, 2, 8.0).unwrap();
    assert!(QloraLinear::new(&w, 4, 8, &QuantConfig::default(), Some(bad_adapter)).is_err());
    let ok = QloraLinear::new(&w, 4, 8, &QuantConfig::default(), None).unwrap();
    assert!(ok.forward(&[1.0; 7], 1).is_err()); // wrong in_dim
}
