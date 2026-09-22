//! LoRA adapters and small row-major matmul helpers.
//!
//! LoRA (Hu et al., 2021) freezes the base weight `W: (n, k)` and learns a
//! low-rank update `dW = B · A` with `A: (r, k)`, `B: (n, r)`, applied as:
//!
//! ```text
//! Y = X · W^T + scale · (X · A^T) · B^T,   scale = alpha / r
//! ```
//!
//! All matrices are row-major. `X` is `(batch, k)` (batch folds the sequence
//! dimension).

use crate::error::QloraError;

/// A LoRA adapter: `A: (r, in_dim)`, `B: (out_dim, r)`, scaling `alpha / r`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapter {
    in_dim: usize,
    out_dim: usize,
    r: usize,
    alpha: f32,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl LoraAdapter {
    /// Build an adapter, checking `a.len() == r * in_dim` etc.
    pub fn new(
        a: &[f32],
        b: &[f32],
        in_dim: usize,
        out_dim: usize,
        r: usize,
        alpha: f32,
    ) -> Result<Self, QloraError> {
        if r == 0 {
            return Err(QloraError::InvalidConfig("rank r must be > 0".to_string()));
        }
        if a.len() != r * in_dim {
            return Err(QloraError::ShapeMismatch(format!(
                "a.len() = {} but r*in_dim = {}",
                a.len(),
                r * in_dim
            )));
        }
        if b.len() != out_dim * r {
            return Err(QloraError::ShapeMismatch(format!(
                "b.len() = {} but out_dim*r = {}",
                b.len(),
                out_dim * r
            )));
        }
        Ok(Self {
            in_dim,
            out_dim,
            r,
            alpha,
            a: a.to_vec(),
            b: b.to_vec(),
        })
    }

    /// The LoRA scaling factor `alpha / r`.
    pub fn scale(&self) -> f32 {
        self.alpha / self.r as f32
    }

    /// Replace the adapter matrices in place (shape-checked). Used by
    /// optimizers after each step; the base weight is untouched.
    pub fn set_weights(&mut self, a: &[f32], b: &[f32]) -> Result<(), QloraError> {
        if a.len() != self.a.len() || b.len() != self.b.len() {
            return Err(QloraError::ShapeMismatch(format!(
                "set_weights: got ({}, {}), expected ({}, {})",
                a.len(),
                b.len(),
                self.a.len(),
                self.b.len()
            )));
        }
        self.a.copy_from_slice(a);
        self.b.copy_from_slice(b);
        Ok(())
    }

    /// Adapter-only term `scale · (X · A^T) · B^T` for `X: (batch, in_dim)`.
    pub fn forward(&self, x: &[f32], batch: usize) -> Result<Vec<f32>, QloraError> {
        if x.len() != batch * self.in_dim {
            return Err(QloraError::ShapeMismatch(format!(
                "x.len() = {} but batch*in_dim = {}",
                x.len(),
                batch * self.in_dim
            )));
        }
        // T1 = X · A^T : (batch, r)
        let t1 = matmul_trans_b(x, &self.a, batch, self.in_dim, self.r);
        // T2 = T1 · B^T : (batch, out_dim)
        let mut t2 = matmul_trans_b(&t1, &self.b, batch, self.r, self.out_dim);
        let s = self.scale();
        for v in &mut t2 {
            *v *= s;
        }
        Ok(t2)
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
    pub fn rank(&self) -> usize {
        self.r
    }
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
    pub fn a(&self) -> &[f32] {
        &self.a
    }
    pub fn b(&self) -> &[f32] {
        &self.b
    }
}

/// Row-major GEMM: `C = A · B` with `A: (m, k)`, `B: (k, n)`, `C: (m, n)`.
///
/// Plain triple loop (cache-friendly `i, k, j` order); the GPU backend in
/// `qlora-wgpu` provides the accelerated version with identical semantics.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// Row-major GEMM with transposed LHS: `C = A^T · B` with `A: (m, k)`,
/// `B: (m, n)`, `C: (k, n)`.
///
/// Used by the backward pass (`dB = dY^T · T1`, `dA = D^T · X`).
pub fn matmul_trans_a(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), m * n);
    let mut c = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..m {
                acc += a[p * k + i] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Row-major GEMM with transposed RHS: `C = A · B^T` with `A: (m, k)`,
/// `B: (n, k)` stored row-major, `C: (m, n)`.
///
/// This matches the transformer convention where weights are stored as
/// `(out, in)` and inputs multiply `W^T`.
pub fn matmul_trans_b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_small_known_answer() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        let c = matmul(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_trans_a_matches_transposed_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (3, 2)
        let b = vec![1.0, -1.0, 0.5, 2.0, 0.0, 1.5]; // (3, 2)
        let c = matmul_trans_a(&a, &b, 3, 2, 2);
        // A^T = [[1, 3, 5], [2, 4, 6]]
        let at = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        assert_eq!(c, matmul(&at, &b, 2, 3, 2));
    }

    #[test]
    fn matmul_trans_b_matches_transposed_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2, 3)
        let b = vec![1.0, 0.0, -1.0, 0.5, 2.0, 1.5]; // (2, 3) stored
        let c = matmul_trans_b(&a, &b, 2, 3, 2);
        // B^T = [[1, 0.5], [0, 2], [-1, 1.5]]
        let bt = vec![1.0, 0.5, 0.0, 2.0, -1.0, 1.5];
        assert_eq!(c, matmul(&a, &bt, 2, 3, 2));
    }
}
