//! Fused QLoRA linear layer: quantized base weight + LoRA adapter.

use crate::error::QloraError;
use crate::lora::{matmul, matmul_trans_a, matmul_trans_b, LoraAdapter};
use crate::quant::{QuantConfig, QuantizedTensor};

/// Gradients produced by [`QloraLinear::backward`].
///
/// The base weight is frozen (that is the point of QLoRA), so only the
/// adapter gradients are produced — plus `grad_x`, needed to backpropagate
/// into earlier layers.
#[derive(Debug, Clone, PartialEq)]
pub struct QloraGrads {
    /// `dL/dA`, shape `(r, in_dim)`; `None` without an adapter.
    pub grad_a: Option<Vec<f32>>,
    /// `dL/dB`, shape `(out_dim, r)`; `None` without an adapter.
    pub grad_b: Option<Vec<f32>>,
    /// `dL/dX`, shape `(batch, in_dim)`.
    pub grad_x: Vec<f32>,
}

/// A fused QLoRA linear layer.
///
/// Holds the frozen NF4-quantized base weight `W: (out_dim, in_dim)` and an
/// optional trainable [`LoraAdapter`]. Forward computes
///
/// ```text
/// Y = X · dequant(W_q)^T + adapter_term(X)
/// ```
///
/// for `X: (batch, in_dim)`, `Y: (batch, out_dim)`.
#[derive(Debug, Clone)]
pub struct QloraLinear {
    weight: QuantizedTensor,
    adapter: Option<LoraAdapter>,
}

impl QloraLinear {
    /// Build a layer from an FP32 base weight (quantized immediately) and an
    /// optional adapter.
    pub fn new(
        weight_fp32: &[f32],
        out_dim: usize,
        in_dim: usize,
        quant: &QuantConfig,
        adapter: Option<LoraAdapter>,
    ) -> Result<Self, QloraError> {
        let weight = QuantizedTensor::quantize(weight_fp32, out_dim, in_dim, quant)?;
        if let Some(ref ad) = adapter {
            if ad.in_dim() != in_dim {
                return Err(QloraError::ShapeMismatch(format!(
                    "adapter in_dim = {} but layer in_dim = {in_dim}",
                    ad.in_dim()
                )));
            }
            if ad.out_dim() != out_dim {
                return Err(QloraError::ShapeMismatch(format!(
                    "adapter out_dim = {} but layer out_dim = {out_dim}",
                    ad.out_dim()
                )));
            }
        }
        Ok(Self { weight, adapter })
    }

    /// Rebuild from an already-quantized weight (e.g. received from Python).
    pub fn from_quantized(
        weight: QuantizedTensor,
        adapter: Option<LoraAdapter>,
    ) -> Result<Self, QloraError> {
        if let Some(ref ad) = adapter {
            if ad.in_dim() != weight.cols() || ad.out_dim() != weight.rows() {
                return Err(QloraError::ShapeMismatch(format!(
                    "adapter ({}, {}) vs weight ({}, {})",
                    ad.out_dim(),
                    ad.in_dim(),
                    weight.rows(),
                    weight.cols()
                )));
            }
        }
        Ok(Self { weight, adapter })
    }

    /// Fused forward for `X: (batch, in_dim)` row-major.
    pub fn forward(&self, x: &[f32], batch: usize) -> Result<Vec<f32>, QloraError> {
        let in_dim = self.weight.cols();
        let out_dim = self.weight.rows();
        if x.len() != batch * in_dim {
            return Err(QloraError::ShapeMismatch(format!(
                "x.len() = {} but batch*in_dim = {}",
                x.len(),
                batch * in_dim
            )));
        }
        // Base term with the dequantized weight.
        let w = self.weight.dequantize();
        let mut y = matmul_trans_b(x, &w, batch, in_dim, out_dim);
        // Adapter term.
        if let Some(ref ad) = self.adapter {
            let d = ad.forward(x, batch)?;
            for (y_i, d_i) in y.iter_mut().zip(d.iter()) {
                *y_i += *d_i;
            }
        }
        Ok(y)
    }

    /// Backward pass for training the adapter.
    ///
    /// Given the layer input `X: (batch, in_dim)` and the upstream gradient
    /// `dY = dL/dY: (batch, out_dim)`, computes gradients w.r.t. the LoRA
    /// matrices (`dB = s·dY^T·T1`, `dA = s·D^T·X` with `T1 = X·A^T`,
    /// `D = dY·B`, `s = alpha/r`) and w.r.t. the input
    /// (`dX = dY·W + s·D·A`). The quantized base weight gets no gradient.
    pub fn backward(&self, x: &[f32], dy: &[f32], batch: usize) -> Result<QloraGrads, QloraError> {
        let in_dim = self.weight.cols();
        let out_dim = self.weight.rows();
        if x.len() != batch * in_dim {
            return Err(QloraError::ShapeMismatch(format!(
                "x.len() = {} but batch*in_dim = {}",
                x.len(),
                batch * in_dim
            )));
        }
        if dy.len() != batch * out_dim {
            return Err(QloraError::ShapeMismatch(format!(
                "dy.len() = {} but batch*out_dim = {}",
                dy.len(),
                batch * out_dim
            )));
        }
        let w = self.weight.dequantize();
        // Base term of dX (LoRA term added below when present).
        let mut grad_x = matmul(dy, &w, batch, out_dim, in_dim);

        let (grad_a, grad_b) = match self.adapter {
            None => (None, None),
            Some(ref ad) => {
                let r = ad.rank();
                let s = ad.scale();
                // Pre-scale dY once; every adapter term carries the factor s.
                let dys: Vec<f32> = dy.iter().map(|v| v * s).collect();
                // T1 = X · A^T : (batch, r)
                let t1 = matmul_trans_b(x, ad.a(), batch, in_dim, r);
                // dB = dYs^T · T1 : (out_dim, r)
                let grad_b = matmul_trans_a(&dys, &t1, batch, out_dim, r);
                // Ds = dYs · B : (batch, r)
                let ds = matmul(&dys, ad.b(), batch, out_dim, r);
                // dA = Ds^T · X : (r, in_dim)
                let grad_a = matmul_trans_a(&ds, x, batch, r, in_dim);
                // dX += Ds · A : (batch, in_dim)
                let dx_lora = matmul(&ds, ad.a(), batch, r, in_dim);
                for (g, d) in grad_x.iter_mut().zip(dx_lora.iter()) {
                    *g += d;
                }
                (Some(grad_a), Some(grad_b))
            }
        };
        Ok(QloraGrads {
            grad_a,
            grad_b,
            grad_x,
        })
    }

    pub fn weight(&self) -> &QuantizedTensor {
        &self.weight
    }
    pub fn adapter(&self) -> Option<&LoraAdapter> {
        self.adapter.as_ref()
    }

    /// Mutable access to the adapter for optimizer updates.
    pub fn adapter_mut(&mut self) -> Option<&mut LoraAdapter> {
        self.adapter.as_mut()
    }
    pub fn in_dim(&self) -> usize {
        self.weight.cols()
    }
    pub fn out_dim(&self) -> usize {
        self.weight.rows()
    }
}
