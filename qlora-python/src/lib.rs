//! Python bindings for the qlora QLoRA library (PyO3).
//!
//! Build & install (inside `guix shell -f ../../guix.scm`, with a venv):
//!
//! ```sh
//! python -m venv .venv && source .venv/bin/activate
//! pip install maturin
//! maturin develop
//! ```
//!
//! All weight matrices are row-major flat lists; shapes are passed
//! explicitly (`rows`, `cols`, `batch`, ...).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use qlora_core::{Adam8bit, AdamConfig, LoraAdapter, QloraLinear, QuantConfig, QuantizedTensor};
use qlora_wgpu_backend::{gpu_qlora_backward, gpu_qlora_forward, GpuContext};

fn to_pyerr(e: qlora_core::QloraError) -> PyErr {
    PyErr::new::<PyValueError, _>(e.to_string())
}

/// An NF4-quantized weight matrix.
///
/// * `codes`: packed nibbles (two per byte, little-nibble-first).
/// * `scales`: per-block fp32 scales (double quantization is decoded on
///   export, so Python always sees plain fp32 scales).
#[pyclass]
#[derive(Clone)]
struct QuantizedWeight {
    #[pyo3(get)]
    rows: usize,
    #[pyo3(get)]
    cols: usize,
    #[pyo3(get)]
    block_size: usize,
    #[pyo3(get)]
    double_quant: bool,
    codes: Vec<u8>,
    #[pyo3(get)]
    scales: Vec<f32>,
}

#[pymethods]
impl QuantizedWeight {
    /// Packed nibble codes as `bytes`.
    fn codes_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.codes)
    }

    /// Packed nibble codes as a list of ints.
    #[getter]
    fn codes_list(&self) -> Vec<u8> {
        self.codes.clone()
    }

    /// Dequantize back to a flat row-major list of floats.
    fn dequantize(&self) -> Vec<f32> {
        let n = self.rows * self.cols;
        let mut out = Vec::with_capacity(n);
        for idx in 0..n {
            let byte = self.codes[idx / 2];
            let code = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out.push(
                qlora_core::quant::NF4_LEVELS[code as usize] * self.scales[idx / self.block_size],
            );
        }
        out
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantizedWeight(rows={}, cols={}, block_size={}, double_quant={})",
            self.rows, self.cols, self.block_size, self.double_quant
        )
    }
}

/// Quantize a flat row-major `rows x cols` matrix to NF4.
#[pyfunction]
#[pyo3(signature = (weights, rows, cols, block_size=64, double_quant=true))]
fn quantize_nf4(
    weights: Vec<f32>,
    rows: usize,
    cols: usize,
    block_size: usize,
    double_quant: bool,
) -> PyResult<QuantizedWeight> {
    let cfg = QuantConfig {
        block_size,
        double_quant,
    };
    let q = QuantizedTensor::quantize(&weights, rows, cols, &cfg).map_err(to_pyerr)?;
    Ok(QuantizedWeight {
        rows: q.rows(),
        cols: q.cols(),
        block_size: q.block_size(),
        double_quant: q.uses_double_quant(),
        codes: q.codes().to_vec(),
        scales: q.block_scales(),
    })
}

/// Fused QLoRA linear forward.
///
/// * `x`: flat row-major `batch x in_dim` input.
/// * `weight`: [`QuantizedWeight`] of shape `out_dim x in_dim`.
/// * `lora_a` / `lora_b`: flat `r x in_dim` / `out_dim x r` (both `None`
///   or both given). `lora_alpha` defaults to `r` (i.e. scale 1).
/// * `use_gpu`: try the WGPU backend first, fall back to CPU when no
///   adapter is available.
///
/// Returns a flat row-major `batch x out_dim` list.
#[pyfunction]
#[pyo3(signature = (x, batch, weight, lora_a=None, lora_b=None, lora_alpha=None, use_gpu=false))]
fn qlora_linear_forward(
    x: Vec<f32>,
    batch: usize,
    weight: &QuantizedWeight,
    lora_a: Option<Vec<f32>>,
    lora_b: Option<Vec<f32>>,
    lora_alpha: Option<f32>,
    use_gpu: bool,
) -> PyResult<Vec<f32>> {
    let layer = rebuild_layer(weight, lora_a, lora_b, lora_alpha)?;
    if use_gpu {
        let ctx = GpuContext::try_new_blocking().map_err(to_pyerr)?;
        Ok(gpu_qlora_forward(ctx.as_ref(), &layer, &x, batch).map_err(to_pyerr)?)
    } else {
        Ok(layer.forward(&x, batch).map_err(to_pyerr)?)
    }
}

/// Shared validation + layer rebuild for forward/backward.
fn rebuild_layer(
    weight: &QuantizedWeight,
    lora_a: Option<Vec<f32>>,
    lora_b: Option<Vec<f32>>,
    lora_alpha: Option<f32>,
) -> PyResult<QloraLinear> {
    let in_dim = weight.cols;
    let out_dim = weight.rows;
    let r = match (&lora_a, &lora_b) {
        (None, None) => 0,
        (Some(a), Some(b)) => {
            if a.len() % in_dim.max(1) != 0 {
                return Err(PyErr::new::<PyValueError, _>(
                    "lora_a length must be a multiple of in_dim",
                ));
            }
            let r = a.len() / in_dim.max(1);
            if b.len() != out_dim * r {
                return Err(PyErr::new::<PyValueError, _>(
                    "lora_b length must equal out_dim * rank",
                ));
            }
            r
        }
        _ => {
            return Err(PyErr::new::<PyValueError, _>(
                "lora_a and lora_b must be given together",
            ))
        }
    };

    // Rebuild the Rust tensor from the Python-side payload.
    let codes = weight.codes.clone();
    let expected_codes = (out_dim * in_dim).div_ceil(2);
    if codes.len() != expected_codes {
        return Err(PyErr::new::<PyValueError, _>(
            "QuantizedWeight codes length does not match rows*cols",
        ));
    }
    // Re-quantize path is lossy-free here: rebuild scales + codes directly.
    // (QuantizedTensor owns its store; reconstruct via dequant->quantize is
    // exact in codes because quantize is deterministic. Cheaper: keep a
    // helper in core. See `QuantizedTensor::from_raw_parts`.)
    let tensor = QuantizedTensor::from_raw_parts(
        weight.rows,
        weight.cols,
        weight.block_size,
        codes,
        weight.scales.clone(),
    )
    .map_err(to_pyerr)?;

    let adapter = if r == 0 {
        None
    } else {
        Some(
            LoraAdapter::new(
                lora_a.as_deref().unwrap(),
                lora_b.as_deref().unwrap(),
                in_dim,
                out_dim,
                r,
                lora_alpha.unwrap_or(r as f32),
            )
            .map_err(to_pyerr)?,
        )
    };
    Ok(QloraLinear::from_quantized(tensor, adapter).map_err(to_pyerr)?)
}

/// Backward pass of a fused QLoRA linear layer.
///
/// Takes the same arguments as [`qlora_linear_forward`] plus the upstream
/// gradient `dy` (flat row-major `batch x out_dim`). Returns a dict with
/// `grad_a` / `grad_b` (`None` without an adapter) and `grad_x`.
#[pyfunction]
#[pyo3(signature = (x, dy, batch, weight, lora_a=None, lora_b=None, lora_alpha=None, use_gpu=false))]
#[allow(clippy::too_many_arguments)]
fn qlora_linear_backward(
    x: Vec<f32>,
    dy: Vec<f32>,
    batch: usize,
    weight: &QuantizedWeight,
    lora_a: Option<Vec<f32>>,
    lora_b: Option<Vec<f32>>,
    lora_alpha: Option<f32>,
    use_gpu: bool,
) -> PyResult<BackwardOut> {
    let layer = rebuild_layer(weight, lora_a, lora_b, lora_alpha)?;
    let g = if use_gpu {
        let ctx = GpuContext::try_new_blocking().map_err(to_pyerr)?;
        gpu_qlora_backward(ctx.as_ref(), &layer, &x, &dy, batch).map_err(to_pyerr)?
    } else {
        layer.backward(&x, &dy, batch).map_err(to_pyerr)?
    };
    Ok(BackwardOut {
        grad_a: g.grad_a,
        grad_b: g.grad_b,
        grad_x: g.grad_x,
    })
}

/// Gradients returned by [`qlora_linear_backward`].
#[pyclass]
struct BackwardOut {
    #[pyo3(get)]
    grad_a: Option<Vec<f32>>,
    #[pyo3(get)]
    grad_b: Option<Vec<f32>>,
    #[pyo3(get)]
    grad_x: Vec<f32>,
}

/// Paged 8-bit Adam optimizer for LoRA adapters.
///
/// Holds one state per parameter position: `step(params, grads)` updates
/// the flat row-major lists in place order and returns the new params.
///
/// ```python
/// opt = qlora_wgpu.PagedAdam(lr=1e-3, max_resident_pages=4)
/// new_a, new_b = opt.step([a, b], [grad_a, grad_b])
/// ```
#[pyclass]
struct PagedAdam {
    opt: Adam8bit,
}

#[pymethods]
impl PagedAdam {
    #[new]
    #[pyo3(signature = (lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, block_size=256, max_resident_pages=None))]
    fn new(
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        block_size: usize,
        max_resident_pages: Option<usize>,
    ) -> PyResult<Self> {
        let opt = Adam8bit::new(AdamConfig {
            lr,
            beta1,
            beta2,
            eps,
            block_size,
            max_resident_pages: max_resident_pages.unwrap_or(usize::MAX),
        })
        .map_err(to_pyerr)?;
        Ok(Self { opt })
    }

    /// One step; returns the updated params (same shapes as inputs).
    fn step(&mut self, params: Vec<Vec<f32>>, grads: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let mut owned: Vec<Vec<f32>> = params;
        self.opt.step(&mut owned, &grads).map_err(to_pyerr)?;
        Ok(owned)
    }

    /// Current 8-bit state bytes (vs 8 bytes/element for fp32 Adam).
    fn state_bytes(&self) -> usize {
        self.opt.state_bytes()
    }
}

/// QLoRA operations with optional WGPU acceleration.
#[pymodule]
fn qlora_wgpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantize_nf4, m)?)?;
    m.add_function(wrap_pyfunction!(qlora_linear_forward, m)?)?;
    m.add_function(wrap_pyfunction!(qlora_linear_backward, m)?)?;
    m.add_class::<QuantizedWeight>()?;
    m.add_class::<BackwardOut>()?;
    m.add_class::<PagedAdam>()?;
    Ok(())
}
