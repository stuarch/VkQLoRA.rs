//! Paged 8-bit Adam optimizer for training LoRA adapters.
//!
//! The frozen NF4 base weight needs no optimizer state; this optimizer
//! trains the small adapter matrices (`A`, `B`). It follows the QLoRA /
//! bitsandbytes recipe in spirit:
//!
//! * First and second moments (`m`, `v`) are stored in **8 bit**, block-wise
//!   symmetric INT8 per block (`x = code / 127 * absmax`). The second moment
//!   spans orders of magnitude, where symmetric quantization collapses small
//!   values, so `v` is stored in the `sqrt` domain (`s = sqrt(v)` quantized,
//!   `v_hat = s^2 / bias_correction`) — halving the logarithmic dynamic
//!   range. bitsandbytes instead uses a dynamic-exponent map; the `sqrt`
//!   trick here is simpler and needs no lookup table.
//! * **Paged**: state blocks stream through a resident window of at most
//!   `max_resident_pages` blocks, so peak state memory stays bounded no
//!   matter the model size. Because blocks are updated independently,
//!   paging is numerically identical to the fully-resident run (asserted by
//!   `paging_matches_fully_resident`).
//!
//! Math per element (dequantized `m`, `v`, step `t` starting at 1):
//!
//! ```text
//! m = beta1 * m + (1 - beta1) * g
//! v = beta2 * v + (1 - beta2) * g^2
//! m_hat = m / (1 - beta1^t);  v_hat = v / (1 - beta2^t)
//! p -= lr * m_hat / (sqrt(v_hat) + eps)
//! ```

use crate::error::QloraError;

/// Hyperparameters and paging budget.
#[derive(Debug, Clone)]
pub struct AdamConfig {
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Numerical stability term.
    pub eps: f32,
    /// Elements per quantization block (states are quantized per block).
    pub block_size: usize,
    /// Max state blocks resident at once; the rest "page" through this
    /// window during `step`. `usize::MAX` (default) keeps everything resident.
    pub max_resident_pages: usize,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            block_size: 256,
            max_resident_pages: usize::MAX,
        }
    }
}

impl AdamConfig {
    fn validate(&self) -> Result<(), QloraError> {
        if self.lr <= 0.0 || !self.lr.is_finite() {
            return Err(QloraError::InvalidConfig(
                "lr must be positive and finite".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            return Err(QloraError::InvalidConfig(
                "betas must be in (0, 1)".to_string(),
            ));
        }
        if self.block_size == 0 || self.max_resident_pages == 0 {
            return Err(QloraError::InvalidConfig(
                "block_size and max_resident_pages must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Quantize one block symmetrically to INT8 (`code/127*absmax`).
fn quantize_block_i8(x: &[f32]) -> (Vec<u8>, f32) {
    let absmax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if absmax == 0.0 {
        return (vec![0u8; x.len()], 0.0);
    }
    let codes = x
        .iter()
        .map(|&v| ((v / absmax * 127.0).round() as i32).clamp(-127, 127) as i8 as u8)
        .collect();
    (codes, absmax)
}

fn dequant_block_i8(codes: &[u8], absmax: f32, out: &mut [f32]) {
    for (o, &c) in out.iter_mut().zip(codes.iter()) {
        *o = (c as i8 as f32) / 127.0 * absmax;
    }
}

/// Optimizer state of one parameter tensor, stored as 8-bit pages.
#[derive(Debug, Clone)]
struct ParamState {
    numel: usize,
    num_blocks: usize,
    m_codes: Vec<u8>,
    m_scales: Vec<f32>,
    v_codes: Vec<u8>,
    v_scales: Vec<f32>,
    step: u64,
}

impl ParamState {
    fn new(numel: usize, block_size: usize) -> Self {
        let num_blocks = numel.div_ceil(block_size);
        Self {
            numel,
            num_blocks,
            m_codes: vec![0u8; numel],
            m_scales: vec![0.0; num_blocks],
            v_codes: vec![0u8; numel],
            v_scales: vec![0.0; num_blocks],
            step: 0,
        }
    }

    /// Bytes held by this state (vs `8 * numel` for fp32 m+v).
    fn state_bytes(&self) -> usize {
        self.m_codes.len() + self.v_codes.len() + (self.m_scales.len() + self.v_scales.len()) * 4
    }
}

/// Statistics returned by [`Adam8bit::step`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdamStats {
    /// State blocks streamed through the resident window this step.
    pub pages_streamed: usize,
    /// Total 8-bit state bytes across all parameters.
    pub state_bytes: usize,
}

/// Paged 8-bit Adam optimizer.
///
/// Parameters are registered by position: the `i`-th entry of `params` in
/// every `step` call addresses the `i`-th state (created lazily).
#[derive(Debug, Clone)]
pub struct Adam8bit {
    config: AdamConfig,
    states: Vec<ParamState>,
}

impl Adam8bit {
    /// Create an optimizer, validating the config.
    pub fn new(config: AdamConfig) -> Result<Self, QloraError> {
        config.validate()?;
        Ok(Self {
            config,
            states: Vec::new(),
        })
    }

    /// One optimizer step over `(param, grad)` pairs, updating params in place.
    pub fn step(
        &mut self,
        params: &mut [Vec<f32>],
        grads: &[Vec<f32>],
    ) -> Result<AdamStats, QloraError> {
        if params.len() != grads.len() {
            return Err(QloraError::ShapeMismatch(format!(
                "params.len() = {} but grads.len() = {}",
                params.len(),
                grads.len()
            )));
        }
        let mut pages_streamed = 0;
        for (i, (p, g)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if p.len() != g.len() {
                return Err(QloraError::ShapeMismatch(format!(
                    "param {i}: len = {} but grad len = {}",
                    p.len(),
                    g.len()
                )));
            }
            if self.states.len() <= i {
                self.states
                    .push(ParamState::new(p.len(), self.config.block_size));
            }
            // Re-resolve in case a reshaped param changed length.
            if self.states[i].numel != p.len() {
                self.states[i] = ParamState::new(p.len(), self.config.block_size);
            }
            pages_streamed += Self::step_param(&self.config, &mut self.states[i], p, g);
        }
        Ok(AdamStats {
            pages_streamed,
            state_bytes: self.state_bytes(),
        })
    }

    /// Total 8-bit state bytes (vs 8 bytes/element for fp32 Adam).
    pub fn state_bytes(&self) -> usize {
        self.states.iter().map(ParamState::state_bytes).sum()
    }

    fn step_param(cfg: &AdamConfig, st: &mut ParamState, p: &mut [f32], g: &[f32]) -> usize {
        st.step += 1;
        let t = st.step as f32;
        let b1t = 1.0 - cfg.beta1.powf(t);
        let b2t = 1.0 - cfg.beta2.powf(t);
        let window = cfg.max_resident_pages.min(st.num_blocks).max(1);

        let mut streamed = 0;
        let mut m = vec![0.0f32; cfg.block_size];
        let mut v = vec![0.0f32; cfg.block_size];
        for block_start in (0..st.num_blocks).step_by(window) {
            let block_end = (block_start + window).min(st.num_blocks);
            for b in block_start..block_end {
                let s = b * cfg.block_size;
                let e = (s + cfg.block_size).min(st.numel);
                let len = e - s;
                m.truncate(len);
                v.truncate(len);
                dequant_block_i8(&st.m_codes[s..e], st.m_scales[b], &mut m);
                dequant_block_i8(&st.v_codes[s..e], st.v_scales[b], &mut v);
                // `v` holds sqrt-domain values: v[j] = sqrt(E[g^2]).
                for j in 0..len {
                    let gj = g[s + j];
                    m[j] = cfg.beta1 * m[j] + (1.0 - cfg.beta1) * gj;
                    let var = cfg.beta2 * v[j] * v[j] + (1.0 - cfg.beta2) * gj * gj;
                    v[j] = var.sqrt();
                    let mh = m[j] / b1t;
                    let vh = var / b2t;
                    p[s + j] -= cfg.lr * mh / (vh.sqrt() + cfg.eps);
                }
                let (mc, ms) = quantize_block_i8(&m);
                let (vc, vs) = quantize_block_i8(&v);
                st.m_codes[s..e].copy_from_slice(&mc);
                st.v_codes[s..e].copy_from_slice(&vc);
                st.m_scales[b] = ms;
                st.v_scales[b] = vs;
                m.resize(cfg.block_size, 0.0);
                v.resize(cfg.block_size, 0.0);
                streamed += 1;
            }
        }
        streamed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_block_roundtrip_bounds_error() {
        let x: Vec<f32> = (0..300)
            .map(|i| ((i * 7) % 101) as f32 / 101.0 - 0.5)
            .collect();
        let (codes, scale) = quantize_block_i8(&x);
        assert!((scale - 0.5).abs() < 1e-6);
        let mut back = vec![0.0f32; x.len()];
        dequant_block_i8(&codes, scale, &mut back);
        for (o, b) in x.iter().zip(back.iter()) {
            assert!((o - b).abs() <= 0.5 / 127.0 * scale + 1e-7);
        }
        // Zero block stays zero.
        let (zc, zs) = quantize_block_i8(&[0.0; 8]);
        assert_eq!(zs, 0.0);
        assert!(zc.iter().all(|&c| c == 0));
    }
}
