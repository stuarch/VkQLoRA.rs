//! NF4 block-wise quantization with optional double quantization.
//!
//! Follows the QLoRA paper (Dettmers et al., 2023) as implemented by
//! `bitsandbytes`:
//!
//! * 4-bit NormalFloat (NF4) codebook: 16 levels, quantile-quantized for a
//!   standard normal distribution in `[-1, 1]`.
//! * Block-wise quantization with block size 64: every block stores its own
//!   FP32 `absmax` scale.
//! * Optional double quantization: the FP32 block scales are themselves
//!   quantized to 8 bit in super-blocks of 256 scales.
//!
//! # Storage layout
//!
//! * `codes`: nibbles packed two per byte, little-nibble-first, i.e. element
//!   `2*i` is the low nibble of byte `i`, element `2*i+1` the high nibble.
//!   The GPU backend reinterprets these bytes as little-endian `u32`s.
//! * Scales cover `ceil(n / block_size)` blocks; the last block may be short.

use crate::error::QloraError;

/// NF4 codebook from the QLoRA paper (ascending order).
///
/// Literals keep the canonical float64 values from the paper; they are
/// rounded to `f32` on use (allowed lint below).
#[allow(clippy::excessive_precision)]
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Code used for all-zero blocks (decodes to exactly `0.0`).
pub const NF4_ZERO_CODE: u8 = 7;

/// Number of block scales covered by one second-level absmax.
pub const DOUBLE_QUANT_BLOCK: usize = 256;

/// Quantization settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantConfig {
    /// Elements per quantization block. QLoRA uses 64.
    pub block_size: usize,
    /// Whether to double-quantize the FP32 block scales to 8 bit.
    pub double_quant: bool,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            double_quant: true,
        }
    }
}

impl QuantConfig {
    /// Validate the config, returning a [`QloraError`] on bad values.
    pub fn validate(&self) -> Result<(), QloraError> {
        if self.block_size == 0 {
            return Err(QloraError::InvalidConfig(
                "block_size must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Compact storage for the per-block FP32 scales.
#[derive(Debug, Clone, PartialEq)]
enum ScaleStore {
    /// One FP32 scale per block.
    F32(Vec<f32>),
    /// 8-bit quantized scales: `scale = (code as i8) / 127 * absmax`.
    Int8 { codes: Vec<u8>, absmax: Vec<f32> },
}

/// A row-major `(rows, cols)` weight matrix quantized to NF4.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    rows: usize,
    cols: usize,
    block_size: usize,
    /// Packed nibbles, little-nibble-first, `ceil(n / 2)` bytes.
    codes: Vec<u8>,
    scales: ScaleStore,
}

impl QuantizedTensor {
    /// Quantize a row-major `(rows, cols)` FP32 matrix.
    pub fn quantize(
        weights: &[f32],
        rows: usize,
        cols: usize,
        config: &QuantConfig,
    ) -> Result<Self, QloraError> {
        config.validate()?;
        let n = rows
            .checked_mul(cols)
            .ok_or_else(|| QloraError::InvalidConfig("rows*cols overflow".to_string()))?;
        if weights.len() != n {
            return Err(QloraError::ShapeMismatch(format!(
                "weights.len() = {} but rows*cols = {n}",
                weights.len()
            )));
        }

        let block_size = config.block_size;
        let num_blocks = n.div_ceil(block_size);
        let mut codes = vec![0u8; n.div_ceil(2)];
        let mut scales_f32 = Vec::with_capacity(num_blocks);

        for (b, block) in weights.chunks(block_size).enumerate() {
            let absmax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            scales_f32.push(absmax);
            for (i, &w) in block.iter().enumerate() {
                let idx = b * block_size + i;
                let code = nearest_nf4_code(w, absmax);
                if idx % 2 == 0 {
                    codes[idx / 2] |= code;
                } else {
                    codes[idx / 2] |= code << 4;
                }
            }
        }

        let scales = if config.double_quant {
            let (codes, absmax) = double_quantize_scales(&scales_f32);
            ScaleStore::Int8 { codes, absmax }
        } else {
            ScaleStore::F32(scales_f32)
        };

        Ok(Self {
            rows,
            cols,
            block_size,
            codes,
            scales,
        })
    }

    /// Dequantize back to a row-major FP32 vector of length `rows * cols`.
    pub fn dequantize(&self) -> Vec<f32> {
        let n = self.rows * self.cols;
        let scales = self.block_scales();
        let mut out = Vec::with_capacity(n);
        for idx in 0..n {
            let byte = self.codes[idx / 2];
            let code = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out.push(NF4_LEVELS[code as usize] * scales[idx / self.block_size]);
        }
        out
    }

    /// Rebuild a tensor from raw parts (used by the Python bindings, which
    /// carry plain fp32 scales; validation mirrors [`Self::quantize`]).
    pub fn from_raw_parts(
        rows: usize,
        cols: usize,
        block_size: usize,
        codes: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self, QloraError> {
        if block_size == 0 {
            return Err(QloraError::InvalidConfig(
                "block_size must be > 0".to_string(),
            ));
        }
        let n = rows
            .checked_mul(cols)
            .ok_or_else(|| QloraError::InvalidConfig("rows*cols overflow".to_string()))?;
        if codes.len() != n.div_ceil(2) {
            return Err(QloraError::ShapeMismatch(format!(
                "codes.len() = {} but ceil(rows*cols/2) = {}",
                codes.len(),
                n.div_ceil(2)
            )));
        }
        if scales.len() != n.div_ceil(block_size) {
            return Err(QloraError::ShapeMismatch(format!(
                "scales.len() = {} but ceil(rows*cols/block_size) = {}",
                scales.len(),
                n.div_ceil(block_size)
            )));
        }
        Ok(Self {
            rows,
            cols,
            block_size,
            codes,
            scales: ScaleStore::F32(scales),
        })
    }

    /// Per-block FP32 scales (double-quantized scales are decoded on the fly).
    ///
    /// The GPU backend uploads exactly this vector alongside the packed codes.
    pub fn block_scales(&self) -> Vec<f32> {
        match &self.scales {
            ScaleStore::F32(s) => s.clone(),
            ScaleStore::Int8 { codes, absmax } => codes
                .iter()
                .enumerate()
                .map(|(i, &c)| (c as i8 as f32) / 127.0 * absmax[i / DOUBLE_QUANT_BLOCK])
                .collect(),
        }
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.rows * self.cols
    }

    /// Bytes used by the quantized payload (codes + scales).
    pub fn storage_bytes(&self) -> usize {
        let scale_bytes = match &self.scales {
            ScaleStore::F32(s) => s.len() * 4,
            ScaleStore::Int8 { codes, absmax } => codes.len() + absmax.len() * 4,
        };
        self.codes.len() + scale_bytes
    }

    /// Whether double quantization is active.
    pub fn uses_double_quant(&self) -> bool {
        matches!(self.scales, ScaleStore::Int8 { .. })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn num_blocks(&self) -> usize {
        self.num_elements().div_ceil(self.block_size)
    }
    /// Packed nibble codes (little-nibble-first).
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }
}

/// Nearest NF4 code for `w` given the block `absmax`.
///
/// A zero block maps every element to [`NF4_ZERO_CODE`] (exact `0.0`).
fn nearest_nf4_code(w: f32, absmax: f32) -> u8 {
    if absmax == 0.0 {
        return NF4_ZERO_CODE;
    }
    let target = w / absmax;
    let mut best = 0u8;
    let mut best_dist = f32::INFINITY;
    for (i, &level) in NF4_LEVELS.iter().enumerate() {
        let d = (level - target).abs();
        if d < best_dist {
            best_dist = d;
            best = i as u8;
        }
    }
    best
}

/// Double-quantize FP32 block scales to 8 bit, 256 scales per super-block.
///
/// Returns `(codes, absmax)` where codes store `i8` values bit-cast to `u8`.
fn double_quantize_scales(scales: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut codes = Vec::with_capacity(scales.len());
    let mut absmax = Vec::with_capacity(scales.len().div_ceil(DOUBLE_QUANT_BLOCK));
    for chunk in scales.chunks(DOUBLE_QUANT_BLOCK) {
        let amax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        absmax.push(amax);
        for &s in chunk {
            let q = if amax == 0.0 {
                0i8
            } else {
                ((s / amax * 127.0).round() as i32).clamp(-127, 127) as i8
            };
            codes.push(q as u8);
        }
    }
    (codes, absmax)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nf4_codebook_is_sorted() {
        let mut sorted = NF4_LEVELS;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, NF4_LEVELS);
        assert_eq!(NF4_LEVELS[0], -1.0);
        assert_eq!(NF4_LEVELS[15], 1.0);
        assert_eq!(NF4_LEVELS[NF4_ZERO_CODE as usize], 0.0);
    }

    #[test]
    fn zero_block_decodes_to_zero() {
        let w = vec![0.0f32; 64];
        let q = QuantizedTensor::quantize(&w, 8, 8, &QuantConfig::default()).unwrap();
        assert_eq!(q.dequantize(), w);
    }
}
