//! Quantization roundtrip properties (deterministic PRNG, no dependencies).

use qlora_core::{QuantConfig, QuantizedTensor};

/// Tiny deterministic PRNG (xorshift64*) so tests need no `rand` crate.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (u64::MAX >> 11) as f64
    }

    /// Approximately standard-normal via Box-Muller.
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

fn roundtrip_stats(back: &[f32], orig: &[f32]) -> (f32, f32) {
    let mut max_err = 0.0f32;
    let mut mse = 0.0f64;
    for (b, o) in back.iter().zip(orig.iter()) {
        let e = (b - o).abs();
        max_err = max_err.max(e);
        mse += (e * e) as f64;
    }
    (max_err, (mse / orig.len() as f64) as f32)
}

#[test]
fn nf4_roundtrip_normal_weights_stays_accurate() {
    // NF4 is designed for (near-)normal data: error must be small.
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);
    let rows = 64;
    let cols = 128;
    let w: Vec<f32> = (0..rows * cols).map(|_| rng.next_normal()).collect();
    for double_quant in [false, true] {
        let cfg = QuantConfig {
            block_size: 64,
            double_quant,
        };
        let q = QuantizedTensor::quantize(&w, rows, cols, &cfg).unwrap();
        assert_eq!(q.uses_double_quant(), double_quant);
        let back = q.dequantize();
        assert_eq!(back.len(), w.len());
        // Nearest-level property: each element's error is bounded by half
        // the widest NF4 gap (0.304 / 2 = 0.152) times its block absmax,
        // plus slack for double-quantized scales.
        let scales = q.block_scales();
        for (i, (b, o)) in back.iter().zip(w.iter()).enumerate() {
            let bound = 0.16 * scales[i / cfg.block_size] + 1e-6;
            assert!(
                (b - o).abs() <= bound,
                "elem {i}: err={} > bound={bound} (dq={double_quant})",
                (b - o).abs()
            );
        }
        let (_, mse) = roundtrip_stats(&back, &w);
        assert!(mse < 0.01, "mse={mse} (dq={double_quant})");
    }
}

#[test]
fn nf4_roundtrip_uniform_weights_bounded() {
    // Adversarial uniform data: NF4 is coarser at the tails, but the error
    // is still bounded by half the widest codebook gap (0.277 / 2).
    let mut rng = Rng(0xDEAD_BEEF_CAFE_1234);
    let rows = 16;
    let cols = 64;
    let w: Vec<f32> = (0..rows * cols)
        .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
        .collect();
    let cfg = QuantConfig {
        block_size: 64,
        double_quant: false,
    };
    let q = QuantizedTensor::quantize(&w, rows, cols, &cfg).unwrap();
    let (max_err, _) = roundtrip_stats(&q.dequantize(), &w);
    assert!(max_err <= 0.15, "max_err={max_err}");
}

#[test]
fn packed_codes_length_and_layout() {
    let w: Vec<f32> = (0..10).map(|i| i as f32 / 10.0).collect();
    let q = QuantizedTensor::quantize(&w, 2, 5, &QuantConfig::default()).unwrap();
    assert_eq!(q.codes().len(), 5); // ceil(10 / 2)
    assert_eq!(q.num_blocks(), 1); // block_size 64 > 10
    assert_eq!(q.block_scales().len(), 1);
    // absmax of 0.0..0.9 is 0.9
    assert!((q.block_scales()[0] - 0.9).abs() < 1e-6);
}

#[test]
fn partial_last_block_is_handled() {
    // 100 elements, block 64 -> blocks of 64 + 36.
    let w: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
    let cfg = QuantConfig {
        block_size: 64,
        double_quant: true,
    };
    let q = QuantizedTensor::quantize(&w, 10, 10, &cfg).unwrap();
    assert_eq!(q.num_blocks(), 2);
    assert_eq!(q.block_scales().len(), 2);
    let back = q.dequantize();
    assert_eq!(back.len(), 100);
}

#[test]
fn double_quant_shrinks_scale_storage() {
    let w: Vec<f32> = (0..65_536)
        .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
        .collect();
    let plain = QuantizedTensor::quantize(
        &w,
        256,
        256,
        &QuantConfig {
            block_size: 64,
            double_quant: false,
        },
    )
    .unwrap();
    let dq = QuantizedTensor::quantize(&w, 256, 256, &QuantConfig::default()).unwrap();
    assert!(dq.storage_bytes() < plain.storage_bytes());
    // 1024 blocks -> 1024 fp32 scales (4096 B) vs 1024 int8 + 4 fp32 (1040 B).
    assert_eq!(plain.storage_bytes() - dq.storage_bytes(), 4096 - 1040);
}

#[test]
fn from_raw_parts_roundtrips_exactly() {
    // What the Python bindings do: quantize -> export codes+scales ->
    // rebuild -> identical dequantized output.
    let w: Vec<f32> = (0..512)
        .map(|i| ((i * 7) % 31) as f32 / 31.0 - 0.5)
        .collect();
    let q = QuantizedTensor::quantize(&w, 16, 32, &QuantConfig::default()).unwrap();
    let rebuilt = QuantizedTensor::from_raw_parts(
        q.rows(),
        q.cols(),
        q.block_size(),
        q.codes().to_vec(),
        q.block_scales(),
    )
    .unwrap();
    assert_eq!(rebuilt.dequantize(), q.dequantize());
    assert!(QuantizedTensor::from_raw_parts(16, 32, 64, vec![0u8; 3], q.block_scales()).is_err());
}

#[test]
fn quantize_rejects_bad_shapes_and_configs() {
    let bad_cfg = QuantConfig {
        block_size: 0,
        double_quant: false,
    };
    assert!(QuantizedTensor::quantize(&[1.0], 1, 1, &bad_cfg).is_err());
    assert!(QuantizedTensor::quantize(&[1.0, 2.0], 1, 1, &QuantConfig::default()).is_err());
}
