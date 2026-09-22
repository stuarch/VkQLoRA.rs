//! Paged 8-bit Adam: f64 reference parity, paging parity, convergence.

use qlora_core::{Adam8bit, AdamConfig};

/// Independent f64 Adam reference (no quantization).
struct RefAdam {
    m: Vec<f64>,
    v: Vec<f64>,
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
}

impl RefAdam {
    fn step(&mut self, p: &mut [f64], g: &[f64], t: u64) {
        for i in 0..p.len() {
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g[i];
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * g[i] * g[i];
            let mh = self.m[i] / (1.0 - self.b1.powi(t as i32));
            let vh = self.v[i] / (1.0 - self.b2.powi(t as i32));
            p[i] -= self.lr * mh / (vh.sqrt() + self.eps);
        }
    }
}

#[test]
fn adam8bit_tracks_fp64_reference() {
    let n = 64;
    let g: Vec<f32> = (0..n)
        .map(|i| ((i * 13) % 17) as f32 / 17.0 - 0.5)
        .collect();
    let mut params = vec![vec![1.0f32; n]];
    let grads = vec![g.clone()];
    let mut opt = Adam8bit::new(AdamConfig {
        lr: 0.01,
        block_size: 32,
        ..Default::default()
    })
    .unwrap();

    let mut pr: Vec<f64> = vec![1.0; n];
    let gr: Vec<f64> = g.iter().map(|&v| v as f64).collect();
    let mut reference = RefAdam {
        m: vec![0.0f64; n],
        v: vec![0.0f64; n],
        lr: 0.01,
        b1: 0.9,
        b2: 0.999,
        eps: 1e-8,
    };
    for t in 1..=5u64 {
        opt.step(&mut params, &grads).unwrap();
        reference.step(&mut pr, &gr, t);
        // 8-bit state noise must stay small vs the exact reference.
        for (got, want) in params[0].iter().zip(pr.iter()) {
            assert!((*got as f64 - *want).abs() < 5e-3, "t={t}: {got} vs {want}");
        }
    }
}

#[test]
fn paging_matches_fully_resident_bitwise() {
    let mk = |pages| {
        Adam8bit::new(AdamConfig {
            lr: 0.01,
            block_size: 16,
            max_resident_pages: pages,
            ..Default::default()
        })
        .unwrap()
    };
    let g: Vec<f32> = (0..256)
        .map(|i| ((i * 7) % 43) as f32 / 43.0 - 0.5)
        .collect();
    let mut pa = vec![vec![0.5f32; 256], vec![-1.0f32; 256]];
    let mut pb = pa.clone();
    let ga = vec![g.clone(), g.clone()];
    let mut oa = mk(1); // one page resident: maximum paging
    let mut ob = mk(usize::MAX); // fully resident
    for _ in 0..3 {
        let sa = oa.step(&mut pa, &ga).unwrap();
        let sb = ob.step(&mut pb, &ga).unwrap();
        assert_eq!(sa.pages_streamed, sb.pages_streamed);
        assert_eq!(pa, pb);
    }
}

#[test]
fn adam8bit_converges_on_quadratic() {
    // Minimize sum(x^2): gradient 2x, recomputed each step.
    let mut params = vec![vec![2.0f32; 32]];
    let mut opt = Adam8bit::new(AdamConfig {
        lr: 0.05,
        block_size: 32,
        ..Default::default()
    })
    .unwrap();
    for _ in 0..300 {
        let g: Vec<f32> = params[0].iter().map(|&x| 2.0 * x).collect();
        opt.step(&mut params, &[g]).unwrap();
    }
    let max_abs = params[0].iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    assert!(max_abs < 0.05, "max|x| = {max_abs}");
}

#[test]
fn adam8bit_state_is_compact_and_validated() {
    let mut opt = Adam8bit::new(AdamConfig::default()).unwrap();
    let mut params = vec![vec![0.1f32; 1024]];
    let grads = vec![vec![0.01f32; 1024]];
    opt.step(&mut params, &grads).unwrap();
    // fp32 Adam would hold 2 * 1024 * 4 = 8192 bytes of state.
    assert!(opt.state_bytes() < 8192);
    // 1024 int8 + 1024 int8 + 2 * 4 blocks * 4 bytes (block 256).
    assert_eq!(opt.state_bytes(), 2048 + 2 * 4 * 4);

    assert!(Adam8bit::new(AdamConfig {
        lr: -0.1,
        ..Default::default()
    })
    .is_err());
    assert!(Adam8bit::new(AdamConfig {
        block_size: 0,
        ..Default::default()
    })
    .is_err());
    let mut bad_params = vec![vec![1.0f32; 4]];
    let bad_grads = vec![vec![1.0f32; 5]];
    assert!(opt.step(&mut bad_params, &bad_grads).is_err());
}
