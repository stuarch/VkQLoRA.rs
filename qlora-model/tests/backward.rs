//! Model backward vs finite differences on a tiny synthetic model.
//!
//! L = sum(logits), so dy is all ones. Every adapter element (A and B of
//! Q/V in each layer) is checked with central differences.

use qlora_core::{LoraAdapter, QloraLinear, QuantConfig};
use qlora_model::{AdapterGrad, LayerWeights, LlamaConfig, LlamaModel, Proj, ProjKind};
use std::collections::HashMap;

fn tiny_cfg() -> LlamaConfig {
    LlamaConfig {
        hidden: 16,
        layers: 2,
        heads: 4,
        kv_heads: 2,
        intermediate: 32,
        vocab: 48,
        rms_eps: 1e-5,
        rope_theta: 10000.0,
    }
}

fn lcg(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (u64::MAX >> 11) as f64 * 2.0 - 1.0) as f32
}

#[derive(Clone)]
struct RawLayer {
    attn_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    mlp_norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

#[derive(Clone)]
struct Tiny {
    cfg: LlamaConfig,
    embed: Vec<f32>,
    layers_raw: Vec<RawLayer>,
    final_norm: Vec<f32>,
    a_vals: HashMap<(usize, ProjKind), Vec<f32>>,
    b_vals: HashMap<(usize, ProjKind), Vec<f32>>,
}

const IDS: [u32; 4] = [1, 7, 3, 40];
const RANK: usize = 2;

impl Tiny {
    fn fresh() -> Self {
        let cfg = tiny_cfg();
        let mut rng = 0xABCD_EF01_2345_6789u64;
        let mut rv = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|_| lcg(&mut rng) * s).collect() };
        let h = cfg.hidden;
        let kv = cfg.kv_heads * (h / cfg.heads);
        let mut layers_raw = Vec::new();
        for _ in 0..cfg.layers {
            layers_raw.push(RawLayer {
                attn_norm: rv(h, 0.2).iter().map(|v| v + 1.0).collect(),
                q: rv(h * h, 0.2),
                k: rv(kv * h, 0.2),
                v: rv(kv * h, 0.2),
                o: rv(h * h, 0.2),
                mlp_norm: rv(h, 0.1).iter().map(|v| v + 1.0).collect(),
                gate: rv(cfg.intermediate * h, 0.2),
                up: rv(cfg.intermediate * h, 0.2),
                down: rv(h * cfg.intermediate, 0.2),
            });
        }
        let mut a_vals = HashMap::new();
        let mut b_vals = HashMap::new();
        for l in 0..cfg.layers {
            for (kind, out_dim) in [(ProjKind::Q, h), (ProjKind::V, kv)] {
                a_vals.insert((l, kind), rv(RANK * h, 0.1));
                b_vals.insert((l, kind), rv(out_dim * RANK, 0.1));
            }
        }
        Self {
            embed: rv(cfg.vocab * h, 0.2),
            final_norm: rv(h, 0.1).iter().map(|v| v + 1.0).collect(),
            layers_raw,
            a_vals,
            b_vals,
            cfg,
        }
    }

    fn build(&self) -> LlamaModel {
        let cfg = &self.cfg;
        let quant = QuantConfig {
            block_size: 64,
            double_quant: false,
        };
        let h = cfg.hidden;
        let kv = cfg.kv_heads * (h / cfg.heads);
        let layers = self
            .layers_raw
            .iter()
            .enumerate()
            .map(|(l, raw)| {
                let qlora = |kind: ProjKind, w: &[f32], out_dim: usize, in_dim: usize| -> Proj {
                    let ad = LoraAdapter::new(
                        &self.a_vals[&(l, kind)],
                        &self.b_vals[&(l, kind)],
                        in_dim,
                        out_dim,
                        RANK,
                        4.0,
                    )
                    .unwrap();
                    Proj::Qlora(QloraLinear::new(w, out_dim, in_dim, &quant, Some(ad)).unwrap())
                };
                LayerWeights {
                    attn_norm: raw.attn_norm.clone(),
                    q: qlora(ProjKind::Q, &raw.q, h, h),
                    k: Proj::Raw(raw.k.clone()),
                    v: qlora(ProjKind::V, &raw.v, kv, h),
                    o: Proj::Raw(raw.o.clone()),
                    mlp_norm: raw.mlp_norm.clone(),
                    gate: Proj::Raw(raw.gate.clone()),
                    up: Proj::Raw(raw.up.clone()),
                    down: Proj::Raw(raw.down.clone()),
                }
            })
            .collect();
        LlamaModel {
            cfg: cfg.clone(),
            embed: self.embed.clone(),
            layers,
            final_norm: self.final_norm.clone(),
        }
    }

    fn loss(&self) -> f32 {
        self.build().forward(&IDS).iter().sum()
    }

    fn fd(&self, layer: usize, kind: ProjKind, is_a: bool, idx: usize) -> f32 {
        let eps = 1e-3f32;
        let mut plus = self.clone();
        let mut minus = self.clone();
        let slot_p = if is_a {
            plus.a_vals.get_mut(&(layer, kind)).unwrap()
        } else {
            plus.b_vals.get_mut(&(layer, kind)).unwrap()
        };
        slot_p[idx] += eps;
        let slot_m = if is_a {
            minus.a_vals.get_mut(&(layer, kind)).unwrap()
        } else {
            minus.b_vals.get_mut(&(layer, kind)).unwrap()
        };
        slot_m[idx] -= eps;
        (plus.loss() - minus.loss()) / (2.0 * eps)
    }
}

#[test]
fn model_backward_matches_finite_differences() {
    let tiny = Tiny::fresh();
    let cfg = tiny_cfg();
    let dy = vec![1.0f32; IDS.len() * cfg.vocab];
    let model = tiny.build();
    let grads: Vec<AdapterGrad> = model.backward(&IDS, &dy);
    assert_eq!(grads.len(), cfg.layers * 2); // Q and V per layer

    for g in &grads {
        for (i, &v) in g.grad_a.iter().enumerate() {
            let want = tiny.fd(g.layer, g.kind, true, i);
            assert!(
                (v - want).abs() < 2e-2,
                "L{} {:?} dA[{i}]: {v} vs {want}",
                g.layer,
                g.kind
            );
        }
        for (i, &v) in g.grad_b.iter().enumerate() {
            let want = tiny.fd(g.layer, g.kind, false, i);
            assert!(
                (v - want).abs() < 2e-2,
                "L{} {:?} dB[{i}]: {v} vs {want}",
                g.layer,
                g.kind
            );
        }
    }
}
