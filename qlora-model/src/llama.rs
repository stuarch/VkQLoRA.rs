//! Llama forward + backward (fp32 CPU reference) with QLoRA adapters.
//!
//! Any of the 7 per-layer projections can carry a LoRA adapter (standard:
//! Q and V). Adapters use zero-init B, so attaching them leaves the base
//! forward bit-identical. Only adapter matrices get gradients; the base
//! weights stay frozen.

use qlora_core::lora::{matmul, matmul_trans_b};
use qlora_core::{LoraAdapter, QloraLinear, QuantConfig};
use qlora_io::{read_safetensors, NamedArray};

/// Architecture hyperparameters.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

impl LlamaConfig {
    /// SmolLM-135M (HuggingFaceTB/SmolLM-135M `config.json`).
    pub fn smollm_135m() -> Self {
        Self {
            hidden: 576,
            layers: 30,
            heads: 9,
            kv_heads: 3,
            intermediate: 1536,
            vocab: 49152,
            rms_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}

/// Which projection carries a LoRA adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjKind {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

/// Adapter attachment: standard QLoRA targets Q and V.
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    pub targets: Vec<ProjKind>,
    pub rank: usize,
    pub alpha: f32,
    pub quant: QuantConfig,
}

impl AdapterConfig {
    /// No adapters (pure base model).
    pub fn none() -> Self {
        Self {
            targets: Vec::new(),
            rank: 8,
            alpha: 16.0,
            quant: QuantConfig::default(),
        }
    }

    /// Classic QLoRA: adapters on Q and V projections.
    pub fn qv(rank: usize, alpha: f32) -> Self {
        Self {
            targets: vec![ProjKind::Q, ProjKind::V],
            rank,
            alpha,
            quant: QuantConfig::default(),
        }
    }
}

/// A projection: frozen fp32 base weight, or a fused QLoRA layer.
#[derive(Debug, Clone)]
pub enum Proj {
    Raw(Vec<f32>),
    Qlora(QloraLinear),
}

/// Deterministic toy PRNG (uniform in `[-1, 1]`) for adapter init.
fn lcg(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    let u = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (u64::MAX >> 11) as f64;
    (u * 2.0 - 1.0) as f32
}

fn proj_forward(p: &Proj, x: &[f32], batch: usize, out_dim: usize) -> Vec<f32> {
    match p {
        Proj::Raw(w) => {
            let in_dim = w.len() / out_dim;
            matmul_trans_b(x, w, batch, in_dim, out_dim)
        }
        Proj::Qlora(l) => l.forward(x, batch).expect("proj shapes validated at load"),
    }
}

/// Returns `(adapter grads (dA, dB) if QLoRA, dX)`.
fn proj_backward(
    p: &Proj,
    x: &[f32],
    dy: &[f32],
    batch: usize,
    out_dim: usize,
) -> (Option<(Vec<f32>, Vec<f32>)>, Vec<f32>) {
    match p {
        Proj::Raw(w) => {
            let in_dim = w.len() / out_dim;
            (None, matmul(dy, w, batch, out_dim, in_dim))
        }
        Proj::Qlora(l) => {
            let g = l
                .backward(x, dy, batch)
                .expect("proj shapes validated at load");
            let ab = match (g.grad_a, g.grad_b) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            };
            (ab, g.grad_x)
        }
    }
}

/// One decoder layer's weights (row-major, `(out, in)`; norms are vectors).
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub q: Proj,
    pub k: Proj,
    pub v: Proj,
    pub o: Proj,
    pub mlp_norm: Vec<f32>,
    pub gate: Proj,
    pub up: Proj,
    pub down: Proj,
}

/// Gradients of one adapter: `(layer, kind, dA (r,in), dB (out,r))`.
#[derive(Debug, Clone)]
pub struct AdapterGrad {
    pub layer: usize,
    pub kind: ProjKind,
    pub grad_a: Vec<f32>,
    pub grad_b: Vec<f32>,
}

/// Full model weights (fp32).
#[derive(Debug, Clone)]
pub struct LlamaModel {
    pub cfg: LlamaConfig,
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadError(pub String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "load: {}", self.0)
    }
}

impl std::error::Error for LoadError {}

impl LlamaModel {
    /// Load from a `.safetensors` buffer, checking every shape.
    ///
    /// Projections named in `adapters.targets` are quantized to NF4 and
    /// wrapped in fused [`QloraLinear`] layers (A random, B zero, so the
    /// adapter branch contributes exactly zero until the first optimizer
    /// step; the base term carries the usual NF4 error).
    pub fn load(
        bytes: &[u8],
        cfg: &LlamaConfig,
        adapters: &AdapterConfig,
    ) -> Result<Self, LoadError> {
        let ts = read_safetensors(bytes).map_err(|e| LoadError(e.to_string()))?;
        Self::from_tensors(&ts, cfg, adapters, 0x1234_5678_9ABC_DEF0)
    }

    /// Build from parsed tensors (same layout as `.safetensors`).
    /// Exposed for tests that synthesize tiny models.
    pub fn from_tensors(
        ts: &[NamedArray],
        cfg: &LlamaConfig,
        adapters: &AdapterConfig,
        seed: u64,
    ) -> Result<Self, LoadError> {
        Self::load_impl(ts, cfg, adapters, seed)
    }

    fn load_impl(
        ts: &[NamedArray],
        cfg: &LlamaConfig,
        adapters: &AdapterConfig,
        seed: u64,
    ) -> Result<Self, LoadError> {
        let get = |name: &str, want: &[usize]| -> Result<Vec<f32>, LoadError> {
            let t = NamedArray::find(&ts, name)
                .ok_or_else(|| LoadError(format!("missing tensor {name}")))?;
            if t.shape != want {
                return Err(LoadError(format!(
                    "{name}: shape {:?} != {want:?}",
                    t.shape
                )));
            }
            Ok(t.data.clone())
        };
        let h = cfg.hidden;
        let inter = cfg.intermediate;
        let kv_dim = cfg.kv_heads * cfg.head_dim();
        let mut rng = seed;
        // Wrap a raw weight in QLoRA when targeted (A random, B zero).
        let mut maybe_qlora = |kind: ProjKind,
                               w: Vec<f32>,
                               out_dim: usize,
                               in_dim: usize|
         -> Result<Proj, LoadError> {
            if !adapters.targets.contains(&kind) {
                return Ok(Proj::Raw(w));
            }
            let r = adapters.rank;
            let a: Vec<f32> = (0..r * in_dim).map(|_| lcg(&mut rng) * 0.02).collect();
            let b = vec![0.0f32; out_dim * r];
            let ad = LoraAdapter::new(&a, &b, in_dim, out_dim, r, adapters.alpha)
                .map_err(|e| LoadError(format!("adapter {kind:?}: {e}")))?;
            let layer = QloraLinear::new(&w, out_dim, in_dim, &adapters.quant, Some(ad))
                .map_err(|e| LoadError(format!("quantize {kind:?}: {e}")))?;
            Ok(Proj::Qlora(layer))
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("model.layers.{l}");
            layers.push(LayerWeights {
                attn_norm: get(&format!("{p}.input_layernorm.weight"), &[h])?,
                q: maybe_qlora(
                    ProjKind::Q,
                    get(&format!("{p}.self_attn.q_proj.weight"), &[h, h])?,
                    h,
                    h,
                )?,
                k: maybe_qlora(
                    ProjKind::K,
                    get(&format!("{p}.self_attn.k_proj.weight"), &[kv_dim, h])?,
                    kv_dim,
                    h,
                )?,
                v: maybe_qlora(
                    ProjKind::V,
                    get(&format!("{p}.self_attn.v_proj.weight"), &[kv_dim, h])?,
                    kv_dim,
                    h,
                )?,
                o: maybe_qlora(
                    ProjKind::O,
                    get(&format!("{p}.self_attn.o_proj.weight"), &[h, h])?,
                    h,
                    h,
                )?,
                mlp_norm: get(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                gate: maybe_qlora(
                    ProjKind::Gate,
                    get(&format!("{p}.mlp.gate_proj.weight"), &[inter, h])?,
                    inter,
                    h,
                )?,
                up: maybe_qlora(
                    ProjKind::Up,
                    get(&format!("{p}.mlp.up_proj.weight"), &[inter, h])?,
                    inter,
                    h,
                )?,
                down: maybe_qlora(
                    ProjKind::Down,
                    get(&format!("{p}.mlp.down_proj.weight"), &[h, inter])?,
                    h,
                    inter,
                )?,
            });
        }
        Ok(Self {
            embed: get("model.embed_tokens.weight", &[cfg.vocab, h])?,
            final_norm: get("model.norm.weight", &[h])?,
            layers,
            cfg: cfg.clone(),
        })
    }

    /// Forward over `ids`, returning row-major `(seq, vocab)` logits.
    pub fn forward(&self, ids: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let s = ids.len();
        let h = cfg.hidden;
        // Embedding gather.
        let mut x = vec![0.0f32; s * h];
        for (t, &id) in ids.iter().enumerate() {
            x[t * h..(t + 1) * h]
                .copy_from_slice(&self.embed[id as usize * h..(id as usize + 1) * h]);
        }
        for layer in &self.layers {
            block_forward(cfg, layer, &mut x, s, None);
        }
        rmsnorm_in_place(&mut x, &self.final_norm, cfg.rms_eps);
        // Tied head: logits = H . E^T.
        matmul_trans_b(&x, &self.embed, s, h, cfg.vocab)
    }

    /// Current `(a, b)` of the adapter at `(layer, kind)` (`None` if raw).
    /// Used by training loops to feed the optimizer.
    pub fn adapter_ab(&self, layer: usize, kind: ProjKind) -> Option<(Vec<f32>, Vec<f32>)> {
        let w = self.layers.get(layer)?;
        let proj = match kind {
            ProjKind::Q => &w.q,
            ProjKind::K => &w.k,
            ProjKind::V => &w.v,
            ProjKind::O => &w.o,
            ProjKind::Gate => &w.gate,
            ProjKind::Up => &w.up,
            ProjKind::Down => &w.down,
        };
        match proj {
            Proj::Qlora(l) => l.adapter().map(|ad| (ad.a().to_vec(), ad.b().to_vec())),
            Proj::Raw(_) => None,
        }
    }

    /// Write fresh `(a, b)` into the adapter at `(layer, kind)`.
    /// Used by training loops after an optimizer step.
    pub fn set_adapter_weights(
        &mut self,
        layer: usize,
        kind: ProjKind,
        a: &[f32],
        b: &[f32],
    ) -> Result<(), LoadError> {
        let w = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| LoadError(format!("no layer {layer}")))?;
        let proj = match kind {
            ProjKind::Q => &mut w.q,
            ProjKind::K => &mut w.k,
            ProjKind::V => &mut w.v,
            ProjKind::O => &mut w.o,
            ProjKind::Gate => &mut w.gate,
            ProjKind::Up => &mut w.up,
            ProjKind::Down => &mut w.down,
        };
        match proj {
            Proj::Qlora(l) => l
                .adapter_mut()
                .ok_or_else(|| LoadError("adapter missing".to_string()))?
                .set_weights(a, b)
                .map_err(|e| LoadError(e.to_string())),
            Proj::Raw(_) => Err(LoadError(format!("layer {layer} {kind:?} has no adapter"))),
        }
    }

    /// Backward pass returning one [`AdapterGrad`] per attached adapter.
    ///
    /// Runs forward with recording, then backpropagates `dlogits`
    /// (`(seq, vocab)`, e.g. from [`softmax_cross_entropy`]) through the
    /// tied head, all blocks, and into every adapter. Base weights stay
    /// frozen (only adapter `dA`/`dB` are produced).
    pub fn backward(&self, ids: &[u32], dlogits: &[f32]) -> Vec<AdapterGrad> {
        let cfg = &self.cfg;
        let s = ids.len();
        let h = cfg.hidden;
        // Forward with recording.
        let mut x = vec![0.0f32; s * h];
        for (t, &id) in ids.iter().enumerate() {
            x[t * h..(t + 1) * h]
                .copy_from_slice(&self.embed[id as usize * h..(id as usize + 1) * h]);
        }
        let mut caches: Vec<BlockCache> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let mut cache = BlockCache::default();
            block_forward(cfg, layer, &mut x, s, Some(&mut cache));
            caches.push(cache);
        }
        let x_pre_final = x.clone();
        rmsnorm_in_place(&mut x, &self.final_norm, cfg.rms_eps);
        // Tied head backward (embedding grads discarded: frozen).
        let dx_head = matmul(dlogits, &self.embed, s, cfg.vocab, h);
        let (mut dx, _) = rmsnorm_backward(&x_pre_final, &self.final_norm, &dx_head, cfg.rms_eps);
        let mut grads = Vec::new();
        for (li, (layer, cache)) in self.layers.iter().zip(caches.iter()).enumerate().rev() {
            dx = block_backward(cfg, layer, cache, &dx, li, &mut grads);
        }
        grads
    }
}

/// One decoder block in place on `x: (seq, hidden)`.
///
/// When `rec` is `Some`, per-projection inputs are recorded for `backward`.
fn block_forward(
    cfg: &LlamaConfig,
    w: &LayerWeights,
    x: &mut [f32],
    seq: usize,
    mut rec: Option<&mut BlockCache>,
) {
    let h = cfg.hidden;
    let kv_dim = cfg.kv_heads * cfg.head_dim();
    if let Some(r) = rec.as_deref_mut() {
        r.x_in = x.to_vec();
    }
    // --- attention ---
    let mut xn = x.to_vec();
    rmsnorm_in_place(&mut xn, &w.attn_norm, cfg.rms_eps);
    let q = proj_forward(&w.q, &xn, seq, h);
    let k = proj_forward(&w.k, &xn, seq, kv_dim);
    let v = proj_forward(&w.v, &xn, seq, kv_dim);
    let attn = attention(cfg, &q, &k, &v, seq);
    let proj = proj_forward(&w.o, &attn, seq, h);
    for (x_i, p) in x.iter_mut().zip(proj.iter()) {
        *x_i += *p;
    }
    if let Some(r) = rec.as_deref_mut() {
        r.xn_attn = xn;
        r.q = q;
        r.k = k;
        r.v = v;
        r.attn_out = attn;
        r.x_mid = x.to_vec();
    }
    // --- MLP ---
    let mut xn = x.to_vec();
    rmsnorm_in_place(&mut xn, &w.mlp_norm, cfg.rms_eps);
    let gate = proj_forward(&w.gate, &xn, seq, cfg.intermediate);
    let up = proj_forward(&w.up, &xn, seq, cfg.intermediate);
    let mut act = vec![0.0f32; seq * cfg.intermediate];
    for ((a, &g), &u) in act.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *a = silu(g) * u;
    }
    let down = proj_forward(&w.down, &act, seq, h);
    for (x_i, d) in x.iter_mut().zip(down.iter()) {
        *x_i += *d;
    }
    if let Some(r) = rec.as_deref_mut() {
        r.xn_mlp = xn;
        r.gate = gate;
        r.up = up;
        r.act = act;
    }
}

/// Recorded per-block tensors for the backward pass.
#[derive(Debug, Clone, Default)]
struct BlockCache {
    x_in: Vec<f32>,
    xn_attn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    x_mid: Vec<f32>,
    xn_mlp: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
}

/// Mean softmax cross-entropy over `seq` positions plus `dlogits`
/// (`(p - onehot) / seq`, row-major `(seq, vocab)`).
pub fn softmax_cross_entropy(
    logits: &[f32],
    targets: &[u32],
    seq: usize,
    vocab: usize,
) -> (f32, Vec<f32>) {
    let mut loss = 0.0f32;
    let mut dlogits = vec![0.0f32; seq * vocab];
    for t in 0..seq {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        for (d, &v) in dlogits[t * vocab..(t + 1) * vocab]
            .iter_mut()
            .zip(row.iter())
        {
            let e = (v - max).exp();
            *d = e;
            sum += e;
        }
        loss -= (row[targets[t] as usize] - max - sum.ln()) / seq as f32;
        for d in dlogits[t * vocab..(t + 1) * vocab].iter_mut() {
            *d = *d / sum / seq as f32;
        }
        dlogits[t * vocab + targets[t] as usize] -= 1.0 / seq as f32;
    }
    (loss, dlogits)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu_prime(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// One decoder block backward: `dx_out` (dL/d(block output)) in, `dx_in`
/// (dL/d(block input)) out; adapter grads appended to `grads`.
fn block_backward(
    cfg: &LlamaConfig,
    w: &LayerWeights,
    c: &BlockCache,
    dx_out: &[f32],
    layer: usize,
    grads: &mut Vec<AdapterGrad>,
) -> Vec<f32> {
    let h = cfg.hidden;
    let inter = cfg.intermediate;
    let kv_dim = cfg.kv_heads * cfg.head_dim();
    let s = c.x_in.len() / h;
    let mut push = |kind: ProjKind, ab: Option<(Vec<f32>, Vec<f32>)>| {
        if let Some((grad_a, grad_b)) = ab {
            grads.push(AdapterGrad {
                layer,
                kind,
                grad_a,
                grad_b,
            });
        }
    };

    // --- MLP branch (reverse) ---
    let (down_ab, d_act) = proj_backward(&w.down, &c.act, dx_out, s, h);
    push(ProjKind::Down, down_ab);
    let mut dg = vec![0.0f32; s * inter];
    let mut du = vec![0.0f32; s * inter];
    for i in 0..s * inter {
        dg[i] = d_act[i] * c.up[i] * silu_prime(c.gate[i]);
        du[i] = d_act[i] * silu(c.gate[i]);
    }
    let (gate_ab, dxn_g) = proj_backward(&w.gate, &c.xn_mlp, &dg, s, inter);
    push(ProjKind::Gate, gate_ab);
    let (up_ab, dxn_u) = proj_backward(&w.up, &c.xn_mlp, &du, s, inter);
    push(ProjKind::Up, up_ab);
    let dxn_mlp: Vec<f32> = dxn_g.iter().zip(dxn_u.iter()).map(|(a, b)| a + b).collect();
    let (dx_mid_norm, _) = rmsnorm_backward(&c.x_mid, &w.mlp_norm, &dxn_mlp, cfg.rms_eps);
    // Residual x_out = x_mid + mlp: dL/dx_mid = norm path + straight path.
    let dx_mid: Vec<f32> = dx_mid_norm
        .iter()
        .zip(dx_out.iter())
        .map(|(a, b)| a + b)
        .collect();

    // --- attention branch (reverse) ---
    let (o_ab, d_attn) = proj_backward(&w.o, &c.attn_out, &dx_mid, s, h);
    push(ProjKind::O, o_ab);
    let (dq, dk, dv) = attention_backward(cfg, &c.q, &c.k, &c.v, &d_attn, s);
    let (q_ab, dxn_q) = proj_backward(&w.q, &c.xn_attn, &dq, s, h);
    push(ProjKind::Q, q_ab);
    let (k_ab, dxn_k) = proj_backward(&w.k, &c.xn_attn, &dk, s, kv_dim);
    push(ProjKind::K, k_ab);
    let (v_ab, dxn_v) = proj_backward(&w.v, &c.xn_attn, &dv, s, kv_dim);
    push(ProjKind::V, v_ab);
    let mut dxn_attn = vec![0.0f32; s * h];
    for ((a, &b), &d) in dxn_attn.iter_mut().zip(dxn_q.iter()).zip(dxn_k.iter()) {
        *a = b + d;
    }
    // dxn_k/dxn_v are (s, h): k/v projs map h -> kv_dim, dx flows back to h.
    for (a, &d) in dxn_attn.iter_mut().zip(dxn_v.iter()) {
        *a += d;
    }
    let (dx_in_norm, _) = rmsnorm_backward(&c.x_in, &w.attn_norm, &dxn_attn, cfg.rms_eps);

    // Residual x_mid = x_in + attn: dL/dx_in = norm path + straight path.
    dx_mid
        .iter()
        .zip(dx_in_norm.iter())
        .map(|(a, b)| a + b)
        .collect()
}

/// RMSNorm backward: returns `(dx, dw)` for `y = x / sqrt(mean(x^2)+eps) * w`.
fn rmsnorm_backward(x: &[f32], w: &[f32], dy: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
    let h = w.len();
    let s = x.len() / h;
    let mut dx = vec![0.0f32; x.len()];
    let mut dw = vec![0.0f32; h];
    for t in 0..s {
        let xr = &x[t * h..(t + 1) * h];
        let dyr = &dy[t * h..(t + 1) * h];
        let mean_sq = xr.iter().map(|&v| v * v).sum::<f32>() / h as f32;
        let ss = mean_sq + eps;
        let r = ss.sqrt().recip();
        // dS = sum(dy * x * w) * (-0.5) * S^-1.5
        let mut ds = 0.0f32;
        for j in 0..h {
            ds += dyr[j] * xr[j] * w[j];
            dw[j] += dyr[j] * xr[j] * r;
        }
        ds *= -0.5 * ss.recip() * r;
        for j in 0..h {
            dx[t * h + j] = dyr[j] * w[j] * r + 2.0 * xr[j] / h as f32 * ds;
        }
    }
    (dx, dw)
}

/// Causal GQA attention backward with RoPE.
///
/// Returns `(dQ (s,h), dK (s,kv_dim), dV (s,kv_dim))`.
fn attention_backward(
    cfg: &LlamaConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_attn: &[f32],
    seq: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim();
    let h = cfg.hidden;
    let kv_dim = cfg.kv_heads * dh;
    let (cos, sin) = rope_tables(cfg, seq);
    let n_rep = cfg.heads / cfg.kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut dq = vec![0.0f32; seq * h];
    let mut dk = vec![0.0f32; seq * kv_dim];
    let mut dv = vec![0.0f32; seq * kv_dim];
    let mut qr = vec![0.0f32; dh];
    let mut kr = vec![0.0f32; dh];
    // Scratch per query: softmax weights, dp = dL/dp.
    let mut p = vec![0.0f32; seq];
    let mut dp = vec![0.0f32; seq];
    for t in 0..seq {
        for head in 0..cfg.heads {
            let kv = head / n_rep;
            rope_apply(
                &q[(t * h + head * dh)..(t * h + (head + 1) * dh)],
                &cos[t],
                &sin[t],
                &mut qr,
            );
            // Recompute forward scores + softmax (matches `attention`).
            let mut max = f32::NEG_INFINITY;
            for i in 0..=t {
                rope_apply(
                    &k[(i * kv_dim + kv * dh)..(i * kv_dim + (kv + 1) * dh)],
                    &cos[i],
                    &sin[i],
                    &mut kr,
                );
                let dot: f32 = qr.iter().zip(kr.iter()).map(|(a, b)| a * b).sum();
                p[i] = dot * scale;
                max = max.max(p[i]);
            }
            let mut denom = 0.0f32;
            for i in 0..=t {
                p[i] = (p[i] - max).exp();
                denom += p[i];
            }
            for i in 0..=t {
                p[i] /= denom;
            }
            let do_row = &d_attn[(t * h + head * dh)..(t * h + (head + 1) * dh)];
            let mut dp_dot_p = 0.0f32;
            for i in 0..=t {
                let vrow = &v[(i * kv_dim + kv * dh)..(i * kv_dim + (kv + 1) * dh)];
                dp[i] = do_row.iter().zip(vrow.iter()).map(|(a, b)| a * b).sum();
                dp_dot_p += dp[i] * p[i];
                dv[(i * kv_dim + kv * dh)..(i * kv_dim + (kv + 1) * dh)]
                    .iter_mut()
                    .zip(do_row.iter())
                    .for_each(|(dv_ij, &do_j)| *dv_ij += p[i] * do_j);
            }
            // ds_i = p_i * (dp_i - dp.p); dqr += ds_i * kr_i; dkr_i = ds_i * qr.
            let mut dqr = vec![0.0f32; dh];
            for i in 0..=t {
                let ds = p[i] * (dp[i] - dp_dot_p) * scale;
                rope_apply(
                    &k[(i * kv_dim + kv * dh)..(i * kv_dim + (kv + 1) * dh)],
                    &cos[i],
                    &sin[i],
                    &mut kr,
                );
                for j in 0..dh {
                    dqr[j] += ds * kr[j];
                }
                // dk[i,kv] += unrope(ds * qr) (R^T of the forward rotation).
                let dk_base = i * kv_dim + kv * dh;
                for j in 0..dh / 2 {
                    let y0 = ds * qr[j];
                    let y1 = ds * qr[j + dh / 2];
                    dk[dk_base + j] += y0 * cos[i][j] + y1 * sin[i][j];
                    dk[dk_base + j + dh / 2] += -y0 * sin[i][j] + y1 * cos[i][j];
                }
            }
            // dq[t,head] = unrope(dqr).
            let dq_base = t * h + head * dh;
            for j in 0..dh / 2 {
                dq[dq_base + j] += dqr[j] * cos[t][j] + dqr[j + dh / 2] * sin[t][j];
                dq[dq_base + j + dh / 2] += -dqr[j] * sin[t][j] + dqr[j + dh / 2] * cos[t][j];
            }
        }
    }
    (dq, dk, dv)
}

/// Row-wise RMSNorm in place: `x = x / sqrt(mean(x^2) + eps) * w`.
fn rmsnorm_in_place(x: &mut [f32], w: &[f32], eps: f32) {
    let h = w.len();
    for row in x.chunks_mut(h) {
        let mean_sq = row.iter().map(|&v| v * v).sum::<f32>() / h as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for (v, &w_i) in row.iter_mut().zip(w.iter()) {
            *v = *v * scale * w_i;
        }
    }
}

/// Causal GQA attention with RoPE. `q: (s, heads*dh)`, `k/v: (s, kv*dh)`.
/// Returns `(s, hidden)` (heads concatenated in order).
fn attention(cfg: &LlamaConfig, q: &[f32], k: &[f32], v: &[f32], seq: usize) -> Vec<f32> {
    let dh = cfg.head_dim();
    let h = cfg.hidden;
    let (cos, sin) = rope_tables(cfg, seq);
    let n_rep = cfg.heads / cfg.kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut out = vec![0.0f32; seq * h];
    let mut scores = vec![0.0f32; seq];
    for t in 0..seq {
        for head in 0..cfg.heads {
            let kv = head / n_rep;
            // Rope'd query row for (t, head).
            let mut qr = vec![0.0f32; dh];
            rope_apply(
                &q[(t * h + head * dh)..(t * h + (head + 1) * dh)],
                &cos[t],
                &sin[t],
                &mut qr,
            );
            // Scores against positions 0..=t.
            let mut max = f32::NEG_INFINITY;
            for i in 0..=t {
                let mut kr = vec![0.0f32; dh];
                rope_apply(
                    &k[(i * cfg.kv_heads * dh + kv * dh)..(i * cfg.kv_heads * dh + (kv + 1) * dh)],
                    &cos[i],
                    &sin[i],
                    &mut kr,
                );
                let dot: f32 = qr.iter().zip(kr.iter()).map(|(a, b)| a * b).sum();
                scores[i] = dot * scale;
                max = max.max(scores[i]);
            }
            let mut denom = 0.0f32;
            for i in 0..=t {
                scores[i] = (scores[i] - max).exp();
                denom += scores[i];
            }
            let base = t * h + head * dh;
            for j in 0..dh {
                let mut acc = 0.0f32;
                for i in 0..=t {
                    acc += scores[i] / denom * v[(i * cfg.kv_heads * dh + kv * dh) + j];
                }
                out[base + j] = acc;
            }
        }
    }
    out
}

/// RoPE cos/sin tables, one `(dh/2)`-vector per position.
fn rope_tables(cfg: &LlamaConfig, seq: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let half = cfg.head_dim() / 2;
    let mut cos = vec![vec![0.0f32; half]; seq];
    let mut sin = vec![vec![0.0f32; half]; seq];
    for pos in 0..seq {
        for i in 0..half {
            let freq = cfg
                .rope_theta
                .powf(-((2 * i) as f32) / cfg.head_dim() as f32);
            let ang = pos as f32 * freq;
            cos[pos][i] = ang.cos();
            sin[pos][i] = ang.sin();
        }
    }
    (cos, sin)
}

/// HF-style RoPE: `out = x * cos + rotate_half(x) * sin` with
/// `rotate_half(x) = cat(-x2, x1)`.
fn rope_apply(x: &[f32], cos: &[f32], sin: &[f32], out: &mut [f32]) {
    let half = cos.len();
    for i in 0..half {
        out[i] = x[i] * cos[i] - x[i + half] * sin[i];
        out[i + half] = x[i + half] * cos[i] + x[i] * sin[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_rotate_half_known_answer() {
        // d=4, theta s.t. freq[0]=1, freq[1]=theta^-0.5... use theta=1: all freqs 1.
        let cfg = LlamaConfig {
            rope_theta: 1.0,
            ..LlamaConfig::smollm_135m()
        };
        let (cos, sin) = rope_tables(&cfg, 2);
        // pos 0: identity.
        assert_eq!(cos[0], vec![1.0; 32]);
        assert_eq!(sin[0], vec![0.0; 32]);
        let mut out = vec![0.0f32; 64];
        let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
        rope_apply(&x, &cos[1], &sin[1], &mut out);
        // cos(1)=0.5403, sin(1)=0.8415 for every pair.
        assert!((out[0] - (0.0 * 0.5403 - 32.0 * 0.8415)).abs() < 1e-3);
        assert!((out[32] - (32.0 * 0.5403 + 0.0 * 0.8415)).abs() < 1e-3);
    }

    #[test]
    fn rmsnorm_known_answer() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        rmsnorm_in_place(&mut x, &[1.0; 4], 0.0);
        let rms = (30.0f32 / 4.0).sqrt();
        assert!((x[0] - 1.0 / rms).abs() < 1e-6);
        assert!((x[3] - 4.0 / rms).abs() < 1e-6);
    }
}
