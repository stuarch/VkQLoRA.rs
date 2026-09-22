//! Kernel dispatch: dequantize, tiled GEMMs, and composed (Q)LoRA passes.
//!
//! Every function takes `Option<&GpuContext>`: `None` runs the `qlora-core`
//! CPU reference, so one code path serves both GPU machines and headless
//! ones. GEMM dispatches are 2D over 16x16 tiles (see `shaders/gemm.wgsl`).

use bytemuck::{Pod, Zeroable};
use qlora_core::{
    lora::{matmul, matmul_trans_a, matmul_trans_b},
    qlora::{QloraGrads, QloraLinear},
    quant::{QuantizedTensor, NF4_LEVELS},
    QloraError,
};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, MapMode,
};

use crate::context::GpuContext;

const DEQUANT_WORKGROUP: u32 = 256;
const TILE: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DequantParams {
    n: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatmulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

/// GPU dequantize of `q` (`None` context -> CPU reference).
pub fn gpu_dequantize(
    ctx: Option<&GpuContext>,
    q: &QuantizedTensor,
) -> Result<Vec<f32>, QloraError> {
    let Some(ctx) = ctx else {
        return Ok(q.dequantize());
    };
    let n = q.num_elements();
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = ctx.device();

    // Pad codes to a whole number of u32 words for the shader.
    let mut codes = q.codes().to_vec();
    codes.resize(codes.len().next_multiple_of(4), 0);
    let scales = q.block_scales();

    let codes_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("qlora.codes"),
        contents: &codes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let scales_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("qlora.scales"),
        contents: bytemuck::cast_slice(&scales),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let lut_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("qlora.nf4lut"),
        contents: bytemuck::cast_slice(&NF4_LEVELS),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&BufferDescriptor {
        label: Some("qlora.dequant_out"),
        size: (n * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = DequantParams {
        n: n as u32,
        block_size: q.block_size() as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("qlora.dequant_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    dispatch(
        ctx,
        ctx.dequant_pipeline(),
        &[&codes_buf, &scales_buf, &lut_buf, &out_buf, &params_buf],
        n.div_ceil(DEQUANT_WORKGROUP as usize) as u32,
        1,
    );
    readback_f32(ctx, &out_buf, n)
}

/// Upload helper: two read-only storage buffers + general output + uniforms.
fn gemm_buffers(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    out_len: usize,
    params: MatmulParams,
) -> (Buffer, Buffer, Buffer, Buffer) {
    let device = ctx.device();
    let a_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("qlora.gemm_a"),
        contents: bytemuck::cast_slice(a),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let b_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("qlora.gemm_b"),
        contents: bytemuck::cast_slice(b),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&BufferDescriptor {
        label: Some("qlora.gemm_out"),
        size: (out_len * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("qlora.gemm_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    (a_buf, b_buf, out_buf, params_buf)
}

/// Tiled GPU `C = A · B` with `A: (m, k)`, `B: (k, n)` row-major
/// (`None` context -> CPU reference).
pub fn gpu_matmul(
    ctx: Option<&GpuContext>,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, QloraError> {
    let Some(ctx) = ctx else {
        return Ok(matmul(a, b, m, k, n));
    };
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if a.len() != m * k || b.len() != k * n {
        return Err(QloraError::ShapeMismatch(format!(
            "gpu_matmul: a.len() = {}, b.len() = {}, expected {m}*{k} and {k}*{n}",
            a.len(),
            b.len()
        )));
    }
    let params = MatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        _pad: 0,
    };
    let (a_buf, b_buf, out_buf, params_buf) = gemm_buffers(ctx, a, b, m * n, params);
    dispatch(
        ctx,
        ctx.mm_nn_pipeline(),
        &[&a_buf, &b_buf, &out_buf, &params_buf],
        n.div_ceil(TILE as usize) as u32,
        m.div_ceil(TILE as usize) as u32,
    );
    readback_f32(ctx, &out_buf, m * n)
}

/// Tiled GPU `C = A · B^T` with `A: (m, k)`, `B: (n, k)` row-major
/// (`None` context -> CPU reference).
pub fn gpu_matmul_trans_b(
    ctx: Option<&GpuContext>,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, QloraError> {
    let Some(ctx) = ctx else {
        return Ok(matmul_trans_b(a, b, m, k, n));
    };
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if a.len() != m * k || b.len() != n * k {
        return Err(QloraError::ShapeMismatch(format!(
            "gpu_matmul_trans_b: a.len() = {}, b.len() = {}, expected {m}*{k} and {n}*{k}",
            a.len(),
            b.len()
        )));
    }
    let params = MatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        _pad: 0,
    };
    let (a_buf, b_buf, out_buf, params_buf) = gemm_buffers(ctx, a, b, m * n, params);
    dispatch(
        ctx,
        ctx.mm_nt_pipeline(),
        &[&a_buf, &b_buf, &out_buf, &params_buf],
        n.div_ceil(TILE as usize) as u32,
        m.div_ceil(TILE as usize) as u32,
    );
    readback_f32(ctx, &out_buf, m * n)
}

/// Tiled GPU `C = A^T · B` with `A: (m, k)`, `B: (m, n)`, `C: (k, n)`
/// (`None` context -> CPU reference).
pub fn gpu_matmul_trans_a(
    ctx: Option<&GpuContext>,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, QloraError> {
    let Some(ctx) = ctx else {
        return Ok(matmul_trans_a(a, b, m, k, n));
    };
    if k == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if a.len() != m * k || b.len() != m * n {
        return Err(QloraError::ShapeMismatch(format!(
            "gpu_matmul_trans_a: a.len() = {}, b.len() = {}, expected {m}*{k} and {m}*{n}",
            a.len(),
            b.len()
        )));
    }
    // Params describe the (k, n) output; k-field carries the inner dim m.
    let params = MatmulParams {
        m: k as u32,
        n: n as u32,
        k: m as u32,
        _pad: 0,
    };
    let (a_buf, b_buf, out_buf, params_buf) = gemm_buffers(ctx, a, b, k * n, params);
    dispatch(
        ctx,
        ctx.mm_tn_pipeline(),
        &[&a_buf, &b_buf, &out_buf, &params_buf],
        n.div_ceil(TILE as usize) as u32,
        k.div_ceil(TILE as usize) as u32,
    );
    readback_f32(ctx, &out_buf, k * n)
}

/// Fused QLoRA forward on GPU: dequant + base GEMM + optional LoRA branch.
///
/// `None` context -> identical CPU result via [`QloraLinear::forward`].
pub fn gpu_qlora_forward(
    ctx: Option<&GpuContext>,
    layer: &QloraLinear,
    x: &[f32],
    batch: usize,
) -> Result<Vec<f32>, QloraError> {
    let Some(ctx) = ctx else {
        return layer.forward(x, batch);
    };
    let in_dim = layer.in_dim();
    let out_dim = layer.out_dim();
    if x.len() != batch * in_dim {
        return Err(QloraError::ShapeMismatch(format!(
            "x.len() = {} but batch*in_dim = {}",
            x.len(),
            batch * in_dim
        )));
    }

    let w = gpu_dequantize(Some(ctx), layer.weight())?;
    let mut y = gpu_matmul_trans_b(Some(ctx), x, &w, batch, in_dim, out_dim)?;

    if let Some(ad) = layer.adapter() {
        // Fold alpha/r into A on the CPU; the GPU then does plain GEMMs.
        let s = ad.scale();
        let a_scaled: Vec<f32> = ad.a().iter().map(|v| v * s).collect();
        let r = ad.rank();
        let t1 = gpu_matmul_trans_b(Some(ctx), x, &a_scaled, batch, in_dim, r)?;
        let t2 = gpu_matmul_trans_b(Some(ctx), &t1, ad.b(), batch, r, out_dim)?;
        for (y_i, t) in y.iter_mut().zip(t2.iter()) {
            *y_i += *t;
        }
    }
    Ok(y)
}

/// QLoRA backward pass on GPU, mirroring [`QloraLinear::backward`].
///
/// `None` context -> identical CPU result.
pub fn gpu_qlora_backward(
    ctx: Option<&GpuContext>,
    layer: &QloraLinear,
    x: &[f32],
    dy: &[f32],
    batch: usize,
) -> Result<QloraGrads, QloraError> {
    let Some(ctx) = ctx else {
        return layer.backward(x, dy, batch);
    };
    let in_dim = layer.in_dim();
    let out_dim = layer.out_dim();
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

    let w = gpu_dequantize(Some(ctx), layer.weight())?;
    let mut grad_x = gpu_matmul(Some(ctx), dy, &w, batch, out_dim, in_dim)?;

    let (grad_a, grad_b) = match layer.adapter() {
        None => (None, None),
        Some(ad) => {
            let r = ad.rank();
            let s = ad.scale();
            let dys: Vec<f32> = dy.iter().map(|v| v * s).collect();
            let t1 = gpu_matmul_trans_b(Some(ctx), x, ad.a(), batch, in_dim, r)?;
            let grad_b = gpu_matmul_trans_a(Some(ctx), &dys, &t1, batch, out_dim, r)?;
            let ds = gpu_matmul(Some(ctx), &dys, ad.b(), batch, out_dim, r)?;
            let grad_a = gpu_matmul_trans_a(Some(ctx), &ds, x, batch, r, in_dim)?;
            let dx_lora = gpu_matmul(Some(ctx), &ds, ad.a(), batch, r, in_dim)?;
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

fn dispatch(
    ctx: &GpuContext,
    pipeline: &ComputePipeline,
    buffers: &[&Buffer],
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let device = ctx.device();
    let layout = pipeline.get_bind_group_layout(0);
    let entries: Vec<BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bg = device.create_bind_group(&BindGroupDescriptor {
        label: Some("qlora.bg"),
        layout: &layout,
        entries: &entries,
    });
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("qlora.enc"),
    });
    {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("qlora.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups_x.max(1), workgroups_y.max(1), 1);
    }
    ctx.queue().submit(Some(enc.finish()));
}

fn readback_f32(ctx: &GpuContext, src: &Buffer, len: usize) -> Result<Vec<f32>, QloraError> {
    let device = ctx.device();
    let size = (len * 4) as u64;
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("qlora.staging"),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("qlora.readback"),
    });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    ctx.queue().submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    // wgpu 22: `Maintain::Wait` blocks until the mapping is ready.
    let _ = device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| QloraError::Gpu(format!("readback channel: {e:?}")))?
        .map_err(|e| QloraError::Gpu(format!("map_async: {e:?}")))?;
    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}
