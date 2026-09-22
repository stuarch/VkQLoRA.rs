//! WGPU device setup: adapter, device, and compiled pipelines.

use qlora_core::QloraError;
use wgpu::{
    Backends, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Features,
    Instance, InstanceDescriptor, Limits, MemoryHints, PipelineCompilationOptions, PowerPreference,
    Queue, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
};

const DEQUANT_WGSL: &str = include_str!("../shaders/dequant_nf4.wgsl");
const GEMM_WGSL: &str = include_str!("../shaders/gemm.wgsl");

/// A ready-to-dispatch GPU context.
///
/// Created with [`GpuContext::try_new_blocking`]; `Ok(None)` means "no
/// usable adapter", in which case callers should use the CPU reference in
/// `qlora-core`.
pub struct GpuContext {
    device: Device,
    queue: Queue,
    dequant_pipeline: ComputePipeline,
    /// Tiled `C = A . B`.
    mm_nn_pipeline: ComputePipeline,
    /// Tiled `C = A . B^T`.
    mm_nt_pipeline: ComputePipeline,
    /// Tiled `C = A^T . B`.
    mm_tn_pipeline: ComputePipeline,
}

impl GpuContext {
    /// Blocking constructor. Returns `Ok(None)` when no GPU adapter is
    /// found instead of failing, so headless machines keep working on CPU.
    pub fn try_new_blocking() -> Result<Option<Self>, QloraError> {
        pollster::block_on(Self::try_new_async())
    }

    async fn try_new_async() -> Result<Option<Self>, QloraError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        // NOTE: wgpu 22 `request_adapter` returns `Option<Adapter>`.
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;
        let adapter = match adapter {
            Some(a) => a,
            None => return Ok(None),
        };
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("qlora-wgpu"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| QloraError::Gpu(format!("request_device: {e:?}")))?;

        let dequant_pipeline = Self::pipeline(&device, "dequant_nf4", DEQUANT_WGSL, "main");
        let mm_nn_pipeline = Self::pipeline(&device, "mm_nn", GEMM_WGSL, "mm_nn");
        let mm_nt_pipeline = Self::pipeline(&device, "mm_nt", GEMM_WGSL, "mm_nt");
        let mm_tn_pipeline = Self::pipeline(&device, "mm_tn", GEMM_WGSL, "mm_tn");
        Ok(Some(Self {
            device,
            queue,
            dequant_pipeline,
            mm_nn_pipeline,
            mm_nt_pipeline,
            mm_tn_pipeline,
        }))
    }

    fn pipeline(device: &Device, label: &str, src: &str, entry: &str) -> ComputePipeline {
        use std::borrow::Cow;
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(Cow::Borrowed(src)),
        });
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: entry,
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }
    pub(crate) fn queue(&self) -> &Queue {
        &self.queue
    }
    pub(crate) fn dequant_pipeline(&self) -> &ComputePipeline {
        &self.dequant_pipeline
    }
    pub(crate) fn mm_nn_pipeline(&self) -> &ComputePipeline {
        &self.mm_nn_pipeline
    }
    pub(crate) fn mm_nt_pipeline(&self) -> &ComputePipeline {
        &self.mm_nt_pipeline
    }
    pub(crate) fn mm_tn_pipeline(&self) -> &ComputePipeline {
        &self.mm_tn_pipeline
    }
}
