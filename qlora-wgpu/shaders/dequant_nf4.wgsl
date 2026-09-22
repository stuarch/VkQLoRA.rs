// NF4 dequantization kernel.
//
// Bindings:
//   0: codes  - packed nibbles, little-nibble-first, reinterpreted as
//               little-endian u32 (8 nibbles per word). The host pads the
//               byte buffer up to a multiple of 4.
//   1: scales - one fp32 absmax per block (double-quantized scales are
//               decoded on the CPU before upload, so both modes work).
//   2: lut    - 16 NF4 levels as fp32 (kept in a buffer so the shader can
//               index it dynamically).
//   3: out    - dequantized fp32 weights, length n.
//   4: params - uniforms { n, block_size }.
//
// One thread per output element.

struct Params {
    n: u32,
    block_size: u32,
};

@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) {
        return;
    }
    let word = codes[idx / 8u];
    let shift = (idx % 8u) * 4u;
    let nib = (word >> shift) & 0xFu;
    let block = idx / params.block_size;
    out[idx] = lut[nib] * scales[block];
}
