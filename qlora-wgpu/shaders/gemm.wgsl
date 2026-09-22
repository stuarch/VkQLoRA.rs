// Tiled row-major GEMMs, TILE x TILE threads per workgroup.
//
// One module, three entry points sharing the same bindings:
//   0: a (read), 1: b (read), 2: c (read_write), 3: params (uniform).
//
// Params always describe the OUTPUT tile: C is (m, n), k is the reduction
// dimension:
//
//   mm_nn: C = A . B        (A: m x k, B: k x n)
//   mm_nt: C = A . B^T      (A: m x k, B: n x k, row-major)
//   mm_tn: C = A^T . B      (A: m_inner x k, B: m_inner x n;
//                            pass m=m_inner, n=n, k=k)
//
// Out-of-tile loads are zero-filled, so any (m, n, k) works, including
// sizes that are not multiples of TILE. Dispatch (ceil(n/16), ceil(m/16)).

const TILE: u32 = 16u;

struct Params {
    m: u32,
    n: u32,
    k: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

// C = A . B
@compute @workgroup_size(16, 16, 1)
fn mm_nn(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    var acc: f32 = 0.0;
    let tiles = (params.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t++) {
        let ak = t * TILE + lid.x;
        let bk = t * TILE + lid.y;
        if (row < params.m && ak < params.k) {
            tile_a[lid.y][lid.x] = a[row * params.k + ak];
        } else {
            tile_a[lid.y][lid.x] = 0.0;
        }
        if (bk < params.k && col < params.n) {
            tile_b[lid.y][lid.x] = b[bk * params.n + col];
        } else {
            tile_b[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var p = 0u; p < TILE; p++) {
            acc += tile_a[lid.y][p] * tile_b[p][lid.x];
        }
        workgroupBarrier();
    }
    if (row < params.m && col < params.n) {
        c[row * params.n + col] = acc;
    }
}

// C = A . B^T  (B stored row-major as n x k)
@compute @workgroup_size(16, 16, 1)
fn mm_nt(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    var acc: f32 = 0.0;
    let tiles = (params.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t++) {
        let kk = t * TILE;
        if (row < params.m && kk + lid.x < params.k) {
            tile_a[lid.y][lid.x] = a[row * params.k + kk + lid.x];
        } else {
            tile_a[lid.y][lid.x] = 0.0;
        }
        if (col < params.n && kk + lid.y < params.k) {
            tile_b[lid.y][lid.x] = b[col * params.k + kk + lid.y];
        } else {
            tile_b[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var p = 0u; p < TILE; p++) {
            acc += tile_a[lid.y][p] * tile_b[p][lid.x];
        }
        workgroupBarrier();
    }
    if (row < params.m && col < params.n) {
        c[row * params.n + col] = acc;
    }
}

// C = A^T . B  (A: m_inner x k, B: m_inner x n, C: k x n;
// params carry m=k, n=n, k=m_inner; each thread reads its own tile row)
@compute @workgroup_size(16, 16, 1)
fn mm_tn(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    var acc: f32 = 0.0;
    let tiles = (params.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t++) {
        // A is read along the reader's own tile row (pp rides lid.x) ...
        let ppa = t * TILE + lid.x;
        if (ppa < params.k && row < params.m) {
            tile_a[lid.y][lid.x] = a[ppa * params.m + row];
        } else {
            tile_a[lid.y][lid.x] = 0.0;
        }
        // ... while B is read along the reader's own tile column
        // (pp rides lid.y), mirroring mm_nt.
        let ppb = t * TILE + lid.y;
        if (ppb < params.k && col < params.n) {
            tile_b[lid.y][lid.x] = b[ppb * params.n + col];
        } else {
            tile_b[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var p = 0u; p < TILE; p++) {
            acc += tile_a[lid.y][p] * tile_b[p][lid.x];
        }
        workgroupBarrier();
    }
    if (row < params.m && col < params.n) {
        c[row * params.n + col] = acc;
    }
}
