// A volume cut by a plane at any angle, drawn from its brick atlas. Every screen pixel
// finds, per level, its point in the level's voxel indices (the plane's placement,
// computed per frame on the CPU in double precision: camera, the layer's transform and
// the level's own placement); voxel centres are at integers. Nearest: the voxel whose
// centre is nearest. Linear: the eight voxels around the point, weighted (trilinear);
// bricks overlap by one voxel, so all eight are in the brick of the lowest one. Per
// active channel the value comes from the frame's level, or else the nearest coarser
// level holding the brick, or else a finer one; contrast, gamma and the channel's colour
// or colour map as in the flat shader; channels add up; output premultiplied by the
// layer's opacity, alpha the coverage.
//
// TEXEL is replaced by u32, i32 or f32 to match the atlas format.

struct Params {
    n_levels: u32,
    n_channels: u32,
    level: u32,
    // page table entries per channel (all levels)
    channel_stride: u32,
    missing: vec4<f32>,
    outside: vec4<f32>,
    opacity: f32,
    linear: u32,
    // projections: the mode (see shader_mip.wgsl) and the opacity per voxel
    mode: u32,
    density: f32,
}

struct Level {
    origin_int: vec4<i32>,
    origin_frac: vec4<f32>,
    ux: vec4<f32>,
    uy: vec4<f32>,
    // voxels along x, y, z; w: the level's first page entry
    size: vec4<i32>,
    // bricks along x, y, z; w: 1 when the level has a page table
    grid: vec4<u32>,
    // projections: the voxel index change per world unit along the view's normal
    dir: vec4<f32>,
}

struct Channel {
    color: vec3<f32>,
    gamma: f32,
    lo: f32,
    hi: f32,
    shown: u32,
    lut: u32,
    // isosurfaces: the level when has_iso, else the middle of the contrast window
    iso: f32,
    has_iso: u32,
    _c0: u32,
    _c1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var atlas: texture_2d_array<TEXEL>;
@group(0) @binding(2) var<storage, read> pages: array<u32>;
@group(0) @binding(3) var<storage, read> levels: array<Level>;
@group(0) @binding(4) var<storage, read> channels: array<Channel>;
@group(0) @binding(5) var luts: texture_2d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;
// per brick slot: its lowest and highest value
@group(0) @binding(7) var<storage, read> ranges: array<vec2<f32>>;

const BRICK: i32 = 32;
const STEP: i32 = 31;
const STRIP_W: i32 = 1024;
const STRIPS_PER_ROW: u32 = 4u;
const PER_LAYER: u32 = 512u;
const LUT_SIZE: f32 = 256.0;

struct Out {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    var o: Out;
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    return o;
}

// The point's fraction part (the integer part is origin_int).
fn frac_at(l: Level, s: vec2<f32>) -> vec3<f32> {
    return l.origin_frac.xyz + s.x * l.ux.xyz + s.y * l.uy.xyz;
}

fn level_order(k: u32) -> u32 {
    let coarser = p.n_levels - 1u - p.level;
    if (k <= coarser) {
        return p.level + k;
    }
    return p.level - (k - coarser);
}

// The slot + 1 of the brick holding voxel v (>= 0) of channel c at level l, or 0.
fn page_of(l: Level, c: u32, v: vec3<i32>) -> u32 {
    let b = vec3<u32>(v / STEP);
    let entry = c * p.channel_stride + u32(l.size.w) + (b.z * l.grid.y + b.y) * l.grid.x + b.x;
    return pages[entry];
}

// Voxel `o` (within its brick) of brick slot `slot`.
fn load(slot: u32, o: vec3<i32>) -> f32 {
    let within = slot % PER_LAYER;
    let sx = i32(within % STRIPS_PER_ROW) * STRIP_W + o.z * BRICK + o.x;
    let sy = i32(within / STRIPS_PER_ROW) * BRICK + o.y;
    return f32(textureLoad(atlas, vec2<i32>(sx, sy), i32(slot / PER_LAYER), 0).r);
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let s = in.pos.xy;
    let top = levels[p.level];
    let here = top.origin_int.xyz + vec3<i32>(floor(frac_at(top, s) + 0.5));
    if (any(here < vec3<i32>(0)) || any(here >= top.size.xyz)) {
        return p.outside;
    }
    var rgb = vec3<f32>(0.0);
    var shown = false;
    var wanted = false;
    for (var c = 0u; c < p.n_channels; c++) {
        let ch = channels[c];
        if (ch.shown == 0u) {
            continue;
        }
        wanted = true;
        for (var k = 0u; k < p.n_levels; k++) {
            let l = levels[level_order(k)];
            if (l.grid.w == 0u) {
                continue;
            }
            let last = l.size.xyz - 1;
            let f = frac_at(l, s);
            var v: f32;
            if (p.linear != 0u) {
                // the point within the level's voxel centres, and the lowest of its eight
                let at = clamp(vec3<f32>(l.origin_int.xyz) + f, vec3<f32>(0.0), vec3<f32>(last));
                let i0 = min(vec3<i32>(floor(at)), last);
                let w = at - vec3<f32>(i0);
                let page = page_of(l, c, i0);
                if (page == 0u) {
                    continue;
                }
                let slot = page - 1u;
                let o0 = i0 - (i0 / STEP) * STEP;
                // the second voxel along an axis: the next one, or the same at the edge
                let d = vec3<i32>(min(i0 + 1, last) - i0);
                let c000 = load(slot, o0);
                let c100 = load(slot, o0 + vec3<i32>(d.x, 0, 0));
                let c010 = load(slot, o0 + vec3<i32>(0, d.y, 0));
                let c110 = load(slot, o0 + vec3<i32>(d.x, d.y, 0));
                let c001 = load(slot, o0 + vec3<i32>(0, 0, d.z));
                let c101 = load(slot, o0 + vec3<i32>(d.x, 0, d.z));
                let c011 = load(slot, o0 + vec3<i32>(0, d.y, d.z));
                let c111 = load(slot, o0 + d);
                let x00 = mix(c000, c100, w.x);
                let x10 = mix(c010, c110, w.x);
                let x01 = mix(c001, c101, w.x);
                let x11 = mix(c011, c111, w.x);
                v = mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
            } else {
                let vox = clamp(l.origin_int.xyz + vec3<i32>(floor(f + 0.5)), vec3<i32>(0), last);
                let page = page_of(l, c, vox);
                if (page == 0u) {
                    continue;
                }
                v = load(page - 1u, vox - (vox / STEP) * STEP);
            }
            var g = clamp((v - ch.lo) / max(ch.hi - ch.lo, 1e-30), 0.0, 1.0);
            if (ch.gamma != 1.0) {
                g = pow(g, ch.gamma);
            }
            if (ch.lut != 0u) {
                let lu = (g * (LUT_SIZE - 1.0) + 0.5) / LUT_SIZE;
                let lv = (f32(c) + 0.5) / f32(p.n_channels);
                rgb += textureSampleLevel(luts, lut_sampler, vec2<f32>(lu, lv), 0.0).rgb;
            } else {
                rgb += g * ch.color;
            }
            shown = true;
            break;
        }
    }
    if (!wanted) {
        return p.outside;
    }
    if (!shown) {
        return vec4<f32>(p.missing.rgb * p.opacity, p.opacity);
    }
    return vec4<f32>(min(rgb, vec3<f32>(1.0)) * p.opacity, p.opacity);
}
