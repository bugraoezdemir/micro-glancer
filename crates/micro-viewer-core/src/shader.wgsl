// An image's plane drawn from a tile atlas. Every screen pixel finds, per level, the
// level pixel whose centre is nearest (from that level's placement, computed per frame on
// the CPU in double precision); per active channel it takes the value from the frame's
// level, or else from the nearest coarser level holding the tile, or else a finer one;
// contrast, gamma and colour map it, and the channels add up.
//
// TEXEL is replaced by u32, i32 or f32 to match the atlas format.

struct Params {
    n_levels: u32,
    n_channels: u32,
    // the frame's level (finest worth drawing at this zoom)
    level: u32,
    // page table entries per channel (all levels)
    channel_stride: u32,
    background: vec4<f32>,
    missing: vec4<f32>,
}

// One level's placement on screen: the level pixel at screen position p (pixels, centres
// at +0.5) is floor(origin + p.x * ux + p.y * uy), origin split into an integer part and a
// fraction so deep zooms keep their precision.
struct Level {
    origin_int: vec2<i32>,
    origin_frac: vec2<f32>,
    ux: vec2<f32>,
    uy: vec2<f32>,
    size: vec2<i32>,
    grid_w: u32,
    page_base: u32,
}

struct Channel {
    color: vec3<f32>,
    gamma: f32,
    lo: f32,
    hi: f32,
    shown: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var atlas: texture_2d_array<TEXEL>;
@group(0) @binding(2) var<storage, read> pages: array<u32>;
@group(0) @binding(3) var<storage, read> levels: array<Level>;
@group(0) @binding(4) var<storage, read> channels: array<Channel>;

const TILE: i32 = 256;
const SLOTS_PER_ROW: u32 = 16u;
const SLOTS_PER_LAYER: u32 = 256u;

struct Out {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    // one triangle covering the screen
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    var o: Out;
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    return o;
}

fn pixel_at(l: Level, s: vec2<f32>) -> vec2<i32> {
    return l.origin_int + vec2<i32>(floor(l.origin_frac + s.x * l.ux + s.y * l.uy));
}

// The `k`-th level to try: the frame's level, then coarser ones, then finer ones.
fn level_order(k: u32) -> u32 {
    let coarser = p.n_levels - 1u - p.level;
    if (k <= coarser) {
        return p.level + k;
    }
    return p.level - (k - coarser);
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let s = in.pos.xy;
    let top = levels[p.level];
    let here = pixel_at(top, s);
    if (any(here < vec2<i32>(0)) || any(here >= top.size)) {
        return p.background;
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
            let px = clamp(pixel_at(l, s), vec2<i32>(0), l.size - 1);
            let t = px / TILE;
            let page = pages[c * p.channel_stride + l.page_base + u32(t.y) * l.grid_w + u32(t.x)];
            if (page == 0u) {
                continue;
            }
            let slot = page - 1u;
            let within = slot % SLOTS_PER_LAYER;
            let base = vec2<i32>(i32(within % SLOTS_PER_ROW), i32(within / SLOTS_PER_ROW)) * TILE;
            let v = f32(textureLoad(atlas, base + px - t * TILE, i32(slot / SLOTS_PER_LAYER), 0).r);
            var g = clamp((v - ch.lo) / max(ch.hi - ch.lo, 1e-30), 0.0, 1.0);
            if (ch.gamma != 1.0) {
                g = pow(g, ch.gamma);
            }
            rgb += g * ch.color;
            shown = true;
            break;
        }
    }
    if (wanted && !shown) {
        return p.missing;
    }
    return vec4<f32>(min(rgb, vec3<f32>(1.0)), 1.0);
}
