// One plane drawn from a tile atlas: every screen pixel finds its image pixel through
// the camera, its tile through the page table, and its value in the atlas; contrast
// limits map it to grey. A tile not loaded yet shows the background.

struct Params {
    // image pixel (x, y) at the screen's top-left corner
    origin: vec2<f32>,
    // image pixels per screen pixel
    scale: f32,
    lo: f32,
    hi: f32,
    width: f32,
    height: f32,
    grid_w: u32,
    tile: u32,
    slots_per_row: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> pages: array<u32>;

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

const BACKGROUND = vec4<f32>(0.08, 0.08, 0.08, 1.0);
const MISSING = vec4<f32>(0.16, 0.16, 0.18, 1.0);

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    // pixel centres: screen pixel (sx + 0.5) covers image x = origin + (sx + 0.5) * scale
    let ip = p.origin + in.pos.xy * p.scale;
    if (ip.x < 0.0 || ip.y < 0.0 || ip.x >= p.width || ip.y >= p.height) {
        return BACKGROUND;
    }
    let px = vec2<u32>(floor(ip));
    let t = px / p.tile;
    let slot = pages[t.y * p.grid_w + t.x];
    if (slot == 0u) {
        return MISSING;
    }
    let s = slot - 1u;
    let base = vec2<u32>(s % p.slots_per_row, s / p.slots_per_row) * p.tile;
    let v = textureLoad(atlas, vec2<i32>(base + px - t * p.tile), 0).r;
    let g = clamp((v - p.lo) / max(p.hi - p.lo, 1e-30), 0.0, 1.0);
    return vec4<f32>(g, g, g, 1.0);
}
