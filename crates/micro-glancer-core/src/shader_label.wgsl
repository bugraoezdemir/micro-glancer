
// Label images, flat (appended to the flat shader; entry fs_label): every screen pixel
// takes the id of the level pixel whose centre is nearest (the frame's level, else a
// coarser level holding its tile, else a finer one), exact: the texture's red and green
// hold the id's low and high 32 bits (green is 0 for narrower ids). Id 0 is the
// background (nothing drawn). Each id has a colour of its own (a hash of the id and the
// layer's seed); with outlines only pixels whose screen neighbours hold another id are
// drawn. A selected id can be highlighted or shown alone. The colour is blended over
// what lies beneath (premultiplied by the opacity).
//
// p.mode bits: 1 outlines, 2 only the selected id, 4 an id is selected; p.extra: the
// selected id's low and high 32 bits, the colour seed.

struct Id {
    found: bool,
    v: vec2<u32>,
}

// The id at screen point s from level l (channel 0), if its tile is cached.
fn id_in(l: Level, s: vec2<f32>) -> Id {
    var r: Id;
    r.found = false;
    r.v = vec2<u32>(0u);
    let px = clamp(pixel_at(l, s), vec2<i32>(0), l.size - 1);
    let t = px / TILE;
    let page = pages[l.page_base + u32(t.y) * l.grid_w + u32(t.x)];
    if (page == 0u) {
        return r;
    }
    let slot = page - 1u;
    let within = slot % SLOTS_PER_LAYER;
    let base = vec2<i32>(i32(within % SLOTS_PER_ROW), i32(within / SLOTS_PER_ROW)) * TILE;
    r.v = textureLoad(atlas, base + px - t * TILE, i32(slot / SLOTS_PER_LAYER), 0).rg;
    r.found = true;
    return r;
}

fn hash_id(id: vec2<u32>) -> u32 {
    var h = (id.x * 0x9E3779B1u) ^ (id.y * 0x85EBCA77u) ^ (p.extra.z * 0xC2B2AE3Du);
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    h *= 0x297A2D39u;
    h ^= h >> 15u;
    return h;
}

fn hsv(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(5.0, 3.0, 1.0);
    let q = abs(fract(vec3<f32>(h) + k / 6.0) * 6.0 - 3.0);
    return v * mix(vec3<f32>(1.0), clamp(q - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

fn label_colour(id: vec2<u32>) -> vec3<f32> {
    let h = hash_id(id);
    let hue = f32(h & 0xFFFFu) / 65535.0;
    let sat = 0.6 + 0.35 * f32((h >> 16u) & 0xFFu) / 255.0;
    let val = 0.85 + 0.15 * f32(h >> 24u) / 255.0;
    return hsv(hue, sat, val);
}

// The label's colour and alpha for `id`, or alpha 0 (nothing drawn); `edge`: whether a
// screen neighbour holds another id (only asked with outlines).
fn label_paint(id: vec2<u32>, edge: bool) -> vec4<f32> {
    if (all(id == vec2<u32>(0u))) {
        return vec4<f32>(0.0);
    }
    let selected = vec2<u32>(p.extra.x, p.extra.y);
    let has = (p.mode & 4u) != 0u;
    let is = has && all(id == selected);
    if ((p.mode & 2u) != 0u && !is) {
        return vec4<f32>(0.0);
    }
    if ((p.mode & 1u) != 0u && !edge) {
        return vec4<f32>(0.0);
    }
    var colour = label_colour(id);
    var a = p.opacity;
    if ((p.mode & 1u) != 0u) {
        a = min(1.0, p.opacity * 1.6);
    }
    if (has && (p.mode & 2u) == 0u) {
        if (is) {
            colour = mix(colour, vec3<f32>(1.0), 0.35);
            a = min(1.0, a * 1.5);
        } else {
            a *= 0.3;
        }
    }
    return vec4<f32>(colour * a, a);
}

@fragment
fn fs_label(in: Out) -> @location(0) vec4<f32> {
    let s = in.pos.xy;
    let top = levels[p.level];
    let here = pixel_at(top, s);
    if (any(here < vec2<i32>(0)) || any(here >= top.size)) {
        return vec4<f32>(0.0);
    }
    for (var k = 0u; k < p.n_levels; k++) {
        let l = levels[level_order(k)];
        let r = id_in(l, s);
        if (!r.found) {
            continue;
        }
        var edge = false;
        if ((p.mode & 1u) != 0u) {
            let n = array<vec2<f32>, 4>(vec2<f32>(1.0, 0.0), vec2<f32>(-1.0, 0.0),
                                        vec2<f32>(0.0, 1.0), vec2<f32>(0.0, -1.0));
            for (var j = 0u; j < 4u; j++) {
                let o = id_in(l, s + n[j]);
                if (o.found && any(o.v != r.v)) {
                    edge = true;
                }
            }
        }
        return label_paint(r.v, edge);
    }
    return vec4<f32>(0.0);
}
