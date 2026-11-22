
// Label volumes cut by a plane at any angle (appended to the brick shader's bindings and
// sampling; entry fs_label): the nearest voxel's id (exact: the texture's red and green
// hold its low and high 32 bits), the frame's level else a coarser one else a finer one;
// colours, outlines (screen neighbours holding another id) and the selected id as for
// flat label images.

struct Id {
    found: bool,
    v: vec2<u32>,
}

fn id_in(l: Level, s: vec2<f32>) -> Id {
    var r: Id;
    r.found = false;
    r.v = vec2<u32>(0u);
    let f = l.origin_frac.xyz + s.x * l.ux.xyz + s.y * l.uy.xyz;
    let vox = clamp(l.origin_int.xyz + vec3<i32>(floor(f + 0.5)), vec3<i32>(0), l.size.xyz - 1);
    let page = page_of(l, 0u, vox);
    if (page == 0u) {
        return r;
    }
    let slot = page - 1u;
    let o = vox - (vox / STEP) * STEP;
    let within = slot % PER_LAYER;
    let sx = i32(within % STRIPS_PER_ROW) * STRIP_W + o.z * BRICK + o.x;
    let sy = i32(within / STRIPS_PER_ROW) * BRICK + o.y;
    r.v = textureLoad(atlas, vec2<i32>(sx, sy), i32(slot / PER_LAYER), 0).rg;
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
    let here = top.origin_int.xyz + vec3<i32>(floor(frac_at(top, s) + 0.5));
    if (any(here < vec3<i32>(0)) || any(here >= top.size.xyz)) {
        return vec4<f32>(0.0);
    }
    for (var k = 0u; k < p.n_levels; k++) {
        let l = levels[level_order(k)];
        if (l.grid.w == 0u) {
            continue;
        }
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
