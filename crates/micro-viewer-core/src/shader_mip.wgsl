
// Maximum intensity projection (appended to the brick shader, whose own fragment entry is
// renamed): every screen pixel casts a ray along the view's normal through the volume
// (orthographic), the part inside the volume's box at the frame's level, in steps of at
// most 0.8 voxel along any axis (at most MAX_STEPS steps). Per active channel the ray
// keeps the highest value it meets (each sample from the frame's level, or else the
// nearest coarser level holding its brick, or else a finer one; nearest voxel or
// trilinear); contrast, gamma and colour map the highest value (they keep the order).

const MAX_STEPS: f32 = 2048.0;

struct Hit {
    found: bool,
    v: f32,
}

// The value at point q (level l's voxel indices, centres at integers) in brick slot
// `slot` (the brick of q's voxel, or of its lowest neighbour when linear).
fn value_in(l: Level, slot: u32, q: vec3<f32>) -> f32 {
    let last = l.size.xyz - 1;
    if (p.linear != 0u) {
        let at = clamp(q, vec3<f32>(0.0), vec3<f32>(last));
        let i0 = min(vec3<i32>(floor(at)), last);
        let w = at - vec3<f32>(i0);
        let o0 = i0 - (i0 / STEP) * STEP;
        let d = vec3<i32>(min(i0 + 1, last) - i0);
        let x00 = mix(load(slot, o0), load(slot, o0 + vec3<i32>(d.x, 0, 0)), w.x);
        let x10 = mix(load(slot, o0 + vec3<i32>(0, d.y, 0)), load(slot, o0 + vec3<i32>(d.x, d.y, 0)), w.x);
        let x01 = mix(load(slot, o0 + vec3<i32>(0, 0, d.z)), load(slot, o0 + vec3<i32>(d.x, 0, d.z)), w.x);
        let x11 = mix(load(slot, o0 + vec3<i32>(0, d.y, d.z)), load(slot, o0 + d), w.x);
        return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
    }
    let vox = clamp(vec3<i32>(floor(q + 0.5)), vec3<i32>(0), last);
    return load(slot, vox - (vox / STEP) * STEP);
}

// The voxel whose brick holds point q's sample (nearest, or the lowest of eight).
fn voxel_of(l: Level, q: vec3<f32>) -> vec3<i32> {
    let last = l.size.xyz - 1;
    if (p.linear != 0u) {
        return min(vec3<i32>(floor(clamp(q, vec3<f32>(0.0), vec3<f32>(last)))), last);
    }
    return clamp(vec3<i32>(floor(q + 0.5)), vec3<i32>(0), last);
}

// Channel c's value at point q of level l, if its brick is cached.
fn sample_at(l: Level, c: u32, q: vec3<f32>) -> Hit {
    var h: Hit;
    h.found = false;
    h.v = 0.0;
    let page = page_of(l, c, voxel_of(l, q));
    if (page == 0u) {
        return h;
    }
    h.v = value_in(l, page - 1u, q);
    h.found = true;
    return h;
}

// The ray's entry and exit (t) through the box lo .. hi along one axis.
fn slab(o: f32, d: f32, lo: f32, hi: f32, t: vec2<f32>) -> vec2<f32> {
    if (abs(d) < 1e-12) {
        if (o < lo || o > hi) {
            return vec2<f32>(1.0, -1.0);
        }
        return t;
    }
    let a = (lo - o) / d;
    let b = (hi - o) / d;
    return vec2<f32>(max(t.x, min(a, b)), min(t.y, max(a, b)));
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let s = in.pos.xy;
    let top = levels[p.level];
    // the ray at the frame's level: its point at t = 0 (on the view's plane) and its
    // direction (voxels per world unit along the normal)
    let o = vec3<f32>(top.origin_int.xyz) + frac_at(top, s);
    let d = top.dir.xyz;
    let hi = vec3<f32>(top.size.xyz) - 0.5;
    var t = vec2<f32>(-1e30, 1e30);
    t = slab(o.x, d.x, -0.5, hi.x, t);
    t = slab(o.y, d.y, -0.5, hi.y, t);
    t = slab(o.z, d.z, -0.5, hi.z, t);
    if (t.x > t.y) {
        return p.outside;
    }
    let most = max(abs(d.x), max(abs(d.y), abs(d.z)));
    let step = max(0.8 / max(most, 1e-30), (t.y - t.x) / MAX_STEPS);
    let n = u32(ceil((t.y - t.x) / step));
    var rgb = vec3<f32>(0.0);
    var shown = false;
    var wanted = false;
    for (var c = 0u; c < p.n_channels; c++) {
        let ch = channels[c];
        if (ch.shown == 0u) {
            continue;
        }
        wanted = true;
        var best = -3.4e38;
        var found = false;
        // the frame level's brick page, looked up when the ray enters a new brick
        var last_b = vec3<i32>(-1);
        var last_page = 0u;
        for (var i = 0u; i < n; i++) {
            let tt = t.x + (f32(i) + 0.5) * step;
            if (top.grid.w != 0u) {
                let q = o + tt * d;
                let vox = voxel_of(top, q);
                let b = vox / STEP;
                if (any(b != last_b)) {
                    last_b = b;
                    last_page = page_of(top, c, vox);
                }
                if (last_page != 0u) {
                    best = max(best, value_in(top, last_page - 1u, q));
                    found = true;
                    continue;
                }
            }
            // that brick is missing: a coarser level, else a finer one
            for (var k = 1u; k < p.n_levels; k++) {
                let l = levels[level_order(k)];
                if (l.grid.w == 0u) {
                    continue;
                }
                let q = vec3<f32>(l.origin_int.xyz) + frac_at(l, s) + tt * l.dir.xyz;
                let h = sample_at(l, c, q);
                if (h.found) {
                    best = max(best, h.v);
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            continue;
        }
        var g = clamp((best - ch.lo) / max(ch.hi - ch.lo, 1e-30), 0.0, 1.0);
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
    }
    if (!wanted) {
        return p.outside;
    }
    if (!shown) {
        return vec4<f32>(p.missing.rgb * p.opacity, p.opacity);
    }
    return vec4<f32>(min(rgb, vec3<f32>(1.0)) * p.opacity, p.opacity);
}
