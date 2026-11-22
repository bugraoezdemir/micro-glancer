
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

fn fs_mip(s: vec2<f32>) -> vec4<f32> {
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

// ------------------------------------------------------------------ shaded projections
//
// p.mode (bits 0-1): 0 maximum intensity, 1 volume (front-to-back compositing: a sample's
// opacity grows with its brightest channel, `p.density` the opacity per voxel at the top
// of the contrast window; its colour the channels' colours weighted by brightness), 2
// isosurfaces (the first point along the ray where a channel reaches the middle of its
// contrast window). Bit 2: shading, from the gradient of the brightest channel (volume)
// or of the surface's channel (iso): a light from the viewer's upper left, both sides lit.

struct Ray {
    ok: bool,
    o: vec3<f32>,
    d: vec3<f32>,
    t0: f32,
    t1: f32,
    step: f32,
    n: u32,
    // voxels along the most changing axis per step (opacity correction)
    vox: f32,
}

fn ray_for(s: vec2<f32>) -> Ray {
    var r: Ray;
    let top = levels[p.level];
    r.o = vec3<f32>(top.origin_int.xyz) + frac_at(top, s);
    r.d = top.dir.xyz;
    let hi = vec3<f32>(top.size.xyz) - 0.5;
    var t = vec2<f32>(-1e30, 1e30);
    t = slab(r.o.x, r.d.x, -0.5, hi.x, t);
    t = slab(r.o.y, r.d.y, -0.5, hi.y, t);
    t = slab(r.o.z, r.d.z, -0.5, hi.z, t);
    r.ok = t.x <= t.y;
    r.t0 = t.x;
    r.t1 = t.y;
    let most = max(abs(r.d.x), max(abs(r.d.y), abs(r.d.z)));
    r.step = max(0.8 / max(most, 1e-30), (t.y - t.x) / MAX_STEPS);
    r.n = u32(ceil(max(t.y - t.x, 0.0) / r.step));
    r.vox = r.step * most;
    return r;
}

// Channel c's raw value on the ray of screen point s at tt (world units along the view's
// normal): the frame's level, else a coarser one, else a finer one.
fn value_on_ray(c: u32, s: vec2<f32>, tt: f32) -> Hit {
    for (var k = 0u; k < p.n_levels; k++) {
        let l = levels[level_order(k)];
        if (l.grid.w == 0u) {
            continue;
        }
        let q = vec3<f32>(l.origin_int.xyz) + frac_at(l, s) + tt * l.dir.xyz;
        let h = sample_at(l, c, q);
        if (h.found) {
            return h;
        }
    }
    var none: Hit;
    none.found = false;
    none.v = 0.0;
    return none;
}

// A raw value through the channel's contrast and gamma (0 .. 1).
fn mapped(ch: Channel, v: f32) -> f32 {
    var g = clamp((v - ch.lo) / max(ch.hi - ch.lo, 1e-30), 0.0, 1.0);
    if (ch.gamma != 1.0) {
        g = pow(g, ch.gamma);
    }
    return g;
}

// The channel's colour at mapped value g: its colour map's, or its colour.
fn tint(ch: Channel, c: u32, g: f32) -> vec3<f32> {
    if (ch.lut != 0u) {
        let lu = (g * (LUT_SIZE - 1.0) + 0.5) / LUT_SIZE;
        let lv = (f32(c) + 0.5) / f32(p.n_channels);
        return textureSampleLevel(luts, lut_sampler, vec2<f32>(lu, lv), 0.0).rgb;
    }
    return ch.color;
}

// The surface normal at level l's point q from channel c's gradient (central
// differences), seen by the camera (x right, y down, z into the screen); zero when the
// gradient is flat or its voxels are not cached.
fn normal_at(l: Level, c: u32, q: vec3<f32>) -> vec3<f32> {
    let ex = vec3<f32>(1.0, 0.0, 0.0);
    let ey = vec3<f32>(0.0, 1.0, 0.0);
    let ez = vec3<f32>(0.0, 0.0, 1.0);
    let x1 = sample_at(l, c, q + ex);
    let x0 = sample_at(l, c, q - ex);
    let y1 = sample_at(l, c, q + ey);
    let y0 = sample_at(l, c, q - ey);
    let z1 = sample_at(l, c, q + ez);
    let z0 = sample_at(l, c, q - ez);
    if (!(x1.found && x0.found && y1.found && y0.found && z1.found && z0.found)) {
        return vec3<f32>(0.0);
    }
    // index-space gradient to the camera's axes: the level's steps per screen pixel along
    // right and down (over the scale) and per world unit along the normal
    let g = vec3<f32>(x1.v - x0.v, y1.v - y0.v, z1.v - z0.v);
    let k = max(l.dir.w, 1e-30);
    let n = vec3<f32>(dot(g, l.ux.xyz) / k, dot(g, l.uy.xyz) / k, dot(g, l.dir.xyz));
    let len = length(n);
    if (len < 1e-20) {
        return vec3<f32>(0.0);
    }
    return n / len;
}

// The normal from the gradient around voxel `vox` of level l, all from its brick slot
// `slot` (one-sided at the brick's and the volume's edges: no more page lookups), seen
// by the camera as in normal_at.
fn normal_in(l: Level, slot: u32, vox: vec3<i32>) -> vec3<f32> {
    let o = vox - (vox / STEP) * STEP;
    let last = l.size.xyz - 1;
    let up = vec3<i32>(select(vec3<i32>(0), vec3<i32>(1), (o < vec3<i32>(STEP)) & (vox < last)));
    let dn = vec3<i32>(select(vec3<i32>(0), vec3<i32>(1), o > vec3<i32>(0)));
    let span = vec3<f32>(max(up + dn, vec3<i32>(1)));
    let g = vec3<f32>(
        load(slot, o + vec3<i32>(up.x, 0, 0)) - load(slot, o - vec3<i32>(dn.x, 0, 0)),
        load(slot, o + vec3<i32>(0, up.y, 0)) - load(slot, o - vec3<i32>(0, dn.y, 0)),
        load(slot, o + vec3<i32>(0, 0, up.z)) - load(slot, o - vec3<i32>(0, 0, dn.z))
    ) / span;
    let k = max(l.dir.w, 1e-30);
    let n = vec3<f32>(dot(g, l.ux.xyz) / k, dot(g, l.uy.xyz) / k, dot(g, l.dir.xyz));
    let len = length(n);
    if (len < 1e-20) {
        return vec3<f32>(0.0);
    }
    return n / len;
}

// Light at a surface with normal n (both sides lit; a flat gradient is lit evenly).
fn shade(n: vec3<f32>) -> f32 {
    if (all(n == vec3<f32>(0.0))) {
        return 1.0;
    }
    let light = normalize(vec3<f32>(-0.35, -0.45, -0.82));
    let diffuse = abs(dot(n, light));
    let half_way = normalize(light + vec3<f32>(0.0, 0.0, -1.0));
    let specular = pow(abs(dot(n, half_way)), 24.0) * 0.25;
    return 0.3 + 0.7 * diffuse + specular;
}

fn fs_volume(s: vec2<f32>) -> vec4<f32> {
    let r = ray_for(s);
    if (!r.ok) {
        return p.outside;
    }
    let top = levels[p.level];
    let shading = (p.mode & 4u) != 0u;
    var acc = vec3<f32>(0.0);
    var a = 0.0;
    var wanted = false;
    var seen = false;
    // per channel: the frame level's brick and its page, looked up when the ray enters it
    var bk: array<vec3<i32>, 16>;
    var pg: array<u32, 16>;
    for (var c = 0u; c < ISO_CHANNELS; c++) {
        bk[c] = vec3<i32>(-1);
        pg[c] = 0u;
    }
    for (var i = 0u; i < r.n; i++) {
        let tt = r.t0 + (f32(i) + 0.5) * r.step;
        let q = r.o + tt * r.d;
        let vox = voxel_of(top, q);
        let b = vox / STEP;
        var rgb = vec3<f32>(0.0);
        var weight = 0.0;
        var gmax = 0.0;
        var brightest = 0u;
        var brightest_slot = 0u;
        var found = false;
        for (var c = 0u; c < p.n_channels; c++) {
            let ch = channels[c];
            if (ch.shown == 0u) {
                continue;
            }
            wanted = true;
            var h: Hit;
            h.found = false;
            var slot1 = 0u;
            if (top.grid.w != 0u && c < ISO_CHANNELS) {
                if (any(b != bk[c])) {
                    bk[c] = b;
                    pg[c] = page_of(top, c, vox);
                }
                if (pg[c] != 0u) {
                    h.v = value_in(top, pg[c] - 1u, q);
                    h.found = true;
                    slot1 = pg[c];
                }
            }
            if (!h.found) {
                h = value_on_ray(c, s, tt);
            }
            if (!h.found) {
                continue;
            }
            found = true;
            let g = mapped(ch, h.v);
            rgb += g * tint(ch, c, g);
            weight += g;
            if (g > gmax) {
                gmax = g;
                brightest = c;
                brightest_slot = slot1;
            }
        }
        if (!found) {
            continue;
        }
        seen = true;
        if (gmax <= 0.0) {
            continue;
        }
        let alpha = 1.0 - pow(1.0 - clamp(gmax * p.density, 0.0, 0.999), r.vox);
        var colour = rgb / max(weight, 1e-30);
        if (shading && alpha > 0.004) {
            if (brightest_slot != 0u) {
                colour *= shade(normal_in(top, brightest_slot - 1u, vox));
            } else {
                colour *= shade(normal_at(top, brightest, q));
            }
        }
        acc += (1.0 - a) * alpha * min(colour, vec3<f32>(1.0));
        a += (1.0 - a) * alpha;
        if (a > 0.995) {
            break;
        }
    }
    if (!wanted) {
        return p.outside;
    }
    if (!seen) {
        return vec4<f32>(p.missing.rgb * p.opacity, p.opacity);
    }
    return vec4<f32>(acc * p.opacity, a * p.opacity);
}

const ISO_CHANNELS: u32 = 16u;

fn fs_iso(s: vec2<f32>) -> vec4<f32> {
    let r = ray_for(s);
    if (!r.ok) {
        return p.outside;
    }
    let top = levels[p.level];
    let shading = (p.mode & 4u) != 0u;
    var prev: array<f32, 16>;
    for (var c = 0u; c < ISO_CHANNELS; c++) {
        prev[c] = -3.4e38;
    }
    var wanted = false;
    var seen = false;
    let nc = min(p.n_channels, ISO_CHANNELS);
    var bk: array<vec3<i32>, 16>;
    var pg: array<u32, 16>;
    for (var c = 0u; c < ISO_CHANNELS; c++) {
        bk[c] = vec3<i32>(-1);
        pg[c] = 0u;
    }
    for (var i = 0u; i < r.n; i++) {
        let tt = r.t0 + (f32(i) + 0.5) * r.step;
        let q = r.o + tt * r.d;
        let vox = voxel_of(top, q);
        let b = vox / STEP;
        for (var c = 0u; c < nc; c++) {
            let ch = channels[c];
            if (ch.shown == 0u) {
                continue;
            }
            wanted = true;
            var h: Hit;
            h.found = false;
            if (top.grid.w != 0u) {
                if (any(b != bk[c])) {
                    bk[c] = b;
                    pg[c] = page_of(top, c, vox);
                }
                if (pg[c] != 0u) {
                    h.v = value_in(top, pg[c] - 1u, q);
                    h.found = true;
                }
            }
            if (!h.found) {
                h = value_on_ray(c, s, tt);
            }
            if (!h.found) {
                continue;
            }
            seen = true;
            let level = ch.lo + 0.5 * (ch.hi - ch.lo);
            if (h.v >= level) {
                // where between this sample and the one before the value crossed it
                var t_hit = tt;
                if (prev[c] > -3.0e38 && h.v > prev[c]) {
                    t_hit = tt - r.step + (level - prev[c]) / (h.v - prev[c]) * r.step;
                }
                var colour = tint(ch, c, 1.0);
                if (shading) {
                    let qh = r.o + t_hit * r.d;
                    let vh = voxel_of(top, qh);
                    let ph = page_of(top, c, vh);
                    if (ph != 0u) {
                        colour *= shade(normal_in(top, ph - 1u, vh));
                    } else {
                        colour *= shade(normal_at(top, c, qh));
                    }
                }
                return vec4<f32>(min(colour, vec3<f32>(1.0)) * p.opacity, p.opacity);
            }
            prev[c] = h.v;
        }
    }
    if (!wanted) {
        return p.outside;
    }
    if (!seen) {
        return vec4<f32>(p.missing.rgb * p.opacity, p.opacity);
    }
    return p.outside;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let mode = p.mode & 3u;
    if (mode == 1u) {
        return fs_volume(in.pos.xy);
    }
    if (mode == 2u) {
        return fs_iso(in.pos.xy);
    }
    return fs_mip(in.pos.xy);
}
