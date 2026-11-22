
// Projections (appended to the brick shader's bindings and sampling): every screen pixel
// casts a ray along the view's normal through the volume (orthographic), the part inside
// the volume's box at the frame's level, in steps of at most 0.8 voxel along any axis (at
// most MAX_STEPS steps). Each sample comes from the frame's level, or else the nearest
// coarser level holding its brick, or else a finer one (nearest voxel or trilinear).
//
// p.mode bits 0-1: 0 maximum intensity (per channel the highest value along the ray,
// then contrast, gamma and colour, which keep the order); 1 volume (front-to-back
// compositing: a channel's opacity per voxel is its opacity curve at the mapped value
// (the colour map's alpha) times p.density, channels combine as layers of glass, the
// colour is the channels' colours weighted by their opacity); 2 isosurfaces (the first
// point where a channel reaches its level: its own, or the middle of its contrast
// window). Bit 2: shading, from the gradient of the most opaque channel (volume) or of
// the surface's channel (iso): a light from the viewer's upper left, both sides lit.
// Bit 3: no skipping (else rays jump over bricks that cannot add: no brighter than the
// highest value so far or below the window (maximum), below every window where the
// opacity curves start at zero (volume), below every level (iso); each brick's lowest
// and highest value come with it).

const MAX_STEPS: f32 = 2048.0;
const MAX_CHANNELS: u32 = 16u;

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

fn skipping() -> bool {
    return (p.mode & 8u) == 0u;
}

// The first sample index past the frame level's brick b on ray r (from sample i): where
// the ray leaves the cells whose samples fall into that brick.
fn after_brick(r: Ray, b: vec3<i32>, i: u32) -> u32 {
    var lo = vec3<f32>(b * STEP) - 0.5;
    if (p.linear != 0u) {
        lo = vec3<f32>(b * STEP);
    }
    let hi = lo + f32(STEP);
    var t_exit = 1e30;
    if (r.d.x > 1e-12) { t_exit = min(t_exit, (hi.x - r.o.x) / r.d.x); }
    if (r.d.x < -1e-12) { t_exit = min(t_exit, (lo.x - r.o.x) / r.d.x); }
    if (r.d.y > 1e-12) { t_exit = min(t_exit, (hi.y - r.o.y) / r.d.y); }
    if (r.d.y < -1e-12) { t_exit = min(t_exit, (lo.y - r.o.y) / r.d.y); }
    if (r.d.z > 1e-12) { t_exit = min(t_exit, (hi.z - r.o.z) / r.d.z); }
    if (r.d.z < -1e-12) { t_exit = min(t_exit, (lo.z - r.o.z) / r.d.z); }
    let next = ceil((t_exit - r.t0) / r.step - 0.5);
    return max(u32(clamp(next, 0.0, f32(r.n))), i + 1u);
}

// Channel c's raw value on the ray of screen point s at tt (world units along the view's
// normal) from the levels other than the frame's (coarser first).
fn value_elsewhere(c: u32, s: vec2<f32>, tt: f32) -> Hit {
    for (var k = 1u; k < p.n_levels; k++) {
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

fn lut_at(c: u32, g: f32) -> vec4<f32> {
    let lu = (g * (LUT_SIZE - 1.0) + 0.5) / LUT_SIZE;
    let lv = (f32(c) + 0.5) / f32(p.n_channels);
    return textureSampleLevel(luts, lut_sampler, vec2<f32>(lu, lv), 0.0);
}

// The channel's colour at mapped value g: its colour map's, or its colour.
fn tint(ch: Channel, c: u32, g: f32) -> vec3<f32> {
    if (ch.lut != 0u) {
        return lut_at(c, g).rgb;
    }
    return ch.color;
}

// The channel's isosurface level (a stored value).
fn iso_level(ch: Channel) -> f32 {
    if (ch.has_iso != 0u) {
        return ch.iso;
    }
    return ch.lo + 0.5 * (ch.hi - ch.lo);
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
    return to_camera(l, vec3<f32>(x1.v - x0.v, y1.v - y0.v, z1.v - z0.v));
}

// The normal from the gradient around voxel `vox` of level l, all from its brick slot
// `slot` (one-sided at the brick's and the volume's edges: no more page lookups).
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
    return to_camera(l, g);
}

// An index-space gradient as a unit normal in the camera's axes: the level's steps per
// screen pixel along right and down (over the scale, in dir.w) and per world unit along
// the normal.
fn to_camera(l: Level, g: vec3<f32>) -> vec3<f32> {
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

fn finish(rgb: vec3<f32>, wanted: bool, shown: bool) -> vec4<f32> {
    if (!wanted) {
        return p.outside;
    }
    if (!shown) {
        return vec4<f32>(p.missing.rgb * p.opacity, p.opacity);
    }
    return vec4<f32>(min(rgb, vec3<f32>(1.0)) * p.opacity, p.opacity);
}

fn fs_mip(s: vec2<f32>) -> vec4<f32> {
    let r = ray_for(s);
    if (!r.ok) {
        return p.outside;
    }
    let top = levels[p.level];
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
        // the frame level's brick and its page, looked up when the ray enters a new brick
        var last_b = vec3<i32>(-1);
        var last_page = 0u;
        var i = 0u;
        loop {
            if (i >= r.n) {
                break;
            }
            let tt = r.t0 + (f32(i) + 0.5) * r.step;
            if (top.grid.w != 0u) {
                let q = r.o + tt * r.d;
                let vox = voxel_of(top, q);
                let b = vox / STEP;
                if (any(b != last_b)) {
                    last_b = b;
                    last_page = page_of(top, c, vox);
                    if (skipping() && last_page != 0u) {
                        let most = ranges[last_page - 1u].y;
                        // nothing brighter here, or all of it below the window (black)
                        if (most <= best || most <= ch.lo) {
                            best = max(best, min(most, max(best, ch.lo)));
                            found = true;
                            i = after_brick(r, b, i);
                            continue;
                        }
                    }
                }
                if (last_page != 0u) {
                    best = max(best, value_in(top, last_page - 1u, q));
                    found = true;
                    i += 1u;
                    continue;
                }
            }
            // that brick is missing: a coarser level, else a finer one
            let h = value_elsewhere(c, s, tt);
            if (h.found) {
                best = max(best, h.v);
                found = true;
            }
            i += 1u;
        }
        if (!found) {
            continue;
        }
        let g = mapped(ch, best);
        if (ch.lut != 0u) {
            rgb += lut_at(c, g).rgb;
        } else {
            rgb += g * ch.color;
        }
        shown = true;
    }
    return finish(rgb, wanted, shown);
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
    // per channel: the frame level's page for the brick the ray is in
    var pg: array<u32, 16>;
    for (var c = 0u; c < MAX_CHANNELS; c++) {
        pg[c] = 0u;
    }
    var last_b = vec3<i32>(-1);
    var i = 0u;
    loop {
        if (i >= r.n) {
            break;
        }
        let tt = r.t0 + (f32(i) + 0.5) * r.step;
        let q = r.o + tt * r.d;
        let vox = voxel_of(top, q);
        let b = vox / STEP;
        if (top.grid.w != 0u && any(b != last_b)) {
            last_b = b;
            // a new brick: every channel's page, and whether all of it is clear
            var clear = skipping();
            for (var c = 0u; c < min(p.n_channels, MAX_CHANNELS); c++) {
                let ch = channels[c];
                if (ch.shown == 0u) {
                    continue;
                }
                pg[c] = page_of(top, c, vox);
                if (pg[c] == 0u || ranges[pg[c] - 1u].y > ch.lo || lut_at(c, 0.0).a > 0.0) {
                    clear = false;
                }
            }
            if (clear && p.n_channels <= MAX_CHANNELS) {
                wanted = true;
                seen = true;
                i = after_brick(r, b, i);
                continue;
            }
        }
        var rgb = vec3<f32>(0.0);
        var weight = 0.0;
        var keep = 1.0;
        var most = 0.0;
        var densest = 0u;
        var densest_slot = 0u;
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
            if (top.grid.w != 0u && c < MAX_CHANNELS && pg[c] != 0u) {
                h.v = value_in(top, pg[c] - 1u, q);
                h.found = true;
                slot1 = pg[c];
            }
            if (!h.found) {
                h = value_elsewhere(c, s, tt);
            }
            if (!h.found) {
                continue;
            }
            found = true;
            let g = mapped(ch, h.v);
            let opacity = lut_at(c, g).a;
            let alpha = 1.0 - pow(1.0 - clamp(opacity * p.density, 0.0, 0.999), r.vox);
            rgb += alpha * tint(ch, c, g);
            weight += alpha;
            keep *= 1.0 - alpha;
            if (alpha > most) {
                most = alpha;
                densest = c;
                densest_slot = slot1;
            }
        }
        if (found) {
            seen = true;
        }
        let alpha = 1.0 - keep;
        if (found && alpha > 0.0) {
            var colour = rgb / max(weight, 1e-30);
            if (shading && alpha > 0.004) {
                if (densest_slot != 0u) {
                    colour *= shade(normal_in(top, densest_slot - 1u, vox));
                } else {
                    colour *= shade(normal_at(top, densest, q));
                }
            }
            acc += (1.0 - a) * alpha * min(colour, vec3<f32>(1.0));
            a += (1.0 - a) * alpha;
            if (a > 0.995) {
                break;
            }
        }
        i += 1u;
    }
    if (!wanted) {
        return p.outside;
    }
    if (!seen) {
        return vec4<f32>(p.missing.rgb * p.opacity, p.opacity);
    }
    return vec4<f32>(acc * p.opacity, a * p.opacity);
}

fn fs_iso(s: vec2<f32>) -> vec4<f32> {
    let r = ray_for(s);
    if (!r.ok) {
        return p.outside;
    }
    let top = levels[p.level];
    let shading = (p.mode & 4u) != 0u;
    var prev: array<f32, 16>;
    var pg: array<u32, 16>;
    for (var c = 0u; c < MAX_CHANNELS; c++) {
        prev[c] = -3.4e38;
        pg[c] = 0u;
    }
    var wanted = false;
    var seen = false;
    let nc = min(p.n_channels, MAX_CHANNELS);
    var last_b = vec3<i32>(-1);
    var i = 0u;
    loop {
        if (i >= r.n) {
            break;
        }
        let tt = r.t0 + (f32(i) + 0.5) * r.step;
        let q = r.o + tt * r.d;
        let vox = voxel_of(top, q);
        let b = vox / STEP;
        if (top.grid.w != 0u && any(b != last_b)) {
            last_b = b;
            var below = skipping();
            for (var c = 0u; c < nc; c++) {
                let ch = channels[c];
                if (ch.shown == 0u) {
                    continue;
                }
                pg[c] = page_of(top, c, vox);
                if (pg[c] == 0u || ranges[pg[c] - 1u].y >= iso_level(ch)) {
                    below = false;
                }
            }
            if (below) {
                // to the brick's last sample: its true value is the next crossing's start
                wanted = true;
                seen = true;
                let next = after_brick(r, b, i);
                if (next > i + 1u) {
                    i = next - 1u;
                    continue;
                }
            }
        }
        for (var c = 0u; c < nc; c++) {
            let ch = channels[c];
            if (ch.shown == 0u) {
                continue;
            }
            wanted = true;
            var h: Hit;
            h.found = false;
            if (top.grid.w != 0u && pg[c] != 0u) {
                h.v = value_in(top, pg[c] - 1u, q);
                h.found = true;
            }
            if (!h.found) {
                h = value_elsewhere(c, s, tt);
            }
            if (!h.found) {
                continue;
            }
            seen = true;
            let level = iso_level(ch);
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
        i += 1u;
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
