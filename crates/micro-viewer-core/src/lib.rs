//! micro-viewer's core (Workplan WP14): OME-Zarr images drawn by the GPU from a tile
//! cache, hosted by any toolkit that can hand over a native window, or read back
//! off-screen. No Qt and no Python here: the bindings and the widget live outside.
//!
//! 2D viewer (tasks V7, V9): layers, each an image placed in a shared world by its own
//! levels and a transform of its own (moved and turned by hand, saved to the image);
//! every level of an image, the finest worth drawing chosen per frame from the zoom,
//! coarser (or finer) cached tiles standing in while tiles load; tiles read in the
//! background, centre first, a ring around and the next and previous plane when the
//! cache has room, and from coarser levels while the view moves when reading is slow;
//! channels with colour or colour map, contrast and gamma, added up; layers added up;
//! the camera turned; the plane chosen on the axes before y and x; picking; the view as
//! JSON.

mod blocks;
mod gpu;
mod loader;
mod luts;
mod source;
mod tiles;
mod view;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub use gpu::GpuOptions;
use gpu::{ChannelGpu, FrameParams, Gpu, LevelGpu, LUT_SIZE, MISSING};
use loader::{Loaded, Loader, Request};
pub use luts::NAMES as COLOUR_MAPS;
pub use raw_window_handle;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
pub use source::{Affine2, ChannelInfo, Level, Source, Texel};
use tiles::{Cache, TileKey, TILE};
pub use view::ViewState;

/// The coordinate system a layer's transform is saved into (NGFF 0.6): an edge from the
/// image's own coordinate system to this one. Opening an image that has such an edge
/// starts its layer at that transform.
pub const ALIGNED: &str = "aligned";

/// The view counts as moving while it changed in two different frames this recently
/// (a drag, a turn of the wheel; not one jump, an open or a step in z); while it moves,
/// slow reads are taken from coarser levels.
const MOVING: Duration = Duration::from_millis(200);
/// ... and at most this many frames apart.
const MOVING_FRAMES: u64 = 4;
/// The reading time (seconds, all reading threads together) a frame's missing tiles may
/// take while the view moves before coarser levels are read instead.
const MOVE_BUDGET: f64 = 0.15;

/// Something the viewer cannot do as asked; the message says why.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewerError(pub String);

impl std::fmt::Display for ViewerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ViewerError {}

fn verr(msg: impl Into<String>) -> ViewerError {
    ViewerError(msg.into())
}

/// What a frame showed (all visible layers together; `level` is the current layer's).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Tiles of the frame's levels in view (all active channels).
    pub visible: u32,
    /// Tiles uploaded for this frame.
    pub loaded: u32,
    /// Tiles requested and not read yet.
    pub pending: u32,
    /// The current layer's level for this frame (0 the finest).
    pub level: u32,
    /// Tiles of the frame's levels in view and not cached yet (coarser or finer levels
    /// stand in for them): 0 means the view is sharp.
    pub missing: u32,
    /// Whether reading at the frame's level waits until the view stops moving (slow
    /// reads: coarser levels are read meanwhile).
    pub deferred: bool,
}

/// The camera: the world point (x, y) at the target's centre, the world units per
/// screen pixel and the turn in degrees (the image appears turned clockwise by it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub center: [f64; 2],
    pub scale: f64,
    pub rotation: f64,
}

impl Camera {
    /// Screen pixel (x, y) of a `w` x `h` target to world (x, y).
    pub fn screen_to_world(&self, w: u32, h: u32) -> Affine2 {
        let (s, c) = self.rotation.to_radians().sin_cos();
        let k = self.scale;
        // world = center + k * turn(-rotation) * (p - size / 2)
        let (a, b, d, e) = (k * c, k * s, -k * s, k * c);
        let (hx, hy) = (w as f64 / 2.0, h as f64 / 2.0);
        Affine2 {
            m: [
                [a, b, self.center[0] - a * hx - b * hy],
                [d, e, self.center[1] - d * hx - e * hy],
            ],
        }
    }
}

/// How one channel is shown.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelDisplay {
    pub label: String,
    pub color: [f32; 3],
    /// A colour map's name ([`COLOUR_MAPS`]) instead of the colour, or None.
    pub lut: Option<String>,
    pub lo: f64,
    pub hi: f64,
    pub gamma: f64,
    pub visible: bool,
}

/// One channel's value at a picked point.
#[derive(Debug, Clone, PartialEq)]
pub struct PickedValue {
    pub channel: usize,
    /// The value as stored, or None when no cached tile holds the point.
    pub value: Option<f64>,
    /// The level the value came from.
    pub level: Option<usize>,
}

/// What one layer has under a screen point.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerPick {
    pub layer: usize,
    /// The level-0 pixel (x, y) whose centre is nearest, if inside the image.
    pub pixel: Option<[u64; 2]>,
    pub values: Vec<PickedValue>,
}

/// What lies under a screen point: the world point and every layer's pixel and values.
#[derive(Debug, Clone, PartialEq)]
pub struct Pick {
    pub world: [f64; 2],
    pub layers: Vec<LayerPick>,
}

/// One image in the view: its source, tile cache and page table, its channels, plane
/// and transform.
pub struct Layer {
    id: u16,
    path: PathBuf,
    source: Source,
    cache: Cache,
    /// The page table as on the GPU: per channel, per level, per tile: slot + 1 or 0.
    pages: Vec<u32>,
    level_base: Vec<u32>,
    grid: Vec<(u32, u32)>,
    stride: u32,
    /// Per level: the current plane's indices and id.
    planes: Vec<(Vec<u64>, u64)>,
    /// Page entries changed since the last upload (first, last).
    dirty: Option<(usize, usize)>,
    pages_all: bool,
    failed: HashSet<TileKey>,
    error: Option<String>,
    channels: Vec<ChannelDisplay>,
    luts_dirty: bool,
    /// Level-0 indices on the image's plane axes.
    position: Vec<f64>,
    /// World to world: where the layer sits on top of its levels' placement.
    transform: Affine2,
    inverse: Affine2,
    /// Whether the layer is drawn.
    pub visible: bool,
    /// 0..1: how strongly the layer adds to the others.
    pub opacity: f32,
    /// Seconds per tile read, smoothed.
    read_s: f64,
}

impl Layer {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn source(&self) -> &Source {
        &self.source
    }

    pub fn channels(&self) -> &[ChannelDisplay] {
        &self.channels
    }

    /// The layer's transform (world to world, on top of its levels' placement).
    pub fn transform(&self) -> Affine2 {
        self.transform
    }

    /// The first tile read error (or a note from opening) since the layer was added.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// The plane's position: per plane axis, its name, the level-0 index and the
    /// number of indices.
    pub fn position(&self) -> Vec<(String, f64, u64)> {
        self.source
            .plane_axes
            .iter()
            .zip(&self.position)
            .map(|(&a, &p)| {
                (
                    self.source.axes[a].clone(),
                    p,
                    self.source.levels[0].shape[a],
                )
            })
            .collect()
    }

    /// The world points of the finest level's outer corners, the transform applied.
    fn corners(&self) -> [(f64, f64); 4] {
        let lv = &self.source.levels[0];
        let (w, h) = (lv.width as f64 - 0.5, lv.height as f64 - 0.5);
        [(-0.5, -0.5), (w, -0.5), (-0.5, h), (w, h)].map(|(ix, iy)| {
            let (x, y) = lv.to_world.apply(ix, iy);
            self.transform.apply(x, y)
        })
    }

    /// The world box (x0, y0, x1, y1) the layer covers.
    pub fn bounds(&self) -> [f64; 4] {
        let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for (x, y) in self.corners() {
            b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
        }
        b
    }

    /// The world point at the image's centre (the transform applied).
    pub fn centre(&self) -> [f64; 2] {
        let lv = &self.source.levels[0];
        let (x, y) = lv.to_world.apply(
            (lv.width as f64 - 1.0) / 2.0,
            (lv.height as f64 - 1.0) / 2.0,
        );
        let (x, y) = self.transform.apply(x, y);
        [x, y]
    }

    /// World units per pixel of level `l`, the transform's scaling included.
    fn pixel(&self, l: usize) -> f64 {
        self.source.levels[l].pixel * self.transform.det().abs().sqrt()
    }

    /// The level a frame at `scale` world units per screen pixel draws: the coarsest
    /// whose pixels are no larger than a screen pixel (the finest when zoomed in further).
    pub fn level_for(&self, scale: f64) -> usize {
        let s = scale * (1.0 + 1e-9);
        (0..self.source.levels.len())
            .filter(|&l| self.pixel(l) <= s)
            .max_by(|&a, &b| self.pixel(a).total_cmp(&self.pixel(b)))
            .unwrap_or(0)
    }

    /// The transform as an NGFF affine over the image's axes: n rows of n + 1 (the last
    /// column the translation), identity outside y and x.
    pub fn transform_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.source.axes.len();
        let (y, x) = (n - 2, n - 1);
        let mut m: Vec<Vec<f64>> = (0..n)
            .map(|r| (0..=n).map(|c| (r == c) as u8 as f64).collect())
            .collect();
        let t = &self.transform.m;
        m[x][x] = t[0][0];
        m[x][y] = t[0][1];
        m[x][n] = t[0][2];
        m[y][x] = t[1][0];
        m[y][y] = t[1][1];
        m[y][n] = t[1][2];
        m
    }

    fn entry(&self, k: &TileKey) -> usize {
        k.channel as usize * self.stride as usize
            + self.level_base[k.level as usize] as usize
            + (k.y * self.grid[k.level as usize].0 + k.x) as usize
    }

    fn current(&self, k: &TileKey) -> bool {
        self.planes[k.level as usize].1 == k.plane
    }

    fn set_page(&mut self, k: &TileKey, v: u32) {
        let e = self.entry(k);
        self.pages[e] = v;
        self.dirty = Some(match self.dirty {
            Some((a, b)) => (a.min(e), b.max(e)),
            None => (e, e),
        });
    }

    /// Every page entry again from the cache (the plane changed).
    fn rebuild_pages(&mut self) {
        self.pages.iter_mut().for_each(|p| *p = 0);
        let present: Vec<(TileKey, u32)> = self
            .cache
            .iter()
            .filter(|(k, _)| self.current(k))
            .map(|(k, s)| (*k, *s))
            .collect();
        for (k, s) in present {
            let e = self.entry(&k);
            self.pages[e] = s + 1;
        }
        self.pages_all = true;
    }

    fn key(&self, c: usize, l: usize, (x, y): (u32, u32)) -> TileKey {
        TileKey {
            layer: self.id,
            plane: self.planes[l].1,
            channel: c as u16,
            level: l as u8,
            x,
            y,
        }
    }

    fn missing(&self, k: &TileKey) -> bool {
        self.cache.peek(k).is_none() && !self.failed.contains(k)
    }

    fn request(&self, k: TileKey, plane: &[u64]) -> Request {
        let lv = &self.source.levels[k.level as usize];
        let (ty, tx) = (k.y as u64 * TILE as u64, k.x as u64 * TILE as u64);
        let h = (lv.height - ty).min(TILE as u64);
        let w = (lv.width - tx).min(TILE as u64);
        let (start, stop) = self.source.region(plane, k.channel as usize, ty, tx, h, w);
        Request {
            key: k,
            gen: 0,
            array: self.source.array(k.level as usize),
            start,
            stop,
        }
    }

    /// The channels' colour maps, one row of LUT_SIZE RGBA entries each.
    fn lut_rows(&self) -> Vec<[u8; 4]> {
        let mut rows = Vec::with_capacity(self.channels.len() * LUT_SIZE as usize);
        for d in &self.channels {
            let table = d
                .lut
                .as_deref()
                .and_then(|n| luts::NAMES.iter().position(|&m| m == n));
            for i in 0..LUT_SIZE as usize {
                rows.push(match table {
                    Some(t) => {
                        let [r, g, b] = luts::TABLES[t][i];
                        [r, g, b, 255]
                    }
                    None => {
                        let f = i as f32 / (LUT_SIZE - 1) as f32;
                        let c = |v: f32| (v.clamp(0.0, 1.0) * f * 255.0).round() as u8;
                        [c(d.color[0]), c(d.color[1]), c(d.color[2]), 255]
                    }
                });
            }
        }
        rows
    }
}

/// The tiles of a level a target of `tw` x `th` pixels shows, given the screen-to-index
/// map `to_index`: (x0, y0, x1, y1), the last exclusive, or None when the view misses
/// the level.
fn visible_tiles(
    lv: &source::Level,
    to_index: &Affine2,
    tw: u32,
    th: u32,
    margin: i64,
) -> Option<(u32, u32, u32, u32)> {
    let (x0, y0, x1, y1) = index_box(to_index, tw, th);
    let (w, h) = (lv.width as f64, lv.height as f64);
    if x1 + 0.5 < 0.0 || y1 + 0.5 < 0.0 || x0 + 0.5 >= w || y0 + 0.5 >= h {
        return None;
    }
    let t = TILE as i64;
    let gw = lv.width.div_ceil(TILE as u64) as i64;
    let gh = lv.height.div_ceil(TILE as u64) as i64;
    let tx0 = ((x0 + 0.5).floor().max(0.0) as i64 / t - margin).max(0);
    let ty0 = ((y0 + 0.5).floor().max(0.0) as i64 / t - margin).max(0);
    let tx1 = (((x1 + 0.5).floor().min(w - 1.0) as i64) / t + 1 + margin).min(gw);
    let ty1 = (((y1 + 0.5).floor().min(h - 1.0) as i64) / t + 1 + margin).min(gh);
    Some((tx0 as u32, ty0 as u32, tx1 as u32, ty1 as u32))
}

/// The box (x0, y0, x1, y1) of index space the target's corners map to.
fn index_box(to_index: &Affine2, tw: u32, th: u32) -> (f64, f64, f64, f64) {
    let corners = [
        (0.0, 0.0),
        (tw as f64, 0.0),
        (0.0, th as f64),
        (tw as f64, th as f64),
    ];
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for (sx, sy) in corners {
        let (ix, iy) = to_index.apply(sx, sy);
        x0 = x0.min(ix);
        y0 = y0.min(iy);
        x1 = x1.max(ix);
        y1 = y1.max(iy);
    }
    (x0, y0, x1, y1)
}

fn count((x0, y0, x1, y1): (u32, u32, u32, u32)) -> usize {
    ((x1 - x0) * (y1 - y0)) as usize
}

/// The tiles of a range, centre first.
fn centre_first((x0, y0, x1, y1): (u32, u32, u32, u32)) -> Vec<(u32, u32)> {
    let mut v: Vec<(u32, u32)> = (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (x, y)))
        .collect();
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    v.sort_by(|a, b| {
        let d = |p: &(u32, u32)| (p.0 as f64 + 0.5 - cx).powi(2) + (p.1 as f64 + 0.5 - cy).powi(2);
        d(a).total_cmp(&d(b))
    });
    v
}

/// What a frame asks the readers for, in priority classes (all layers' coarse tiles
/// first, then the frame's levels, then rings, then the planes next to this one).
#[derive(Default)]
struct Wants {
    coarse: Vec<Request>,
    here: Vec<Request>,
    ring: Vec<Request>,
    planes: Vec<Request>,
}

/// One frame's setting, shared by the layers.
struct FrameCtx {
    frame: u64,
    tw: u32,
    th: u32,
    screen: Affine2,
    scale: f64,
    moving: bool,
    threads: f64,
}

/// One layer's part of a frame: keep the cached tiles the view uses, place the tiles
/// read since the last frame, update the page table, and ask for what is still missing.
/// Returns the layer's parameters when it is drawn.
fn prepare_layer(
    o: &mut Layer,
    gpu: &mut Gpu,
    loaded: Vec<Loaded>,
    cx: &FrameCtx,
    wants: &mut Wants,
    stats: &mut FrameStats,
) -> (Option<FrameParams>, usize) {
    let frame = cx.frame;
    let nl = o.source.levels.len();
    let level = o.level_for(cx.scale);
    // the plane on every level
    let mut moved = false;
    for l in 0..nl {
        let p = o.source.plane(l, &o.position);
        if p.1 != o.planes[l].1 {
            o.planes[l] = p;
            moved = true;
        }
    }
    if moved {
        o.rebuild_pages();
    }
    // per level: screen pixel to level index
    let to_index: Vec<Affine2> = o
        .source
        .levels
        .iter()
        .map(|lv| lv.to_index.after(&o.inverse).after(&cx.screen))
        .collect();
    let active: Vec<usize> = (0..o.channels.len())
        .filter(|&c| o.channels[c].visible)
        .collect();
    let drawn = o.visible && !active.is_empty();
    // keep what the view uses: the frame's level and the coarser ones
    let mut views = vec![None; nl];
    let mut order = Vec::new();
    let mut missing_here = Vec::new();
    if drawn {
        for (l, v) in views.iter_mut().enumerate().skip(level) {
            *v = visible_tiles(&o.source.levels[l], &to_index[l], cx.tw, cx.th, 0);
        }
        order = views[level].map(centre_first).unwrap_or_default();
        for &t in &order {
            for &c in &active {
                stats.visible += 1;
                let k = o.key(c, level, t);
                if o.cache.touch(&k, frame).is_none() {
                    missing_here.push(k);
                }
            }
        }
        for (l, view) in views.iter().enumerate().skip(level + 1) {
            if let Some(r) = *view {
                for t in centre_first(r) {
                    for &c in &active {
                        let k = o.key(c, l, t);
                        o.cache.touch(&k, frame);
                    }
                }
            }
        }
    }
    // place the tiles read since the last frame
    for t in loaded {
        o.read_s = if o.read_s == 0.0 {
            t.took
        } else {
            0.8 * o.read_s + 0.2 * t.took
        };
        match t.data {
            Err(e) => {
                o.error.get_or_insert(e);
                o.failed.insert(t.key);
            }
            Ok(data) => {
                let Some((slot, old)) = o.cache.place(t.key, frame) else {
                    continue; // every slot is in use by this frame
                };
                if let Some(old) = old {
                    if o.current(&old) {
                        o.set_page(&old, 0);
                    }
                }
                gpu.upload_tile(o.id, slot, &data, t.w, t.h);
                if o.current(&t.key) {
                    o.set_page(&t.key, slot + 1);
                }
                stats.loaded += 1;
            }
        }
    }
    if drawn {
        // slow reads while the view moves: read the finest level whose tiles in view
        // fit the budget, and the frame's level once the view stops
        let threads = cx.threads.max(1.0);
        let cost = |n: usize| n as f64 * o.read_s / threads;
        let todo = missing_here.iter().filter(|k| o.missing(k)).count();
        if cx.moving && cost(todo) > MOVE_BUDGET {
            stats.deferred = true;
            let fetch = (level + 1..nl)
                .find(|&l| cost(views[l].map_or(0, count) * active.len()) <= MOVE_BUDGET)
                .unwrap_or(nl - 1);
            if let Some(r) = views[fetch] {
                for t in centre_first(r) {
                    for &c in &active {
                        let k = o.key(c, fetch, t);
                        if o.missing(&k) {
                            wants.here.push(o.request(k, &o.planes[fetch].0));
                        }
                    }
                }
            }
        } else {
            // a coarse level first (few tiles cover the view), then the frame's level
            // centre first, then a ring around and the planes next to this one while the
            // cache has room
            let coarse = (level + 2).min(nl - 1);
            if coarse != level {
                if let Some(r) = views[coarse] {
                    for t in centre_first(r) {
                        for &c in &active {
                            let k = o.key(c, coarse, t);
                            if o.missing(&k) {
                                wants.coarse.push(o.request(k, &o.planes[coarse].0));
                            }
                        }
                    }
                }
            }
            for k in &missing_here {
                if o.missing(k) {
                    wants.here.push(o.request(*k, &o.planes[level].0));
                }
            }
            let room = o.cache.capacity() as usize;
            let here = order.len() * active.len();
            if here * 2 < room {
                if let (Some(inner), Some(outer)) = (
                    views[level],
                    visible_tiles(&o.source.levels[level], &to_index[level], cx.tw, cx.th, 1),
                ) {
                    for t in centre_first(outer) {
                        let inside =
                            t.0 >= inner.0 && t.0 < inner.2 && t.1 >= inner.1 && t.1 < inner.3;
                        if inside {
                            continue;
                        }
                        for &c in &active {
                            let k = o.key(c, level, t);
                            if o.missing(&k) {
                                wants.ring.push(o.request(k, &o.planes[level].0));
                            } else {
                                o.cache.touch(&k, frame);
                            }
                        }
                    }
                }
            }
            if here * 4 < room {
                // the next and previous plane on each plane axis (z, t, ...)
                for k_axis in 0..o.source.plane_axes.len() {
                    for step in [1.0, -1.0] {
                        let mut pos = o.position.clone();
                        pos[k_axis] += step;
                        let a = o.source.plane_axes[k_axis];
                        if pos[k_axis] < 0.0
                            || pos[k_axis] > (o.source.levels[0].shape[a] - 1) as f64
                        {
                            continue;
                        }
                        let (idx, id) = o.source.plane(level, &pos);
                        if id == o.planes[level].1 {
                            continue;
                        }
                        for &t in &order {
                            for &c in &active {
                                let k = TileKey {
                                    layer: o.id,
                                    plane: id,
                                    channel: c as u16,
                                    level: level as u8,
                                    x: t.0,
                                    y: t.1,
                                };
                                if o.missing(&k) {
                                    wants.planes.push(o.request(k, &idx));
                                } else {
                                    o.cache.touch(&k, frame);
                                }
                            }
                        }
                    }
                }
            }
        }
        stats.missing += missing_here
            .iter()
            .filter(|k| o.cache.peek(k).is_none())
            .count() as u32;
    }
    // the page table, levels, channels and colour maps for the shader
    if o.pages_all {
        gpu.write_pages(o.id, o.pages.len(), 0, &o.pages);
        o.pages_all = false;
        o.dirty = None;
    } else if let Some((a, b)) = o.dirty.take() {
        if gpu.write_pages(o.id, o.pages.len(), a, &o.pages[a..=b]) {
            gpu.write_pages(o.id, o.pages.len(), 0, &o.pages);
        }
    }
    if !drawn {
        return (None, level);
    }
    let levels: Vec<LevelGpu> = o
        .source
        .levels
        .iter()
        .enumerate()
        .map(|(l, lv)| {
            let m = &to_index[l];
            let (ox, oy) = m.apply(0.0, 0.0);
            let (ox, oy) = (ox + 0.5, oy + 0.5); // nearest centre: floor(i + 0.5)
            let (fx, fy) = (ox.floor(), oy.floor());
            let ux = m.linear(1.0, 0.0);
            let uy = m.linear(0.0, 1.0);
            LevelGpu {
                origin_int: [fx as i32, fy as i32],
                origin_frac: [(ox - fx) as f32, (oy - fy) as f32],
                ux: [ux.0 as f32, ux.1 as f32],
                uy: [uy.0 as f32, uy.1 as f32],
                size: [lv.width as i32, lv.height as i32],
                grid_w: o.grid[l].0,
                page_base: o.level_base[l],
            }
        })
        .collect();
    let channels: Vec<ChannelGpu> = o
        .channels
        .iter()
        .map(|d| ChannelGpu {
            color: d.color,
            gamma: d.gamma as f32,
            lo: d.lo as f32,
            hi: d.hi as f32,
            active: d.visible as u32,
            lut: d.lut.is_some() as u32,
        })
        .collect();
    gpu.write_levels(o.id, &levels);
    gpu.write_channels(o.id, &channels);
    if o.luts_dirty {
        gpu.write_luts(o.id, &o.lut_rows());
        o.luts_dirty = false;
    }
    let params = FrameParams {
        n_levels: nl as u32,
        n_channels: channels.len() as u32,
        level: level as u32,
        channel_stride: o.stride,
        missing: MISSING,
        outside: [0.0; 4],
        opacity: o.opacity.clamp(0.0, 1.0),
        _pad: [0.0; 3],
    };
    (Some(params), level)
}

/// A 2-D transform from an NGFF edge's matrix over `n` axes, if it stays in the y-x
/// plane (only y and x mixed, the other axes left alone).
fn plane_transform(m: &[Vec<f64>], n: usize) -> Option<Affine2> {
    let (y, x) = (n - 2, n - 1);
    for (r, row) in m.iter().enumerate().take(n) {
        for (c, &v) in row.iter().enumerate().take(n + 1) {
            let inside = (r == y || r == x) && (c == y || c == x || c == n);
            let want = if r == c { 1.0 } else { 0.0 };
            if !inside && (v - want).abs() > 1e-9 {
                return None;
            }
        }
    }
    Some(Affine2 {
        m: [[m[x][x], m[x][y], m[x][n]], [m[y][x], m[y][y], m[y][n]]],
    })
}

/// A viewer: a GPU target, its layers, a camera.
pub struct Viewer {
    gpu: Gpu,
    loader: Loader,
    threads: usize,
    layers: Vec<Layer>,
    current: usize,
    next_id: u16,
    pub camera: Camera,
    frame: u64,
    /// When the view last changed, in which frame, and when it changed in an earlier
    /// frame before that.
    moved_at: Option<Instant>,
    moved_frame: u64,
    moved_before: Option<Instant>,
    moved_before_frame: u64,
    deferred: bool,
}

impl Viewer {
    /// A viewer drawing into a native window of `width` x `height` physical pixels.
    ///
    /// # Safety
    /// The window must stay valid as long as the viewer.
    pub unsafe fn for_window(
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
        options: GpuOptions,
    ) -> Result<Self, ViewerError> {
        Ok(Self::with(Gpu::for_window(
            display, window, width, height, options,
        )?))
    }

    /// A viewer drawing off-screen (frames read back as RGBA8).
    pub fn offscreen(width: u32, height: u32, options: GpuOptions) -> Result<Self, ViewerError> {
        Ok(Self::with(Gpu::offscreen(width, height, options)?))
    }

    fn with(gpu: Gpu) -> Self {
        let threads = Loader::default_threads();
        Self {
            gpu,
            loader: Loader::new(threads),
            threads,
            layers: Vec::new(),
            current: 0,
            next_id: 0,
            camera: Camera {
                center: [0.0, 0.0],
                scale: 1.0,
                rotation: 0.0,
            },
            frame: 0,
            moved_at: None,
            moved_frame: 0,
            moved_before: None,
            moved_before_frame: 0,
            deferred: false,
        }
    }

    /// The adapter drawing the frames (name, backend, kind).
    pub fn adapter(&self) -> &str {
        &self.gpu.adapter_info
    }

    /// The atlas slots a layer of `channels` channels wants for this target: the frame's
    /// level shows up to two level pixels per screen pixel and axis, the coarser levels
    /// a third more, each active channel its own, and room for the ring and planes.
    fn slots_wanted(&self, channels: usize) -> u32 {
        let (w, h) = self.gpu.size();
        let per = (2 * w / TILE + 3) * (2 * h / TILE + 3);
        (per as f64 * 1.34 * channels.max(1) as f64 * 1.5) as u32
    }

    /// Layer `i`'s share of the atlas budget (MiB): by channels.
    fn budget_mib(&self, channels: usize) -> u32 {
        let total: usize = self.layers.iter().map(|l| l.channels.len()).sum::<usize>();
        let total = total.max(channels).max(1);
        ((self.gpu.atlas_mib() as u64 * channels.max(1) as u64 / total as u64) as u32).max(16)
    }

    /// Give each layer the atlas its share allows: larger when the target grew or a
    /// layer left, smaller when a layer came (its tiles are read again). A layer the GPU
    /// cannot hold is removed.
    fn rebalance(&mut self) -> Result<(), ViewerError> {
        let mut i = 0;
        while i < self.layers.len() {
            let (id, texel, nc, have) = {
                let l = &self.layers[i];
                (l.id, l.source.texel, l.channels.len(), l.cache.capacity())
            };
            let budget = self.budget_mib(nc);
            let want = self.gpu.slots_for(texel, self.slots_wanted(nc), budget);
            let cap = self.gpu.slots_for(texel, u32::MAX, budget);
            if want > have || have > cap {
                match self.gpu.prepare_layer(id, texel, want, budget) {
                    Ok(slots) => {
                        let l = &mut self.layers[i];
                        l.cache = Cache::new(slots);
                        l.rebuild_pages();
                    }
                    Err(e) => {
                        self.remove(i);
                        return Err(e);
                    }
                }
            }
            i += 1;
        }
        Ok(())
    }

    /// Open the image at `path` (a folder, or a .zip / .ozx) as the only layer: fit it
    /// into the target.
    pub fn open(&mut self, path: &Path) -> Result<(), ViewerError> {
        while !self.layers.is_empty() {
            self.remove(self.layers.len() - 1);
        }
        self.camera.rotation = 0.0;
        self.add(path)?;
        self.fit();
        Ok(())
    }

    /// Add the image at `path` as a new layer on top (channels from its metadata:
    /// contrast from omero's window, else from the coarsest level's middle tile; the
    /// plane at omero's default z and t; the transform saved in its "aligned" coordinate
    /// system, if any). The first layer is fitted into the target. It becomes the
    /// current layer; returns its index.
    pub fn add(&mut self, path: &Path) -> Result<usize, ViewerError> {
        let source = Source::open(path)?;
        let (transform, note) = saved_transform(&source);
        let nc = source.channels.len();
        let id = self.fresh_id();
        let budget = self.budget_mib(nc);
        let slots = self
            .gpu
            .prepare_layer(id, source.texel, self.slots_wanted(nc), budget)?;
        let mut level_base = Vec::new();
        let mut grid = Vec::new();
        let mut stride = 0u32;
        for lv in &source.levels {
            let g = (
                lv.width.div_ceil(TILE as u64) as u32,
                lv.height.div_ceil(TILE as u64) as u32,
            );
            level_base.push(stride);
            stride += g.0 * g.1;
            grid.push(g);
        }
        let position = source.default_position.clone();
        let planes = (0..source.levels.len())
            .map(|l| source.plane(l, &position))
            .collect();
        let channels = Self::initial_display(&source, &position);
        let pages = vec![0u32; stride as usize * nc];
        let first = self.layers.is_empty();
        self.layers.push(Layer {
            id,
            path: path.to_path_buf(),
            source,
            cache: Cache::new(slots),
            pages,
            level_base,
            grid,
            stride,
            planes,
            dirty: None,
            pages_all: true,
            failed: HashSet::new(),
            error: note,
            channels,
            luts_dirty: true,
            position,
            transform,
            inverse: transform.inverse().unwrap_or(Affine2::IDENTITY),
            visible: true,
            opacity: 1.0,
            read_s: 0.0,
        });
        self.current = self.layers.len() - 1;
        // the others give up atlas room for the new one
        self.rebalance()?;
        if first {
            self.fit();
        }
        Ok(self.layers.len() - 1)
    }

    fn fresh_id(&mut self) -> u16 {
        loop {
            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1);
            if self.layers.iter().all(|l| l.id != id) {
                return id;
            }
        }
    }

    /// Remove layer `i` (nothing happens for an index past the end).
    pub fn remove(&mut self, i: usize) {
        if i >= self.layers.len() {
            return;
        }
        let l = self.layers.remove(i);
        self.gpu.remove_layer(l.id);
        // its arrays are freed: forget their decoded blocks (and what was in flight)
        self.loader.reset();
        if self.current >= self.layers.len() {
            self.current = self.layers.len().saturating_sub(1);
        }
        let _ = self.rebalance();
    }

    /// Read layer `i`'s image again (after its metadata changed, e.g. a saved
    /// transform): its placement and saved transform come from the file again; its
    /// channels (when their number is the same), plane, visibility and opacity stay.
    pub fn reload(&mut self, i: usize) -> Result<(), ViewerError> {
        let (path, channels, position, visible, opacity) = {
            let l = self.layer(i)?;
            (
                l.path.clone(),
                l.channels.clone(),
                l.position(),
                l.visible,
                l.opacity,
            )
        };
        let current = self.current;
        let n = self.add(&path)?;
        let fresh = self.layers.remove(n);
        self.layers.insert(i + 1, fresh);
        self.remove(i);
        self.current = current.min(self.layers.len().saturating_sub(1));
        if self.layers[i].channels.len() == channels.len() {
            self.layers[i].channels = channels;
            self.layers[i].luts_dirty = true;
        }
        for (axis, p, _) in position {
            let _ = self.set_position(i, &axis, p);
        }
        self.set_layer(i, visible, opacity)
    }

    /// Channels as the image describes them; contrast from omero's window, else the
    /// 0.1 and 99.9 percentiles of the coarsest level's middle tile.
    fn initial_display(source: &Source, position: &[f64]) -> Vec<ChannelDisplay> {
        let last = source.levels.len() - 1;
        let lv = &source.levels[last];
        let (h, w) = (lv.height.min(TILE as u64), lv.width.min(TILE as u64));
        let (ty, tx) = ((lv.height - h) / 2, (lv.width - w) / 2);
        let (plane, _) = source.plane(last, position);
        source
            .channels
            .iter()
            .enumerate()
            .map(|(c, info)| {
                let (lo, hi) = info.window.unwrap_or_else(|| {
                    let (start, stop) = source.region(&plane, c, ty, tx, h, w);
                    source
                        .read(last, &start, &stop)
                        .ok()
                        .and_then(|d| source::quantiles(source.texel, &d, 0.001, 0.999))
                        .filter(|(a, b)| a < b)
                        .unwrap_or_else(|| source.texel.range())
                });
                ChannelDisplay {
                    label: info.label.clone(),
                    color: info.color,
                    lut: None,
                    lo,
                    hi,
                    gamma: 1.0,
                    visible: info.active,
                }
            })
            .collect()
    }

    /// The layers, bottom first.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    fn layer_mut(&mut self, i: usize) -> Result<&mut Layer, ViewerError> {
        let n = self.layers.len();
        self.layers
            .get_mut(i)
            .ok_or_else(|| verr(format!("layer {i}: the view has {n}")))
    }

    pub fn layer(&self, i: usize) -> Result<&Layer, ViewerError> {
        self.layers
            .get(i)
            .ok_or_else(|| verr(format!("layer {i}: the view has {}", self.layers.len())))
    }

    /// The current layer's index (the one input and the single-layer calls act on).
    pub fn current(&self) -> usize {
        self.current
    }

    pub fn set_current(&mut self, i: usize) -> Result<(), ViewerError> {
        self.layer(i)?;
        self.current = i;
        Ok(())
    }

    /// Show or hide layer `i` and set its opacity (0..1).
    pub fn set_layer(&mut self, i: usize, visible: bool, opacity: f32) -> Result<(), ViewerError> {
        if !(0.0..=1.0).contains(&opacity) {
            return Err(verr(format!("opacity {opacity}: it is 0 to 1")));
        }
        let l = self.layer_mut(i)?;
        l.visible = visible;
        l.opacity = opacity;
        Ok(())
    }

    /// The current layer's image, if any.
    pub fn source(&self) -> Option<&Source> {
        self.layers.get(self.current).map(|l| &l.source)
    }

    /// The current layer's path, if any.
    pub fn path(&self) -> Option<&Path> {
        self.layers.get(self.current).map(|l| l.path.as_path())
    }

    /// Change channel `c` of layer `i`.
    pub fn set_channel(
        &mut self,
        i: usize,
        c: usize,
        d: ChannelDisplay,
    ) -> Result<(), ViewerError> {
        if !(d.lo.is_finite() && d.hi.is_finite() && d.gamma > 0.0 && d.gamma.is_finite()) {
            return Err(verr(format!(
                "channel {c}: contrast {} .. {} and gamma {} are not usable",
                d.lo, d.hi, d.gamma
            )));
        }
        if let Some(n) = &d.lut {
            if !luts::NAMES.contains(&n.as_str()) {
                return Err(verr(format!(
                    "colour map {n:?}: there are {:?}",
                    luts::NAMES
                )));
            }
        }
        let l = self.layer_mut(i)?;
        let n = l.channels.len();
        let slot = l
            .channels
            .get_mut(c)
            .ok_or_else(|| verr(format!("channel {c}: the image has {n}")))?;
        if slot.color != d.color || slot.lut != d.lut {
            l.luts_dirty = true;
        }
        *slot = d;
        Ok(())
    }

    /// Move layer `i`'s plane to level-0 index `index` on axis `axis` (z, t, ...).
    pub fn set_position(&mut self, i: usize, axis: &str, index: f64) -> Result<(), ViewerError> {
        let o = self.layer_mut(i)?;
        let k = o
            .source
            .plane_axes
            .iter()
            .position(|&a| o.source.axes[a] == axis)
            .ok_or_else(|| {
                let names: Vec<&str> = o
                    .source
                    .plane_axes
                    .iter()
                    .map(|&a| o.source.axes[a].as_str())
                    .collect();
                verr(format!("no plane axis {axis:?}: the image's are {names:?}"))
            })?;
        let n = o.source.levels[0].shape[o.source.plane_axes[k]];
        if !(index.is_finite() && index >= 0.0 && index <= (n.max(1) - 1) as f64) {
            return Err(verr(format!(
                "{axis} = {index}: the image has {n} indices on {axis}"
            )));
        }
        o.position[k] = index;
        self.moved();
        Ok(())
    }

    /// Set layer `i`'s transform (world to world, on top of its levels' placement).
    pub fn set_transform(&mut self, i: usize, t: Affine2) -> Result<(), ViewerError> {
        let inverse = t
            .inverse()
            .filter(|_| t.is_finite())
            .ok_or_else(|| verr(format!("a transform that cannot be inverted: {:?}", t.m)))?;
        let l = self.layer_mut(i)?;
        l.transform = t;
        l.inverse = inverse;
        self.moved();
        Ok(())
    }

    /// Move layer `i` by `dx`, `dy` screen pixels (it follows the pointer).
    pub fn move_layer(&mut self, i: usize, dx: f64, dy: f64) -> Result<(), ViewerError> {
        let (tw, th) = self.gpu.size();
        let (wx, wy) = self.camera.screen_to_world(tw, th).linear(dx, dy);
        let t = Affine2::translation(wx, wy).after(&self.layer(i)?.transform);
        self.set_transform(i, t)
    }

    /// Turn layer `i` by `degrees` (clockwise on an unturned view) about world point
    /// `about`, by default the image's centre.
    pub fn rotate_layer(
        &mut self,
        i: usize,
        degrees: f64,
        about: Option<[f64; 2]>,
    ) -> Result<(), ViewerError> {
        let l = self.layer(i)?;
        let c = about.unwrap_or_else(|| l.centre());
        let t = Affine2::rotation(degrees, c[0], c[1]).after(&l.transform);
        self.set_transform(i, t)
    }

    /// The world box (x0, y0, x1, y1) the visible layers cover.
    pub fn bounds(&self) -> Option<[f64; 4]> {
        let mut out: Option<[f64; 4]> = None;
        for l in self.layers.iter().filter(|l| l.visible) {
            let b = l.bounds();
            out = Some(match out {
                None => b,
                Some(a) => [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ],
            });
        }
        out
    }

    /// Show every visible layer whole, centred (the camera's turn stays).
    pub fn fit(&mut self) {
        let (tw, th) = self.gpu.size();
        let turn = Affine2::rotation(self.camera.rotation, 0.0, 0.0);
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for l in self.layers.iter().filter(|l| l.visible) {
            for (x, y) in l.corners() {
                let (u, v) = turn.apply(x, y); // as the screen's axes see it
                x0 = x0.min(u);
                y0 = y0.min(v);
                x1 = x1.max(u);
                y1 = y1.max(v);
            }
        }
        if x0 > x1 {
            return;
        }
        let scale = ((x1 - x0) / tw as f64)
            .max((y1 - y0) / th as f64)
            .max(1e-12);
        let back = Affine2::rotation(-self.camera.rotation, 0.0, 0.0);
        let (cx, cy) = back.apply((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        self.camera.center = [cx, cy];
        self.camera.scale = scale;
        self.moved();
    }

    /// The world point at the target's centre.
    pub fn center(&self) -> [f64; 2] {
        self.camera.center
    }

    /// Put world point `center` at the target's centre, `scale` world units per pixel.
    pub fn look_at(&mut self, center: [f64; 2], scale: f64) {
        self.camera.center = center;
        self.camera.scale = scale;
        self.moved();
    }

    /// The world point at screen pixel (`x`, `y`).
    pub fn screen_to_world(&self, x: f64, y: f64) -> [f64; 2] {
        let (tw, th) = self.gpu.size();
        let (u, v) = self.camera.screen_to_world(tw, th).apply(x, y);
        [u, v]
    }

    /// Move the view by `dx`, `dy` screen pixels (the image follows the pointer).
    pub fn pan(&mut self, dx: f64, dy: f64) {
        let (tw, th) = self.gpu.size();
        let (wx, wy) = self.camera.screen_to_world(tw, th).linear(dx, dy);
        self.camera.center = [self.camera.center[0] - wx, self.camera.center[1] - wy];
        self.moved();
    }

    /// Zoom by `factor` (> 1: closer) keeping the world point under screen (`x`, `y`).
    /// The scale stays between 1/64 of the finest pixel and 64 times the whole view.
    pub fn zoom(&mut self, factor: f64, x: f64, y: f64) {
        let finest = self
            .layers
            .iter()
            .map(|l| l.pixel(0))
            .fold(f64::MAX, f64::min);
        let (lo, hi) = match self.bounds() {
            Some(b) if finest < f64::MAX => (finest / 64.0, 64.0 * (b[2] - b[0]).max(b[3] - b[1])),
            _ => (1e-12, 1e12),
        };
        let before = self.screen_to_world(x, y);
        self.camera.scale = (self.camera.scale / factor).clamp(lo, hi.max(lo));
        let after = self.screen_to_world(x, y);
        self.camera.center[0] += before[0] - after[0];
        self.camera.center[1] += before[1] - after[1];
        self.moved();
    }

    /// Turn the view by `degrees` (the image turns clockwise) keeping the world point
    /// under screen (`x`, `y`).
    pub fn rotate(&mut self, degrees: f64, x: f64, y: f64) {
        let before = self.screen_to_world(x, y);
        self.camera.rotation = (self.camera.rotation + degrees).rem_euclid(360.0);
        let after = self.screen_to_world(x, y);
        self.camera.center[0] += before[0] - after[0];
        self.camera.center[1] += before[1] - after[1];
        self.moved();
    }

    fn moved(&mut self) {
        if self.moved_frame != self.frame {
            self.moved_before_frame = self.moved_frame;
            self.moved_before = self.moved_at;
            self.moved_frame = self.frame;
        }
        self.moved_at = Some(Instant::now());
    }

    /// Whether the view is moving (see [`MOVING`]): it changed in two frames close
    /// together, the last of them a few frames ago at most.
    fn moving(&self) -> bool {
        let recent = |t: Option<Instant>| t.is_some_and(|t| t.elapsed() < MOVING);
        recent(self.moved_at)
            && recent(self.moved_before)
            && self.moved_frame - self.moved_before_frame <= MOVING_FRAMES
            && self.frame - self.moved_frame <= MOVING_FRAMES
    }

    /// Resize the target; the world point at the centre stays. A target larger than the
    /// atlases were made for gets larger atlases (their tiles are read again).
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), ViewerError> {
        self.gpu.resize(width, height);
        self.rebalance()
    }

    /// The current layer's level at the current zoom.
    pub fn level(&self) -> usize {
        self.layers
            .get(self.current)
            .map_or(0, |l| l.level_for(self.camera.scale))
    }

    /// The first error (tile read, or a note from opening) of any layer.
    pub fn error(&self) -> Option<&str> {
        self.layers.iter().find_map(|l| l.error.as_deref())
    }

    /// Whether read tiles are waiting for the next frame (for the host to repaint).
    pub fn has_results(&self) -> bool {
        self.loader.has_results()
    }

    /// Whether the host should draw again soon: tiles on their way or waiting, or
    /// reading deferred until the view stops moving.
    pub fn wants_frame(&self) -> bool {
        self.loader.pending() > 0 || self.loader.has_results() || self.deferred
    }

    /// Tiles requested and not read yet.
    pub fn pending(&self) -> usize {
        self.loader.pending()
    }

    /// A wake-up called from a reading thread after each tile, for a host whose event
    /// loop can be woken from another thread (it should only schedule a repaint).
    pub fn set_wakeup(&self, f: Option<Box<dyn Fn() + Send>>) {
        self.loader.set_notify(f);
    }

    /// Every layer's part of the frame (see [`prepare_layer`]); the requests of all
    /// layers go to the readers in priority classes.
    fn prepare(&mut self) -> (Vec<(u16, FrameParams)>, FrameStats) {
        self.frame += 1;
        let (tw, th) = self.gpu.size();
        let cx = FrameCtx {
            frame: self.frame,
            tw,
            th,
            screen: self.camera.screen_to_world(tw, th),
            scale: self.camera.scale,
            moving: self.moving(),
            threads: self.threads as f64,
        };
        let mut results = self.loader.take();
        let mut stats = FrameStats::default();
        let mut wants = Wants::default();
        let mut frames = Vec::new();
        for (i, l) in self.layers.iter_mut().enumerate() {
            let (mine, rest): (Vec<Loaded>, Vec<Loaded>) =
                results.into_iter().partition(|t| t.key.layer == l.id);
            results = rest;
            let (params, level) =
                prepare_layer(l, &mut self.gpu, mine, &cx, &mut wants, &mut stats);
            if i == self.current {
                stats.level = level as u32;
            }
            if let Some(p) = params {
                frames.push((l.id, p));
            }
        }
        let mut all = wants.coarse;
        all.extend(wants.here);
        all.extend(wants.ring);
        all.extend(wants.planes);
        self.loader.want(all);
        stats.pending = self.loader.pending() as u32;
        self.deferred = stats.deferred;
        (frames, stats)
    }

    /// A GPU error from the last frame, as an error.
    fn gpu_error(&self) -> Result<(), ViewerError> {
        match self.gpu.take_error() {
            Some(e) => Err(verr(format!("GPU error: {e}"))),
            None => Ok(()),
        }
    }

    /// Draw one frame into the window.
    pub fn render(&mut self) -> Result<FrameStats, ViewerError> {
        let (frames, stats) = self.prepare();
        self.gpu.present(&frames)?;
        self.gpu_error()?;
        Ok(stats)
    }

    /// Draw one frame off-screen into `out` (RGBA8 rows, width x height x 4 bytes).
    pub fn render_into(&mut self, out: &mut [u8]) -> Result<FrameStats, ViewerError> {
        let (frames, stats) = self.prepare();
        self.gpu.render_into(&frames, out)?;
        self.gpu_error()?;
        Ok(stats)
    }

    /// Draw one frame off-screen and read it back (RGBA8 rows).
    pub fn render_rgba(&mut self) -> Result<(Vec<u8>, FrameStats), ViewerError> {
        let (w, h) = self.gpu.size();
        let mut out = vec![0u8; (4 * w * h) as usize];
        let stats = self.render_into(&mut out)?;
        Ok((out, stats))
    }

    /// Prepare frames until the view is sharp (every tile of the frame's levels in view
    /// is read; with `read_ahead`, also everything read ahead: the ring and the planes
    /// next to this one), or `timeout` passes. For tests, screenshots and the harness;
    /// returns the last frame's statistics (a tile that failed to read counts as done).
    pub fn settle(
        &mut self,
        timeout: Duration,
        read_ahead: bool,
    ) -> Result<FrameStats, ViewerError> {
        let t0 = Instant::now();
        loop {
            let (_, stats) = self.prepare();
            let failed = self.layers.iter().map(|l| l.failed.len()).sum::<usize>();
            let done = if read_ahead {
                stats.pending == 0 && !self.loader.has_results() && !stats.deferred
            } else {
                stats.missing == 0 || (failed > 0 && stats.pending == 0 && !stats.deferred)
            };
            if done || t0.elapsed() > timeout {
                return Ok(stats);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Set the contrast of layer `i`'s channel `channel` (every visible channel when
    /// None) from what is in view: the `lo` and `hi` quantiles (0..1) of the current
    /// plane over the view's box, read at the finest level that keeps it within about a
    /// million pixels.
    pub fn auto_contrast(
        &mut self,
        i: usize,
        channel: Option<usize>,
        lo: f64,
        hi: f64,
    ) -> Result<(), ViewerError> {
        if !(0.0..=1.0).contains(&lo) || !(0.0..=1.0).contains(&hi) || lo >= hi {
            return Err(verr(format!("quantiles {lo} and {hi}: 0 <= lo < hi <= 1")));
        }
        let (tw, th) = self.gpu.size();
        let screen = self.camera.screen_to_world(tw, th);
        let l = self.layer(i)?;
        let nl = l.source.levels.len();
        let mut pick = None;
        for lv in 0..nl {
            let level = &l.source.levels[lv];
            let m = level.to_index.after(&l.inverse).after(&screen);
            let (x0, y0, x1, y1) = index_box(&m, tw, th);
            let cx0 = (x0 + 0.5).floor().max(0.0) as u64;
            let cy0 = (y0 + 0.5).floor().max(0.0) as u64;
            let cx1 = ((x1 + 0.5).floor() + 1.0).min(level.width as f64).max(0.0) as u64;
            let cy1 = ((y1 + 0.5).floor() + 1.0).min(level.height as f64).max(0.0) as u64;
            if cx1 <= cx0 || cy1 <= cy0 {
                return Err(verr("the image is not in view"));
            }
            pick = Some((lv, cy0, cx0, cy1 - cy0, cx1 - cx0));
            if (cx1 - cx0) * (cy1 - cy0) <= 1 << 20 {
                break;
            }
        }
        let (lv, ty, tx, h, w) = pick.ok_or_else(|| verr("the image has no levels"))?;
        let (plane, _) = l.source.plane(lv, &l.position);
        let chans: Vec<usize> = match channel {
            Some(c) if c < l.channels.len() => vec![c],
            Some(c) => {
                return Err(verr(format!(
                    "channel {c}: the image has {}",
                    l.channels.len()
                )))
            }
            None => (0..l.channels.len())
                .filter(|&c| l.channels[c].visible)
                .collect(),
        };
        let mut found = Vec::new();
        for c in chans {
            let (start, stop) = l.source.region(&plane, c, ty, tx, h, w);
            let data = l.source.read(lv, &start, &stop)?;
            if let Some((a, b)) = source::quantiles(l.source.texel, &data, lo, hi) {
                let b = if b > a { b } else { a + 1.0 };
                found.push((c, a, b));
            }
        }
        for (c, a, b) in found {
            let mut d = self.layers[i].channels[c].clone();
            d.lo = a;
            d.hi = b;
            self.set_channel(i, c, d)?;
        }
        Ok(())
    }

    /// What lies under screen point (`x`, `y`): the world point, and per layer the
    /// level-0 pixel and every channel's value from the finest cached tile holding it
    /// (read back from the GPU; the frame's level first, then coarser, then finer).
    pub fn pick(&self, x: f64, y: f64) -> Result<Pick, ViewerError> {
        let world = self.screen_to_world(x, y);
        let mut layers = Vec::new();
        for (li, o) in self.layers.iter().enumerate() {
            let (wx, wy) = o.inverse.apply(world[0], world[1]);
            let nearest = |l: usize| -> Option<(u64, u64)> {
                let lv = &o.source.levels[l];
                let (ix, iy) = lv.to_index.apply(wx, wy);
                let (ix, iy) = ((ix + 0.5).floor(), (iy + 0.5).floor());
                (ix >= 0.0 && iy >= 0.0 && ix < lv.width as f64 && iy < lv.height as f64)
                    .then_some((ix as u64, iy as u64))
            };
            let pixel = nearest(0).map(|(x, y)| [x, y]);
            let level = o.level_for(self.camera.scale);
            let nl = o.source.levels.len();
            let order: Vec<usize> = (level..nl).chain((0..level).rev()).collect();
            let mut values = Vec::new();
            for ch in 0..o.channels.len() {
                let mut found = PickedValue {
                    channel: ch,
                    value: None,
                    level: None,
                };
                if pixel.is_some() {
                    for &l in &order {
                        let Some((ix, iy)) = nearest(l) else { continue };
                        let k = o.key(
                            ch,
                            l,
                            ((ix / TILE as u64) as u32, (iy / TILE as u64) as u32),
                        );
                        if let Some(slot) = o.cache.peek(&k) {
                            let bytes = self.gpu.read_texel(
                                o.id,
                                slot,
                                (ix % TILE as u64) as u32,
                                (iy % TILE as u64) as u32,
                            )?;
                            found.value = Some(o.source.texel.value(&bytes, 0));
                            found.level = Some(l);
                            break;
                        }
                    }
                }
                values.push(found);
            }
            layers.push(LayerPick {
                layer: li,
                pixel,
                values,
            });
        }
        Ok(Pick { world, layers })
    }

    /// The view as a [`ViewState`] (serialises to JSON).
    pub fn view_state(&self) -> ViewState {
        ViewState::of(self)
    }

    /// Apply a view state (see [`ViewState::apply`]).
    pub fn set_view_state(&mut self, state: &ViewState) -> Result<(), ViewerError> {
        state.apply(self)
    }

    /// Wait for the GPU (for timing).
    pub fn finish(&self) -> Result<(), ViewerError> {
        self.gpu.finish()
    }

    /// The target's size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        self.gpu.size()
    }

    /// Atlas slots (tiles the GPU holds at once) of every layer together.
    pub fn atlas_slots(&self) -> u32 {
        self.gpu.all_slots()
    }
}

/// The transform an image's "aligned" coordinate system gives it (an NGFF 0.6 edge
/// from its own coordinate system to "aligned"), and a note when one is there but cannot
/// be shown in a plane.
fn saved_transform(source: &Source) -> (Affine2, Option<String>) {
    match &source.aligned {
        None => (Affine2::IDENTITY, None),
        Some(m) => match plane_transform(m, source.axes.len()).filter(|t| t.inverse().is_some()) {
            Some(t) => (t, None),
            None => (
                Affine2::IDENTITY,
                Some(format!(
                    "the image's {ALIGNED:?} transform leaves the y-x plane (or cannot be \
                     inverted); the image is shown without it"
                )),
            ),
        },
    }
}
