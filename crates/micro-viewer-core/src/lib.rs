//! micro-viewer's core (Workplan WP14): OME-Zarr images drawn by the GPU from a tile
//! cache, hosted by any toolkit that can hand over a native window, or read back
//! off-screen. No Qt and no Python here: the bindings and the widget live outside.
//!
//! 2D viewer (task V7): every level of an image, the finest worth drawing chosen per
//! frame from the zoom, coarser (or finer) cached tiles standing in while tiles load;
//! tiles read in the background, centre first, a ring around and the next and previous
//! plane when the cache has room; channels with colour, contrast and gamma, added up;
//! the plane chosen on the axes before y and x; picking; the view as JSON.

mod blocks;
mod gpu;
mod loader;
mod source;
mod tiles;
mod view;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub use gpu::GpuOptions;
use gpu::{ChannelGpu, FrameParams, Gpu, LevelGpu, BACKGROUND, MISSING};
use loader::{Loader, Request};
pub use raw_window_handle;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
pub use source::{ChannelInfo, Level, Source, Texel};
use tiles::{Cache, TileKey, TILE};
pub use view::ViewState;

/// Something the viewer cannot do as asked; the message says why.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewerError(pub String);

impl std::fmt::Display for ViewerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ViewerError {}

/// What a frame showed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Tiles of the frame's level in view (all active channels).
    pub visible: u32,
    /// Tiles uploaded for this frame.
    pub loaded: u32,
    /// Tiles requested and not read yet.
    pub pending: u32,
    /// The frame's level (0 the finest).
    pub level: u32,
    /// Tiles of the frame's level in view and not cached yet (coarser or finer levels
    /// stand in for them): 0 means the view is sharp.
    pub missing: u32,
}

/// The camera: the world point (x, y) at the target's top-left corner and the world
/// units per screen pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub origin: [f64; 2],
    pub scale: f64,
}

/// How one channel is shown.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelDisplay {
    pub label: String,
    pub color: [f32; 3],
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

/// What lies under a screen point.
#[derive(Debug, Clone, PartialEq)]
pub struct Pick {
    pub world: [f64; 2],
    /// The level-0 pixel (x, y) whose centre is nearest, if inside the image.
    pub pixel: Option<[u64; 2]>,
    pub values: Vec<PickedValue>,
}

struct Opened {
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
}

impl Opened {
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
}

/// A viewer: a GPU target, at most one open image, a camera, the plane's position and
/// the channels' display.
pub struct Viewer {
    gpu: Gpu,
    loader: Loader,
    opened: Option<Opened>,
    pub camera: Camera,
    channels: Vec<ChannelDisplay>,
    /// Level-0 indices on the image's plane axes.
    position: Vec<f64>,
    frame: u64,
}

/// The tiles of `level` a target of `tw` x `th` pixels shows: (x0, y0, x1, y1), the last
/// exclusive, or None when the view misses the level.
fn visible_tiles(
    lv: &source::Level,
    c: &Camera,
    tw: u32,
    th: u32,
    margin: i64,
) -> Option<(u32, u32, u32, u32)> {
    let corners = [
        (0.0, 0.0),
        (tw as f64, 0.0),
        (0.0, th as f64),
        (tw as f64, th as f64),
    ];
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for (sx, sy) in corners {
        let (ix, iy) = lv
            .to_index
            .apply(c.origin[0] + sx * c.scale, c.origin[1] + sy * c.scale);
        x0 = x0.min(ix);
        y0 = y0.min(iy);
        x1 = x1.max(ix);
        y1 = y1.max(iy);
    }
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
        Self {
            gpu,
            loader: Loader::new(Loader::default_threads()),
            opened: None,
            camera: Camera {
                origin: [0.0, 0.0],
                scale: 1.0,
            },
            channels: Vec::new(),
            position: Vec::new(),
            frame: 0,
        }
    }

    /// The adapter drawing the frames (name, backend, kind).
    pub fn adapter(&self) -> &str {
        &self.gpu.adapter_info
    }

    /// The atlas slots needed for a target of this size: the frame's level shows up to
    /// two level pixels per screen pixel and axis, the coarser levels a third more, each
    /// active channel its own, and room for the ring and planes around.
    fn slots_wanted(&self, channels: usize) -> u32 {
        let (w, h) = self.gpu.size();
        let per = (2 * w / TILE + 3) * (2 * h / TILE + 3);
        (per as f64 * 1.34 * channels.max(1) as f64 * 1.5) as u32
    }

    /// Open the image at `path` (a folder, or a .zip / .ozx): fit it into the target,
    /// channels from its metadata (contrast from omero's window, else from the coarsest
    /// level's middle tile), the plane at omero's default z and t.
    pub fn open(&mut self, path: &Path) -> Result<(), ViewerError> {
        let source = Source::open(path)?;
        self.loader.reset();
        let nc = source.channels.len();
        let slots = match self.gpu.prepare_image(source.texel, self.slots_wanted(nc)) {
            Ok(s) => s,
            Err(e) => {
                // the old image's atlas is gone too: nothing is open now
                self.opened = None;
                self.channels.clear();
                self.position.clear();
                return Err(e);
            }
        };
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
        self.gpu.write_pages(pages.len(), 0, &pages);
        self.opened = Some(Opened {
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
            error: None,
        });
        self.channels = channels;
        self.position = position;
        self.fit();
        Ok(())
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
                    lo,
                    hi,
                    gamma: 1.0,
                    visible: info.active,
                }
            })
            .collect()
    }

    /// The open image, if any.
    pub fn source(&self) -> Option<&Source> {
        self.opened.as_ref().map(|o| &o.source)
    }

    /// The open image's path, if any.
    pub fn path(&self) -> Option<&Path> {
        self.opened.as_ref().map(|o| o.path.as_path())
    }

    pub fn channels(&self) -> &[ChannelDisplay] {
        &self.channels
    }

    /// Change channel `c`'s display.
    pub fn set_channel(&mut self, c: usize, d: ChannelDisplay) -> Result<(), ViewerError> {
        let n = self.channels.len();
        let slot = self
            .channels
            .get_mut(c)
            .ok_or_else(|| ViewerError(format!("channel {c}: the image has {n}")))?;
        if !(d.lo.is_finite() && d.hi.is_finite() && d.gamma > 0.0 && d.gamma.is_finite()) {
            return Err(ViewerError(format!(
                "channel {c}: contrast {} .. {} and gamma {} are not usable",
                d.lo, d.hi, d.gamma
            )));
        }
        *slot = d;
        Ok(())
    }

    /// The plane's position: per plane axis, its name, the level-0 index and the
    /// number of indices.
    pub fn position(&self) -> Vec<(String, f64, u64)> {
        let Some(o) = &self.opened else {
            return Vec::new();
        };
        o.source
            .plane_axes
            .iter()
            .zip(&self.position)
            .map(|(&a, &p)| (o.source.axes[a].clone(), p, o.source.levels[0].shape[a]))
            .collect()
    }

    /// Move the plane to level-0 index `index` on axis `axis` (z, t, ...).
    pub fn set_position(&mut self, axis: &str, index: f64) -> Result<(), ViewerError> {
        let o = self
            .opened
            .as_ref()
            .ok_or_else(|| ViewerError("no image is open".into()))?;
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
                ViewerError(format!("no plane axis {axis:?}: the image's are {names:?}"))
            })?;
        let n = o.source.levels[0].shape[o.source.plane_axes[k]];
        if !(index.is_finite() && index >= 0.0 && index <= (n.max(1) - 1) as f64) {
            return Err(ViewerError(format!(
                "{axis} = {index}: the image has {n} indices on {axis}"
            )));
        }
        self.position[k] = index;
        Ok(())
    }

    /// The world box (x0, y0, x1, y1) the image's finest level covers.
    pub fn bounds(&self) -> Option<[f64; 4]> {
        let o = self.opened.as_ref()?;
        let lv = &o.source.levels[0];
        let (w, h) = (lv.width as f64 - 0.5, lv.height as f64 - 0.5);
        let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for (ix, iy) in [(-0.5, -0.5), (w, -0.5), (-0.5, h), (w, h)] {
            let (x, y) = lv.to_world.apply(ix, iy);
            b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
        }
        Some(b)
    }

    /// Show the whole image, centred.
    pub fn fit(&mut self) {
        let Some(b) = self.bounds() else { return };
        let (tw, th) = self.gpu.size();
        let (w, h) = (b[2] - b[0], b[3] - b[1]);
        let scale = (w / tw as f64).max(h / th as f64).max(1e-12);
        self.camera = Camera {
            origin: [
                b[0] + (w - tw as f64 * scale) / 2.0,
                b[1] + (h - th as f64 * scale) / 2.0,
            ],
            scale,
        };
    }

    /// The world point at the target's centre.
    pub fn center(&self) -> [f64; 2] {
        let (tw, th) = self.gpu.size();
        let c = &self.camera;
        [
            c.origin[0] + tw as f64 / 2.0 * c.scale,
            c.origin[1] + th as f64 / 2.0 * c.scale,
        ]
    }

    /// Put world point `center` at the target's centre, `scale` world units per pixel.
    pub fn look_at(&mut self, center: [f64; 2], scale: f64) {
        let (tw, th) = self.gpu.size();
        self.camera = Camera {
            origin: [
                center[0] - tw as f64 / 2.0 * scale,
                center[1] - th as f64 / 2.0 * scale,
            ],
            scale,
        };
    }

    /// Move the view by `dx`, `dy` screen pixels (the image follows the pointer).
    pub fn pan(&mut self, dx: f64, dy: f64) {
        self.camera.origin[0] -= dx * self.camera.scale;
        self.camera.origin[1] -= dy * self.camera.scale;
    }

    /// Zoom by `factor` (> 1: closer) keeping the world point under screen (`x`, `y`).
    /// The scale stays between 1/64 of the finest pixel and 64 times the whole image.
    pub fn zoom(&mut self, factor: f64, x: f64, y: f64) {
        let (lo, hi) = match (&self.opened, self.bounds()) {
            (Some(o), Some(b)) => (
                o.source.levels[0].pixel / 64.0,
                64.0 * (b[2] - b[0]).max(b[3] - b[1]),
            ),
            _ => (1e-12, 1e12),
        };
        let c = &mut self.camera;
        let (wx, wy) = (c.origin[0] + x * c.scale, c.origin[1] + y * c.scale);
        c.scale = (c.scale / factor).clamp(lo, hi.max(lo));
        c.origin = [wx - x * c.scale, wy - y * c.scale];
    }

    /// Resize the target; the world point at the centre stays. A target larger than the
    /// atlas was made for gets a larger atlas (its tiles are read again).
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), ViewerError> {
        let center = self.center();
        self.gpu.resize(width, height);
        self.look_at(center, self.camera.scale);
        let Some(o) = &self.opened else {
            return Ok(());
        };
        let wanted = self.slots_wanted(o.source.channels.len());
        let texel = o.source.texel;
        if wanted > self.gpu.slots() {
            match self.gpu.prepare_image(texel, wanted) {
                Ok(slots) => {
                    let o = self.opened.as_mut().expect("checked above");
                    o.cache = Cache::new(slots);
                    o.rebuild_pages();
                }
                Err(e) => {
                    // the old atlas is gone too: nothing is open now
                    self.opened = None;
                    self.channels.clear();
                    self.position.clear();
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// The level a frame at the current zoom draws: the coarsest whose pixels are no
    /// larger than a screen pixel (the finest when zoomed in further).
    pub fn level(&self) -> usize {
        let Some(o) = &self.opened else { return 0 };
        let s = self.camera.scale * (1.0 + 1e-9);
        o.source
            .levels
            .iter()
            .enumerate()
            .filter(|(_, l)| l.pixel <= s)
            .map(|(i, _)| i)
            .max_by(|&a, &b| {
                o.source.levels[a]
                    .pixel
                    .total_cmp(&o.source.levels[b].pixel)
            })
            .unwrap_or(0)
    }

    /// The first tile read error since the image was opened, if any (the tile stays
    /// missing; coarser levels stand in).
    pub fn error(&self) -> Option<&str> {
        self.opened.as_ref().and_then(|o| o.error.as_deref())
    }

    /// Whether read tiles are waiting for the next frame (for the host to repaint).
    pub fn has_results(&self) -> bool {
        self.loader.has_results()
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

    /// Tiles for this frame: keep the cached tiles the view uses, place the tiles read
    /// since the last frame, update the page table, and ask for what is still missing.
    fn prepare(&mut self) -> (Option<FrameParams>, FrameStats) {
        self.frame += 1;
        let frame = self.frame;
        let (tw, th) = self.gpu.size();
        let level = self.level();
        let camera = self.camera;
        let Some(o) = &mut self.opened else {
            return (None, FrameStats::default());
        };
        let nl = o.source.levels.len();
        // the plane on every level
        let mut moved = false;
        for l in 0..nl {
            let p = o.source.plane(l, &self.position);
            if p.1 != o.planes[l].1 {
                o.planes[l] = p;
                moved = true;
            }
        }
        if moved {
            o.rebuild_pages();
        }
        let active: Vec<usize> = (0..self.channels.len())
            .filter(|&c| self.channels[c].visible)
            .collect();
        let key = |o: &Opened, c: usize, l: usize, (x, y): (u32, u32)| TileKey {
            plane: o.planes[l].1,
            channel: c as u16,
            level: l as u8,
            x,
            y,
        };
        // keep what the view uses: the frame's level and the coarser ones
        let mut stats = FrameStats {
            level: level as u32,
            ..Default::default()
        };
        let mut views = vec![None; nl];
        for (l, v) in views.iter_mut().enumerate().skip(level) {
            *v = visible_tiles(&o.source.levels[l], &camera, tw, th, 0);
        }
        let order = views[level].map(centre_first).unwrap_or_default();
        let mut missing_here = Vec::new();
        for &t in &order {
            for &c in &active {
                stats.visible += 1;
                let k = key(o, c, level, t);
                if o.cache.touch(&k, frame).is_none() {
                    missing_here.push(k);
                }
            }
        }
        for (l, view) in views.iter().enumerate().skip(level + 1) {
            if let Some(r) = *view {
                for t in centre_first(r) {
                    for &c in &active {
                        let k = key(o, c, l, t);
                        o.cache.touch(&k, frame);
                    }
                }
            }
        }
        // place the tiles read since the last frame
        for t in self.loader.take() {
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
                    self.gpu.upload_tile(slot, &data, t.w, t.h);
                    if o.current(&t.key) {
                        o.set_page(&t.key, slot + 1);
                    }
                    stats.loaded += 1;
                }
            }
        }
        // ask for what is missing: a coarse level first (few tiles cover the view),
        // then the frame's level centre first, then a ring around and the planes next to
        // this one while the cache has room
        let source = &o.source;
        let request = |k: TileKey, plane: &[u64]| -> Request {
            let lv = &source.levels[k.level as usize];
            let (ty, tx) = (k.y as u64 * TILE as u64, k.x as u64 * TILE as u64);
            let h = (lv.height - ty).min(TILE as u64);
            let w = (lv.width - tx).min(TILE as u64);
            let (start, stop) = source.region(plane, k.channel as usize, ty, tx, h, w);
            Request {
                key: k,
                gen: 0,
                array: source.array(k.level as usize),
                start,
                stop,
            }
        };
        let mut wanted = Vec::new();
        let missing = |o: &Opened, k: &TileKey| o.cache.peek(k).is_none() && !o.failed.contains(k);
        let coarse = (level + 2).min(nl - 1);
        if coarse != level {
            if let Some(r) = views[coarse] {
                for t in centre_first(r) {
                    for &c in &active {
                        let k = key(o, c, coarse, t);
                        if missing(o, &k) {
                            wanted.push(request(k, &o.planes[coarse].0));
                        }
                    }
                }
            }
        }
        for k in &missing_here {
            if missing(o, k) {
                wanted.push(request(*k, &o.planes[level].0));
            }
        }
        let room = o.cache.capacity() as usize;
        if stats.visible as usize * 2 < room {
            if let (Some(inner), Some(outer)) = (
                views[level],
                visible_tiles(&source.levels[level], &camera, tw, th, 1),
            ) {
                for t in centre_first(outer) {
                    let inside = t.0 >= inner.0 && t.0 < inner.2 && t.1 >= inner.1 && t.1 < inner.3;
                    if inside {
                        continue;
                    }
                    for &c in &active {
                        let k = key(o, c, level, t);
                        if missing(o, &k) {
                            wanted.push(request(k, &o.planes[level].0));
                        } else {
                            o.cache.touch(&k, frame);
                        }
                    }
                }
            }
        }
        if stats.visible as usize * 4 < room {
            // the next and previous plane on each plane axis (z, t, ...)
            for k_axis in 0..source.plane_axes.len() {
                for step in [1.0, -1.0] {
                    let mut pos = self.position.clone();
                    pos[k_axis] += step;
                    let a = source.plane_axes[k_axis];
                    if pos[k_axis] < 0.0 || pos[k_axis] > (source.levels[0].shape[a] - 1) as f64 {
                        continue;
                    }
                    let (idx, id) = source.plane(level, &pos);
                    if id == o.planes[level].1 {
                        continue;
                    }
                    for &t in &order {
                        for &c in &active {
                            let k = TileKey {
                                plane: id,
                                channel: c as u16,
                                level: level as u8,
                                x: t.0,
                                y: t.1,
                            };
                            if missing(o, &k) {
                                wanted.push(request(k, &idx));
                            } else {
                                o.cache.touch(&k, frame);
                            }
                        }
                    }
                }
            }
        }
        self.loader.want(wanted);
        stats.pending = self.loader.pending() as u32;
        stats.missing = missing_here
            .iter()
            .filter(|k| o.cache.peek(k).is_none())
            .count() as u32;
        // the page table, levels and channels for the shader
        if o.pages_all {
            self.gpu.write_pages(o.pages.len(), 0, &o.pages);
            o.pages_all = false;
            o.dirty = None;
        } else if let Some((a, b)) = o.dirty.take() {
            if self.gpu.write_pages(o.pages.len(), a, &o.pages[a..=b]) {
                self.gpu.write_pages(o.pages.len(), 0, &o.pages);
            }
        }
        let levels: Vec<LevelGpu> = source
            .levels
            .iter()
            .enumerate()
            .map(|(l, lv)| {
                let (ox, oy) = lv.to_index.apply(camera.origin[0], camera.origin[1]);
                let (ox, oy) = (ox + 0.5, oy + 0.5); // nearest centre: floor(i + 0.5)
                let (fx, fy) = (ox.floor(), oy.floor());
                let ux = lv.to_index.linear(camera.scale, 0.0);
                let uy = lv.to_index.linear(0.0, camera.scale);
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
        let channels: Vec<ChannelGpu> = self
            .channels
            .iter()
            .map(|d| ChannelGpu {
                color: d.color,
                gamma: d.gamma as f32,
                lo: d.lo as f32,
                hi: d.hi as f32,
                active: d.visible as u32,
                _pad: 0,
            })
            .collect();
        self.gpu.write_levels(&levels);
        self.gpu.write_channels(&channels);
        let params = FrameParams {
            n_levels: nl as u32,
            n_channels: channels.len() as u32,
            level: level as u32,
            channel_stride: o.stride,
            background: BACKGROUND,
            missing: MISSING,
        };
        (Some(params), stats)
    }

    /// A GPU error from the last frame, as an error.
    fn gpu_error(&self) -> Result<(), ViewerError> {
        match self.gpu.take_error() {
            Some(e) => Err(ViewerError(format!("GPU error: {e}"))),
            None => Ok(()),
        }
    }

    /// Draw one frame into the window.
    pub fn render(&mut self) -> Result<FrameStats, ViewerError> {
        let (params, stats) = self.prepare();
        self.gpu.present(params.as_ref())?;
        self.gpu_error()?;
        Ok(stats)
    }

    /// Draw one frame off-screen into `out` (RGBA8 rows, width x height x 4 bytes).
    pub fn render_into(&mut self, out: &mut [u8]) -> Result<FrameStats, ViewerError> {
        let (params, stats) = self.prepare();
        self.gpu.render_into(params.as_ref(), out)?;
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

    /// Prepare frames until the view is sharp (every tile of the frame's level in view
    /// is read; with `read_ahead`, also everything read ahead: the ring and the planes
    /// next to this one), or `timeout` passes. For tests, screenshots and the harness;
    /// returns the last frame's statistics (a tile that failed to read counts as done).
    pub fn settle(
        &mut self,
        timeout: std::time::Duration,
        read_ahead: bool,
    ) -> Result<FrameStats, ViewerError> {
        let t0 = std::time::Instant::now();
        loop {
            let (_, stats) = self.prepare();
            let failed = self.opened.as_ref().map_or(0, |o| o.failed.len());
            let done = if read_ahead {
                stats.pending == 0 && !self.loader.has_results()
            } else {
                stats.missing == 0 || (failed > 0 && stats.pending == 0)
            };
            if done {
                return Ok(stats);
            }
            if t0.elapsed() > timeout {
                return Ok(stats);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// What lies under screen point (`x`, `y`): the world point, the level-0 pixel and
    /// every channel's value from the finest cached tile holding it (read back from the
    /// GPU; the frame's level first, then coarser, then finer).
    pub fn pick(&self, x: f64, y: f64) -> Result<Pick, ViewerError> {
        let c = &self.camera;
        let world = [c.origin[0] + x * c.scale, c.origin[1] + y * c.scale];
        let Some(o) = &self.opened else {
            return Ok(Pick {
                world,
                pixel: None,
                values: Vec::new(),
            });
        };
        let nearest = |l: usize| -> Option<(u64, u64)> {
            let lv = &o.source.levels[l];
            let (ix, iy) = lv.to_index.apply(world[0], world[1]);
            let (ix, iy) = ((ix + 0.5).floor(), (iy + 0.5).floor());
            (ix >= 0.0 && iy >= 0.0 && ix < lv.width as f64 && iy < lv.height as f64)
                .then_some((ix as u64, iy as u64))
        };
        let pixel = nearest(0).map(|(x, y)| [x, y]);
        let level = self.level();
        let nl = o.source.levels.len();
        let order: Vec<usize> = (level..nl).chain((0..level).rev()).collect();
        let mut values = Vec::new();
        for ch in 0..self.channels.len() {
            let mut found = PickedValue {
                channel: ch,
                value: None,
                level: None,
            };
            if pixel.is_some() {
                for &l in &order {
                    let Some((ix, iy)) = nearest(l) else { continue };
                    let k = TileKey {
                        plane: o.planes[l].1,
                        channel: ch as u16,
                        level: l as u8,
                        x: (ix / TILE as u64) as u32,
                        y: (iy / TILE as u64) as u32,
                    };
                    if let Some(slot) = o.cache.peek(&k) {
                        let bytes = self.gpu.read_texel(
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
        Ok(Pick {
            world,
            pixel,
            values,
        })
    }

    /// The view as a [`ViewState`] (serialises to JSON).
    pub fn view_state(&self) -> ViewState {
        ViewState::of(self)
    }

    /// Apply a view state: open its image when it names another one, then its camera,
    /// position and channels (fields it leaves out stay as they are).
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

    /// Atlas slots (tiles the GPU holds at once).
    pub fn atlas_slots(&self) -> u32 {
        self.gpu.slots()
    }
}
