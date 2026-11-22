//! micro-viewer's core (Workplan WP14): OME-Zarr images drawn by the GPU from a tile
//! cache, hosted by any toolkit that can hand over a native window, or read back
//! off-screen. No Qt and no Python here: the bindings and the widget live outside.
//!
//! The view (tasks V7, V9, V4): layers, each an image placed in a shared 3-D world by
//! its own levels and a transform of its own (moved and turned by hand, saved to the
//! image); a camera with a centre, a scale and an orientation, so the view is a plane at
//! any angle.
//!
//! - Flat: a 2-D image, or a volume seen along z (the plane cuts its z planes), is drawn
//!   from 2-D tiles of the plane: every level, the finest worth drawing chosen per frame
//!   from the zoom, coarser (or finer) cached tiles standing in while tiles load; tiles
//!   read in the background, centre first, a ring around and the next and previous plane
//!   when the cache has room.
//! - Volume: a volume cut at an angle (or seen along x or y) is drawn from bricks (3-D
//!   pieces, [`bricks`]) the plane cuts, nearest voxel or trilinear.
//!
//! Both read from coarser levels while the view moves when reading is slow. Channels with
//! colour or colour map, contrast and gamma, added up; layers added up; picking; the view
//! as JSON. z is a world coordinate shared by all layers (the camera's centre); a layer's
//! z index is where the camera's plane cuts it.

mod blocks;
mod bricks;
mod geom;
mod gpu;
mod loader;
mod luts;
mod source;
mod tiles;
mod view;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use geom::{add, cross, dot, mul, norm, sub, turn, unit};
pub use geom::{Affine3, Plane3, V3};
pub use gpu::GpuOptions;
use gpu::{ChannelGpu, Draw, FrameParams, Gpu, LevelGpu, LevelGpu3, LUT_SIZE, MISSING};
use loader::{Loaded, Loader, Request};
pub use luts::NAMES as COLOUR_MAPS;
pub use raw_window_handle;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
pub use source::{Affine2, ChannelInfo, Level, Source, Texel};
use tiles::{Cache, TileKey, TILE};
pub use view::{parse_transform, ViewState};

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
/// The most bricks a level's page table holds (per channel); finer levels of larger
/// volumes are not drawn as volumes (coarser ones stand in).
const BRICK_PAGES_MAX: u64 = 1 << 22;

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
    /// Tiles (or bricks) of the frame's levels in view (all active channels).
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

/// The camera: the world point at the target's centre, the world units per screen
/// pixel, and the screen's right and down directions in the world (unit vectors; the
/// camera looks along right x down).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub center: V3,
    pub scale: f64,
    pub right: V3,
    pub down: V3,
}

impl Camera {
    fn new() -> Self {
        Camera {
            center: [0.0; 3],
            scale: 1.0,
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
        }
    }

    /// The direction the camera looks along.
    pub fn normal(&self) -> V3 {
        cross(self.right, self.down)
    }

    /// The plane the screen shows: screen pixel (sx, sy) to world (x, y, z).
    pub fn plane(&self, w: u32, h: u32) -> Plane3 {
        let ux = mul(self.right, self.scale);
        let uy = mul(self.down, self.scale);
        Plane3 {
            origin: sub(
                self.center,
                add(mul(ux, w as f64 / 2.0), mul(uy, h as f64 / 2.0)),
            ),
            ux,
            uy,
        }
    }

    /// Screen pixel (x, y) to world (x, y) (exact for a view along z).
    pub fn screen_to_world(&self, w: u32, h: u32) -> Affine2 {
        self.plane(w, h).to_2d()
    }

    /// Whether the camera looks along z (any turn about z).
    pub fn is_flat(&self) -> bool {
        self.normal()[2].abs() > 1.0 - 1e-9
    }

    /// The turn about the view's normal, degrees (0 .. 360): the image appears turned
    /// clockwise by it (measured as for a view along z).
    pub fn rotation(&self) -> f64 {
        (-self.right[1])
            .atan2(self.right[0])
            .to_degrees()
            .rem_euclid(360.0)
            .min(360.0 - 1e-12)
    }

    /// Turn the image clockwise by `degrees` about the view's normal.
    fn roll(&mut self, degrees: f64) {
        let n = self.normal();
        self.right = turn(self.right, n, -degrees);
        self.down = turn(self.down, n, -degrees);
        self.tidy();
    }

    /// Turn the camera: `yaw` degrees about the screen's down axis, `pitch` degrees about
    /// its right axis.
    fn turn(&mut self, yaw: f64, pitch: f64) {
        self.right = turn(self.right, self.down, yaw);
        self.down = turn(self.down, self.right, pitch);
        self.tidy();
    }

    /// Unit, perpendicular directions again (drift from many small turns).
    fn tidy(&mut self) {
        self.right = unit(self.right);
        self.down = unit(sub(self.down, mul(self.right, dot(self.down, self.right))));
        // snap tiny components (a view along z stays exactly along z)
        for v in [&mut self.right, &mut self.down] {
            for c in v.iter_mut() {
                if c.abs() < 1e-12 {
                    *c = 0.0;
                }
            }
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
    /// For a volume: the level-0 voxel (x, y, z) whose centre is nearest, if inside.
    pub voxel: Option<[u64; 3]>,
    pub values: Vec<PickedValue>,
}

/// What lies under a screen point: the world point and every layer's pixel and values.
#[derive(Debug, Clone, PartialEq)]
pub struct Pick {
    pub world: V3,
    pub layers: Vec<LayerPick>,
}

/// How a layer is drawn for the current camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// From 2-D tiles (a 2-D image, or a volume seen along z).
    Flat,
    /// From bricks (a volume cut at an angle).
    Volume,
    /// Not drawn: a 2-D image seen at an angle.
    Off,
}

/// A volume's bricks: cache, page table and per-level layout.
struct Bricks {
    cache: Cache,
    /// Per channel, per level, per brick: slot + 1 or 0.
    pages: Vec<u32>,
    dirty: Option<(usize, usize)>,
    pages_all: bool,
    /// Per level: the lead indices (without z) and their id.
    leads: Vec<(Vec<u64>, u64)>,
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
    /// Level-0 indices on the image's plane axes (z among them for a volume: where the
    /// camera's plane cuts it).
    position: Vec<f64>,
    /// World to world: where the layer sits on top of its levels' placement.
    transform: Affine3,
    inverse: Affine3,
    /// Whether the layer is drawn.
    pub visible: bool,
    /// 0..1: how strongly the layer adds to the others.
    pub opacity: f32,
    /// Volumes: trilinear sampling between voxel centres (else the nearest voxel).
    pub linear: bool,
    /// Seconds per tile read, smoothed.
    read_s: f64,
    /// A volume's brick layout per level: bricks along x, y, z, first page entry, and
    /// whether the level has a page table.
    bgrid: Vec<([u32; 3], u32, bool)>,
    bstride: u32,
    /// The bricks, once a plane at an angle needed them.
    bricks: Option<Bricks>,
    /// Bricks could not be made (the error says why): not tried again.
    no_bricks: bool,
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

    /// Whether the image is a volume (it has a z axis).
    pub fn is_volume(&self) -> bool {
        self.source.z_axis.is_some()
    }

    /// The layer's transform (world to world, on top of its levels' placement).
    pub fn transform(&self) -> Affine3 {
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

    /// The position's index of the z axis, for a volume.
    fn z_slot(&self) -> Option<usize> {
        let z = self.source.z_axis?;
        self.source.plane_axes.iter().position(|&a| a == z)
    }

    /// Level `l`'s index (x, y, z) of a world point: the layer's transform undone, then
    /// the level's own placement.
    fn to_index(&self, l: usize) -> Affine3 {
        self.source.levels[l].to_index3.after(&self.inverse)
    }

    /// The world points of the finest level's outer corners, the transform applied (8 for
    /// a volume, 4 for a 2-D image, at z 0).
    fn corners(&self) -> Vec<V3> {
        let lv = &self.source.levels[0];
        let (w, h, d) = (
            lv.width as f64 - 0.5,
            lv.height as f64 - 0.5,
            lv.depth as f64 - 0.5,
        );
        let zs: &[f64] = if self.is_volume() { &[-0.5, d] } else { &[0.0] };
        let mut out = Vec::new();
        for &z in zs {
            for (x, y) in [(-0.5, -0.5), (w, -0.5), (-0.5, h), (w, h)] {
                out.push(self.transform.apply(lv.to_world3.apply([x, y, z])));
            }
        }
        out
    }

    /// The world box (x0, y0, x1, y1) the layer covers.
    pub fn bounds(&self) -> [f64; 4] {
        let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for p in self.corners() {
            b = [
                b[0].min(p[0]),
                b[1].min(p[1]),
                b[2].max(p[0]),
                b[3].max(p[1]),
            ];
        }
        b
    }

    /// The world point at the image's centre (the transform applied; z of the middle
    /// plane for a volume, 0 for a 2-D image).
    pub fn centre(&self) -> V3 {
        let lv = &self.source.levels[0];
        let z = if self.is_volume() {
            (lv.depth as f64 - 1.0) / 2.0
        } else {
            0.0
        };
        self.transform.apply(lv.to_world3.apply([
            (lv.width as f64 - 1.0) / 2.0,
            (lv.height as f64 - 1.0) / 2.0,
            z,
        ]))
    }

    /// World units per pixel of level `l`, the transform's plane scaling included.
    fn pixel(&self, l: usize) -> f64 {
        self.source.levels[l].pixel * self.transform.to_2d().det().abs().sqrt()
    }

    /// The level a flat frame at `scale` world units per screen pixel draws: the coarsest
    /// whose pixels are no larger than a screen pixel (the finest when zoomed in further).
    pub fn level_for(&self, scale: f64) -> usize {
        let s = scale * (1.0 + 1e-9);
        (0..self.source.levels.len())
            .filter(|&l| self.pixel(l) <= s)
            .max_by(|&a, &b| self.pixel(a).total_cmp(&self.pixel(b)))
            .unwrap_or(0)
    }

    /// How the layer is drawn by `camera`.
    pub fn mode(&self, camera: &Camera) -> Mode {
        if !self.is_volume() {
            return if camera.is_flat() {
                Mode::Flat
            } else {
                Mode::Off
            };
        }
        let a = self.to_index(0);
        // the camera's plane is one of the volume's z planes
        if camera.is_flat() && a.m[2][0].abs() < 1e-9 && a.m[2][1].abs() < 1e-9 {
            Mode::Flat
        } else {
            Mode::Volume
        }
    }

    /// The level a volume frame draws for the camera's plane `plane` (world): the
    /// coarsest whose voxels along the screen are no larger than a screen pixel, among
    /// the levels with a page table.
    fn volume_level(&self, plane: &Plane3) -> usize {
        let usable: Vec<usize> = (0..self.source.levels.len())
            .filter(|&l| self.bgrid[l].2)
            .collect();
        let step = |l: usize| {
            let p = plane.through(&self.to_index(l));
            norm(p.ux).max(norm(p.uy))
        };
        usable
            .iter()
            .copied()
            .filter(|&l| step(l) >= 1.0 - 1e-9)
            .max_by(|&a, &b| step(b).total_cmp(&step(a)))
            .or_else(|| usable.first().copied())
            .unwrap_or(0)
    }

    /// The transform as an NGFF affine over the image's axes: n rows of n + 1 (the last
    /// column the translation), the identity outside x, y (and z for a volume).
    pub fn transform_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.source.axes.len();
        let (y, x) = (n - 2, n - 1);
        let mut m: Vec<Vec<f64>> = (0..n)
            .map(|r| (0..=n).map(|c| (r == c) as u8 as f64).collect())
            .collect();
        let t = &self.transform.m;
        let mut spatial = vec![(x, 0usize), (y, 1usize)];
        if let Some(z) = self.source.z_axis {
            spatial.push((z, 2));
        }
        for &(ra, ri) in &spatial {
            for &(ca, ci) in &spatial {
                m[ra][ca] = t[ri][ci];
            }
            m[ra][n] = t[ri][3];
        }
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

    fn brick_entry(&self, k: &TileKey) -> usize {
        let (g, base, _) = self.bgrid[k.level as usize];
        k.channel as usize * self.bstride as usize
            + base as usize
            + ((k.z * g[1] + k.y) * g[0] + k.x) as usize
    }

    fn brick_current(&self, k: &TileKey) -> bool {
        self.bricks
            .as_ref()
            .is_some_and(|b| b.leads[k.level as usize].1 == k.plane)
    }

    fn set_brick_page(&mut self, k: &TileKey, v: u32) {
        let e = self.brick_entry(k);
        if let Some(b) = &mut self.bricks {
            b.pages[e] = v;
            b.dirty = Some(match b.dirty {
                Some((x, y)) => (x.min(e), y.max(e)),
                None => (e, e),
            });
        }
    }

    /// Every brick page entry again from the brick cache (the lead indices changed).
    fn rebuild_brick_pages(&mut self) {
        let Some(b) = &self.bricks else { return };
        let present: Vec<(TileKey, u32)> = b
            .cache
            .iter()
            .filter(|(k, _)| b.leads[k.level as usize].1 == k.plane)
            .map(|(k, s)| (*k, *s))
            .collect();
        let entries: Vec<(usize, u32)> = present
            .iter()
            .map(|(k, s)| (self.brick_entry(k), s + 1))
            .collect();
        if let Some(b) = &mut self.bricks {
            b.pages.iter_mut().for_each(|p| *p = 0);
            for (e, v) in entries {
                b.pages[e] = v;
            }
            b.pages_all = true;
        }
    }

    fn key(&self, c: usize, l: usize, (x, y): (u32, u32)) -> TileKey {
        TileKey {
            layer: self.id,
            brick: false,
            plane: self.planes[l].1,
            channel: c as u16,
            level: l as u8,
            x,
            y,
            z: 0,
        }
    }

    fn brick_key(&self, c: usize, l: usize, (x, y, z): (u32, u32, u32)) -> TileKey {
        TileKey {
            layer: self.id,
            brick: true,
            plane: self.bricks.as_ref().map_or(0, |b| b.leads[l].1),
            channel: c as u16,
            level: l as u8,
            x,
            y,
            z,
        }
    }

    fn missing(&self, k: &TileKey) -> bool {
        let cached = if k.brick {
            self.bricks
                .as_ref()
                .is_some_and(|b| b.cache.peek(k).is_some())
        } else {
            self.cache.peek(k).is_some()
        };
        !cached && !self.failed.contains(k)
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

    fn brick_request(&self, k: TileKey) -> Request {
        let lv = &self.source.levels[k.level as usize];
        let lead = &self.bricks.as_ref().expect("bricks").leads[k.level as usize].0;
        let (start, stop) = self.source.region3(
            lead,
            k.channel as usize,
            bricks::span(k.z, lv.depth),
            bricks::span(k.y, lv.height),
            bricks::span(k.x, lv.width),
        );
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

    fn channel_gpu(&self) -> Vec<ChannelGpu> {
        self.channels
            .iter()
            .map(|d| ChannelGpu {
                color: d.color,
                gamma: d.gamma as f32,
                lo: d.lo as f32,
                hi: d.hi as f32,
                active: d.visible as u32,
                lut: d.lut.is_some() as u32,
            })
            .collect()
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
    /// The view's size and its top-left corner in the target.
    tw: u32,
    th: u32,
    ox: f64,
    oy: f64,
    camera: Camera,
    moving: bool,
    threads: f64,
}

/// The most bricks a level may have to be projected (every brick is listed per frame).
const MIP_BRICKS: u64 = 1 << 16;

/// One level's placement for the brick shaders: the plane `p` (the view's own pixels)
/// moved to target pixels by the view's corner (`ox`, `oy`), the level's size, brick
/// layout, and the voxel change per world unit along the view's normal `dir`.
fn level_gpu3(
    p: &Plane3,
    ox: f64,
    oy: f64,
    s: [u64; 3],
    (g, base, usable): ([u32; 3], u32, bool),
    dir: V3,
) -> LevelGpu3 {
    let origin = p.at(-ox, -oy);
    let oi = origin.map(f64::floor);
    LevelGpu3 {
        origin_int: [oi[0] as i32, oi[1] as i32, oi[2] as i32, 0],
        origin_frac: [
            (origin[0] - oi[0]) as f32,
            (origin[1] - oi[1]) as f32,
            (origin[2] - oi[2]) as f32,
            0.0,
        ],
        ux: [p.ux[0] as f32, p.ux[1] as f32, p.ux[2] as f32, 0.0],
        uy: [p.uy[0] as f32, p.uy[1] as f32, p.uy[2] as f32, 0.0],
        size: [s[0] as i32, s[1] as i32, s[2] as i32, base as i32],
        grid: [g[0], g[1], g[2], usable as u32],
        dir: [dir[0] as f32, dir[1] as f32, dir[2] as f32, 0.0],
    }
}

/// Place tiles and bricks read since the last frame into their caches and atlases.
fn place_loaded(
    o: &mut Layer,
    gpu: &mut Gpu,
    loaded: Vec<Loaded>,
    frame: u64,
    stats: &mut FrameStats,
) {
    for t in loaded {
        o.read_s = if o.read_s == 0.0 {
            t.took
        } else {
            0.8 * o.read_s + 0.2 * t.took
        };
        let data = match t.data {
            Err(e) => {
                o.error.get_or_insert(e);
                o.failed.insert(t.key);
                continue;
            }
            Ok(d) => d,
        };
        if t.key.brick {
            let Some(b) = &mut o.bricks else { continue };
            let Some((slot, old)) = b.cache.place(t.key, frame) else {
                continue;
            };
            if let Some(old) = old {
                if o.brick_current(&old) {
                    o.set_brick_page(&old, 0);
                }
            }
            gpu.upload_brick(o.id, slot, &data);
            if o.brick_current(&t.key) {
                o.set_brick_page(&t.key, slot + 1);
            }
        } else {
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
        }
        stats.loaded += 1;
    }
}

/// The channels and colour maps for the shaders (shared by both kinds of drawing).
fn write_display(o: &mut Layer, gpu: &mut Gpu) {
    gpu.write_channels(o.id, &o.channel_gpu());
    if o.luts_dirty {
        gpu.write_luts(o.id, &o.lut_rows());
        o.luts_dirty = false;
    }
}

/// One layer's flat part of a frame: keep the cached tiles the view uses, update the
/// page table, and ask for what is still missing. Returns the layer's parameters when it
/// is drawn, and its level.
fn prepare_flat(
    o: &mut Layer,
    gpu: &mut Gpu,
    cx: &FrameCtx,
    wants: &mut Wants,
    stats: &mut FrameStats,
) -> (Option<FrameParams>, usize) {
    let frame = cx.frame;
    let nl = o.source.levels.len();
    let level = o.level_for(cx.camera.scale);
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
    // per level: screen pixel to level index (x, y)
    let plane = cx.camera.plane(cx.tw, cx.th);
    let to_index: Vec<Affine2> = (0..nl)
        .map(|l| plane.through(&o.to_index(l)).to_2d())
        .collect();
    let active: Vec<usize> = (0..o.channels.len())
        .filter(|&c| o.channels[c].visible)
        .collect();
    // a volume's z: the camera's plane must cut the volume
    let inside = match o.z_slot() {
        Some(k) => {
            let z = o.position[k];
            z >= -0.5 && z < o.source.levels[0].depth as f64 - 0.5
        }
        None => true,
    };
    let drawn = o.visible && !active.is_empty() && inside;
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
                        pos[k_axis] = pos[k_axis].round() + step;
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
                                    plane: id,
                                    ..o.key(c, level, t)
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
            let (ox, oy) = m.apply(-cx.ox, -cx.oy); // the view's corner in the target
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
    gpu.write_levels(o.id, &levels);
    write_display(o, gpu);
    let params = FrameParams {
        n_levels: nl as u32,
        n_channels: o.channels.len() as u32,
        level: level as u32,
        channel_stride: o.stride,
        missing: MISSING,
        outside: [0.0; 4],
        opacity: o.opacity.clamp(0.0, 1.0),
        linear: 0,
        _pad: [0.0; 2],
    };
    (Some(params), level)
}

/// One volume layer's part of a frame drawn from bricks: the bricks the camera's plane
/// cuts (or, for a projection `mip`, every brick whose centre projects into the view),
/// centre first, at the level the zoom wants (coarser when the brick atlas cannot hold
/// them); coarser levels stand in. Returns the parameters and the level.
fn prepare_volume(
    o: &mut Layer,
    gpu: &mut Gpu,
    cx: &FrameCtx,
    wants: &mut Wants,
    stats: &mut FrameStats,
    mip: bool,
) -> (Option<FrameParams>, usize) {
    let frame = cx.frame;
    let nl = o.source.levels.len();
    let plane = cx.camera.plane(cx.tw, cx.th);
    // the lead indices (t, ...) on every level
    let leads: Vec<(Vec<u64>, u64)> = (0..nl).map(|l| o.source.lead(l, &o.position)).collect();
    let changed = o
        .bricks
        .as_ref()
        .is_some_and(|b| b.leads.iter().zip(&leads).any(|(a, b)| a.1 != b.1));
    if let Some(b) = &mut o.bricks {
        b.leads = leads;
    }
    if changed {
        o.rebuild_brick_pages();
    }
    let planes: Vec<Plane3> = (0..nl).map(|l| plane.through(&o.to_index(l))).collect();
    let size = |l: usize| {
        let lv = &o.source.levels[l];
        [lv.width, lv.height, lv.depth]
    };
    let active: Vec<usize> = (0..o.channels.len())
        .filter(|&c| o.channels[c].visible)
        .collect();
    let capacity = o.bricks.as_ref().map_or(0, |b| b.cache.capacity()) as usize;
    // a projection lists every brick of a level: levels with few enough bricks
    let mip_fits = |l: usize| {
        o.bgrid[l].2 && o.bgrid[l].0.iter().map(|&v| v as u64).product::<u64>() <= MIP_BRICKS
    };
    let find = |l: usize| {
        if !mip {
            bricks::visible(&planes[l], cx.tw, cx.th, size(l))
        } else if mip_fits(l) {
            bricks::projected(&planes[l], cx.tw, cx.th, size(l))
        } else {
            Vec::new()
        }
    };
    let mut level = o.volume_level(&plane);
    while mip && level + 1 < nl && !mip_fits(level) {
        level += 1;
    }
    let mut views: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); nl];
    views[level] = find(level);
    // coarser while the frame's bricks would not fit the atlas
    while level + 1 < nl && views[level].len() * active.len().max(1) * 10 > capacity * 8 {
        level += 1;
        while level + 1 < nl
            && !(if mip {
                mip_fits(level)
            } else {
                o.bgrid[level].2
            })
        {
            level += 1;
        }
        views[level] = find(level);
    }
    let drawn = o.visible && !active.is_empty() && o.bricks.is_some();
    let mut missing_here = Vec::new();
    if drawn {
        for (l, view) in views.iter_mut().enumerate().skip(level + 1) {
            if o.bgrid[l].2 {
                *view = find(l);
            }
        }
        for &b in &views[level] {
            for &c in &active {
                stats.visible += 1;
                let k = o.brick_key(c, level, b);
                let hit = o
                    .bricks
                    .as_mut()
                    .and_then(|br| br.cache.touch(&k, frame))
                    .is_some();
                if !hit {
                    missing_here.push(k);
                }
            }
        }
        for (l, view) in views.iter().enumerate().skip(level + 1) {
            for &b in view {
                for &c in &active {
                    let k = o.brick_key(c, l, b);
                    if let Some(br) = &mut o.bricks {
                        br.cache.touch(&k, frame);
                    }
                }
            }
        }
        let threads = cx.threads.max(1.0);
        let cost = |n: usize| n as f64 * o.read_s / threads;
        let todo = missing_here.iter().filter(|k| o.missing(k)).count();
        if cx.moving && cost(todo) > MOVE_BUDGET {
            stats.deferred = true;
            let fetch = (level + 1..nl)
                .filter(|&l| o.bgrid[l].2)
                .find(|&l| cost(views[l].len() * active.len()) <= MOVE_BUDGET)
                .unwrap_or(nl - 1);
            for &b in &views[fetch] {
                for &c in &active {
                    let k = o.brick_key(c, fetch, b);
                    if o.missing(&k) {
                        wants.here.push(o.brick_request(k));
                    }
                }
            }
        } else {
            let coarse = (level + 2).min(nl - 1);
            if coarse != level && o.bgrid[coarse].2 {
                for &b in &views[coarse] {
                    for &c in &active {
                        let k = o.brick_key(c, coarse, b);
                        if o.missing(&k) {
                            wants.coarse.push(o.brick_request(k));
                        }
                    }
                }
            }
            for k in &missing_here {
                if o.missing(k) {
                    wants.here.push(o.brick_request(*k));
                }
            }
        }
        stats.missing += missing_here
            .iter()
            .filter(|k| o.bricks.as_ref().is_some_and(|b| b.cache.peek(k).is_none()))
            .count() as u32;
    }
    // the brick page table, levels, channels and colour maps for the shader
    let id = o.id;
    let bstride = o.bstride;
    if let Some(b) = &mut o.bricks {
        if b.pages_all {
            gpu.write_brick_pages(id, b.pages.len(), 0, &b.pages);
            b.pages_all = false;
            b.dirty = None;
        } else if let Some((x, y)) = b.dirty.take() {
            if gpu.write_brick_pages(id, b.pages.len(), x, &b.pages[x..=y]) {
                gpu.write_brick_pages(id, b.pages.len(), 0, &b.pages);
            }
        }
    }
    if !drawn {
        return (None, level);
    }
    let normal = cx.camera.normal();
    let levels: Vec<LevelGpu3> = (0..nl)
        .map(|l| {
            let dir = o.to_index(l).linear(normal);
            level_gpu3(&planes[l], cx.ox, cx.oy, size(l), o.bgrid[l], dir)
        })
        .collect();
    gpu.write_brick_levels(id, &levels);
    write_display(o, gpu);
    let params = FrameParams {
        n_levels: nl as u32,
        n_channels: o.channels.len() as u32,
        level: level as u32,
        channel_stride: bstride,
        missing: MISSING,
        outside: [0.0; 4],
        opacity: o.opacity.clamp(0.0, 1.0),
        linear: o.linear as u32,
        _pad: [0.0; 2],
    };
    (Some(params), level)
}

/// The spatial part of an NGFF edge's matrix over `n` axes as a 3-D transform, if it
/// leaves the other axes alone (x and y only for a 2-D image, x, y and z for a volume).
fn space_transform(m: &[Vec<f64>], n: usize, z_axis: Option<usize>) -> Option<Affine3> {
    let (y, x) = (n - 2, n - 1);
    let spatial = |a: usize| a == x || a == y || Some(a) == z_axis;
    for (r, row) in m.iter().enumerate().take(n) {
        for (c, &v) in row.iter().enumerate().take(n + 1) {
            let inside = spatial(r) && (c == n || spatial(c));
            let want = if r == c { 1.0 } else { 0.0 };
            if !inside && (v - want).abs() > 1e-9 {
                return None;
            }
        }
    }
    let axes: Vec<Option<usize>> = vec![Some(x), Some(y), z_axis];
    let mut t = Affine3::IDENTITY;
    for (ri, ra) in axes.iter().enumerate() {
        let Some(ra) = *ra else { continue };
        for (ci, ca) in axes.iter().enumerate() {
            if let Some(ca) = *ca {
                t.m[ri][ci] = m[ra][ca];
            }
        }
        t.m[ri][3] = m[ra][n];
    }
    Some(t)
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
    /// Linked views: the main view, its YZ and XZ companions and the projection, in
    /// quarters of the target.
    linked: bool,
    /// The main view shows the projection (maximum intensity) instead of the plane.
    mip: bool,
}

/// One view of the target: where it sits (None: the whole target), its size, camera, and
/// whether it shows the projection.
#[derive(Debug, Clone, Copy)]
struct View {
    rect: Option<[u32; 4]>,
    w: u32,
    h: u32,
    camera: Camera,
    mip: bool,
    /// Drawn smaller than its place: the place (x, y, width, height) it is scaled into.
    place: Option<[u32; 4]>,
}

/// The pixels a projection is drawn with at most (then scaled into its place), still and
/// while the view moves: its rays are long.
const MIP_PIXELS: f64 = (1 << 20) as f64;
const MIP_PIXELS_MOVING: f64 = (1 << 18) as f64;

impl View {
    fn corner(&self) -> (f64, f64) {
        self.rect.map_or((0.0, 0.0), |r| (r[0] as f64, r[1] as f64))
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        match self.rect {
            None => true,
            Some([rx, ry, rw, rh]) => {
                x >= rx as f64 && y >= ry as f64 && x < (rx + rw) as f64 && y < (ry + rh) as f64
            }
        }
    }
}

/// The gap between linked views, in target pixels.
const GAP_PX: u32 = 2;

/// The companions of a view (linked views): `which` 0 turns it to look across its
/// screen's down axis (XZ for a view along z), 1 across its right axis (YZ).
fn companion(c: &Camera, which: u8) -> Camera {
    let mut out = *c;
    if which == 0 {
        out.down = turn(c.down, c.right, 90.0);
    } else {
        out.right = turn(c.right, c.down, -90.0);
    }
    out.tidy();
    out
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
            camera: Camera::new(),
            frame: 0,
            moved_at: None,
            moved_frame: 0,
            moved_before: None,
            moved_before_frame: 0,
            deferred: false,
            linked: false,
            mip: false,
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

    /// The brick slots a volume of `channels` channels wants for this target: the plane
    /// cuts about two bricks deep per screen-sized patch of bricks, the coarser levels a
    /// third more.
    fn brick_slots_wanted(&self, channels: usize) -> u32 {
        let (w, h) = self.gpu.size();
        let step = bricks::STEP as u32;
        let per = (w / step + 2) * (h / step + 2) * 2;
        (per as f64 * 1.34 * channels.max(1) as f64 * 1.2) as u32
    }

    /// A layer's share of the atlas budget (MiB): by channels.
    fn budget_mib(&self, channels: usize) -> u32 {
        let total: usize = self.layers.iter().map(|l| l.channels.len()).sum::<usize>();
        let total = total.max(channels).max(1);
        ((self.gpu.atlas_mib() as u64 * channels.max(1) as u64 / total as u64) as u32).max(16)
    }

    /// Give each layer the atlases its share allows (a volume with bricks splits it
    /// between tiles and bricks): larger when the target grew or a layer left, smaller
    /// when a layer came (their contents are read again). A layer the GPU cannot hold is
    /// removed.
    fn rebalance(&mut self) -> Result<(), ViewerError> {
        let mut i = 0;
        while i < self.layers.len() {
            let (id, texel, nc, have, bricks_have) = {
                let l = &self.layers[i];
                (
                    l.id,
                    l.source.texel,
                    l.channels.len(),
                    l.cache.capacity(),
                    l.bricks.as_ref().map(|b| b.cache.capacity()),
                )
            };
            let budget = self.budget_mib(nc);
            let tiles_budget = if bricks_have.is_some() {
                (budget / 2).max(16)
            } else {
                budget
            };
            let want = self
                .gpu
                .slots_for(texel, self.slots_wanted(nc), tiles_budget);
            let cap = self.gpu.slots_for(texel, u32::MAX, tiles_budget);
            if want > have || have > cap {
                match self.gpu.prepare_layer(id, texel, want, tiles_budget) {
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
            if let Some(bhave) = bricks_have {
                let bbudget = budget.saturating_sub(tiles_budget).max(16);
                let want = self
                    .gpu
                    .brick_slots_for(texel, self.brick_slots_wanted(nc), bbudget);
                let cap = self.gpu.brick_slots_for(texel, u32::MAX, bbudget);
                if want > bhave || bhave > cap {
                    match self.gpu.prepare_bricks(id, want, bbudget) {
                        Ok(slots) => {
                            let l = &mut self.layers[i];
                            if let Some(b) = &mut l.bricks {
                                b.cache = Cache::new(slots);
                            }
                            l.rebuild_brick_pages();
                        }
                        Err(e) => {
                            // the layer stays: without bricks it is drawn along z only
                            let l = &mut self.layers[i];
                            l.bricks = None;
                            l.no_bricks = true;
                            l.error.get_or_insert(format!(
                                "no bricks for planes at an angle: {}",
                                e.0
                            ));
                        }
                    }
                }
            }
            i += 1;
        }
        Ok(())
    }

    /// Give layer `i` its bricks (the first time a plane at an angle needs them).
    fn enable_bricks(&mut self, i: usize) -> Result<(), ViewerError> {
        let l = &mut self.layers[i];
        if l.bricks.is_some() || !l.is_volume() {
            return Ok(());
        }
        let nc = l.channels.len();
        let nl = l.source.levels.len();
        l.bricks = Some(Bricks {
            cache: Cache::new(0),
            pages: vec![0; l.bstride as usize * nc],
            dirty: None,
            pages_all: true,
            leads: (0..nl).map(|lv| l.source.lead(lv, &l.position)).collect(),
        });
        self.rebalance()
    }

    /// Open the image at `path` (a folder, or a .zip / .ozx) as the only layer: fit it
    /// into the target, looking along z.
    pub fn open(&mut self, path: &Path) -> Result<(), ViewerError> {
        while !self.layers.is_empty() {
            self.remove(self.layers.len() - 1);
        }
        self.camera = Camera::new();
        self.add(path)?;
        self.fit();
        Ok(())
    }

    /// Add the image at `path` as a new layer on top (channels from its metadata:
    /// contrast from omero's window, else from the coarsest level's middle tile; the
    /// plane at omero's default z and t; the transform saved in its "aligned" coordinate
    /// system, if any). The first layer is fitted into the target (and the camera's
    /// plane put at its default z). It becomes the current layer; returns its index.
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
        let mut bgrid = Vec::new();
        let mut bstride = 0u32;
        for lv in &source.levels {
            let g = (
                lv.width.div_ceil(TILE as u64) as u32,
                lv.height.div_ceil(TILE as u64) as u32,
            );
            level_base.push(stride);
            stride += g.0 * g.1;
            grid.push(g);
            let b = [
                bricks::count(lv.width),
                bricks::count(lv.height),
                bricks::count(lv.depth),
            ];
            let n = b.iter().map(|&v| v as u64).product::<u64>();
            let usable = source.z_axis.is_some() && n <= BRICK_PAGES_MAX;
            bgrid.push((b, bstride, usable));
            if usable {
                bstride += n as u32;
            }
        }
        let position = source.default_position.clone();
        let planes = (0..source.levels.len())
            .map(|l| source.plane(l, &position))
            .collect();
        let channels = Self::initial_display(&source, &position);
        let pages = vec![0u32; stride as usize * nc];
        let first = self.layers.is_empty();
        let layer = Layer {
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
            inverse: transform.inverse().unwrap_or(Affine3::IDENTITY),
            visible: true,
            opacity: 1.0,
            linear: false,
            read_s: 0.0,
            bgrid,
            bstride,
            bricks: None,
            no_bricks: false,
        };
        if first {
            // the camera's plane where the image opens (its default z)
            let lv = &layer.source.levels[0];
            let z = layer.z_slot().map_or(0.0, |k| layer.position[k]);
            self.camera.center = layer.transform.apply(lv.to_world3.apply([
                (lv.width as f64 - 1.0) / 2.0,
                (lv.height as f64 - 1.0) / 2.0,
                z,
            ]));
        }
        self.layers.push(layer);
        self.current = self.layers.len() - 1;
        // the others give up atlas room for the new one
        self.rebalance()?;
        if first {
            self.fit();
        }
        self.sync_z();
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
    /// channels (when their number is the same), plane, visibility, opacity and sampling
    /// stay.
    pub fn reload(&mut self, i: usize) -> Result<(), ViewerError> {
        let (path, channels, position, visible, opacity, linear) = {
            let l = self.layer(i)?;
            (
                l.path.clone(),
                l.channels.clone(),
                l.position(),
                l.visible,
                l.opacity,
                l.linear,
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
        let z = self.layers[i]
            .source
            .z_axis
            .map(|a| self.layers[i].source.axes[a].clone());
        for (axis, p, _) in position {
            if Some(&axis) != z.as_ref() {
                let _ = self.set_position(i, &axis, p);
            }
        }
        self.layers[i].linear = linear;
        self.sync_z();
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

    /// Volumes: sample layer `i` trilinearly between voxel centres (`linear`), or take
    /// the nearest voxel.
    pub fn set_linear(&mut self, i: usize, linear: bool) -> Result<(), ViewerError> {
        self.layer_mut(i)?.linear = linear;
        Ok(())
    }

    /// How layer `i` is drawn for the current camera.
    pub fn mode(&self, i: usize) -> Result<Mode, ViewerError> {
        Ok(self.layer(i)?.mode(&self.camera))
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

    /// Move layer `i`'s plane to level-0 index `index` on axis `axis` (z, t, ...). For a
    /// volume's z, the camera's plane moves there (z is shared by all layers).
    pub fn set_position(&mut self, i: usize, axis: &str, index: f64) -> Result<(), ViewerError> {
        let o = self.layer(i)?;
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
        if o.z_slot() == Some(k) {
            // the camera's centre goes to that plane, where it cuts the layer now
            let to_index = o.to_index(0);
            let mut p = to_index.apply(self.camera.center);
            p[2] = index;
            let world = o.transform.apply(o.source.levels[0].to_world3.apply(p));
            self.camera.center = world;
        } else {
            self.layer_mut(i)?.position[k] = index;
        }
        self.moved();
        Ok(())
    }

    /// Every volume's z index from the camera's centre (z is a world coordinate).
    fn sync_z(&mut self) {
        let c = self.camera.center;
        for l in &mut self.layers {
            if let Some(k) = l.z_slot() {
                let z = l.to_index(0).apply(c)[2];
                // a hair off an index (from turning maps both ways) is that index
                let r = z.round();
                l.position[k] = if (z - r).abs() < 1e-9 { r } else { z };
            }
        }
    }

    /// Move the camera's plane by `steps` voxels of the current layer's finest level
    /// along the view's normal (in a view along z: one z plane per step).
    pub fn step_depth(&mut self, steps: f64) -> Result<(), ViewerError> {
        self.step_along(self.camera.normal(), steps)
    }

    /// Move the camera's centre by `steps` voxels of the current layer along `normal`.
    fn step_along(&mut self, normal: V3, steps: f64) -> Result<(), ViewerError> {
        let l = self.layer(self.current)?;
        let v = l.to_index(0).linear(normal);
        let most = v.iter().fold(0.0f64, |m, c| m.max(c.abs()));
        if most == 0.0 {
            return Ok(());
        }
        self.camera.center = add(self.camera.center, mul(normal, steps / most));
        self.moved();
        Ok(())
    }

    /// Set layer `i`'s transform (world to world, on top of its levels' placement). A
    /// 2-D image moves and turns in its plane only.
    pub fn set_transform(&mut self, i: usize, t: Affine3) -> Result<(), ViewerError> {
        let inverse = t
            .inverse()
            .filter(|_| t.is_finite())
            .ok_or_else(|| verr(format!("a transform that cannot be inverted: {:?}", t.m)))?;
        let l = self.layer_mut(i)?;
        if !l.is_volume() && !t.keeps_planes(1e-9) {
            return Err(verr(
                "a 2-D image moves and turns in its plane only (this transform tilts it)",
            ));
        }
        l.transform = t;
        l.inverse = inverse;
        self.moved();
        Ok(())
    }

    /// Move layer `i` by `dx`, `dy` screen pixels (it follows the pointer).
    pub fn move_layer(&mut self, i: usize, dx: f64, dy: f64) -> Result<(), ViewerError> {
        let c = &self.camera;
        let mut d = mul(add(mul(c.right, dx), mul(c.down, dy)), c.scale);
        let l = self.layer(i)?;
        if !l.is_volume() {
            d[2] = 0.0;
        }
        let t = Affine3::translation(d).after(&l.transform);
        self.set_transform(i, t)
    }

    /// Turn layer `i` by `degrees` about the view's normal (clockwise on screen) through
    /// world point `about`, by default the image's centre. A 2-D image turns about z.
    pub fn rotate_layer(
        &mut self,
        i: usize,
        degrees: f64,
        about: Option<V3>,
    ) -> Result<(), ViewerError> {
        let l = self.layer(i)?;
        let c = about.unwrap_or_else(|| l.centre());
        let axis = if l.is_volume() {
            self.camera.normal()
        } else {
            [0.0, 0.0, 1.0]
        };
        let t = Affine3::rotation(axis, degrees, c).after(&l.transform);
        self.set_transform(i, t)
    }

    /// Tilt volume layer `i`: `yaw` degrees about the screen's down axis and `pitch`
    /// degrees about its right axis, through the image's centre.
    pub fn turn_layer(&mut self, i: usize, yaw: f64, pitch: f64) -> Result<(), ViewerError> {
        let l = self.layer(i)?;
        if !l.is_volume() {
            return Err(verr("a 2-D image turns in its plane only (rotate_layer)"));
        }
        let c = l.centre();
        let t = Affine3::rotation(self.camera.right, pitch, c)
            .after(&Affine3::rotation(self.camera.down, yaw, c))
            .after(&l.transform);
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

    /// Show every visible layer the camera draws whole, centred (the orientation and the
    /// plane's depth stay).
    pub fn fit(&mut self) {
        if self.linked {
            return self.fit_linked();
        }
        let (tw, th) = self.gpu.size();
        let c = self.camera;
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for l in self.layers.iter().filter(|l| l.visible) {
            if l.mode(&c) == Mode::Off {
                continue;
            }
            for p in l.corners() {
                let d = sub(p, c.center);
                let (u, v) = (dot(d, c.right), dot(d, c.down)); // as the screen sees it
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
        self.camera.center = add(
            c.center,
            add(mul(c.right, (x0 + x1) / 2.0), mul(c.down, (y0 + y1) / 2.0)),
        );
        self.camera.scale = scale;
        self.moved();
    }

    /// Linked views: centre on the visible layers' middle and choose the scale that
    /// shows them whole in every view.
    fn fit_linked(&mut self) {
        let corners: Vec<V3> = self
            .layers
            .iter()
            .filter(|l| l.visible)
            .flat_map(|l| l.corners())
            .collect();
        if corners.is_empty() {
            return;
        }
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for p in &corners {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let centre = [
            (lo[0] + hi[0]) / 2.0,
            (lo[1] + hi[1]) / 2.0,
            (lo[2] + hi[2]) / 2.0,
        ];
        self.camera.center = centre;
        let mut scale: f64 = 1e-12;
        for v in self.views().iter().filter(|v| !v.mip) {
            let (mut u0, mut v0, mut u1, mut v1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for p in &corners {
                let d = sub(*p, centre);
                let (u, w) = (dot(d, v.camera.right), dot(d, v.camera.down));
                u0 = u0.min(u);
                v0 = v0.min(w);
                u1 = u1.max(u);
                v1 = v1.max(w);
            }
            scale = scale
                .max(2.0 * u0.abs().max(u1.abs()) / v.w as f64)
                .max(2.0 * v0.abs().max(v1.abs()) / v.h as f64);
        }
        self.camera.scale = scale;
        self.moved();
    }

    /// The world point (x, y) at the target's centre.
    pub fn center(&self) -> [f64; 2] {
        [self.camera.center[0], self.camera.center[1]]
    }

    /// Put world point `center` (x, y) at the target's centre, `scale` world units per
    /// pixel (the plane's depth stays).
    pub fn look_at(&mut self, center: [f64; 2], scale: f64) {
        self.camera.center[0] = center[0];
        self.camera.center[1] = center[1];
        self.camera.scale = scale;
        self.moved();
    }

    /// Put world point `center` (x, y, z) at the target's centre, `scale` world units per
    /// pixel.
    pub fn look_at_3d(&mut self, center: V3, scale: f64) {
        self.camera.center = center;
        self.camera.scale = scale;
        self.moved();
    }

    /// The world point (x, y) at screen pixel (`x`, `y`).
    pub fn screen_to_world(&self, x: f64, y: f64) -> [f64; 2] {
        let p = self.screen_to_world_3d(x, y);
        [p[0], p[1]]
    }

    /// The world point (x, y, z) at screen pixel (`x`, `y`).
    pub fn screen_to_world_3d(&self, x: f64, y: f64) -> V3 {
        let (v, lx, ly) = self.view_at(x, y);
        v.camera.plane(v.w, v.h).at(lx, ly)
    }

    /// Move the view under target pixel (`x`, `y`) by `dx`, `dy` screen pixels (in
    /// linked views every view follows: they share their centre).
    pub fn pan_at(&mut self, x: f64, y: f64, dx: f64, dy: f64) {
        let (v, _, _) = self.view_at(x, y);
        let c = &v.camera;
        let d = mul(add(mul(c.right, dx), mul(c.down, dy)), c.scale);
        self.camera.center = sub(self.camera.center, d);
        self.moved();
    }

    /// Move the plane of the view under target pixel (`x`, `y`) by `steps` voxels of the
    /// current layer along that view's normal.
    pub fn step_depth_at(&mut self, x: f64, y: f64, steps: f64) -> Result<(), ViewerError> {
        let (v, _, _) = self.view_at(x, y);
        self.step_along(v.camera.normal(), steps)
    }

    /// Move layer `i` by `dx`, `dy` screen pixels of the view under target pixel (`x`,
    /// `y`).
    pub fn move_layer_at(
        &mut self,
        i: usize,
        x: f64,
        y: f64,
        dx: f64,
        dy: f64,
    ) -> Result<(), ViewerError> {
        let (v, _, _) = self.view_at(x, y);
        let c = &v.camera;
        let mut d = mul(add(mul(c.right, dx), mul(c.down, dy)), c.scale);
        let l = self.layer(i)?;
        if !l.is_volume() {
            d[2] = 0.0;
        }
        let t = Affine3::translation(d).after(&l.transform);
        self.set_transform(i, t)
    }

    /// Move the view by `dx`, `dy` screen pixels (the image follows the pointer).
    pub fn pan(&mut self, dx: f64, dy: f64) {
        let c = &self.camera;
        let d = mul(add(mul(c.right, dx), mul(c.down, dy)), c.scale);
        self.camera.center = sub(self.camera.center, d);
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
        let before = self.screen_to_world_3d(x, y);
        self.camera.scale = (self.camera.scale / factor).clamp(lo, hi.max(lo));
        let after = self.screen_to_world_3d(x, y);
        self.camera.center = add(self.camera.center, sub(before, after));
        self.moved();
    }

    /// Turn the view by `degrees` about its normal (the image turns clockwise) keeping
    /// the world point under screen (`x`, `y`).
    pub fn rotate(&mut self, degrees: f64, x: f64, y: f64) {
        let before = self.screen_to_world_3d(x, y);
        self.camera.roll(degrees);
        let after = self.screen_to_world_3d(x, y);
        self.camera.center = add(self.camera.center, sub(before, after));
        self.moved();
    }

    /// Turn the view's plane: `yaw` degrees about the screen's down axis and `pitch`
    /// degrees about its right axis, keeping the world point under screen (`x`, `y`).
    pub fn turn(&mut self, yaw: f64, pitch: f64, x: f64, y: f64) {
        let before = self.screen_to_world_3d(x, y);
        self.camera.turn(yaw, pitch);
        let after = self.screen_to_world_3d(x, y);
        self.camera.center = add(self.camera.center, sub(before, after));
        self.moved();
    }

    /// Set the view's orientation: the screen's right and down directions in the world
    /// (made unit and perpendicular); the world point at the centre stays.
    pub fn set_orientation(&mut self, right: V3, down: V3) -> Result<(), ViewerError> {
        if norm(cross(right, down)) < 1e-9 || !right.iter().chain(&down).all(|v| v.is_finite()) {
            return Err(verr(format!(
                "an orientation needs two directions that are not parallel: {right:?}, {down:?}"
            )));
        }
        self.camera.right = right;
        self.camera.down = down;
        self.camera.tidy();
        self.moved();
        Ok(())
    }

    /// Look along a named axis: "xy" (along z), "xz" (x right, z down) or "yz" (z right,
    /// y down); the world point at the centre stays.
    pub fn look_along(&mut self, name: &str) -> Result<(), ViewerError> {
        let (r, d) = match name {
            "xy" => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            "xz" => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            "yz" => ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            _ => {
                return Err(verr(format!(
                    "view {name:?}: there are \"xy\", \"xz\" and \"yz\""
                )))
            }
        };
        self.set_orientation(r, d)
    }

    fn moved(&mut self) {
        if self.moved_frame != self.frame {
            self.moved_before_frame = self.moved_frame;
            self.moved_before = self.moved_at;
            self.moved_frame = self.frame;
        }
        self.moved_at = Some(Instant::now());
        self.sync_z();
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

    /// The current layer's level at the current zoom (and orientation).
    pub fn level(&self) -> usize {
        let Some(l) = self.layers.get(self.current) else {
            return 0;
        };
        if l.mode(&self.camera) == Mode::Volume {
            let (tw, th) = self.gpu.size();
            l.volume_level(&self.camera.plane(tw, th))
        } else {
            l.level_for(self.camera.scale)
        }
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

    /// The views of the target: the main view alone, or linked views (the main view top
    /// left, its YZ companion top right, its XZ companion bottom left, the projection
    /// bottom right).
    fn views(&self) -> Vec<View> {
        let (w, h) = self.gpu.size();
        if !self.linked || w < 2 * GAP_PX + 2 || h < 2 * GAP_PX + 2 {
            return vec![View {
                rect: None,
                w,
                h,
                camera: self.camera,
                mip: self.mip,
                place: None,
            }];
        }
        let (qw, qh) = ((w - GAP_PX) / 2, (h - GAP_PX) / 2);
        let (rw, rh) = (w - qw - GAP_PX, h - qh - GAP_PX);
        let view = |x: u32, y: u32, vw: u32, vh: u32, camera: Camera, mip: bool| View {
            rect: Some([x, y, vw, vh]),
            w: vw,
            h: vh,
            camera,
            mip,
            place: None,
        };
        vec![
            view(0, 0, qw, qh, self.camera, false),
            view(qw + GAP_PX, 0, rw, qh, companion(&self.camera, 1), false),
            view(0, qh + GAP_PX, qw, rh, companion(&self.camera, 0), false),
            view(qw + GAP_PX, qh + GAP_PX, rw, rh, self.camera, true),
        ]
    }

    /// The views as drawn: a projection with more pixels than it may have is drawn
    /// smaller (fewer while the view moves) and scaled into its place.
    fn drawn_views(&self) -> Vec<View> {
        let budget = if self.moving() {
            MIP_PIXELS_MOVING
        } else {
            MIP_PIXELS
        };
        self.views()
            .into_iter()
            .map(|v| {
                let f = ((v.w as f64 * v.h as f64) / budget).sqrt();
                if !v.mip || f <= 1.0 {
                    return v;
                }
                let (w, h) = (
                    ((v.w as f64 / f).ceil() as u32).max(1),
                    ((v.h as f64 / f).ceil() as u32).max(1),
                );
                let mut camera = v.camera;
                camera.scale *= v.w as f64 / w as f64;
                View {
                    rect: None,
                    w,
                    h,
                    camera,
                    mip: true,
                    place: Some(v.rect.unwrap_or([0, 0, v.w, v.h])),
                }
            })
            .collect()
    }

    /// The view under target pixel (`x`, `y`), and the point in its own pixels.
    fn view_at(&self, x: f64, y: f64) -> (View, f64, f64) {
        let views = self.views();
        let v = views
            .iter()
            .copied()
            .find(|v| v.contains(x, y))
            .unwrap_or(views[0]);
        let (ox, oy) = v.corner();
        (v, x - ox, y - oy)
    }

    /// Whether the views are linked (see [`Viewer::set_linked`]).
    pub fn linked(&self) -> bool {
        self.linked
    }

    /// Linked views: the main view, its YZ and XZ companions (all three centred on the
    /// same world point, with a crosshair) and the projection, in quarters of the target.
    pub fn set_linked(&mut self, linked: bool) {
        self.linked = linked;
        self.moved();
    }

    /// Whether the main view shows the projection (maximum intensity along the view).
    pub fn mip(&self) -> bool {
        self.mip
    }

    /// Show the projection (maximum intensity along the view through each volume) in the
    /// main view instead of the plane.
    pub fn set_mip(&mut self, mip: bool) {
        self.mip = mip;
        self.moved();
    }

    /// How layer `l` is drawn in view `v`.
    fn mode_in(l: &Layer, v: &View) -> Mode {
        if v.mip {
            if l.is_volume() {
                Mode::Volume
            } else {
                Mode::Off
            }
        } else {
            l.mode(&v.camera)
        }
    }

    /// The start of a frame: bricks for the volumes the views cut at an angle, then the
    /// tiles and bricks read since the last frame into their atlases.
    fn begin_frame(&mut self, views: &[View], stats: &mut FrameStats) {
        self.frame += 1;
        for i in 0..self.layers.len() {
            let l = &self.layers[i];
            let needs = l.visible
                && l.bricks.is_none()
                && !l.no_bricks
                && views.iter().any(|v| Self::mode_in(l, v) == Mode::Volume);
            if needs {
                if let Err(e) = self.enable_bricks(i) {
                    if let Some(l) = self.layers.get_mut(i) {
                        l.error.get_or_insert(e.0);
                    }
                }
            }
        }
        let mut results = self.loader.take();
        let frame = self.frame;
        for l in self.layers.iter_mut() {
            let (mine, rest): (Vec<Loaded>, Vec<Loaded>) =
                results.into_iter().partition(|t| t.key.layer == l.id);
            results = rest;
            place_loaded(l, &mut self.gpu, mine, frame, stats);
        }
    }

    /// Every layer's part of view `v` (flat, plane from bricks, or projection); its
    /// requests join `wants`. The main view (`main`) gives the frame's level.
    fn prepare_view(
        &mut self,
        v: &View,
        main: bool,
        wants: &mut Wants,
        stats: &mut FrameStats,
    ) -> Vec<Draw> {
        let (ox, oy) = v.corner();
        let cx = FrameCtx {
            frame: self.frame,
            tw: v.w,
            th: v.h,
            ox,
            oy,
            camera: v.camera,
            moving: self.moving(),
            threads: self.threads as f64,
        };
        let mut draws = Vec::new();
        for (i, l) in self.layers.iter_mut().enumerate() {
            let mode = Self::mode_in(l, v);
            let (params, level) = match mode {
                Mode::Flat => prepare_flat(l, &mut self.gpu, &cx, wants, stats),
                Mode::Volume => prepare_volume(l, &mut self.gpu, &cx, wants, stats, v.mip),
                Mode::Off => (None, 0),
            };
            if main && i == self.current {
                stats.level = level as u32;
            }
            if let Some(p) = params {
                draws.push(match mode {
                    Mode::Volume if v.mip => Draw::Mip(l.id, p),
                    Mode::Volume => Draw::Volume(l.id, p),
                    _ => Draw::Flat(l.id, p),
                });
            }
        }
        draws
    }

    /// The end of a frame's preparation: every view's requests to the readers, in
    /// priority classes.
    fn end_frame(&mut self, wants: Wants, stats: &mut FrameStats) {
        let mut all = wants.coarse;
        all.extend(wants.here);
        all.extend(wants.ring);
        all.extend(wants.planes);
        self.loader.want(all);
        stats.pending = self.loader.pending() as u32;
        self.deferred = stats.deferred;
    }

    /// A frame's preparation without drawing (for settle).
    fn prepare(&mut self) -> FrameStats {
        let views = self.drawn_views();
        let mut stats = FrameStats::default();
        let mut wants = Wants::default();
        self.begin_frame(&views, &mut stats);
        for (k, v) in views.iter().enumerate() {
            self.prepare_view(v, k == 0, &mut wants, &mut stats);
        }
        self.end_frame(wants, &mut stats);
        stats
    }

    /// Prepare and draw a frame, view by view; present it, or read it back into `out`.
    fn draw_frame(&mut self, out: Option<&mut [u8]>) -> Result<FrameStats, ViewerError> {
        let views = self.drawn_views();
        let mut stats = FrameStats::default();
        let mut wants = Wants::default();
        self.begin_frame(&views, &mut stats);
        let target = self.gpu.begin()?;
        for (k, v) in views.iter().enumerate() {
            let draws = self.prepare_view(v, k == 0, &mut wants, &mut stats);
            if let Some(f) = &target {
                let cross = (self.linked && !v.mip).then(|| {
                    let (ox, oy) = v.corner();
                    [
                        (ox + v.w as f64 / 2.0) as f32,
                        (oy + v.h as f64 / 2.0) as f32,
                    ]
                });
                match v.place {
                    Some(place) => {
                        let small = self.gpu.scratch_frame(v.w, v.h);
                        self.gpu.draw_view(&small, &draws, None, true, None);
                        self.gpu.blit(f, place, k == 0);
                    }
                    None => self.gpu.draw_view(f, &draws, v.rect, k == 0, cross),
                }
            }
        }
        self.end_frame(wants, &mut stats);
        if let Some(f) = target {
            if views.len() > 1 {
                self.gpu.fill_gaps(&f);
            }
            self.gpu.end(f, out)?;
        }
        Ok(stats)
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
        let stats = self.draw_frame(None)?;
        self.gpu_error()?;
        Ok(stats)
    }

    /// Draw one frame off-screen into `out` (RGBA8 rows, width x height x 4 bytes).
    pub fn render_into(&mut self, out: &mut [u8]) -> Result<FrameStats, ViewerError> {
        let stats = self.draw_frame(Some(out))?;
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
            let stats = self.prepare();
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
    /// None) from what is in view: the `lo` and `hi` quantiles (0..1) over the view, read
    /// at the finest level that keeps it within about a million pixels (for a volume cut
    /// at an angle: the voxels within half a voxel of the plane, of a box of at most
    /// about four million voxels).
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
        let plane = self.camera.plane(tw, th);
        let l = self.layer(i)?;
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
        let found = if l.mode(&self.camera) == Mode::Volume {
            volume_quantiles(l, &plane, tw, th, &chans, lo, hi)?
        } else {
            flat_quantiles(l, &plane, tw, th, &chans, lo, hi)?
        };
        for (c, a, b) in found {
            let mut d = self.layers[i].channels[c].clone();
            d.lo = a;
            d.hi = b;
            self.set_channel(i, c, d)?;
        }
        Ok(())
    }

    /// What lies under screen point (`x`, `y`): the world point, and per layer the
    /// level-0 pixel (and voxel) and every channel's value from the finest cached tile or
    /// brick holding it (read back from the GPU; the frame's level first, then coarser,
    /// then finer).
    pub fn pick(&self, x: f64, y: f64) -> Result<Pick, ViewerError> {
        let (view, lx, ly) = self.view_at(x, y);
        let cam = view.camera;
        let plane = cam.plane(view.w, view.h);
        let world = plane.at(lx, ly);
        let mut layers = Vec::new();
        for (li, o) in self.layers.iter().enumerate() {
            let mode = o.mode(&cam);
            let nearest = |l: usize| -> Option<[u64; 3]> {
                let lv = &o.source.levels[l];
                let p = o.to_index(l).apply(world);
                let v = p.map(|c| (c + 0.5).floor());
                let depth_ok = !o.is_volume() || (v[2] >= 0.0 && v[2] < lv.depth as f64);
                (v[0] >= 0.0
                    && v[1] >= 0.0
                    && v[0] < lv.width as f64
                    && v[1] < lv.height as f64
                    && depth_ok)
                    .then_some([v[0] as u64, v[1] as u64, v[2].max(0.0) as u64])
            };
            let at0 = if mode == Mode::Off { None } else { nearest(0) };
            let pixel = at0.map(|v| [v[0], v[1]]);
            let voxel = at0.filter(|_| o.is_volume());
            let level = match mode {
                Mode::Volume => o.volume_level(&plane),
                _ => o.level_for(self.camera.scale),
            };
            let nl = o.source.levels.len();
            let order: Vec<usize> = (level..nl).chain((0..level).rev()).collect();
            let mut values = Vec::new();
            for ch in 0..o.channels.len() {
                let mut found = PickedValue {
                    channel: ch,
                    value: None,
                    level: None,
                };
                if at0.is_some() {
                    for &l in &order {
                        let Some(v) = nearest(l) else { continue };
                        let bytes = if mode == Mode::Volume {
                            if !o.bgrid[l].2 {
                                continue;
                            }
                            let step = bricks::STEP;
                            let b = (
                                (v[0] / step) as u32,
                                (v[1] / step) as u32,
                                (v[2] / step) as u32,
                            );
                            let k = o.brick_key(ch, l, b);
                            let Some(slot) = o.bricks.as_ref().and_then(|br| br.cache.peek(&k))
                            else {
                                continue;
                            };
                            self.gpu.read_brick_texel(
                                o.id,
                                slot,
                                (v[0] % step) as u32,
                                (v[1] % step) as u32,
                                (v[2] % step) as u32,
                            )?
                        } else {
                            let k = o.key(
                                ch,
                                l,
                                ((v[0] / TILE as u64) as u32, (v[1] / TILE as u64) as u32),
                            );
                            let Some(slot) = o.cache.peek(&k) else {
                                continue;
                            };
                            self.gpu.read_texel(
                                o.id,
                                slot,
                                (v[0] % TILE as u64) as u32,
                                (v[1] % TILE as u64) as u32,
                            )?
                        };
                        found.value = Some(o.source.texel.value(&bytes, 0));
                        found.level = Some(l);
                        break;
                    }
                }
                values.push(found);
            }
            layers.push(LayerPick {
                layer: li,
                pixel,
                voxel,
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

    /// Brick slots of layer `i` (0 before a plane at an angle needed bricks).
    pub fn brick_slots(&self, i: usize) -> Result<u32, ViewerError> {
        Ok(self.gpu.brick_slots(self.layer(i)?.id))
    }
}

type Found = Vec<(usize, f64, f64)>;

/// Quantiles over a flat view: the view's box of the current plane at the finest level
/// within about a million pixels.
fn flat_quantiles(
    l: &Layer,
    plane: &Plane3,
    tw: u32,
    th: u32,
    chans: &[usize],
    lo: f64,
    hi: f64,
) -> Result<Found, ViewerError> {
    let nl = l.source.levels.len();
    let mut pick = None;
    for lv in 0..nl {
        let level = &l.source.levels[lv];
        let m = plane.through(&l.to_index(lv)).to_2d();
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
    let (plane_idx, _) = l.source.plane(lv, &l.position);
    let mut found = Vec::new();
    for &c in chans {
        let (start, stop) = l.source.region(&plane_idx, c, ty, tx, h, w);
        let data = l.source.read(lv, &start, &stop)?;
        if let Some((a, b)) = source::quantiles(l.source.texel, &data, lo, hi) {
            found.push((c, a, if b > a { b } else { a + 1.0 }));
        }
    }
    Ok(found)
}

/// Quantiles over a volume cut at an angle: the voxels within half a voxel of the plane
/// in the box the view spans, at the finest level whose box holds about four million
/// voxels at most.
fn volume_quantiles(
    l: &Layer,
    plane: &Plane3,
    tw: u32,
    th: u32,
    chans: &[usize],
    lo: f64,
    hi: f64,
) -> Result<Found, ViewerError> {
    let nl = l.source.levels.len();
    let mut pick = None;
    for lv in 0..nl {
        let level = &l.source.levels[lv];
        let p = plane.through(&l.to_index(lv));
        let corners = [
            p.at(0.0, 0.0),
            p.at(tw as f64, 0.0),
            p.at(0.0, th as f64),
            p.at(tw as f64, th as f64),
        ];
        let size = [level.width, level.height, level.depth];
        let mut b = [(0u64, 0u64); 3];
        for a in 0..3 {
            let mn = corners.iter().map(|c| c[a]).fold(f64::MAX, f64::min);
            let mx = corners.iter().map(|c| c[a]).fold(f64::MIN, f64::max);
            let s = (mn + 0.5).floor().max(0.0) as u64;
            let e = (((mx + 0.5).floor() + 1.0).max(0.0) as u64).min(size[a]);
            if e <= s {
                return Err(verr("the image is not in view"));
            }
            b[a] = (s, e);
        }
        let voxels: u64 = b.iter().map(|(s, e)| e - s).product();
        pick = Some((lv, b, p));
        if voxels <= 1 << 22 {
            break;
        }
    }
    let (lv, b, p) = pick.ok_or_else(|| verr("the image has no levels"))?;
    let n = cross(p.ux, p.uy);
    let nn = norm(n).max(1e-300);
    let (lead, _) = l.source.lead(lv, &l.position);
    let item = l.source.texel.bytes();
    let mut found = Vec::new();
    for &c in chans {
        let (start, stop) = l.source.region3(&lead, c, b[2], b[1], b[0]);
        let data = l.source.read(lv, &start, &stop)?;
        let (w, h) = (b[0].1 - b[0].0, b[1].1 - b[1].0);
        let mut near = Vec::new();
        let total = data.len() / item;
        for i in 0..total {
            let (x, y, z) = (
                b[0].0 + (i as u64 % w),
                b[1].0 + (i as u64 / w % h),
                b[2].0 + (i as u64 / (w * h)),
            );
            let d = sub([x as f64, y as f64, z as f64], p.origin);
            if (dot(d, n) / nn).abs() <= 0.5 {
                near.extend_from_slice(&data[i * item..(i + 1) * item]);
            }
        }
        if let Some((a, bb)) = source::quantiles(l.source.texel, &near, lo, hi) {
            found.push((c, a, if bb > a { bb } else { a + 1.0 }));
        }
    }
    Ok(found)
}

/// The transform an image's "aligned" coordinate system gives it (an NGFF 0.6 edge
/// from its own coordinate system to "aligned"), and a note when one is there but cannot
/// be shown (it touches other axes than the spatial ones, or cannot be inverted).
fn saved_transform(source: &Source) -> (Affine3, Option<String>) {
    match &source.aligned {
        None => (Affine3::IDENTITY, None),
        Some(m) => {
            let t = space_transform(m, source.axes.len(), source.z_axis)
                .filter(|t| t.inverse().is_some())
                .filter(|t| source.z_axis.is_some() || t.keeps_planes(1e-9));
            match t {
                Some(t) => (t, None),
                None => (
                    Affine3::IDENTITY,
                    Some(format!(
                        "the image's {ALIGNED:?} transform touches more than its spatial axes \
                         (or cannot be inverted); the image is shown without it"
                    )),
                ),
            }
        }
    }
}
