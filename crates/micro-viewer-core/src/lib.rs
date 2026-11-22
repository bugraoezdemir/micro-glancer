//! micro-viewer's core (Workplan WP14): OME-Zarr images drawn by the GPU from a tile
//! cache, hosted by any toolkit that can hand over a native window, or read back
//! off-screen. No Qt and no Python here: the bindings and the widget live outside.
//!
//! Spike scope (task V3): one plane of one level of an image, one channel, contrast in
//! the shader, pan and zoom; tiles read on demand, on the drawing thread. Level choice,
//! the scheduler, channels and colour maps come in task V7.

mod gpu;
mod source;
mod tiles;

use std::path::Path;

pub use gpu::GpuOptions;
use gpu::{Gpu, Params};
pub use raw_window_handle;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
pub use source::Plane;
use tiles::{TileTable, TILE};

/// Something the viewer cannot do as asked; the message says why.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewerError(pub String);

impl std::fmt::Display for ViewerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ViewerError {}

/// What a frame showed: tiles drawn, read and uploaded for it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub visible: u32,
    pub loaded: u32,
}

/// The camera: the image pixel at the target's top-left corner and the image pixels
/// per screen pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub origin: [f64; 2],
    pub scale: f64,
}

struct Opened {
    plane: Plane,
    table: TileTable,
}

/// A viewer: a GPU target, at most one open plane, a camera and contrast limits.
pub struct Viewer {
    gpu: Gpu,
    opened: Option<Opened>,
    pub camera: Camera,
    pub contrast: (f32, f32),
    frame: u64,
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
            opened: None,
            camera: Camera {
                origin: [0.0, 0.0],
                scale: 1.0,
            },
            contrast: (0.0, 1.0),
            frame: 0,
        }
    }

    /// The adapter drawing the frames (name, backend, kind).
    pub fn adapter(&self) -> &str {
        &self.gpu.adapter_info
    }

    /// Open level `level` of the image at `path` (a folder, or a .zip / .ozx), fit it
    /// into the target and set the contrast from its first tile.
    pub fn open(&mut self, path: &Path, level: usize) -> Result<(), ViewerError> {
        let plane = Plane::open(path, level)?;
        let table = TileTable::new(plane.width, plane.height);
        self.gpu.upload_pages(&table.pages);
        let (h, w) = (plane.height.min(TILE as u64), plane.width.min(TILE as u64));
        let first = plane.read_tile(0, 0, h, w)?;
        let (lo, hi) = first
            .iter()
            .filter(|v| v.is_finite())
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        self.contrast = if lo < hi { (lo, hi) } else { (0.0, 1.0) };
        self.opened = Some(Opened { plane, table });
        self.fit();
        Ok(())
    }

    /// The open plane, if any.
    pub fn plane(&self) -> Option<&Plane> {
        self.opened.as_ref().map(|o| &o.plane)
    }

    /// Show the whole plane, centred.
    pub fn fit(&mut self) {
        let Some(o) = &self.opened else { return };
        let (tw, th) = self.gpu.size();
        let (w, h) = (o.plane.width as f64, o.plane.height as f64);
        let scale = (w / tw as f64).max(h / th as f64).max(1e-9);
        self.camera = Camera {
            origin: [(w - tw as f64 * scale) / 2.0, (h - th as f64 * scale) / 2.0],
            scale,
        };
    }

    /// Move the view by `dx`, `dy` screen pixels (the image follows the pointer).
    pub fn pan(&mut self, dx: f64, dy: f64) {
        self.camera.origin[0] -= dx * self.camera.scale;
        self.camera.origin[1] -= dy * self.camera.scale;
    }

    /// Zoom by `factor` (> 1: closer) keeping the image point under screen (`x`, `y`).
    pub fn zoom(&mut self, factor: f64, x: f64, y: f64) {
        let c = &mut self.camera;
        let (ix, iy) = (c.origin[0] + x * c.scale, c.origin[1] + y * c.scale);
        c.scale = (c.scale / factor).clamp(1.0 / 64.0, 1e6);
        c.origin = [ix - x * c.scale, iy - y * c.scale];
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
    }

    /// Load the tiles the view needs and are not cached (read on this thread), and the
    /// shader's parameters for the frame.
    fn prepare(&mut self) -> Result<(Params, FrameStats, bool), ViewerError> {
        self.frame += 1;
        let (tw, th) = self.gpu.size();
        let Some(o) = &mut self.opened else {
            return Ok((Params::default(), FrameStats::default(), false));
        };
        let c = self.camera;
        let t = TILE as f64;
        let x0 = (c.origin[0] / t).floor().max(0.0) as i64;
        let y0 = (c.origin[1] / t).floor().max(0.0) as i64;
        let x1 = ((c.origin[0] + tw as f64 * c.scale) / t).ceil() as i64;
        let y1 = ((c.origin[1] + th as f64 * c.scale) / t).ceil() as i64;
        let (gw, gh) = (o.table.grid_w as i64, o.table.grid_h as i64);
        let mut stats = FrameStats::default();
        let mut changed = false;
        // centre first, so the middle sharpens first once loading moves off this thread
        let mut want: Vec<(i64, i64)> = (y0.max(0)..y1.min(gh))
            .flat_map(|y| (x0.max(0)..x1.min(gw)).map(move |x| (x, y)))
            .collect();
        let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
        want.sort_by(|a, b| {
            let d =
                |p: &(i64, i64)| (p.0 as f64 + 0.5 - cx).powi(2) + (p.1 as f64 + 0.5 - cy).powi(2);
            d(a).total_cmp(&d(b))
        });
        for (x, y) in want {
            stats.visible += 1;
            let id = (y * gw + x) as u32;
            if o.table.touch(id, self.frame) {
                continue;
            }
            let Some(slot) = o.table.place(id, self.frame) else {
                break; // the atlas is full of this frame's tiles: zoomed out too far
            };
            let (ty, tx) = (y as u64 * TILE as u64, x as u64 * TILE as u64);
            let h = (o.plane.height - ty).min(TILE as u64);
            let w = (o.plane.width - tx).min(TILE as u64);
            let data = o.plane.read_tile(ty, tx, h, w)?;
            self.gpu.upload_tile(slot, &data, w as u32, h as u32);
            stats.loaded += 1;
            changed = true;
        }
        if changed {
            self.gpu.upload_pages(&o.table.pages);
        }
        let params = Params {
            origin: [c.origin[0] as f32, c.origin[1] as f32],
            scale: c.scale as f32,
            lo: self.contrast.0,
            hi: self.contrast.1,
            width: o.plane.width as f32,
            height: o.plane.height as f32,
            grid_w: o.table.grid_w,
            tile: TILE,
            slots_per_row: tiles::SLOTS_PER_ROW,
            _pad: [0; 2],
        };
        Ok((params, stats, true))
    }

    /// Draw one frame into the window.
    pub fn render(&mut self) -> Result<FrameStats, ViewerError> {
        let (params, stats, plane) = self.prepare()?;
        self.gpu.present(&params, plane)?;
        Ok(stats)
    }

    /// Draw one frame off-screen and read it back (RGBA8 rows).
    pub fn render_rgba(&mut self) -> Result<(Vec<u8>, FrameStats), ViewerError> {
        let (params, stats, plane) = self.prepare()?;
        Ok((self.gpu.render_rgba(&params, plane)?, stats))
    }

    /// Draw one frame off-screen into `out` (RGBA8 rows, width x height x 4 bytes).
    pub fn render_into(&mut self, out: &mut [u8]) -> Result<FrameStats, ViewerError> {
        let (params, stats, plane) = self.prepare()?;
        self.gpu.render_into(&params, plane, out)?;
        Ok(stats)
    }

    /// Wait for the GPU (for timing).
    pub fn finish(&self) -> Result<(), ViewerError> {
        self.gpu.finish()
    }

    /// The target's size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        self.gpu.size()
    }
}
