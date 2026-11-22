//! Python bindings of micro-viewer-core (module `micro_viewer._core`): a `Viewer` that
//! draws into a native window (a Qt widget's `winId()`) or off-screen. The Qt widget
//! itself is Python (`micro_viewer.qt`).

use std::path::PathBuf;
use std::time::Duration;

use micro_viewer_core as core;
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

create_exception!(micro_viewer._core, ViewerError, PyValueError);

fn err(e: core::ViewerError) -> PyErr {
    ViewerError::new_err(e.0)
}

/// The raw handles of a native window on this platform: `window` is Qt's `winId()`
/// (an HWND, an NSView, an X11 window), `display` the X11 Display pointer (Linux only).
fn handles(window: usize, display: usize) -> PyResult<(RawDisplayHandle, RawWindowHandle)> {
    #[cfg(target_os = "windows")]
    {
        let _ = display;
        let hwnd = std::num::NonZeroIsize::new(window as isize)
            .ok_or_else(|| ViewerError::new_err("a window handle of 0"))?;
        Ok((
            RawDisplayHandle::Windows(raw_window_handle::WindowsDisplayHandle::new()),
            RawWindowHandle::Win32(raw_window_handle::Win32WindowHandle::new(hwnd)),
        ))
    }
    #[cfg(target_os = "macos")]
    {
        let _ = display;
        let view = std::ptr::NonNull::new(window as *mut std::ffi::c_void)
            .ok_or_else(|| ViewerError::new_err("a view handle of 0"))?;
        Ok((
            RawDisplayHandle::AppKit(raw_window_handle::AppKitDisplayHandle::new()),
            RawWindowHandle::AppKit(raw_window_handle::AppKitWindowHandle::new(view)),
        ))
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let dpy = std::ptr::NonNull::new(display as *mut std::ffi::c_void).ok_or_else(|| {
            ViewerError::new_err(
                "a native window on Linux needs the X11 display; use the off-screen mode \
                 (Wayland, or when the display is not known)",
            )
        })?;
        Ok((
            RawDisplayHandle::Xlib(raw_window_handle::XlibDisplayHandle::new(Some(dpy), 0)),
            RawWindowHandle::Xlib(raw_window_handle::XlibWindowHandle::new(window as _)),
        ))
    }
}

/// A frame's statistics: (tiles visible, loaded for it, pending, level, missing: visible
/// tiles of the frame's level not read yet; 0 when sharp).
type Stats = (u32, u32, u32, u32, u32);

fn stats(s: core::FrameStats) -> Stats {
    (s.visible, s.loaded, s.pending, s.level, s.missing)
}

/// A viewer: one GPU target, the open image, camera, plane position and channels.
#[pyclass(unsendable, module = "micro_viewer._core")]
pub struct Viewer {
    inner: core::Viewer,
}

#[pymethods]
impl Viewer {
    /// A viewer drawing into the native window `window` (Qt's `winId()`), `width` x
    /// `height` physical pixels. `software` uses the system's software adapter;
    /// `no_vsync` presents without waiting for the display (frame-time measurements);
    /// `atlas_mib` caps the tile atlas's GPU memory (0: 1024 MiB).
    #[staticmethod]
    #[pyo3(signature = (window, width, height, display = 0, software = false, no_vsync = false, atlas_mib = 0))]
    fn for_window(
        window: usize,
        width: u32,
        height: u32,
        display: usize,
        software: bool,
        no_vsync: bool,
        atlas_mib: u32,
    ) -> PyResult<Self> {
        let (d, w) = handles(window, display)?;
        let options = core::GpuOptions {
            software,
            no_vsync,
            atlas_mib,
        };
        // SAFETY: the Qt widget that owns the window owns this viewer and drops it first
        let inner =
            unsafe { core::Viewer::for_window(d, w, width, height, options) }.map_err(err)?;
        Ok(Self { inner })
    }

    /// A viewer drawing off-screen; frames come back from `render_into` / `render_rgba`.
    #[staticmethod]
    #[pyo3(signature = (width, height, software = false, atlas_mib = 0))]
    fn offscreen(width: u32, height: u32, software: bool, atlas_mib: u32) -> PyResult<Self> {
        let options = core::GpuOptions {
            software,
            no_vsync: false,
            atlas_mib,
        };
        Ok(Self {
            inner: core::Viewer::offscreen(width, height, options).map_err(err)?,
        })
    }

    /// Open the OME-Zarr image at `path` (a folder, or .zip / .ozx) and fit it.
    fn open(&mut self, path: PathBuf) -> PyResult<()> {
        self.inner.open(&path).map_err(err)
    }

    /// The open image: path, NGFF version, dtype, axes, channel axis, plane axes and
    /// levels (path, shape, world units per pixel); None without one.
    #[getter]
    fn image<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(s) = self.inner.source() else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item(
            "path",
            self.inner.path().map(|p| p.to_string_lossy().into_owned()),
        )?;
        d.set_item("version", &s.version)?;
        d.set_item("dtype", &s.dtype)?;
        d.set_item("axes", &s.axes)?;
        d.set_item("channel_axis", s.channel_axis.map(|a| s.axes[a].clone()))?;
        let plane_axes: Vec<&String> = s.plane_axes.iter().map(|&a| &s.axes[a]).collect();
        d.set_item("plane_axes", plane_axes)?;
        let levels = PyList::empty(py);
        for lv in &s.levels {
            let l = PyDict::new(py);
            l.set_item("path", &lv.path)?;
            l.set_item("shape", &lv.shape)?;
            l.set_item("pixel", lv.pixel)?;
            levels.append(l)?;
        }
        d.set_item("levels", levels)?;
        Ok(Some(d))
    }

    fn fit(&mut self) {
        self.inner.fit();
    }

    fn pan(&mut self, dx: f64, dy: f64) {
        self.inner.pan(dx, dy);
    }

    fn zoom(&mut self, factor: f64, x: f64, y: f64) {
        self.inner.zoom(factor, x, y);
    }

    fn resize(&mut self, width: u32, height: u32) -> PyResult<()> {
        self.inner.resize(width, height).map_err(err)
    }

    /// Set every channel's contrast limits.
    fn set_contrast(&mut self, lo: f64, hi: f64) -> PyResult<()> {
        for c in 0..self.inner.channels().len() {
            let mut d = self.inner.channels()[c].clone();
            d.lo = lo;
            d.hi = hi;
            self.inner.set_channel(c, d).map_err(err)?;
        }
        Ok(())
    }

    /// The channels' display: label, color (r, g, b in 0..1), lo, hi, gamma, visible.
    #[getter]
    fn channels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for c in self.inner.channels() {
            let d = PyDict::new(py);
            d.set_item("label", &c.label)?;
            d.set_item("color", (c.color[0], c.color[1], c.color[2]))?;
            d.set_item("lo", c.lo)?;
            d.set_item("hi", c.hi)?;
            d.set_item("gamma", c.gamma)?;
            d.set_item("visible", c.visible)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// Change channel `index`'s display; arguments left out stay.
    #[pyo3(signature = (index, *, color = None, lo = None, hi = None, gamma = None, visible = None, label = None))]
    #[allow(clippy::too_many_arguments)]
    fn set_channel(
        &mut self,
        index: usize,
        color: Option<(f32, f32, f32)>,
        lo: Option<f64>,
        hi: Option<f64>,
        gamma: Option<f64>,
        visible: Option<bool>,
        label: Option<String>,
    ) -> PyResult<()> {
        let mut d = self.inner.channels().get(index).cloned().ok_or_else(|| {
            ViewerError::new_err(format!(
                "channel {index}: the image has {}",
                self.inner.channels().len()
            ))
        })?;
        if let Some((r, g, b)) = color {
            d.color = [r, g, b];
        }
        d.lo = lo.unwrap_or(d.lo);
        d.hi = hi.unwrap_or(d.hi);
        d.gamma = gamma.unwrap_or(d.gamma);
        d.visible = visible.unwrap_or(d.visible);
        if let Some(l) = label {
            d.label = l;
        }
        self.inner.set_channel(index, d).map_err(err)
    }

    /// The plane's position: [(axis, level-0 index, number of indices)] for the axes
    /// before y and x other than the channel axis.
    #[getter]
    fn position(&self) -> Vec<(String, f64, u64)> {
        self.inner.position()
    }

    /// Move the plane to level-0 index `index` on `axis` ("z", "t", ...).
    fn set_position(&mut self, axis: &str, index: f64) -> PyResult<()> {
        self.inner.set_position(axis, index).map_err(err)
    }

    /// (origin x, origin y, world units per screen pixel): the world point at the
    /// top-left corner.
    #[getter]
    fn camera(&self) -> (f64, f64, f64) {
        let c = self.inner.camera;
        (c.origin[0], c.origin[1], c.scale)
    }

    /// Set the camera: the world point at the top-left corner, world units per pixel.
    fn set_camera(&mut self, x: f64, y: f64, scale: f64) -> PyResult<()> {
        if !(scale > 0.0 && scale.is_finite() && x.is_finite() && y.is_finite()) {
            return Err(ViewerError::new_err(format!(
                "a camera at ({x}, {y}) with scale {scale}"
            )));
        }
        self.inner.camera = core::Camera {
            origin: [x, y],
            scale,
        };
        Ok(())
    }

    /// The world point at the view's centre.
    #[getter]
    fn center(&self) -> (f64, f64) {
        let c = self.inner.center();
        (c[0], c[1])
    }

    /// Put world point (`x`, `y`) at the view's centre, `scale` world units per pixel.
    fn look_at(&mut self, x: f64, y: f64, scale: f64) -> PyResult<()> {
        if !(scale > 0.0 && scale.is_finite() && x.is_finite() && y.is_finite()) {
            return Err(ViewerError::new_err(format!(
                "a view centred at ({x}, {y}) with scale {scale}"
            )));
        }
        self.inner.look_at([x, y], scale);
        Ok(())
    }

    /// The world box (x0, y0, x1, y1) of the image, or None.
    #[getter]
    fn bounds(&self) -> Option<(f64, f64, f64, f64)> {
        self.inner.bounds().map(|b| (b[0], b[1], b[2], b[3]))
    }

    /// The level the current zoom draws (0 the finest).
    #[getter]
    fn level(&self) -> usize {
        self.inner.level()
    }

    #[getter]
    fn adapter(&self) -> String {
        self.inner.adapter().to_string()
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        self.inner.size()
    }

    /// Tiles the GPU atlas holds at once.
    #[getter]
    fn atlas_slots(&self) -> u32 {
        self.inner.atlas_slots()
    }

    /// Tiles requested and not read yet.
    #[getter]
    fn pending(&self) -> usize {
        self.inner.pending()
    }

    /// Whether read tiles wait for the next frame (repaint when true).
    fn has_results(&self) -> bool {
        self.inner.has_results()
    }

    /// The first tile read error since the image was opened, or None.
    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.error().map(str::to_string)
    }

    /// Draw one frame into the window: (visible, loaded, pending, level).
    fn render(&mut self) -> PyResult<Stats> {
        self.inner.render().map(stats).map_err(err)
    }

    /// Draw one frame off-screen into `buffer`, a writable contiguous buffer of width x
    /// height x 4 bytes (e.g. a bytearray the widget keeps): (visible, loaded, pending,
    /// level).
    fn render_into(&mut self, buffer: &Bound<'_, PyAny>) -> PyResult<Stats> {
        let buf = pyo3::buffer::PyBuffer::<u8>::get(buffer)?;
        if buf.readonly() || !buf.is_c_contiguous() {
            return Err(ViewerError::new_err(
                "render_into needs a writable, contiguous buffer",
            ));
        }
        // SAFETY: the buffer stays exported (alive, not resizable) while `buf` is held,
        // it is writable, and the GIL keeps other Python code off it during the call
        let out =
            unsafe { std::slice::from_raw_parts_mut(buf.buf_ptr() as *mut u8, buf.len_bytes()) };
        self.inner.render_into(out).map(stats).map_err(err)
    }

    /// Draw one frame off-screen: (RGBA8 bytes, width, height, stats).
    fn render_rgba<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyBytes>, u32, u32, Stats)> {
        let (rgba, s) = self.inner.render_rgba().map_err(err)?;
        let (w, h) = self.inner.size();
        Ok((PyBytes::new(py, &rgba), w, h, stats(s)))
    }

    /// Wait (up to `timeout` seconds) until the view is sharp: every tile of the
    /// frame's level in view read, so the next frame is complete; with `read_ahead`,
    /// also the tiles read ahead (ring, next planes). Tests, screenshots, the harness.
    #[pyo3(signature = (timeout = 30.0, read_ahead = false))]
    fn settle(&mut self, timeout: f64, read_ahead: bool) -> PyResult<Stats> {
        self.inner
            .settle(Duration::from_secs_f64(timeout.max(0.0)), read_ahead)
            .map(stats)
            .map_err(err)
    }

    /// What lies under screen point (`x`, `y`): {"world": (x, y), "pixel": (x, y) of
    /// level 0 or None, "values": [{"channel", "value", "level"}]} with each channel's
    /// value from the finest cached tile.
    fn pick<'py>(&self, py: Python<'py>, x: f64, y: f64) -> PyResult<Bound<'py, PyDict>> {
        let p = self.inner.pick(x, y).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("world", (p.world[0], p.world[1]))?;
        d.set_item("pixel", p.pixel.map(|q| (q[0], q[1])))?;
        let values = PyList::empty(py);
        for v in p.values {
            let e = PyDict::new(py);
            e.set_item("channel", v.channel)?;
            e.set_item("value", v.value)?;
            e.set_item("level", v.level)?;
            values.append(e)?;
        }
        d.set_item("values", values)?;
        Ok(d)
    }

    /// The view state as JSON (image, camera, position, channels).
    fn view_state(&self) -> String {
        self.inner.view_state().to_json()
    }

    /// Apply a view state (JSON); fields it leaves out stay.
    fn set_view_state(&mut self, state: &str) -> PyResult<()> {
        let s = core::ViewState::from_json(state).map_err(err)?;
        self.inner.set_view_state(&s).map_err(err)
    }

    /// Wait until the GPU has finished the frames submitted (for timing).
    fn finish(&self) -> PyResult<()> {
        self.inner.finish().map_err(err)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("ViewerError", m.py().get_type::<ViewerError>())?;
    m.add_class::<Viewer>()?;
    Ok(())
}
