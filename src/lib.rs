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

fn rows(t: &core::Affine2) -> Vec<Vec<f64>> {
    t.m.iter().map(|r| r.to_vec()).collect()
}

fn finite(v: &[f64]) -> PyResult<()> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(ViewerError::new_err(format!("not finite: {v:?}")))
    }
}

fn channel_list<'py>(
    py: Python<'py>,
    chans: &[core::ChannelDisplay],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for c in chans {
        let d = PyDict::new(py);
        d.set_item("label", &c.label)?;
        d.set_item("color", (c.color[0], c.color[1], c.color[2]))?;
        d.set_item("lut", c.lut.as_deref())?;
        d.set_item("lo", c.lo)?;
        d.set_item("hi", c.hi)?;
        d.set_item("gamma", c.gamma)?;
        d.set_item("visible", c.visible)?;
        out.append(d)?;
    }
    Ok(out)
}

/// A viewer: one GPU target, its layers (images, each with channels, plane and
/// transform) and a camera.
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

    /// Open the OME-Zarr image at `path` (a folder, or .zip / .ozx) as the only layer
    /// and fit it.
    fn open(&mut self, path: PathBuf) -> PyResult<()> {
        self.inner.open(&path).map_err(err)
    }

    /// Add the image at `path` as a new layer on top; it becomes the current layer.
    /// Returns its index.
    fn add(&mut self, path: PathBuf) -> PyResult<usize> {
        self.inner.add(&path).map_err(err)
    }

    /// Read layer `index`'s image again (after its metadata changed): placement and
    /// saved transform from the file; channels, plane, visibility and opacity stay.
    fn reload(&mut self, index: usize) -> PyResult<()> {
        self.inner.reload(index).map_err(err)
    }

    /// Remove layer `index`.
    fn remove(&mut self, index: usize) -> PyResult<()> {
        self.inner.layer(index).map_err(err)?;
        self.inner.remove(index);
        Ok(())
    }

    /// The layers, bottom first: path, visible, opacity, transform (2 x 3), position,
    /// channels, level and error of each.
    #[getter]
    fn layers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        let scale = self.inner.camera.scale;
        for l in self.inner.layers() {
            let d = PyDict::new(py);
            d.set_item("path", l.path().to_string_lossy().into_owned())?;
            d.set_item("version", &l.source().version)?;
            d.set_item("axes", &l.source().axes)?;
            d.set_item("visible", l.visible)?;
            d.set_item("opacity", l.opacity)?;
            d.set_item("transform", rows(&l.transform()))?;
            d.set_item("position", l.position())?;
            d.set_item("channels", channel_list(py, l.channels())?)?;
            d.set_item("level", l.level_for(scale))?;
            d.set_item("error", l.error())?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// The current layer's index (input and the calls without `layer=` act on it).
    #[getter]
    fn current(&self) -> usize {
        self.inner.current()
    }

    #[setter]
    fn set_current(&mut self, index: usize) -> PyResult<()> {
        self.inner.set_current(index).map_err(err)
    }

    /// Show or hide a layer, or set its opacity (0..1); arguments left out stay.
    #[pyo3(signature = (index, *, visible = None, opacity = None))]
    fn set_layer(
        &mut self,
        index: usize,
        visible: Option<bool>,
        opacity: Option<f32>,
    ) -> PyResult<()> {
        let l = self.inner.layer(index).map_err(err)?;
        let (v, o) = (visible.unwrap_or(l.visible), opacity.unwrap_or(l.opacity));
        self.inner.set_layer(index, v, o).map_err(err)
    }

    /// The current layer's image: path, NGFF version, dtype, axes, channel axis, plane
    /// axes and levels (path, shape, world units per pixel); None without one.
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

    /// Turn the view by `degrees` (the image turns clockwise) about screen point
    /// (`x`, `y`), by default the centre.
    #[pyo3(signature = (degrees, x = None, y = None))]
    fn rotate(&mut self, degrees: f64, x: Option<f64>, y: Option<f64>) -> PyResult<()> {
        finite(&[degrees])?;
        let (w, h) = self.inner.size();
        self.inner.rotate(
            degrees,
            x.unwrap_or(w as f64 / 2.0),
            y.unwrap_or(h as f64 / 2.0),
        );
        Ok(())
    }

    /// Degrees the image appears turned clockwise (0 <= rotation < 360).
    #[getter]
    fn rotation(&self) -> f64 {
        self.inner.camera.rotation
    }

    /// Turn the view to `degrees` about the centre.
    fn set_rotation(&mut self, degrees: f64) -> PyResult<()> {
        finite(&[degrees])?;
        let (w, h) = self.inner.size();
        let by = degrees - self.inner.camera.rotation;
        self.inner.rotate(by, w as f64 / 2.0, h as f64 / 2.0);
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> PyResult<()> {
        self.inner.resize(width, height).map_err(err)
    }

    /// Set every channel's contrast limits (of the current layer, or `layer`).
    #[pyo3(signature = (lo, hi, layer = None))]
    fn set_contrast(&mut self, lo: f64, hi: f64, layer: Option<usize>) -> PyResult<()> {
        let i = self.layer_index(layer)?;
        for c in 0..self.inner.layer(i).map_err(err)?.channels().len() {
            let mut d = self.inner.layer(i).map_err(err)?.channels()[c].clone();
            d.lo = lo;
            d.hi = hi;
            self.inner.set_channel(i, c, d).map_err(err)?;
        }
        Ok(())
    }

    /// Contrast from what is in view: the `lo` and `hi` quantiles of channel `channel`
    /// (every visible channel when None) over the view, at a level of about a million
    /// pixels.
    #[pyo3(signature = (channel = None, *, lo = 0.001, hi = 0.999, layer = None))]
    fn auto_contrast(
        &mut self,
        channel: Option<usize>,
        lo: f64,
        hi: f64,
        layer: Option<usize>,
    ) -> PyResult<()> {
        let i = self.layer_index(layer)?;
        self.inner.auto_contrast(i, channel, lo, hi).map_err(err)
    }

    /// The current layer's channels: label, color (r, g, b in 0..1), lut (a colour
    /// map's name or None), lo, hi, gamma, visible.
    #[getter]
    fn channels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match self.inner.layers().get(self.inner.current()) {
            Some(l) => channel_list(py, l.channels()),
            None => Ok(PyList::empty(py)),
        }
    }

    /// Change channel `index`'s display (of the current layer, or `layer`); arguments
    /// left out stay. `lut` names a colour map (`COLOUR_MAPS`); "" goes back to the
    /// colour.
    #[pyo3(signature = (index, *, color = None, lut = None, lo = None, hi = None, gamma = None, visible = None, label = None, layer = None))]
    #[allow(clippy::too_many_arguments)]
    fn set_channel(
        &mut self,
        index: usize,
        color: Option<(f32, f32, f32)>,
        lut: Option<String>,
        lo: Option<f64>,
        hi: Option<f64>,
        gamma: Option<f64>,
        visible: Option<bool>,
        label: Option<String>,
        layer: Option<usize>,
    ) -> PyResult<()> {
        let i = self.layer_index(layer)?;
        let chans = self.inner.layer(i).map_err(err)?.channels();
        let mut d = chans.get(index).cloned().ok_or_else(|| {
            ViewerError::new_err(format!("channel {index}: the image has {}", chans.len()))
        })?;
        if let Some((r, g, b)) = color {
            d.color = [r, g, b];
        }
        if let Some(n) = lut {
            d.lut = (!n.is_empty()).then_some(n);
        }
        d.lo = lo.unwrap_or(d.lo);
        d.hi = hi.unwrap_or(d.hi);
        d.gamma = gamma.unwrap_or(d.gamma);
        d.visible = visible.unwrap_or(d.visible);
        if let Some(l) = label {
            d.label = l;
        }
        self.inner.set_channel(i, index, d).map_err(err)
    }

    /// The current layer's plane: [(axis, level-0 index, number of indices)] for the
    /// axes before y and x other than the channel axis.
    #[getter]
    fn position(&self) -> Vec<(String, f64, u64)> {
        self.inner
            .layers()
            .get(self.inner.current())
            .map(|l| l.position())
            .unwrap_or_default()
    }

    /// Move the plane (of the current layer, or `layer`) to level-0 index `index` on
    /// `axis` ("z", "t", ...).
    #[pyo3(signature = (axis, index, layer = None))]
    fn set_position(&mut self, axis: &str, index: f64, layer: Option<usize>) -> PyResult<()> {
        let i = self.layer_index(layer)?;
        self.inner.set_position(i, axis, index).map_err(err)
    }

    /// A layer's transform (world to world, on top of its levels' placement) as the
    /// 2-D affine over (x, y): [[a, b, tx], [c, d, ty]].
    #[pyo3(signature = (layer = None))]
    fn transform(&self, layer: Option<usize>) -> PyResult<Vec<Vec<f64>>> {
        let i = self.layer_index(layer)?;
        Ok(rows(&self.inner.layer(i).map_err(err)?.transform()))
    }

    /// Set a layer's transform: [[a, b, tx], [c, d, ty]] over (x, y).
    #[pyo3(signature = (matrix, layer = None))]
    fn set_transform(&mut self, matrix: Vec<Vec<f64>>, layer: Option<usize>) -> PyResult<()> {
        let i = self.layer_index(layer)?;
        let ok = matrix.len() == 2 && matrix.iter().all(|r| r.len() == 3);
        if !ok {
            return Err(ViewerError::new_err(
                "a transform is 2 rows of 3: [[a, b, tx], [c, d, ty]]",
            ));
        }
        let m = [
            [matrix[0][0], matrix[0][1], matrix[0][2]],
            [matrix[1][0], matrix[1][1], matrix[1][2]],
        ];
        self.inner
            .set_transform(i, core::Affine2 { m })
            .map_err(err)
    }

    /// Take a layer back to where its image places it (the identity transform).
    #[pyo3(signature = (layer = None))]
    fn reset_transform(&mut self, layer: Option<usize>) -> PyResult<()> {
        let i = self.layer_index(layer)?;
        self.inner
            .set_transform(i, core::Affine2::IDENTITY)
            .map_err(err)
    }

    /// Move a layer by `dx`, `dy` screen pixels.
    #[pyo3(signature = (dx, dy, layer = None))]
    fn move_layer(&mut self, dx: f64, dy: f64, layer: Option<usize>) -> PyResult<()> {
        finite(&[dx, dy])?;
        let i = self.layer_index(layer)?;
        self.inner.move_layer(i, dx, dy).map_err(err)
    }

    /// Turn a layer by `degrees` (clockwise on an unturned view) about world point
    /// `about` (x, y), by default the image's centre.
    #[pyo3(signature = (degrees, about = None, layer = None))]
    fn rotate_layer(
        &mut self,
        degrees: f64,
        about: Option<(f64, f64)>,
        layer: Option<usize>,
    ) -> PyResult<()> {
        finite(&[degrees])?;
        let i = self.layer_index(layer)?;
        self.inner
            .rotate_layer(i, degrees, about.map(|(x, y)| [x, y]))
            .map_err(err)
    }

    /// A layer's transform as an NGFF affine over its image's axes: n rows of n + 1
    /// (the last column the translation, world units), the identity outside y and x.
    #[pyo3(signature = (layer = None))]
    fn transform_matrix(&self, layer: Option<usize>) -> PyResult<Vec<Vec<f64>>> {
        let i = self.layer_index(layer)?;
        Ok(self.inner.layer(i).map_err(err)?.transform_matrix())
    }

    /// (origin x, origin y, world units per screen pixel): the world point at the
    /// top-left corner.
    #[getter]
    fn camera(&self) -> (f64, f64, f64) {
        let o = self.inner.screen_to_world(0.0, 0.0);
        (o[0], o[1], self.inner.camera.scale)
    }

    /// Set the camera: the world point at the top-left corner, world units per pixel
    /// (the turn stays).
    fn set_camera(&mut self, x: f64, y: f64, scale: f64) -> PyResult<()> {
        if !(scale > 0.0 && scale.is_finite() && x.is_finite() && y.is_finite()) {
            return Err(ViewerError::new_err(format!(
                "a camera at ({x}, {y}) with scale {scale}"
            )));
        }
        let (w, h) = self.inner.size();
        let mut cam = self.inner.camera;
        cam.scale = scale;
        cam.center = [0.0, 0.0];
        let (dx, dy) = cam
            .screen_to_world(w, h)
            .linear(w as f64 / 2.0, h as f64 / 2.0);
        self.inner.look_at([x + dx, y + dy], scale);
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

    /// The world point at screen pixel (`x`, `y`).
    fn screen_to_world(&self, x: f64, y: f64) -> (f64, f64) {
        let w = self.inner.screen_to_world(x, y);
        (w[0], w[1])
    }

    /// The world box (x0, y0, x1, y1) of the visible layers, or None.
    #[getter]
    fn bounds(&self) -> Option<(f64, f64, f64, f64)> {
        self.inner.bounds().map(|b| (b[0], b[1], b[2], b[3]))
    }

    /// The level the current zoom draws for the current layer (0 the finest).
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

    /// Tiles the GPU atlases hold at once (all layers).
    #[getter]
    fn atlas_slots(&self) -> u32 {
        self.inner.atlas_slots()
    }

    /// Tiles requested and not read yet.
    #[getter]
    fn pending(&self) -> usize {
        self.inner.pending()
    }

    /// Whether read tiles wait for the next frame.
    fn has_results(&self) -> bool {
        self.inner.has_results()
    }

    /// Whether the host should draw again soon (tiles on their way or waiting, or
    /// reading deferred until the view stops moving): repaint while true.
    fn wants_frame(&self) -> bool {
        self.inner.wants_frame()
    }

    /// The first error of any layer (a tile read, or a note from opening), or None.
    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.error().map(str::to_string)
    }

    /// Draw one frame into the window: (visible, loaded, pending, level, missing).
    fn render(&mut self) -> PyResult<Stats> {
        self.inner.render().map(stats).map_err(err)
    }

    /// Draw one frame off-screen into `buffer`, a writable contiguous buffer of width x
    /// height x 4 bytes (e.g. a bytearray the widget keeps): (visible, loaded, pending,
    /// level, missing).
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
    /// frame's levels in view read, so the next frame is complete; with `read_ahead`,
    /// also the tiles read ahead (ring, next planes). Tests, screenshots, the harness.
    #[pyo3(signature = (timeout = 30.0, read_ahead = false))]
    fn settle(&mut self, timeout: f64, read_ahead: bool) -> PyResult<Stats> {
        self.inner
            .settle(Duration::from_secs_f64(timeout.max(0.0)), read_ahead)
            .map(stats)
            .map_err(err)
    }

    /// What lies under screen point (`x`, `y`): {"world": (x, y), "pixel" and "values"
    /// of the current layer, "layers": [{"layer", "pixel", "values"}] of all}; "pixel"
    /// is the level-0 pixel (x, y) or None, "values" [{"channel", "value", "level"}]
    /// with each channel's value from the finest cached tile.
    fn pick<'py>(&self, py: Python<'py>, x: f64, y: f64) -> PyResult<Bound<'py, PyDict>> {
        let p = self.inner.pick(x, y).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("world", (p.world[0], p.world[1]))?;
        let layers = PyList::empty(py);
        let mut current = (None, PyList::empty(py));
        for lp in p.layers {
            let values = PyList::empty(py);
            for v in &lp.values {
                let e = PyDict::new(py);
                e.set_item("channel", v.channel)?;
                e.set_item("value", v.value)?;
                e.set_item("level", v.level)?;
                values.append(e)?;
            }
            let pixel = lp.pixel.map(|q| (q[0], q[1]));
            if lp.layer == self.inner.current() {
                current = (pixel, values.clone());
            }
            let l = PyDict::new(py);
            l.set_item("layer", lp.layer)?;
            l.set_item("pixel", pixel)?;
            l.set_item("values", values)?;
            layers.append(l)?;
        }
        d.set_item("pixel", current.0)?;
        d.set_item("values", current.1)?;
        d.set_item("layers", layers)?;
        Ok(d)
    }

    /// The view state as JSON (camera, layers).
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

impl Viewer {
    /// `layer`, or the current layer; an error when there is none.
    fn layer_index(&self, layer: Option<usize>) -> PyResult<usize> {
        let i = layer.unwrap_or_else(|| self.inner.current());
        self.inner.layer(i).map_err(err)?;
        Ok(i)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("ViewerError", m.py().get_type::<ViewerError>())?;
    m.add("COLOUR_MAPS", core::COLOUR_MAPS.to_vec())?;
    m.add("ALIGNED", core::ALIGNED)?;
    m.add_class::<Viewer>()?;
    Ok(())
}
