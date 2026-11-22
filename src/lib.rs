//! Python bindings of micro-viewer-core (module `micro_viewer._core`): a `Viewer` that
//! draws into a native window (a Qt widget's `winId()`) or off-screen. The Qt widget
//! itself is Python (`micro_viewer.qt`).

use std::path::PathBuf;

use micro_viewer_core as core;
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
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

/// A viewer: one GPU target, the open plane, camera and contrast.
#[pyclass(unsendable, module = "micro_viewer._core")]
pub struct Viewer {
    inner: core::Viewer,
}

#[pymethods]
impl Viewer {
    /// A viewer drawing into the native window `window` (Qt's `winId()`), `width` x
    /// `height` physical pixels. `software` uses the system's software adapter;
    /// `no_vsync` presents without waiting for the display (frame-time measurements).
    #[staticmethod]
    #[pyo3(signature = (window, width, height, display = 0, software = false, no_vsync = false))]
    fn for_window(
        window: usize,
        width: u32,
        height: u32,
        display: usize,
        software: bool,
        no_vsync: bool,
    ) -> PyResult<Self> {
        let (d, w) = handles(window, display)?;
        let options = core::GpuOptions { software, no_vsync };
        // SAFETY: the Qt widget that owns the window owns this viewer and drops it first
        let inner =
            unsafe { core::Viewer::for_window(d, w, width, height, options) }.map_err(err)?;
        Ok(Self { inner })
    }

    /// A viewer drawing off-screen; frames come back from `render_rgba`.
    #[staticmethod]
    #[pyo3(signature = (width, height, software = false))]
    fn offscreen(width: u32, height: u32, software: bool) -> PyResult<Self> {
        let options = core::GpuOptions {
            software,
            no_vsync: false,
        };
        Ok(Self {
            inner: core::Viewer::offscreen(width, height, options).map_err(err)?,
        })
    }

    /// Open level `level` of the OME-Zarr image at `path` (a folder, or .zip / .ozx).
    #[pyo3(signature = (path, level = 0))]
    fn open(&mut self, path: PathBuf, level: usize) -> PyResult<()> {
        self.inner.open(&path, level).map_err(err)
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

    fn resize(&mut self, width: u32, height: u32) {
        self.inner.resize(width, height);
    }

    fn set_contrast(&mut self, lo: f32, hi: f32) {
        self.inner.contrast = (lo, hi);
    }

    #[getter]
    fn contrast(&self) -> (f32, f32) {
        self.inner.contrast
    }

    /// (origin x, origin y, image pixels per screen pixel)
    #[getter]
    fn camera(&self) -> (f64, f64, f64) {
        let c = self.inner.camera;
        (c.origin[0], c.origin[1], c.scale)
    }

    /// Set the camera: the image pixel at the top-left corner, image pixels per screen
    /// pixel.
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

    #[getter]
    fn adapter(&self) -> String {
        self.inner.adapter().to_string()
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        self.inner.size()
    }

    /// The open plane: (height, width, dtype, NGFF version, level path), or None.
    #[getter]
    fn plane(&self) -> Option<(u64, u64, String, String, String)> {
        self.inner.plane().map(|p| {
            (
                p.height,
                p.width,
                p.dtype.clone(),
                p.version.clone(),
                p.level.clone(),
            )
        })
    }

    /// Draw one frame into the window: (tiles visible, tiles loaded for it).
    fn render(&mut self) -> PyResult<(u32, u32)> {
        let s = self.inner.render().map_err(err)?;
        Ok((s.visible, s.loaded))
    }

    /// Draw one frame off-screen: (RGBA8 bytes, width, height, tiles visible, loaded).
    fn render_rgba<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyBytes>, u32, u32, u32, u32)> {
        let (rgba, s) = self.inner.render_rgba().map_err(err)?;
        let (w, h) = self.inner.size();
        Ok((PyBytes::new(py, &rgba), w, h, s.visible, s.loaded))
    }

    /// Draw one frame off-screen into `buffer`, a writable contiguous buffer of width x
    /// height x 4 bytes (e.g. a bytearray the widget keeps): (tiles visible, loaded).
    fn render_into(&mut self, buffer: &Bound<'_, PyAny>) -> PyResult<(u32, u32)> {
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
        let s = self.inner.render_into(out).map_err(err)?;
        Ok((s.visible, s.loaded))
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
