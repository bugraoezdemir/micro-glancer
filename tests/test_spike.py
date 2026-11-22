"""micro-viewer spike (Workplan WP14, task V3): one plane drawn by the GPU from a tile
atlas, read through ozp-in-Rust (DR9) and dyna's arrays crate.

Off-screen frames are compared pixel by pixel with the image itself: a screen pixel
(sx, sy) shows image pixel floor(origin + (s + 0.5) * scale), its value through the
contrast limits, as an 8-bit grey (one step of rounding allowed).
"""
import os
import sys
import zipfile

import numpy as np
import pytest

from micro_viewer import Viewer, ViewerError

ozp = pytest.importorskip("ome_zarr_pyramid")

BACKGROUND = round(0.08 * 255)
H, W = 700, 900                      # 3 x 4 tiles of 256, edge tiles partial


@pytest.fixture(scope="module")
def image():
    yy, xx = np.mgrid[0:H, 0:W]
    return ((yy * 37 + xx * 61) % 65536).astype("uint16")


def _write(path, data, version):
    pyr = ozp.Pyramid().from_arrays([data], axis_order="yx", scales=[[1.0, 1.0]],
                                    version=version)
    ozp.IO().write_pyramid(pyr, str(path), overwrite=True)
    return str(path)


@pytest.fixture(scope="module")
def stores(tmp_path_factory, image):
    tmp = tmp_path_factory.mktemp("spike")
    out = {v: _write(tmp / f"img{v}.ome.zarr", image, v) for v in ("0.4", "0.5", "0.6")}
    zpath = tmp / "img.ozx"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(out["0.5"]):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, out["0.5"]).replace(os.sep, "/"))
    out["ozx"] = str(zpath)
    return out


def _frame(v):
    data, w, h, visible, loaded = v.render_rgba()
    rgba = np.frombuffer(data, np.uint8).reshape(h, w, 4)
    return rgba, visible, loaded


def _expected(image, w, h, x0, y0, scale, lo, hi):
    sx = x0 + (np.arange(w) + 0.5) * scale
    sy = y0 + (np.arange(h) + 0.5) * scale
    ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
    inside = (iy[:, None] >= 0) & (iy[:, None] < image.shape[0]) & \
             (ix[None, :] >= 0) & (ix[None, :] < image.shape[1])
    v = image[np.clip(iy, 0, image.shape[0] - 1)][:, np.clip(ix, 0, image.shape[1] - 1)]
    g = np.clip((v.astype(np.float32) - np.float32(lo)) / np.float32(hi - lo), 0, 1)
    return np.where(inside, np.round(g * 255), BACKGROUND).astype(int)


@pytest.mark.parametrize("camera", [(0.0, 0.0, 1.0), (37.0, 11.0, 0.5), (-50.0, -20.0, 1.75),
                                    (600.0, 400.0, 0.25)])
def test_pixels_are_the_image_through_the_contrast(stores, image, camera):
    v = Viewer.offscreen(320, 240)
    v.open(stores["0.5"])
    v.set_contrast(1000.0, 60000.0)
    v.set_camera(*camera)
    rgba, visible, loaded = _frame(v)
    want = _expected(image, 320, 240, *camera, 1000.0, 60000.0)
    grey = rgba[..., 0].astype(int)
    assert np.abs(grey - want).max() <= 1, np.argwhere(np.abs(grey - want) > 1)[:5]
    assert (rgba[..., 0] == rgba[..., 1]).all() and (rgba[..., 3] == 255).all()
    assert visible >= 1 and loaded == visible            # first frame reads every tile
    _, visible2, loaded2 = _frame(v)
    assert visible2 == visible and loaded2 == 0          # then the atlas holds them


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6", "ozx"])
def test_every_version_and_a_zip_open_alike(stores, image, version):
    v = Viewer.offscreen(200, 160)
    v.open(stores[version])
    h, w, dtype, ngff, level = v.plane
    assert (h, w, dtype, level) == (H, W, "uint16", "0")
    assert ngff == ("0.5" if version == "ozx" else version)
    v.set_contrast(0.0, 65535.0)
    v.set_camera(100.0, 50.0, 1.0)
    rgba, _, _ = _frame(v)
    want = _expected(image, 200, 160, 100.0, 50.0, 1.0, 0.0, 65535.0)
    assert np.abs(rgba[..., 0].astype(int) - want).max() <= 1


def test_the_software_adapter_draws_the_same(stores):
    try:
        soft = Viewer.offscreen(256, 192, software=True)
    except ViewerError as e:
        pytest.skip(f"no software adapter here: {e}")
    gpu = Viewer.offscreen(256, 192)
    for v in (soft, gpu):
        v.open(stores["0.5"])
        v.set_contrast(0.0, 65535.0)
        v.set_camera(13.0, 7.0, 1.5)
    a, b = _frame(soft)[0], _frame(gpu)[0]
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1
    print("software:", soft.adapter, "| gpu:", gpu.adapter)


def test_zoom_keeps_the_point_under_the_pointer(stores):
    v = Viewer.offscreen(400, 300)
    v.open(stores["0.5"])
    x0, y0, s0 = v.camera
    v.zoom(2.0, 100.0, 50.0)
    x1, y1, s1 = v.camera
    assert s1 == pytest.approx(s0 / 2)
    assert x0 + 100 * s0 == pytest.approx(x1 + 100 * s1)
    assert y0 + 50 * s0 == pytest.approx(y1 + 50 * s1)
    v.pan(10.0, -4.0)
    assert v.camera[0] == pytest.approx(x1 - 10 * s1)


def test_fit_shows_the_whole_plane(stores, image):
    v = Viewer.offscreen(450, 450)
    v.open(stores["0.5"])
    x, y, s = v.camera
    assert s == pytest.approx(W / 450)                    # the wider side fills the width
    rgba, visible, _ = _frame(v)
    assert visible == 12                                   # all 3 x 4 tiles


@pytest.mark.parametrize("width", [320, 211])            # 211: rows padded on the GPU
def test_render_into_fills_the_callers_buffer(stores, width):
    v = Viewer.offscreen(width, 97)
    v.open(stores["0.5"])
    v.set_camera(21.0, 5.0, 0.75)
    data = v.render_rgba()[0]
    frame = bytearray(4 * width * 97)
    for _ in range(2):                                     # the readback buffer is reused
        visible, loaded = v.render_into(frame)
        assert visible >= 1 and loaded == 0                # tiles cached by the first frame
        assert bytes(frame) == data
    with pytest.raises(ViewerError, match="needs"):
        v.render_into(bytearray(10))
    with pytest.raises((ViewerError, BufferError, TypeError)):
        v.render_into(bytes(len(frame)))                   # read-only


def test_errors_name_the_way_out(tmp_path):
    v = Viewer.offscreen(64, 64)
    with pytest.raises(ViewerError):
        v.open(str(tmp_path / "missing.zarr"))
    with pytest.raises(ViewerError, match="scale"):
        v.set_camera(0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def qapp():
    qt = pytest.importorskip("PyQt6.QtWidgets")
    return qt.QApplication.instance() or qt.QApplication([])


def test_the_qt_widget_offscreen(qapp, stores, image):
    from micro_viewer.qt import ViewerWidget
    w = ViewerWidget(mode="offscreen")
    w.resize(240, 180)
    w.open(stores["0.5"])
    w.viewer.set_contrast(0.0, 65535.0)
    w.viewer.set_camera(0.0, 0.0, 1.0)
    pix = w.grab().toImage()
    r = w.devicePixelRatioF()
    assert pix.width() == round(240 * r)
    want = _expected(image, round(240 * r), round(180 * r), 0.0, 0.0, 1.0, 0.0, 65535.0)
    got = pix.pixelColor(10, 20).red()
    assert abs(got - want[20, 10]) <= 1


@pytest.mark.skipif(sys.platform not in ("win32", "darwin"), reason="native hosting: Windows, macOS")
def test_the_qt_widget_native(qapp, stores):
    from micro_viewer.qt import ViewerWidget
    w = ViewerWidget(mode="native")
    w.resize(320, 240)
    w.open(stores["0.5"])
    w.show()
    qapp.processEvents()
    w.repaint()
    qapp.processEvents()
    assert w.viewer.size[0] >= 320
    assert w.last_stats[0] >= 1
    w.close()
