"""micro-glancer's transfer functions and histograms (Workplan WP14, task V4): opacity
curves for the volume projection, the isosurface level, skipping empty bricks, the
histogram (of the view, the whole plane, the whole volume) and the channel panel.
"""
import json

import numpy as np
import pytest

from micro_glancer import Viewer, ViewerError

ozp = pytest.importorskip("ome_zarr_pyramid")

BACKGROUND = round(0.08 * 255)
D, H, W = 50, 120, 150
ZS = 2.0


def _write(path, levels, axes, scales):
    pyr = ozp.Pyramid().from_arrays(levels, axis_order=axes, scales=scales, version="0.5")
    ozp.IO().write_pyramid(pyr, str(path), overwrite=True)
    return str(path)


@pytest.fixture(scope="module")
def sparse():
    """Empty but for a block in the middle (most bricks hold only zeros)."""
    v = np.zeros((D, H, W), "uint16")
    zz, yy, xx = np.mgrid[0:12, 0:40, 0:30]
    v[20:32, 40:80, 60:90] = (5000 + zz * 900 + yy * 40 + xx * 70).astype("uint16")
    return v


@pytest.fixture(scope="module")
def smooth():
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    r2 = ((zz - 25) * 2.0) ** 2 + (yy - 60) ** 2 + (xx - 75) ** 2
    return (20000 * np.exp(-r2 / 1800.0) + xx * 10).astype("uint16")


@pytest.fixture(scope="module")
def flat():
    yy, xx = np.mgrid[0:300, 0:400]
    return ((yy * 37 + xx * 61) % 9000).astype("uint16")


@pytest.fixture(scope="module")
def stores(tmp_path_factory, sparse, smooth, flat):
    tmp = tmp_path_factory.mktemp("transfer")
    out = {}
    for name, data in (("sparse", sparse), ("smooth", smooth)):
        out[name] = _write(tmp / f"{name}.ome.zarr", [data], "zyx", [[ZS, 1.0, 1.0]])
    out["flat"] = _write(tmp / "flat.ome.zarr", [flat], "yx", [[1.0, 1.0]])
    out["uniform"] = _write(tmp / "uniform.ome.zarr", [np.full((D, 40, 60), 1000, "uint16")],
                            "zyx", [[ZS, 1.0, 1.0]])
    return out


def _frame(v):
    v.settle(30.0)
    data, w, h, _ = v.render_rgba()
    return np.frombuffer(data, np.uint8).reshape(h, w, 4)


def _volume(path, w=120, h=80):
    v = Viewer.offscreen(w, h)
    v.open(path)
    return v


# ------------------------------------------------------------------ opacity curves

def test_opacity_curves_shape_the_volume(stores):
    v = _volume(stores["uniform"])
    v.set_contrast(0.0, 2000.0)                          # g = 0.5 everywhere
    v.look_at(29.5, 19.5, 1.0)
    v.projection, v.density, v.shading, v.mip = "volume", 0.02, False, True
    v.set_channel(0, opacity=[(0.0, 0.0), (1.0, 0.0)])   # clear: the background shows
    assert (_frame(v)[25:55, 35:85, :3] == BACKGROUND).all()
    v.set_channel(0, opacity=[(0.0, 1.0), (1.0, 1.0)])   # opaque per voxel x density
    inner = _frame(v)[25:55, 35:85, 0].astype(int)
    alpha = 1 - (1 - 1.0 * 0.02) ** 0.8
    a = 1 - (1 - alpha) ** int(np.ceil(D / 0.8))
    assert np.abs(inner - round((a + 0.08 * (1 - a)) * 255)).max() <= 2
    assert v.channels[0]["opacity"] == [(0.0, 1.0), (1.0, 1.0)]
    v.set_channel(0, opacity=[])                         # back to a straight rise
    assert v.channels[0]["opacity"] is None
    with pytest.raises(ViewerError, match="opacity curve"):
        v.set_channel(0, opacity=[(0.5, 0.2), (0.1, 0.9)])
    with pytest.raises(ViewerError, match="opacity curve"):
        v.set_channel(0, opacity=[(0.0, 1.5)])


# ------------------------------------------------------------------ the isosurface level

def test_the_isosurface_level(stores, smooth):
    v = _volume(stores["smooth"], 200, 150)
    v.set_contrast(0.0, 24000.0)
    v.look_at(75.25, 60.25, 1.0)
    v.projection, v.shading, v.mip = "iso", False, True
    v.set_channel(0, iso=8000.0)
    assert v.channels[0]["iso"] == 8000.0
    rgba = _frame(v)
    X, Y = np.meshgrid(np.arange(200) + 0.5 - 100, np.arange(150) + 0.5 - 75)
    ix = np.clip(np.floor(75.25 + X + 0.5).astype(int), 0, W - 1)
    iy = np.clip(np.floor(60.25 + Y + 0.5).astype(int), 0, H - 1)
    top = smooth.max(axis=0)[iy, ix]
    got = rgba[..., 0]
    assert (got[top >= 8000 * 1.03] == 255).mean() > 0.99
    assert (got[top <= 8000 * 0.97] == BACKGROUND).mean() > 0.99
    v.set_channel(0, iso=float("nan"))                   # back to the middle (12000)
    assert v.channels[0]["iso"] is None
    got = _frame(v)[..., 0]
    assert (got[(top >= 9000) & (top <= 11000)] == BACKGROUND).mean() > 0.99


# ------------------------------------------------------------------ skipping empty bricks

@pytest.mark.parametrize("mode", ["mip", "volume", "iso"])
@pytest.mark.parametrize("linear", [False, True])
def test_skipping_empty_bricks_changes_nothing(stores, mode, linear):
    frames = []
    for skip in (True, False):
        v = _volume(stores["sparse"], 220, 160)
        v.set_contrast(100.0, 20000.0)                   # the zeros below the window
        v.look_at_3d(75.0, 60.0, 50.0, 0.8)
        v.turn(25.0, 30.0)
        v.projection, v.density, v.mip = mode, 0.3, True
        v.set_interpolation("linear" if linear else "nearest")
        v.skip_empty = skip
        assert v.skip_empty == skip
        frames.append(_frame(v))
    assert np.array_equal(frames[0], frames[1])
    assert (frames[0][..., :3] != BACKGROUND).any()      # the block shows


# ------------------------------------------------------------------ histograms

def test_histograms_of_the_whole_image_and_the_view(stores, flat):
    v = Viewer.offscreen(200, 150)
    v.open(stores["flat"])
    h = v.histogram(0, bins=64, whole=True)
    want, edges = np.histogram(flat, bins=64, range=(flat.min(), flat.max()))
    assert h["counts"] == want.tolist() and np.allclose(h["edges"], edges)
    assert (h["lo"], h["hi"], h["level"]) == (flat.min(), flat.max(), 0)
    v.look_at(100.25, 80.25, 1.0)                        # 200 x 150 pixels in view
    h = v.histogram(0, bins=32, range=(0.0, 9000.0))
    box = flat[5:156, 0:201]                             # the view's box, a pixel around
    inner = flat[6:154, 1:199]
    total = sum(h["counts"])
    assert inner.size <= total <= box.size
    with pytest.raises(ViewerError, match="bins"):
        v.histogram(0, bins=0)


def test_a_histogram_job_runs_on_another_thread_while_the_viewer_draws(stores, flat):
    # the panel reads histograms this way: a whole plate in view reads every field
    from concurrent.futures import ThreadPoolExecutor

    v = Viewer.offscreen(200, 150)
    v.open(stores["flat"])
    for kw in ({"bins": 64, "whole": True}, {"bins": 32, "range": (0.0, 9000.0)}):
        job = v.histogram_job(0, **kw)
        with ThreadPoolExecutor(1) as pool:
            running = pool.submit(job.run)
            v.render_rgba()                              # the viewer is free meanwhile
            h = running.result(timeout=30)
        assert h == v.histogram(0, **kw)
        assert job.level == h["level"]
    with pytest.raises(ViewerError, match="bins"):
        v.histogram_job(0, bins=0)


def test_the_histogram_of_a_whole_volume(stores, sparse):
    v = _volume(stores["sparse"])
    h = v.histogram(0, bins=50, whole=True)
    want, _ = np.histogram(sparse, bins=50, range=(sparse.min(), sparse.max()))
    assert h["counts"] == want.tolist()


def test_curves_and_levels_in_the_view_state(stores):
    a = _volume(stores["smooth"])
    a.set_channel(0, opacity=[(0.1, 0.0), (0.6, 0.8)], iso=5000.0)
    b = Viewer.offscreen(120, 80)
    b.set_view_state(a.view_state())
    assert b.channels[0]["opacity"] == [(0.1, 0.0), (0.6, 0.8)]
    assert b.channels[0]["iso"] == 5000.0
    doc = json.loads(a.view_state())
    assert doc["layers"][0]["channels"][0]["iso"] == 5000.0


# ------------------------------------------------------------------ the channel panel

@pytest.fixture(scope="module")
def qapp():
    qt = pytest.importorskip("PyQt6.QtWidgets")
    return qt.QApplication.instance() or qt.QApplication([])


def test_the_panel_drags_contrast_level_and_curve(qapp, stores):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from micro_glancer.panel import ChannelPanel
    from micro_glancer.qt import ViewerWidget

    w = ViewerWidget(mode="offscreen")
    w.resize(200, 150)
    w.open(stores["smooth"])
    w.show()
    qapp.processEvents()
    panel = ChannelPanel(w)
    panel.resize(320, 240)
    panel.show()
    qapp.processEvents()
    panel.refresh()
    hist = panel.hist
    assert hist.wait()                     # read on a background thread
    assert hist._hist is not None and sum(hist._hist["counts"]) > 0
    v = w.viewer
    v.set_contrast(1000.0, 15000.0)
    hist.update()
    y = hist.height() // 2
    x = round(hist._x(15000.0))
    QTest.mousePress(hist, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x, y))
    QTest.mouseMove(hist, QPoint(x - 40, y))
    QTest.mouseRelease(hist, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       QPoint(x - 40, y))
    assert v.channels[0]["hi"] < 15000.0 and v.channels[0]["lo"] == 1000.0
    # the isosurface level (dashed line, at the window's middle at first)
    v.projection, v.mip = "iso", True
    hist.update()
    ch = v.channels[0]
    xi = round(hist._x(ch["lo"] + (ch["hi"] - ch["lo"]) / 2))
    QTest.mousePress(hist, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(xi, y))
    QTest.mouseMove(hist, QPoint(xi + 15, y))
    QTest.mouseRelease(hist, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       QPoint(xi + 15, y))
    assert v.channels[0]["iso"] is not None
    # the opacity curve: a double-click adds a point, a right-click takes it away
    v.projection = "volume"
    hist.update()
    xm = round(hist._x(ch["lo"] + 0.5 * (ch["hi"] - ch["lo"])))
    QTest.mouseDClick(hist, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      QPoint(xm, 10))
    pts = v.channels[0]["opacity"]
    assert pts is not None and len(pts) == 3 and pts[1][1] > 0.8
    QTest.mouseClick(hist, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(xm, 10))
    assert len(v.channels[0]["opacity"]) == 2
    panel.close()
    w.close()
