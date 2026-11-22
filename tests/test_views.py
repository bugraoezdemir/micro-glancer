"""micro-viewer's linked views and projection (Workplan WP14, task V4): the main view, its
YZ and XZ companions and the maximum intensity projection in quarters of one target,
sharing their centre; the projection compared with NumPy.
"""
import numpy as np
import pytest

from micro_viewer import Viewer

ozp = pytest.importorskip("ome_zarr_pyramid")

BACKGROUND = round(0.08 * 255)
GAP = (round(0.03 * 255), round(0.03 * 255), round(0.035 * 255))
D, H, W = 50, 120, 150
ZS = 2.0


def _write(path, levels, axes, scales):
    pyr = ozp.Pyramid().from_arrays(levels, axis_order=axes, scales=scales, version="0.5")
    ozp.IO().write_pyramid(pyr, str(path), overwrite=True)
    return str(path)


@pytest.fixture(scope="module")
def vol():
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    return ((zz * 4099 + yy * 37 + xx * 61) % 65536).astype("uint16")


@pytest.fixture(scope="module")
def smooth():
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    # a bump in the middle: the projection's maximum lies inside the volume
    r2 = ((zz - 25) * 2.0) ** 2 + (yy - 60) ** 2 + (xx - 75) ** 2
    return (20000 * np.exp(-r2 / 1800.0) + xx * 10).astype("uint16")


@pytest.fixture(scope="module")
def stores(tmp_path_factory, vol, smooth):
    tmp = tmp_path_factory.mktemp("views")
    out = {}
    for name, data in (("vol", vol), ("smooth", smooth)):
        levels = [data, np.ascontiguousarray(data[:, ::2, ::2])]
        out[name] = _write(tmp / f"{name}.ome.zarr", levels, "zyx",
                           [[ZS, 1.0, 1.0], [ZS, 2.0, 2.0]])
    return out


def _frame(v):
    v.settle(30.0)
    data, w, h, _ = v.render_rgba()
    return np.frombuffer(data, np.uint8).reshape(h, w, 4)


def _single(path, w, h, centre, scale, orientation=None, mip=False):
    v = Viewer.offscreen(w, h)
    v.open(path)
    v.set_contrast(0.0, 65535.0)
    if orientation:
        v.set_orientation(orientation)
    v.look_at_3d(*centre, scale)
    v.mip = mip
    return _frame(v)


def test_linked_views_show_the_three_planes_and_the_projection(stores):
    centre, scale = (75.3, 60.2, 50.7), 0.9
    v = Viewer.offscreen(402, 302)                       # quarters of 200 x 150
    v.open(stores["vol"])
    v.set_contrast(0.0, 65535.0)
    v.look_at_3d(*centre, scale)
    v.linked = True
    assert v.linked and v.brick_slots() == 0
    rgba = _frame(v)
    quarters = {"xy": rgba[:150, :200], "yz": rgba[:150, 202:], "xz": rgba[152:, :200],
                "mip": rgba[152:, 202:]}
    # the crosshair lies on the centre row and column of the three planes
    keep = np.ones((150, 200), bool)
    keep[74:76, :] = False
    keep[:, 99:101] = False
    for name in ("xy", "yz", "xz"):
        want = _single(stores["vol"], 200, 150, centre, scale, name)
        got = quarters[name]
        bad = (np.abs(got[..., :3].astype(int) - want[..., :3]) > 1).any(axis=-1) & keep
        assert bad.mean() <= 0.002, (name, bad.mean())
    want = _single(stores["vol"], 200, 150, centre, scale, mip=True)
    assert np.array_equal(quarters["mip"][..., :3], want[..., :3])
    # the gaps between the views
    assert (rgba[150:152, :, :3] == GAP).all() and (rgba[:, 200:202, :3] == GAP).all()
    assert (rgba[..., 3] == 255).all()


def test_linked_views_move_together(stores):
    v = Viewer.offscreen(402, 302)
    v.open(stores["vol"])
    v.look_at_3d(75.0, 60.0, 50.0, 1.0)
    v.linked = True
    v.pan(10.0, 4.0, 100.0, 220.0)                       # in the XZ view (bottom left)
    x, y, z = v.center_3d
    assert (x, y, z) == pytest.approx((65.0, 60.0, 46.0))
    v.step_depth(3, 100.0, 220.0)                        # along -y: the XZ view's normal
    assert v.center_3d[1] == pytest.approx(57.0)
    v.step_depth(2, 300.0, 70.0)                         # along x in the YZ view
    assert abs(v.center_3d[0] - 65.0) == pytest.approx(2.0)
    v.fit()
    assert v.center_3d == pytest.approx(((W - 1) / 2, (H - 1) / 2, (D - 1) / 2 * ZS), abs=1e-6)
    p = v.pick(302.0, 75.0)                              # the YZ view's centre: the shared point
    assert p["world"] == pytest.approx(v.center_3d, abs=1.0)


def _mip_reference(volume, v):
    """The projection as the shader takes it: per pixel the ray through the volume's box
    (level 0), samples every 0.8 voxel at most, the nearest voxel's highest value."""
    w, h = v.size
    c = np.array(v.center_3d)
    right, down, normal = (np.array(r) for r in v.orientation)
    s = v.camera[2]
    X, Y = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
    world = c + s * (X[..., None] * right + Y[..., None] * down)
    to_index = np.array([1.0, 1.0, 1.0 / ZS])
    o = world * to_index
    d = normal * to_index
    hi = np.array([W, H, D]) - 0.5
    t0 = np.full((h, w), -1e30)
    t1 = np.full((h, w), 1e30)
    for a in range(3):
        if abs(d[a]) < 1e-12:
            out = (o[..., a] < -0.5) | (o[..., a] > hi[a])
            t0[out], t1[out] = 1.0, -1.0
            continue
        ta, tb = (-0.5 - o[..., a]) / d[a], (hi[a] - o[..., a]) / d[a]
        t0, t1 = np.maximum(t0, np.minimum(ta, tb)), np.minimum(t1, np.maximum(ta, tb))
    inside = t0 <= t1
    step = 0.8 / np.abs(d).max()
    n = np.where(inside, np.ceil((t1 - t0) / step), 0).astype(int)
    best = np.full((h, w), -1.0)
    for i in range(n.max()):
        t = t0 + (i + 0.5) * step
        q = np.floor(o + t[..., None] * d + 0.5).astype(int)
        q = np.clip(q, 0, np.array([W, H, D]) - 1)
        val = volume[q[..., 2], q[..., 1], q[..., 0]].astype(float)
        best = np.where(i < n, np.maximum(best, val), best)
    return best, inside


def _grey(v, lo, hi):
    g = np.clip((np.asarray(v, np.float32) - np.float32(lo)) / np.float32(hi - lo), 0, 1)
    return np.round(g * 255).astype(int)


def test_the_projection_along_z_is_the_maximum_over_z(stores, vol):
    v = Viewer.offscreen(200, 150)
    v.open(stores["vol"])
    v.set_contrast(0.0, 65535.0)
    v.look_at(75.25, 60.25, 1.0)
    v.mip = True
    assert v.layers[0]["mode"] == "flat"                 # the layer itself: the camera
    rgba = _frame(v)
    X, Y = np.meshgrid(np.arange(200) + 0.5 - 100, np.arange(150) + 0.5 - 75)
    ix = np.floor(75.25 + X + 0.5).astype(int)
    iy = np.floor(60.25 + Y + 0.5).astype(int)
    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    want = vol.max(axis=0)[np.clip(iy, 0, H - 1), np.clip(ix, 0, W - 1)]
    got = rgba[..., 0].astype(int)
    assert (np.abs(got - _grey(want, 0, 65535))[inside] <= 1).all()
    assert (got[~inside] == BACKGROUND).all()


@pytest.mark.parametrize("yaw, pitch", [(35.0, 25.0), (90.0, 0.0)])
def test_an_oblique_projection(stores, smooth, yaw, pitch):
    v = Viewer.offscreen(200, 150)
    v.open(stores["smooth"])
    v.set_contrast(0.0, 22000.0)
    v.look_at_3d(75.0, 60.0, 50.0, 1.1)
    v.turn(yaw, pitch)
    v.mip = True
    rgba = _frame(v)
    best, inside = _mip_reference(smooth, v)
    got = rgba[..., 0].astype(int)
    bad = (np.abs(got - _grey(best, 0, 22000)) > 1) & inside
    bad |= (got != BACKGROUND) & ~inside
    assert bad.mean() <= 0.01, bad.mean()


def test_qt_keys_toggle_linked_views_and_the_projection(stores):
    qt = pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from micro_viewer.qt import ViewerWidget
    app = qt.QApplication.instance() or qt.QApplication([])
    w = ViewerWidget(mode="offscreen")
    w.resize(240, 180)
    w.open(stores["vol"])
    w.show()
    app.processEvents()
    QTest.keyClick(w, Qt.Key.Key_L)
    assert w.viewer.linked
    QTest.keyClick(w, Qt.Key.Key_M)
    assert w.viewer.mip
    QTest.keyClick(w, Qt.Key.Key_L)
    QTest.keyClick(w, Qt.Key.Key_M)
    assert not w.viewer.linked and not w.viewer.mip
    w.close()


def test_a_large_projection_is_drawn_smaller_and_scaled(stores, smooth):
    """Over a million pixels the projection is drawn with fewer and scaled into place:
    close to the full one on a smooth volume."""
    v = Viewer.offscreen(1400, 1000)
    v.open(stores["smooth"])
    v.set_contrast(0.0, 22000.0)
    v.look_at_3d(75.0, 60.0, 50.0, 0.12)
    v.turn(30.0, 20.0)
    v.mip = True
    rgba = _frame(v)
    best, inside = _mip_reference(smooth, v)
    core = inside.copy()
    core[[0, -1], :] = False
    core[:, [0, -1]] = False
    diff = np.abs(rgba[..., 0].astype(int) - _grey(best, 0, 22000))[core]
    assert diff.mean() < 1.5 and np.percentile(diff, 99) <= 6
