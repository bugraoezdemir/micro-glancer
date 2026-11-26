"""micro-glancer's volumes cut at any angle (Workplan WP14, task V4 part 1): bricks, XZ and
YZ views, oblique planes, nearest and trilinear sampling, z shared by the layers, 3-D
layer transforms saved into NGFF 0.6, and the view state's orientation.

Frames are compared with NumPy (and SciPy for trilinear sampling): a screen pixel's
centre goes to the world through the camera (centre, scale, the screen's right and down
directions), then to the level's voxel indices (scale and translation per axis; voxel
centres at integers): the nearest voxel, or trilinear between the eight around it. On
planes at an angle a few pixels on the edges between two voxels may land on the other
one (float32 on the GPU): a small share is allowed.
"""
import json
import os
import shutil

import numpy as np
import pytest

from micro_glancer import Viewer, ViewerError
from micro_glancer.save import save_transform

ozp = pytest.importorskip("ome_zarr_pyramid")

BACKGROUND = round(0.08 * 255)
D, H, W = 50, 120, 150
ZS = 2.0                                     # world units per z plane (x, y: 1)


def _write(path, levels, axes, scales, version="0.5"):
    pyr = ozp.Pyramid().from_arrays(levels, axis_order=axes, scales=scales, version=version)
    ozp.IO().write_pyramid(pyr, str(path), overwrite=True)
    return str(path)


@pytest.fixture(scope="module")
def vol():
    """Every voxel its own value (wrapping), so a wrong voxel shows."""
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    return ((zz * 4099 + yy * 37 + xx * 61) % 65536).astype("uint16")


@pytest.fixture(scope="module")
def smooth():
    """A slowly changing volume, for trilinear sampling."""
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    return (zz * 300 + yy * 50 + xx * 20).astype("uint16")


@pytest.fixture(scope="module")
def stores(tmp_path_factory, vol, smooth):
    tmp = tmp_path_factory.mktemp("volume")
    out = {}
    for name, data in (("vol", vol), ("smooth", smooth)):
        levels = [data, np.ascontiguousarray(data[:, ::2, ::2])]
        out[name] = _write(tmp / f"{name}.ome.zarr", levels, "zyx",
                           [[ZS, 1.0, 1.0], [ZS, 2.0, 2.0]])
    out["vol06"] = _write(tmp / "vol06.ome.zarr", [vol], "zyx", [[ZS, 1.0, 1.0]], "0.6")
    yy, xx = np.mgrid[0:H, 0:W]
    out["flat"] = _write(tmp / "flat.ome.zarr", [((yy * 37 + xx * 61) % 65536).astype("uint16")],
                         "yx", [[1.0, 1.0]])
    return out


def _frame(v):
    v.settle(30.0)
    data, w, h, _ = v.render_rgba()
    return np.frombuffer(data, np.uint8).reshape(h, w, 4)


def _world(v):
    """Every screen pixel centre's world point (h, w, 3), from the viewer's camera."""
    w, h = v.size
    c = np.array(v.center_3d)
    right, down, _ = (np.array(r) for r in v.orientation)
    s = v.camera[2]
    X, Y = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
    return c + s * (X[..., None] * right + Y[..., None] * down)


def _index(world, transform=None):
    """World points to level-0 voxel indices (x, y, z), a layer's 3 x 4 transform undone."""
    p = world
    if transform is not None:
        a = np.vstack([np.array(transform), [0, 0, 0, 1]])
        inv = np.linalg.inv(a)
        p = p @ inv[:3, :3].T + inv[:3, 3]
    return p / np.array([1.0, 1.0, ZS])


def _nearest(volume, idx):
    v = np.floor(idx + 0.5).astype(int)
    inside = ((v >= 0) & (v < np.array([W, H, D]))).all(axis=-1)
    c = [np.clip(v[..., a], 0, n - 1) for a, n in enumerate((W, H, D))]
    return volume[c[2], c[1], c[0]], inside


def _grey(v, lo=0.0, hi=65535.0):
    g = np.clip((np.asarray(v, np.float32) - np.float32(lo)) / np.float32(hi - lo), 0, 1)
    return np.round(g * 255).astype(int)


def _match(got, want, inside, share=0.0, tol=1):
    bad = (np.abs(got.astype(int) - want) > tol) & inside
    bad |= (got != BACKGROUND) & ~inside
    assert bad.mean() <= share, (bad.mean(), np.argwhere(bad)[:5])


# ------------------------------------------------------------------ views along the axes

@pytest.mark.parametrize("name", ["xz", "yz"])
def test_xz_and_yz_views_cut_the_volume(stores, vol, name):
    v = Viewer.offscreen(200, 120)
    v.open(stores["vol"])
    v.set_contrast(0.0, 65535.0)
    v.set_orientation(name)
    v.look_at_3d(75.25, 60.25, 50.3, 1.0)
    assert v.layers[0]["mode"] == "volume" and v.layers[0]["volume"]
    rgba = _frame(v)
    assert v.brick_slots() > 0 and v.level == 0
    val, inside = _nearest(vol, _index(_world(v)))
    assert inside.any() and (~inside).any()
    _match(rgba[..., 0], _grey(val), inside)


def test_the_xy_view_of_a_volume_stays_flat(stores, vol):
    v = Viewer.offscreen(200, 120)
    v.open(stores["vol"])
    assert v.layers[0]["mode"] == "flat"
    v.set_contrast(0.0, 65535.0)
    v.set_position("z", 17)
    assert v.center_3d[2] == pytest.approx(17 * ZS)
    v.look_at(75.25, 60.25, 1.0)
    rgba = _frame(v)
    val, inside = _nearest(vol, _index(_world(v)))
    _match(rgba[..., 0], _grey(val), inside)
    assert v.brick_slots() == 0                          # no bricks for flat views


@pytest.mark.parametrize("yaw, pitch", [(30.0, 20.0), (-55.0, 70.0)])
def test_an_oblique_plane(stores, vol, yaw, pitch):
    v = Viewer.offscreen(220, 160)
    v.open(stores["vol"])
    v.set_contrast(0.0, 65535.0)
    v.look_at_3d(75.25, 60.25, 50.3, 0.8)
    v.turn(yaw, pitch)
    o = np.array(v.orientation)
    assert np.allclose(o @ o.T, np.eye(3), atol=1e-9)    # unit and perpendicular
    assert np.allclose(np.cross(o[0], o[1]), o[2])
    assert v.center_3d == pytest.approx((75.25, 60.25, 50.3))
    rgba = _frame(v)
    val, inside = _nearest(vol, _index(_world(v)))
    _match(rgba[..., 0], _grey(val), inside, share=0.01)


def test_trilinear_sampling(stores, smooth):
    from scipy import ndimage
    v = Viewer.offscreen(200, 140)
    v.open(stores["smooth"])
    v.set_contrast(0.0, 24000.0)
    v.look_at_3d(70.3, 55.1, 48.7, 0.6)
    v.turn(25.0, 35.0)
    v.set_interpolation("linear")
    assert v.layers[0]["interpolation"] == "linear"
    rgba = _frame(v)
    idx = _index(_world(v))
    inside = _nearest(smooth, idx)[1]
    coords = np.stack([np.clip(idx[..., 2], 0, D - 1), np.clip(idx[..., 1], 0, H - 1),
                       np.clip(idx[..., 0], 0, W - 1)])
    want = ndimage.map_coordinates(smooth.astype(np.float64), coords, order=1)
    _match(rgba[..., 0], _grey(want, 0.0, 24000.0), inside, share=0.01, tol=2)
    v.set_interpolation("nearest")
    with pytest.raises(ViewerError, match="interpolation"):
        v.set_interpolation("cubic")


# ------------------------------------------------------------------ picking, z, stepping

def test_picking_a_voxel(stores, vol):
    v = Viewer.offscreen(200, 120)
    v.open(stores["vol"])
    v.set_orientation("xz")
    v.look_at_3d(75.25, 60.25, 50.3, 1.0)
    v.settle(30.0)
    p = v.pick(120.0, 70.0)
    x, y, z = p["voxel"]
    wx, wy, wz = p["world"]
    assert (x, y, z) == (int(np.floor(wx + 0.5)), int(np.floor(wy + 0.5)),
                         int(np.floor(wz / ZS + 0.5)))
    assert p["values"][0]["value"] == vol[z, y, x] and p["values"][0]["level"] == 0


def test_z_is_shared_and_steps_go_along_the_view(stores):
    v = Viewer.offscreen(200, 120)
    v.open(stores["vol"])
    v.add(stores["smooth"])
    v.set_position("z", 10, layer=0)                     # moves the camera: both layers
    zs = [dict((n, i) for n, i, _ in l["position"])["z"] for l in v.layers]
    assert zs == [10.0, 10.0]
    v.step_depth(1)
    assert v.position[0][1] == pytest.approx(11.0)
    v.set_orientation("xz")                              # looking along y now
    y0 = v.center_3d[1]
    v.step_depth(3)
    assert abs(v.center_3d[1] - y0) == pytest.approx(3.0)
    with pytest.raises(ViewerError, match="xy"):
        v.set_orientation("sideways")


def test_a_two_d_image_is_off_at_an_angle(stores):
    v = Viewer.offscreen(160, 120)
    v.open(stores["flat"])
    v.add(stores["vol"])
    v.turn(20.0, 15.0)
    modes = [l["mode"] for l in v.layers]
    assert modes == ["off", "volume"]
    v.set_orientation("xy")
    assert [l["mode"] for l in v.layers] == ["flat", "flat"]
    with pytest.raises(ViewerError, match="plane"):
        v.set_transform([[1, 0, 0, 0], [0, 0.9, 0.4, 0], [0, -0.4, 0.9, 0]], layer=0)


# ------------------------------------------------------------------ 3-D layer transforms

def test_a_tilted_layer_is_drawn_and_saved_into_0_6(stores, vol, tmp_path):
    path = str(tmp_path / "tilt06.ome.zarr")
    shutil.copytree(stores["vol06"], path)
    v = Viewer.offscreen(200, 140)
    v.open(path)
    v.set_contrast(0.0, 65535.0)
    v.look_at_3d(75.25, 60.25, 50.3, 1.0)
    v.turn_layer(12.0, -8.0)
    t = v.transform_3d()
    assert v.layers[0]["mode"] == "volume"               # tilted: drawn from bricks
    rgba = _frame(v)
    val, inside = _nearest(vol, _index(_world(v), t))
    _match(rgba[..., 0], _grey(val), inside, share=0.01)
    assert "affine" in save_transform(v)
    with open(os.path.join(path, "zarr.json")) as f:
        edge = json.load(f)["attributes"]["ome"]["multiscales"][0]["coordinateTransformations"][0]
    m = np.array(edge["affine"])
    assert m.shape == (3, 4) and abs(m[0, 1]) > 1e-3     # z, y, x mixed
    w = Viewer.offscreen(200, 140)
    w.open(path)
    assert np.allclose(w.transform_3d(), t)


# ------------------------------------------------------------------ the view state

def test_the_view_state_keeps_the_orientation(stores):
    a = Viewer.offscreen(180, 130)
    a.open(stores["vol"])
    a.look_at_3d(70.0, 50.0, 40.0, 0.9)
    a.turn(40.0, -25.0)
    a.set_interpolation("linear")
    state = a.view_state()
    doc = json.loads(state)
    assert doc["version"] == 3 and len(doc["camera"]["center"]) == 3
    assert doc["layers"][0]["interpolation"] == "linear"
    assert len(doc["layers"][0]["transform"]) == 3
    b = Viewer.offscreen(180, 130)
    b.set_view_state(state)
    assert np.allclose(b.orientation, a.orientation)
    assert np.array_equal(_frame(a), _frame(b))
    # a version 2 camera (a centre in the plane and a turn) still applies
    b.set_view_state(json.dumps({"version": 2, "camera": {"center": [10, 20], "scale": 2.0,
                                                          "rotation": 90.0}}))
    assert b.rotation == pytest.approx(90.0) and b.center == pytest.approx((10.0, 20.0))
    assert b.layers[0]["mode"] == "flat"


# ------------------------------------------------------------------ the Qt widget

@pytest.fixture(scope="module")
def qapp():
    qt = pytest.importorskip("PyQt6.QtWidgets")
    return qt.QApplication.instance() or qt.QApplication([])


def test_the_qt_widget_turns_the_view(qapp, stores):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from micro_glancer.qt import ViewerWidget

    w = ViewerWidget(mode="offscreen")
    w.resize(240, 180)
    w.open(stores["vol"])
    w.show()
    qapp.processEvents()
    v = w.viewer
    alt = Qt.KeyboardModifier.AltModifier
    QTest.mousePress(w, Qt.MouseButton.LeftButton, alt, QPoint(100, 100))
    QTest.mouseMove(w, QPoint(140, 80))
    QTest.mouseRelease(w, Qt.MouseButton.LeftButton, alt, QPoint(140, 80))
    assert v.layers[0]["mode"] == "volume"               # turned out of the xy plane
    QTest.keyClick(w, Qt.Key.Key_2)                      # look along y: x right, z down
    assert np.allclose(v.orientation[0], [1, 0, 0]) and np.allclose(v.orientation[1], [0, 0, 1])
    y0 = v.center_3d[1]
    QTest.keyClick(w, Qt.Key.Key_PageUp)                 # one voxel along the view
    assert abs(v.center_3d[1] - y0) == pytest.approx(1.0)
    QTest.keyClick(w, Qt.Key.Key_I)                      # trilinear on, off
    assert v.layers[0]["interpolation"] == "linear"
    QTest.keyClick(w, Qt.Key.Key_1)
    assert v.layers[0]["mode"] == "flat"
    w.close()


def test_deep_chunks_flat_then_turned(tmp_path, vol):
    """Chunks through all of z (as MouseBrain's 1 x 41 x 96 x 128): a flat tile at
    z = 0 and a brick starting at z = 0 share their block's corner, not its depth."""
    pyr = ozp.Pyramid().from_arrays([vol], axis_order="zyx", scales=[[ZS, 1.0, 1.0]],
                                    version="0.5")
    path = str(tmp_path / "deep.ome.zarr")
    ozp.IO().write_pyramid(pyr, path, overwrite=True, chunk_shape=(D, 48, 64))
    v = Viewer.offscreen(200, 140)
    v.open(path)
    v.set_contrast(0.0, 65535.0)
    v.set_position("z", 0)
    v.look_at(75.25, 60.25, 1.0)
    v.settle(30.0)                                       # flat tiles at z = 0
    v.look_at_3d(75.25, 60.25, 20.3, 1.0)
    v.turn(30.0, 20.0)
    rgba = _frame(v)
    assert v.error is None
    val, inside = _nearest(vol, _index(_world(v)))
    _match(rgba[..., 0], _grey(val), inside, share=0.01)


# ------------------------------------------------------------------ deep chunks (V12)

@pytest.fixture(scope="module")
def deep(tmp_path_factory, vol):
    """The volume in chunks 16 planes deep, sharded 2 x 2 x 2: a plane's tiles are put
    together from whole decode units (the inner chunks), each serving 16 planes."""
    path = tmp_path_factory.mktemp("deep") / "deep.ome.zarr"
    pyr = ozp.Pyramid().from_arrays([vol], axis_order="zyx", scales=[[ZS, 1.0, 1.0]],
                                    version="0.5")
    ozp.IO().write_pyramid(pyr, str(path), overwrite=True, chunk_shape=(16, 32, 32),
                           shard_coefficients=(2, 2, 2))
    with open(os.path.join(str(path), "0", "zarr.json")) as fh:
        codecs = json.load(fh)["codecs"]
    assert codecs[0]["name"] == "sharding_indexed"
    assert codecs[0]["configuration"]["chunk_shape"] == [16, 32, 32]
    return str(path)


def test_planes_of_deep_chunks_come_out_right(deep, vol):
    # every plane, inside a unit and across unit boundaries, from the cached units
    v = Viewer.offscreen(200, 150)
    v.open(deep)
    v.look_at(75.25, 60.25, 1.0)
    for z in (0, 1, 15, 16, 17, 31, 32, 49, 30, 2):
        v.set_position("z", z)
        v.settle(30.0)
        for sx, sy in ((100.0, 75.0), (60.0, 40.0), (160.0, 120.0)):     # all inside
            p = v.pick(sx, sy)
            x, y = p["pixel"]
            assert p["values"][0]["value"] == vol[z, y, x], (z, x, y)


def test_a_plane_change_never_shows_an_empty_frame(deep):
    # a step and a jump: the frame right after shows the previous plane where the new
    # one has nothing yet (an empty frame was the "wink"), then the new plane
    v = Viewer.offscreen(200, 150)
    v.open(deep)
    v.fit()
    v.set_position("z", 5)
    v.settle(30.0)
    for target in (6, 40, 41, 3):
        v.set_position("z", target)
        data, w, h, _ = v.render_rgba()                  # before the new tiles arrive
        first = np.frombuffer(data, np.uint8).reshape(h, w, 4)[..., :3]
        assert first.max() > 0, target
        v.settle(30.0)
        assert v.position[0][1] == pytest.approx(target)


# ------------------------------------------------------------------ view controls (V14)

def test_the_depth_range_and_setting_the_depth(stores):
    v = Viewer.offscreen(200, 150)
    v.open(stores["vol"])
    lo, hi, now, step = v.depth_range                     # looking along z
    assert step == pytest.approx(ZS)
    assert (lo, hi) == (pytest.approx(-0.5 * ZS), pytest.approx((D - 0.5) * ZS))
    v.set_depth(10 * ZS)
    assert v.position[0][1] == pytest.approx(10.0)
    assert v.depth_range[2] == pytest.approx(10 * ZS)
    v.turn(30.0, 0.0)                                     # the range follows the view
    lo2, hi2, _, step2 = v.depth_range
    assert hi2 - lo2 > hi - lo                            # across the volume's diagonal
    v.step_depth(1)                                       # a step is what the keys move
    assert v.depth_range[2] - _ == pytest.approx(step2)


def test_the_dataset_axes_follow_the_layer_not_the_view(stores):
    v = Viewer.offscreen(200, 150)
    v.open(stores["vol"])
    assert np.allclose(v.axes(), np.eye(3))
    v.turn(40.0, 20.0)                                    # the view turns: axes stay
    assert np.allclose(v.axes(), np.eye(3))
    v.turn_layer(30.0, 0.0)                               # the layer turns: axes turn
    assert not np.allclose(v.axes(), np.eye(3))
    a = np.array(v.axes())
    assert np.allclose(a @ a.T, np.eye(3), atol=1e-9)    # still unit and perpendicular


def test_the_view_bar_drives_the_viewer(qapp, stores):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from micro_glancer.controls import ViewBar
    from micro_glancer.qt import ViewerWidget

    w = ViewerWidget(mode="offscreen")
    w.resize(240, 180)
    w.open(stores["vol"])
    bar = ViewBar(w)
    w.show()
    bar.show()
    qapp.processEvents()
    w._changed()
    qapp.processEvents()
    v = w.viewer
    assert not bar.isHidden()
    # Depth: one step per plane along z
    assert bar.depth.slider.maximum() == D - 1
    bar.depth.slider.setValue(7)
    assert v.position[0][1] == pytest.approx(7.0)
    assert bar.depth.readout.text() == f"depth 7 of {D - 1}"
    # Tilt + by the 5 degree step: the viewing direction turns by 5 degrees
    n0 = np.array(v.orientation[2])
    row = [r for r in range(1, 4) if bar.angles.layout().itemAtPosition(r, 0).widget().text() == "Tilt"][0]
    bar.angles.layout().itemAtPosition(row, 3).widget().click()
    n1 = np.array(v.orientation[2])
    assert np.degrees(np.arccos(np.clip(n0 @ n1, -1, 1))) == pytest.approx(5.0, abs=1e-6)
    # Roll by an exact 12.5 degrees: the direction stays, the screen's right turns
    r0 = np.array(v.orientation[0])
    field = bar.angles.fields["Roll"]
    field.setValue(12.5)
    field.editingFinished.emit()
    assert np.allclose(v.orientation[2], n1)
    r1 = np.array(v.orientation[0])
    assert np.degrees(np.arccos(np.clip(r0 @ r1, -1, 1))) == pytest.approx(12.5, abs=1e-6)
    assert field.value() == 0.0
    assert "from z" in bar.angles.readout.text()
    # the compass draws, and a double-click on z's arrow looks along z again
    bar.compass.grab()
    tip = bar.compass._tips["z"]
    QTest.mouseDClick(bar.compass, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      QPoint(round(tip.x()), round(tip.y())))
    assert abs(np.array(v.orientation[2]) @ np.array(v.axes()[2])) == pytest.approx(1.0)
    # a drag inside the ball, started sideways, only turns (no tilt mixed in)
    c = bar.compass.rect().center()
    QTest.mousePress(bar.compass, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, c)
    for dx in range(2, 42, 4):
        QTest.mouseMove(bar.compass, QPoint(c.x() + dx, c.y() + (1 if dx % 8 else -1)))
    QTest.mouseRelease(bar.compass, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       QPoint(c.x() + 40, c.y()))
    down = np.array(v.orientation[1])
    assert abs(down @ np.array(v.axes()[1])) == pytest.approx(1.0)   # turned about down only
    bar.close()
    w.close()
