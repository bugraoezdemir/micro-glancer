"""micro-glancer's layers, transforms and the turned camera (Workplan WP14, task V9 part 1),
colour maps and contrast from the view (V7's rest), reading coarser while the view moves
on slow sources, and saving a layer's transform into its image.

Frames are compared with NumPy: a screen pixel's centre goes to the world through the
camera (centre, scale, turn), through the inverse of the layer's transform, and to the
level pixel whose centre is nearest. Off the 90-degree turns a few pixels on the edges
between two image pixels may land on the other one (float32 on the GPU): a small share
is allowed.
"""
import json
import os
import shutil
import time
import zipfile

import numpy as np
import pytest

from micro_glancer import ALIGNED, COLOUR_MAPS, Viewer, ViewerError
from micro_glancer.save import save_transform

ozp = pytest.importorskip("ome_zarr_pyramid")

BACKGROUND = round(0.08 * 255)
H, W = 700, 900
VIRIDIS_0, VIRIDIS_1 = (68, 1, 84), (253, 231, 37)


def _write(path, levels, axes, scales, version="0.5"):
    pyr = ozp.Pyramid().from_arrays(levels, axis_order=axes, scales=scales, version=version)
    ozp.IO().write_pyramid(pyr, str(path), overwrite=True)
    return str(path)


@pytest.fixture(scope="module")
def image():
    yy, xx = np.mgrid[0:H, 0:W]
    return ((yy * 37 + xx * 61) % 65536).astype("uint16")


@pytest.fixture(scope="module")
def stores(tmp_path_factory, image):
    tmp = tmp_path_factory.mktemp("layers")
    out = {v: _write(tmp / f"img{v}.ome.zarr", [image], "yx", [[1.0, 1.0]], v)
           for v in ("0.4", "0.5", "0.6")}
    levels = [image, np.ascontiguousarray(image[::2, ::2]),
              np.ascontiguousarray(image[::4, ::4])]
    out["pyramid"] = _write(tmp / "pyr.ome.zarr", levels, "yx",
                            [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    return out


def _frame(v, settle=True):
    if settle:
        v.settle(30.0)
    data, w, h, _ = v.render_rgba()
    return np.frombuffer(data, np.uint8).reshape(h, w, 4)


def _world(w, h, center, scale, rotation):
    """Every screen pixel centre's world point (x, y)."""
    t = np.radians(rotation)
    c, s = np.cos(t), np.sin(t)
    X, Y = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
    return center[0] + scale * (c * X + s * Y), center[1] + scale * (-s * X + c * Y)


def _look(level, wx, wy, transform=None):
    """The level-0 pixel (scale 1, no translation) under world points, the layer's
    `transform` (2 x 3 over x, y) undone, and which points the image covers."""
    if transform is not None:
        a = np.array(transform + [[0.0, 0.0, 1.0]])
        inv = np.linalg.inv(a)
        wx, wy = (inv[0, 0] * wx + inv[0, 1] * wy + inv[0, 2],
                  inv[1, 0] * wx + inv[1, 1] * wy + inv[1, 2])
    ix = np.floor(wx + 0.5).astype(int)
    iy = np.floor(wy + 0.5).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < level.shape[1]) & (iy < level.shape[0])
    v = level[np.clip(iy, 0, level.shape[0] - 1), np.clip(ix, 0, level.shape[1] - 1)]
    return v, inside


def _grey(v, lo=0.0, hi=65535.0):
    g = np.clip((v.astype(np.float32) - np.float32(lo)) / np.float32(hi - lo), 0, 1)
    return np.round(g * 255).astype(int)


def _match(got, want, inside, share=0.0):
    """`got` matches `want` within one step inside, the background outside; off by
    more on at most `share` of the pixels (edges between two image pixels)."""
    bad = (np.abs(got.astype(int) - want) > 1) & inside
    bad |= (got != BACKGROUND) & ~inside
    assert bad.mean() <= share, (bad.mean(), np.argwhere(bad)[:5])


# ------------------------------------------------------------------ the camera turns

@pytest.mark.parametrize("rotation, share", [(90.0, 0.0), (180.0, 0.0), (30.0, 0.004)])
def test_the_camera_turns(stores, image, rotation, share):
    v = Viewer.offscreen(320, 240)
    v.open(stores["0.5"])
    v.set_contrast(0.0, 65535.0)
    v.look_at(450.25, 350.25, 1.25)
    v.set_rotation(rotation)
    assert v.rotation == pytest.approx(rotation)
    assert v.center == pytest.approx((450.25, 350.25))
    rgba = _frame(v)
    val, inside = _look(image, *_world(320, 240, (450.25, 350.25), 1.25, rotation))
    _match(rgba[..., 0], _grey(val), inside, share)


def test_turning_keeps_the_point_under_the_pointer(stores):
    v = Viewer.offscreen(320, 240)
    v.open(stores["0.5"])
    before = v.screen_to_world(50.0, 60.0)
    v.rotate(17.0, 50.0, 60.0)
    assert v.screen_to_world(50.0, 60.0) == pytest.approx(before)
    v.rotate(-377.0, 50.0, 60.0)                        # wraps to 0 .. 360
    assert v.rotation == pytest.approx(0.0, abs=1e-9) or v.rotation == pytest.approx(360.0)
    v.fit()                                              # fits the turned view too
    v.set_rotation(45.0)
    v.fit()
    x0, y0, x1, y1 = v.bounds
    for x, y in ((x0, y0), (x1, y1)):
        # the image's corners stay inside the target
        sx, sy = _to_screen(v, x, y)
        assert -1.0 <= sx <= 321.0 and -1.0 <= sy <= 241.0


def _to_screen(v, x, y):
    """Screen point of a world point (inverse of screen_to_world)."""
    o = np.array(v.screen_to_world(0.0, 0.0))
    ex = np.array(v.screen_to_world(1.0, 0.0)) - o
    ey = np.array(v.screen_to_world(0.0, 1.0)) - o
    return np.linalg.solve(np.stack([ex, ey], axis=1), np.array([x, y]) - o)


# ------------------------------------------------------------------ layers

def test_a_layer_moves_and_turns(stores, image):
    v = Viewer.offscreen(320, 240)
    v.open(stores["0.5"])
    v.set_contrast(0.0, 65535.0)
    v.look_at(450.25, 350.25, 2.0)
    v.move_layer(15.0, -10.0)                            # screen pixels: 30, -20 world
    assert np.allclose(v.transform(), [[1.0, 0.0, 30.0], [0.0, 1.0, -20.0]])
    v.rotate_layer(90.0)                                 # about the image's centre
    t = v.transform()
    rgba = _frame(v)
    val, inside = _look(image, *_world(320, 240, (450.25, 350.25), 2.0, 0.0), t)
    _match(rgba[..., 0], _grey(val), inside)
    # the NGFF form over the image's axes (y, x): rows y then x, translation last
    m = v.transform_matrix()
    assert m[1] == pytest.approx([t[0][1], t[0][0], t[0][2]])
    assert m[0] == pytest.approx([t[1][1], t[1][0], t[1][2]])
    v.reset_transform()
    assert v.transform() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    with pytest.raises(ViewerError, match="inverted"):
        v.set_transform([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    with pytest.raises(ViewerError, match="2 rows of 3"):
        v.set_transform([[1.0, 0.0], [0.0, 1.0]])


def test_two_layers_add_up(stores, image):
    v = Viewer.offscreen(300, 220)
    v.open(stores["0.5"])
    assert v.add(stores["pyramid"]) == 1 and v.current == 1 and len(v.layers) == 2
    for i, colour in ((0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0))):
        v.set_channel(0, color=colour, lo=0.0, hi=65535.0, layer=i)
    v.set_transform([[1.0, 0.0, 40.0], [0.0, 1.0, 25.0]], layer=1)
    v.look_at(880.0, 680.0, 1.0)                         # the corner: both, one, none
    rgba = _frame(v)
    wx, wy = _world(300, 220, (880.0, 680.0), 1.0, 0.0)
    a, in_a = _look(image, wx, wy)
    b, in_b = _look(image, wx, wy, [[1.0, 0.0, 40.0], [0.0, 1.0, 25.0]])
    assert (in_a & in_b).any() and (~in_a & in_b).any() and (~in_a & ~in_b).any()
    either = in_a | in_b
    assert (np.abs(rgba[..., 0][in_a].astype(int) - _grey(a)[in_a]) <= 1).all()
    assert (rgba[..., 0][~in_a & in_b] == 0).all()
    assert (np.abs(rgba[..., 1][in_b].astype(int) - _grey(b)[in_b]) <= 1).all()
    assert (rgba[..., 1][in_a & ~in_b] == 0).all()
    assert (rgba[..., :3][~either] == BACKGROUND).all() and (rgba[..., 3] == 255).all()
    # opacity weighs a layer (where it is alone, half the background shows through),
    # a hidden layer adds nothing
    v.set_layer(1, opacity=0.5)
    half = _frame(v)[..., 1].astype(int)
    both = in_a & in_b
    assert (np.abs(half[both] - np.round(_grey(b)[both] * 0.5)) <= 1).all()
    alone = ~in_a & in_b
    assert (np.abs(half[alone] - np.round(_grey(b)[alone] * 0.5 + BACKGROUND * 0.5)) <= 1).all()
    v.set_layer(1, visible=False)
    assert (_frame(v)[..., 1][in_a] == 0).all()
    with pytest.raises(ViewerError, match="opacity"):
        v.set_layer(0, opacity=2.0)
    v.remove(1)
    assert len(v.layers) == 1 and v.current == 0
    with pytest.raises(ViewerError, match="layer 3"):
        v.set_channel(0, lo=1.0, layer=3)


def test_pick_reports_every_layer(stores, image):
    v = Viewer.offscreen(200, 200)
    v.open(stores["0.5"])
    v.add(stores["pyramid"])
    v.set_transform([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]], layer=1)
    v.look_at(100.0, 100.0, 1.0)
    v.settle(30.0)
    p = v.pick(100.0, 100.0)                             # world (100, 100)
    assert [lp["pixel"] for lp in p["layers"]] == [(100, 100), (90, 100)]
    assert p["layers"][0]["values"][0]["value"] == image[100, 100]
    assert p["layers"][1]["values"][0]["value"] == image[100, 90]
    assert p["pixel"] == (90, 100)                       # the current layer's


# ------------------------------------------------------------------ colour maps, contrast

def test_colour_maps(stores, image):
    assert "viridis" in COLOUR_MAPS and "gray" in COLOUR_MAPS
    v = Viewer.offscreen(200, 160)
    v.open(stores["0.5"])
    v.set_channel(0, lut="viridis", lo=30000.0, hi=30001.0)
    assert v.channels[0]["lut"] == "viridis"
    v.look_at(300.0, 200.0, 1.0)
    rgba = _frame(v)[..., :3].reshape(-1, 3).astype(int)
    val, inside = _look(image, *_world(200, 160, (300.0, 200.0), 1.0, 0.0))
    low, high = (val <= 30000).ravel() & inside.ravel(), (val >= 30001).ravel() & inside.ravel()
    assert low.any() and high.any()
    assert (np.abs(rgba[low] - VIRIDIS_0) <= 1).all()
    assert (np.abs(rgba[high] - VIRIDIS_1) <= 1).all()
    v.set_channel(0, lut="")                             # back to the colour (white)
    assert v.channels[0]["lut"] is None
    assert (_frame(v)[..., :3].reshape(-1, 3)[high] == 255).all()
    with pytest.raises(ViewerError, match="colour map"):
        v.set_channel(0, lut="rainbow-unicorn")


def test_contrast_from_the_view(stores, image):
    v = Viewer.offscreen(320, 240)
    v.open(stores["0.5"])
    v.auto_contrast(lo=0.0, hi=1.0)                      # the whole image is in view
    c = v.channels[0]
    assert (c["lo"], c["hi"]) == (image.min(), image.max())
    v.look_at(150.0, 100.0, 0.25)                        # 80 x 60 pixels in view
    v.auto_contrast(0, lo=0.0, hi=1.0)
    shown = image[70:131, 110:191]                       # the view's box, a pixel around
    inner = image[71:129, 111:189]
    c = v.channels[0]
    assert shown.min() <= c["lo"] <= inner.min() and inner.max() <= c["hi"] <= shown.max()
    v.look_at(-5000.0, -5000.0, 1.0)
    with pytest.raises(ViewerError, match="not in view"):
        v.auto_contrast()
    with pytest.raises(ViewerError, match="quantiles"):
        v.auto_contrast(lo=0.9, hi=0.1)


# ------------------------------------------------------------------ the view state

def test_the_view_state_keeps_layers_and_turns(stores):
    a = Viewer.offscreen(300, 200)
    a.open(stores["0.5"])
    a.add(stores["pyramid"])
    a.set_channel(0, lut="magma", layer=0)
    a.set_channel(0, color=(0.0, 1.0, 0.0), layer=1)
    a.set_transform([[0.9, -0.2, 30.0], [0.2, 0.9, -12.0]], layer=1)
    a.set_layer(1, opacity=0.6)
    a.look_at(400.0, 300.0, 1.5)
    a.set_rotation(30.0)
    a.current = 0
    state = a.view_state()
    doc = json.loads(state)
    assert doc["version"] == 3 and doc["current"] == 0 and len(doc["layers"]) == 2
    assert doc["camera"]["rotation"] == pytest.approx(30.0)
    assert doc["layers"][1]["transform"][0] == pytest.approx([0.9, -0.2, 0.0, 30.0])
    assert doc["layers"][0]["channels"][0]["lut"] == "magma"
    b = Viewer.offscreen(300, 200)
    b.set_view_state(state)
    assert b.view_state() == state
    assert np.array_equal(_frame(a), _frame(b))
    # a state of version 1 (one image) still opens
    c = Viewer.offscreen(300, 200)
    c.set_view_state(json.dumps({"version": 1, "image": stores["0.5"],
                                 "camera": {"center": [10, 20], "scale": 2.0},
                                 "channels": [{"lo": 5.0, "hi": 500.0}]}))
    assert len(c.layers) == 1 and c.channels[0]["hi"] == 500.0 and c.rotation == 0.0


# ------------------------------------------------------------------ slow sources

def test_slow_reads_take_coarser_levels_while_the_view_moves(stores, monkeypatch):
    monkeypatch.setenv("MICRO_GLANCER_SLOW_READS_MS", "400")
    v = Viewer.offscreen(450, 350)
    v.open(stores["pyramid"])                            # fitted: level 1 (scale 2)
    assert v.level == 1
    v.settle(30.0)                                       # level 1 cached; reads measured
    v.pan(0.0, 0.0)                                      # a drag starts (a frame earlier)
    v.render_rgba()
    v.look_at(450.0, 350.0, 1.0)                         # level 0: 9 tiles, slow
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.8:                # the view keeps moving
        v.pan(0.0, 0.0)
        v.render_rgba()
        assert v.pending == 0 and v.wants_frame()        # level 0 waits
        time.sleep(0.02)
    centre = v.pick(225.0, 175.0)["values"][0]
    assert centre["level"] == 1                          # the coarser level stands in
    stats = v.settle(30.0)                               # still: level 0 is read
    assert stats[4] == 0 and v.pick(225.0, 175.0)["values"][0]["level"] == 0


# ------------------------------------------------------------------ saving transforms

@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_shift_is_saved_into_0_4_and_0_5(stores, tmp_path, image, version):
    path = str(tmp_path / f"copy{version}.ome.zarr")
    shutil.copytree(stores[version], path)
    v = Viewer.offscreen(200, 150)
    v.open(path)
    v.look_at(100.0, 80.0, 1.0)
    v.set_transform([[1.0, 0.0, 12.0], [0.0, 1.0, -7.0]])
    before = _frame(v)
    said = save_transform(v)
    assert "translation" in said
    assert v.transform() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]   # now in the levels
    assert np.array_equal(_frame(v), before)
    pyr = ozp.IO().read_pyramid(path, include_labels=False)
    assert pyr.meta.get_translation("0") == pytest.approx([-7.0, 12.0])
    w = Viewer.offscreen(200, 150)                       # reopened: the same place
    w.open(path)
    w.look_at(100.0, 80.0, 1.0)
    assert np.array_equal(_frame(w), before)
    w.rotate_layer(10.0)
    with pytest.raises(ViewerError, match="0.6"):
        save_transform(w)


def test_a_turn_is_saved_into_0_6(stores, tmp_path):
    path = str(tmp_path / "copy06.ome.zarr")
    shutil.copytree(stores["0.6"], path)
    v = Viewer.offscreen(240, 180)
    v.open(path)
    v.look_at(450.0, 350.0, 2.0)
    v.rotate_layer(30.0)
    v.move_layer(5.0, 3.0)
    t = v.transform()
    before = _frame(v)
    assert "affine" in save_transform(v)
    with open(os.path.join(path, "zarr.json")) as f:
        ms = json.load(f)["attributes"]["ome"]["multiscales"][0]
    names = [s["name"] for s in ms["coordinateSystems"]]
    assert names == ["physical", ALIGNED]
    (edge,) = ms["coordinateTransformations"]
    assert edge["type"] == "affine" and edge["input"] == {"name": "physical"}
    assert edge["output"] == {"name": ALIGNED}
    assert np.allclose(v.transform(), t)                 # read back from the file
    assert np.array_equal(_frame(v), before)
    ozp.IO().read_pyramid(path, include_labels=False)    # ozp still reads it
    w = Viewer.offscreen(240, 180)
    w.open(path)
    w.look_at(450.0, 350.0, 2.0)
    assert np.allclose(w.transform(), t) and np.array_equal(_frame(w), before)
    # moved only: a translation; back where the image puts it: removed
    w.reset_transform()
    w.move_layer(4.0, 0.0)
    assert "translation" in save_transform(w)
    w.reset_transform()
    assert "removed" in save_transform(w)
    with open(os.path.join(path, "zarr.json")) as f:
        ms = json.load(f)["attributes"]["ome"]["multiscales"][0]
    assert "coordinateTransformations" not in ms and len(ms["coordinateSystems"]) == 1


def test_a_zip_is_not_changed(stores, tmp_path):
    zpath = tmp_path / "img.ozx"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(stores["0.5"]):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, stores["0.5"]).replace(os.sep, "/"))
    v = Viewer.offscreen(64, 64)
    v.open(str(zpath))
    v.move_layer(3.0, 0.0)
    with pytest.raises(ViewerError, match="zip"):
        save_transform(v)


# ------------------------------------------------------------------ the Qt widget

@pytest.fixture(scope="module")
def qapp():
    qt = pytest.importorskip("PyQt6.QtWidgets")
    return qt.QApplication.instance() or qt.QApplication([])


def test_the_qt_widget_moves_and_turns_layers(qapp, stores, tmp_path):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QTest
    from micro_glancer.qt import ViewerWidget

    w = ViewerWidget(mode="offscreen")
    w.resize(240, 180)
    w.open(stores["0.5"])
    w.add(stores["pyramid"])
    w.show()
    qapp.processEvents()
    v = w.viewer
    assert len(v.layers) == 2 and v.current == 1
    r = w.devicePixelRatioF()
    scale = v.camera[2]
    ctrl = Qt.KeyboardModifier.ControlModifier
    QTest.mousePress(w, Qt.MouseButton.LeftButton, ctrl, QPoint(100, 100))
    QTest.mouseMove(w, QPoint(130, 90))
    QTest.mouseRelease(w, Qt.MouseButton.LeftButton, ctrl, QPoint(130, 90))
    t = v.transform()
    assert np.allclose([t[0][2], t[1][2]], [30 * r * scale, -10 * r * scale], atol=1e-6)
    assert v.transform(layer=0) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    def wheel(mods, notches=1):
        pos = QPointF(120, 90)
        e = QWheelEvent(pos, w.mapToGlobal(pos), QPoint(), QPoint(0, 120 * notches),
                        Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False)
        w.wheelEvent(e)

    wheel(ctrl, 3)                                       # the layer turns 3 degrees
    t = v.transform()
    assert np.degrees(np.arctan2(t[1][0], t[0][0])) == pytest.approx(3.0)
    wheel(Qt.KeyboardModifier.ShiftModifier, 2)          # the view turns 10 degrees
    assert v.rotation == pytest.approx(10.0)
    QTest.keyClick(w, Qt.Key.Key_Tab)
    assert v.current == 0
    QTest.keyClick(w, Qt.Key.Key_0, ctrl)                 # layer 0 back (it was)
    QTest.keyClick(w, Qt.Key.Key_0)                       # the view turned back
    assert v.rotation == pytest.approx(0.0)
    # saving reports instead of crashing (a zip cannot be changed)
    zpath = tmp_path / "z.ozx"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(stores["0.5"]):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, stores["0.5"]).replace(os.sep, "/"))
    w.add(str(zpath))
    errors, saved = [], []
    w.failed.connect(errors.append)
    w.saved.connect(saved.append)
    QTest.keyClick(w, Qt.Key.Key_S, ctrl)
    assert errors and "zip" in errors[-1] and not saved
    w.close()


def test_one_jump_on_a_slow_source_reads_at_once(stores, monkeypatch):
    """A jump, an open or a step in z is not a moving view: the frame's level is
    read straight away, even when reads are slow."""
    monkeypatch.setenv("MICRO_GLANCER_SLOW_READS_MS", "400")
    v = Viewer.offscreen(450, 350)
    v.open(stores["pyramid"])
    v.settle(30.0)
    v.look_at(450.0, 350.0, 1.0)
    v.render_rgba()
    assert v.pending > 0                                 # level 0 asked for at once
