"""micro-glancer's 2D viewer (Workplan WP14, task V7): OME-Zarr images drawn by the GPU
from a tile cache, read through ozp-in-Rust (DR9) and dyna's arrays crate.

Off-screen frames are compared pixel by pixel with NumPy. A screen pixel (sx, sy) shows,
for level L placed by index -> world = scale * i + translation, the level pixel
i = floor((origin + (s + 0.5) * camera_scale - translation) / scale + 0.5) (NGFF:
indices are pixel centres; the nearest centre wins), through each channel's contrast,
gamma and colour, channels added up, as 8-bit RGB (one step of rounding allowed).
"""
import json
import os
import sys
import zipfile

import numpy as np
import pytest

from micro_glancer import Viewer, ViewerError

ozp = pytest.importorskip("ome_zarr_pyramid")

BACKGROUND = round(0.08 * 255)
MISSING = (round(0.16 * 255), round(0.16 * 255), round(0.18 * 255))
H, W = 700, 900                      # 3 x 4 tiles of 256 at level 0, edge tiles partial


def _write(path, levels, axes, scales, version="0.5", omero=None):
    pyr = ozp.Pyramid().from_arrays(levels, axis_order=axes, scales=scales, version=version)
    ozp.IO().write_pyramid(pyr, str(path), overwrite=True)
    if omero is not None:
        meta = os.path.join(str(path), "zarr.json" if version != "0.4" else ".zattrs")
        with open(meta) as f:
            doc = json.load(f)
        attrs = doc["attributes"]["ome"] if version != "0.4" else doc
        attrs["omero"] = omero
        with open(meta, "w") as f:
            json.dump(doc, f)
    return str(path)


def _pyramid(l0, n=3):
    """Level 0 and every second pixel of the one before, n levels."""
    out = [l0]
    for _ in range(n - 1):
        out.append(np.ascontiguousarray(out[-1][..., ::2, ::2]))
    return out


@pytest.fixture(scope="module")
def image():
    yy, xx = np.mgrid[0:H, 0:W]
    return ((yy * 37 + xx * 61) % 65536).astype("uint16")


@pytest.fixture(scope="module")
def two():
    """Two channels (c, y, x)."""
    yy, xx = np.mgrid[0:H, 0:W]
    return np.stack([(yy * 37 + xx * 61) % 65536, (yy * 11 + xx * 5) % 4000]).astype("uint16")


@pytest.fixture(scope="module")
def stack():
    """Five planes (z, y, x), each plane its own pattern."""
    zz, yy, xx = np.mgrid[0:5, 0:300, 0:400]
    return ((zz * 9000 + yy * 13 + xx * 7) % 50000).astype("uint16")


@pytest.fixture(scope="module")
def stores(tmp_path_factory, image, two, stack):
    tmp = tmp_path_factory.mktemp("viewer")
    out = {}
    for v in ("0.4", "0.5", "0.6"):
        out[v] = _write(tmp / f"img{v}.ome.zarr", [image], "yx", [[1.0, 1.0]], v)
    zpath = tmp / "img.ozx"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(out["0.5"]):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, out["0.5"]).replace(os.sep, "/"))
    out["ozx"] = str(zpath)
    out["pyramid"] = _write(tmp / "pyr.ome.zarr", _pyramid(image), "yx",
                            [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    out["two"] = _write(tmp / "two.ome.zarr", _pyramid(two), "cyx",
                        [[1.0, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 4.0, 4.0]], "0.4",
                        omero={"channels": [
                            {"color": "FF0000", "label": "A",
                             "window": {"start": 0.0, "end": 65535.0}},
                            {"color": "00FF00", "label": "B",
                             "window": {"start": 0.0, "end": 4000.0}}],
                            "rdefs": {"model": "color"}})
    out["stack"] = _write(tmp / "stack.ome.zarr", [stack], "zyx", [[1.0, 1.0, 1.0]], "0.4",
                          omero={"channels": [{"window": {"start": 0.0, "end": 50000.0}}],
                                 "rdefs": {"defaultZ": 3}})
    return out


def _frame(v):
    data, w, h, stats = v.render_rgba()
    return np.frombuffer(data, np.uint8).reshape(h, w, 4), stats


def _settled(v):
    v.settle(30.0)
    return _frame(v)


def _sample(level, w, h, camera, scale=1.0):
    """The level pixel each screen pixel shows (level placed by `scale`, no
    translation), and which screen pixels the level covers."""
    x0, y0, s = camera
    ix = np.floor((x0 + (np.arange(w) + 0.5) * s) / scale + 0.5).astype(int)
    iy = np.floor((y0 + (np.arange(h) + 0.5) * s) / scale + 0.5).astype(int)
    inside = ((iy[:, None] >= 0) & (iy[:, None] < level.shape[-2])
              & (ix[None, :] >= 0) & (ix[None, :] < level.shape[-1]))
    v = level[..., np.clip(iy, 0, level.shape[-2] - 1), :][..., np.clip(ix, 0, level.shape[-1] - 1)]
    return v, inside


def _grey(v, lo, hi, gamma=1.0):
    g = np.clip((v.astype(np.float32) - np.float32(lo)) / np.float32(hi - lo), 0, 1)
    return np.round(g ** gamma * 255).astype(int)


def _close(got, want, inside):
    """Inside the image `got` (H, W) matches `want` within one step; outside it is the
    background."""
    diff = np.abs(got.astype(int) - want)
    assert diff[inside].max(initial=0) <= 1, np.argwhere((diff > 1) & inside)[:5]
    assert (got[~inside] == BACKGROUND).all()


# ------------------------------------------------------------------ one channel, level 0

@pytest.mark.parametrize("camera", [(0.0, 0.0, 1.0), (37.0, 11.0, 0.5), (-50.0, -20.0, 1.75),
                                    (600.0, 400.0, 0.25)])
def test_pixels_are_the_image_through_the_contrast(stores, image, camera):
    v = Viewer.offscreen(320, 240)
    v.open(stores["0.5"])
    v.set_contrast(1000.0, 60000.0)
    v.set_camera(*camera)
    rgba, (visible, loaded, pending, level, missing) = _settled(v)
    assert level == 0 and missing == 0
    val, inside = _sample(image, 320, 240, camera)
    _close(rgba[..., 0], _grey(val, 1000.0, 60000.0), inside)
    assert (rgba[..., 0] == rgba[..., 1]).all() and (rgba[..., 3] == 255).all()


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6", "ozx"])
def test_every_version_and_a_zip_open_alike(stores, image, version):
    v = Viewer.offscreen(200, 160)
    v.open(stores[version])
    info = v.image
    assert info["dtype"] == "uint16" and info["axes"] == ["y", "x"]
    assert info["levels"][0]["shape"] == [H, W]
    assert info["version"] == ("0.5" if version == "ozx" else version)
    v.set_contrast(0.0, 65535.0)
    v.set_camera(100.0, 50.0, 1.0)
    rgba, _ = _settled(v)
    val, inside = _sample(image, 200, 160, (100.0, 50.0, 1.0))
    _close(rgba[..., 0], _grey(val, 0.0, 65535.0), inside)


def test_the_software_adapter_draws_the_same(stores):
    try:
        soft = Viewer.offscreen(256, 192, software=True)
    except ViewerError as e:
        pytest.skip(f"no software adapter here: {e}")
    gpu = Viewer.offscreen(256, 192)
    for v in (soft, gpu):
        v.open(stores["two"])
        v.set_camera(13.0, 7.0, 1.5)
        v.set_channel(1, gamma=0.7)
    a, b = _settled(soft)[0], _settled(gpu)[0]
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1
    print("software:", soft.adapter, "| gpu:", gpu.adapter)


# ------------------------------------------------------------------ levels

@pytest.mark.parametrize("scale, level", [(0.5, 0), (1.0, 0), (1.9, 0), (2.0, 1), (3.5, 1),
                                          (4.0, 2), (9.0, 2)])
def test_the_level_follows_the_zoom(stores, image, scale, level):
    v = Viewer.offscreen(240, 180)
    v.open(stores["pyramid"])
    v.set_contrast(0.0, 65535.0)
    camera = (21.0, -13.0, scale)
    v.set_camera(*camera)
    assert v.level == level
    rgba, stats = _settled(v)
    assert stats[3] == level
    data = _pyramid(image)[level]
    val, inside = _sample(data, 240, 180, camera, 2.0 ** level)
    _close(rgba[..., 0], _grey(val, 0.0, 65535.0), inside)


def test_coarser_tiles_stand_in_while_finer_ones_load(stores, image):
    v = Viewer.offscreen(240, 180)
    v.open(stores["pyramid"])
    v.set_contrast(0.0, 65535.0)
    v.set_camera(0.0, 0.0, 2.0)           # level 1, settled: level 0 never read
    _settled(v)
    camera = (100.0, 80.0, 1.0)
    v.set_camera(*camera)                 # level 0: its tiles are only requested now
    rgba, (visible, loaded, pending, level, missing) = _frame(v)
    assert level == 0 and loaded == 0 and pending > 0 and missing == visible
    val, inside = _sample(_pyramid(image)[1], 240, 180, camera, 2.0)
    _close(rgba[..., 0], _grey(val, 0.0, 65535.0), inside)
    rgba, _ = _settled(v)                 # then level 0 itself
    val, inside = _sample(image, 240, 180, camera)
    _close(rgba[..., 0], _grey(val, 0.0, 65535.0), inside)


def test_tiles_load_in_the_background(stores):
    v = Viewer.offscreen(320, 240)
    v.open(stores["0.5"])
    v.set_camera(0.0, 0.0, 1.0)
    rgba, (visible, loaded, pending, _, missing) = _frame(v)
    assert visible == missing == 2 and loaded == 0 and pending >= visible  # nothing yet
    assert (rgba[..., :3] == MISSING).all()
    v.settle(30.0)
    rgba, (visible, loaded, pending, _, missing) = _frame(v)
    assert missing == 0
    assert not (rgba[..., :3] == MISSING).all(-1).any()


def test_a_tiny_atlas_still_draws(stores, image):
    v = Viewer.offscreen(1200, 900, atlas_mib=1)       # one layer: 256 slots
    assert v.atlas_slots == 0                          # made when an image opens
    v.open(stores["pyramid"])
    assert v.atlas_slots == 256
    v.set_contrast(0.0, 65535.0)
    camera = (-10.0, -5.0, 0.8)
    v.set_camera(*camera)
    rgba, _ = _settled(v)
    val, inside = _sample(image, 1200, 900, camera)
    _close(rgba[..., 0], _grey(val, 0.0, 65535.0), inside)


def test_a_larger_target_gets_a_larger_atlas(stores, image):
    v = Viewer.offscreen(64, 48)
    v.open(stores["pyramid"])
    small = v.atlas_slots
    v.resize(3000, 2600)
    assert v.atlas_slots > small
    v.set_contrast(0.0, 65535.0)
    camera = (-1000.0, -900.0, 1.0)
    v.set_camera(*camera)
    rgba, (visible, loaded, pending, level, missing) = _settled(v)
    assert level == 0 and missing == 0
    val, inside = _sample(image, 3000, 2600, camera)
    _close(rgba[..., 0], _grey(val, 0.0, 65535.0), inside)


# ------------------------------------------------------------------ channels

def test_channels_add_up_in_their_colours(stores, two):
    v = Viewer.offscreen(300, 200)
    v.open(stores["two"])
    ch = v.channels
    assert [c["label"] for c in ch] == ["A", "B"]
    assert ch[0]["color"] == (1.0, 0.0, 0.0) and (ch[1]["lo"], ch[1]["hi"]) == (0.0, 4000.0)
    v.set_channel(1, color=(0.0, 1.0, 1.0), gamma=0.5)
    camera = (5.0, 9.0, 1.0)
    v.set_camera(*camera)
    rgba, _ = _settled(v)
    val, inside = _sample(two, 300, 200, camera)
    r = _grey(val[0], 0.0, 65535.0)
    g = _grey(val[1], 0.0, 4000.0, 0.5)
    want = np.stack([r, np.minimum(g, 255), g], -1)
    for k in range(3):
        _close(rgba[..., k], want[..., k], inside)
    v.set_channel(0, visible=False)
    rgba, _ = _frame(v)
    assert (rgba[..., 0][inside] == 0).all()
    with pytest.raises(ViewerError, match="channel 5"):
        v.set_channel(5, lo=1.0)
    with pytest.raises(ViewerError, match="gamma"):
        v.set_channel(0, gamma=0.0)


@pytest.mark.parametrize("dtype", ["uint8", "int16", "float32", "float64", "int32"])
def test_types_kept_or_cast_to_float(tmp_path, dtype):
    yy, xx = np.mgrid[0:300, 0:310]
    base = (yy * 3 + xx * 5) % 200
    data = (base - 100).astype(dtype) if dtype.startswith("int") else base.astype(dtype)
    if dtype.startswith("float"):
        data = data / 7.0
    path = _write(tmp_path / f"{dtype}.ome.zarr", [data], "yx", [[1.0, 1.0]])
    v = Viewer.offscreen(200, 150)
    v.open(path)
    lo, hi = float(data.min()), float(data.max())
    v.set_contrast(lo, hi)
    camera = (40.0, 30.0, 1.25)
    v.set_camera(*camera)
    rgba, _ = _settled(v)
    val, inside = _sample(data.astype(np.float32), 200, 150, camera)
    _close(rgba[..., 0], _grey(val, lo, hi), inside)


# ------------------------------------------------------------------ plane position

def test_the_plane_moves_on_z(stores, stack):
    v = Viewer.offscreen(200, 160)
    v.open(stores["stack"])
    assert v.image["plane_axes"] == ["z"]
    assert v.position == [("z", 3.0, 5)]                  # omero's defaultZ
    camera = (10.0, 20.0, 1.0)
    for z in (3, 0, 4):
        v.set_position("z", float(z))
        v.set_camera(*camera)
        rgba, _ = _settled(v)
        val, inside = _sample(stack[z], 200, 160, camera)
        _close(rgba[..., 0], _grey(val, 0.0, 50000.0), inside)
    with pytest.raises(ViewerError, match="no plane axis"):
        v.set_position("t", 0.0)
    with pytest.raises(ViewerError, match="5 indices"):
        v.set_position("z", 5.0)


def test_the_next_planes_are_read_ahead(stores):
    v = Viewer.offscreen(200, 160)
    v.open(stores["stack"])
    v.set_camera(0.0, 0.0, 1.0)
    v.settle(30.0, read_ahead=True)                       # z = 3, and 2 and 4 ahead
    v.set_position("z", 4.0)
    _, (visible, loaded, pending, _, missing) = _frame(v)
    assert visible == 1 and loaded == 0                   # nothing to read for z = 4
    rgba, _ = _frame(v)
    assert not (rgba[..., :3] == MISSING).all(-1).any()


# ------------------------------------------------------------------ camera, picking, state

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
    c = v.center
    v.resize(500, 200)
    assert v.center == pytest.approx(c)                   # resizing keeps the centre


def test_fit_shows_the_whole_image(stores):
    v = Viewer.offscreen(450, 450)
    v.open(stores["pyramid"])
    assert v.bounds == (-0.5, -0.5, W - 0.5, H - 0.5)
    x, y, s = v.camera
    assert s == pytest.approx(W / 450)                    # the wider side fills the width
    assert v.level == 1


def test_pick_reads_values_from_the_cache(stores, two):
    v = Viewer.offscreen(300, 200)
    v.open(stores["two"])
    v.set_camera(100.0, 50.0, 1.0)
    p = v.pick(10.0, 20.0)
    assert p["pixel"] == (110, 70) and p["values"][0]["value"] is None   # nothing cached
    _settled(v)
    p = v.pick(10.0, 20.0)
    assert p["world"] == (110.0, 70.0, 0.0)
    assert [e["value"] for e in p["values"]] == [float(two[0, 70, 110]), float(two[1, 70, 110])]
    assert all(e["level"] == 0 for e in p["values"])
    assert v.pick(-500.0, 0.0)["pixel"] is None


def test_the_view_state_round_trips(stores):
    a = Viewer.offscreen(300, 200)
    a.open(stores["two"])
    a.look_at(300.0, 200.0, 0.75)
    a.set_channel(1, color=(0.0, 0.0, 1.0), lo=10.0, hi=3000.0, gamma=1.4)
    a.set_channel(0, visible=False)
    state = a.view_state()
    doc = json.loads(state)
    assert doc["version"] == 3
    assert doc["camera"] == {"center": [300.0, 200.0, 0.0], "scale": 0.75, "rotation": 0.0,
                             "orientation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
    assert doc["layers"][0]["channels"][1]["color"] == "#0000ff"
    b = Viewer.offscreen(300, 200)
    b.set_view_state(state)                               # opens the image too
    assert b.view_state() == state
    assert np.array_equal(_settled(a)[0], _settled(b)[0])
    b.set_view_state('{"camera": {"center": [0, 0], "scale": 2.0}}')   # partial: camera only
    assert b.center == (0.0, 0.0) and b.channels == a.channels
    with pytest.raises(ViewerError, match="channels"):
        b.set_view_state('{"channels": [{}, {}, {}]}')
    with pytest.raises(ViewerError, match="version"):
        b.set_view_state('{"version": 7}')


def test_render_into_fills_the_callers_buffer(stores):
    for width in (320, 211):                              # 211: rows padded on the GPU
        v = Viewer.offscreen(width, 97)
        v.open(stores["0.5"])
        v.set_camera(21.0, 5.0, 0.75)
        v.settle(30.0)
        data = v.render_rgba()[0]
        frame = bytearray(4 * width * 97)
        for _ in range(2):                                # the readback buffer is reused
            v.render_into(frame)
            assert bytes(frame) == data
        with pytest.raises(ViewerError, match="needs"):
            v.render_into(bytearray(10))
        with pytest.raises((ViewerError, BufferError, TypeError)):
            v.render_into(bytes(len(frame)))              # read-only


def test_errors_name_the_way_out(tmp_path):
    v = Viewer.offscreen(64, 64)
    with pytest.raises(ViewerError):
        v.open(str(tmp_path / "missing.zarr"))
    with pytest.raises(ViewerError, match="scale"):
        v.set_camera(0.0, 0.0, 0.0)
    rgba, stats = _frame(v)                               # nothing open: the background
    assert (rgba[..., 0] == BACKGROUND).all() and stats == (0, 0, 0, 0, 0)


@pytest.mark.skipif(not os.environ.get("MICRO_GLANCER_SAMPLE"),
                    reason="set MICRO_GLANCER_SAMPLE to an OME-Zarr image with omero colours")
def test_a_real_image():
    v = Viewer.offscreen(400, 300)
    v.open(os.environ["MICRO_GLANCER_SAMPLE"])
    rgba, (visible, loaded, pending, level, missing) = _settled(v)
    assert missing == 0 and v.error is None
    assert (rgba[..., :3] > BACKGROUND + 10).any()
    print(v.image, v.channels, v.position)


# ------------------------------------------------------------------ Qt

@pytest.fixture(scope="module")
def qapp():
    qt = pytest.importorskip("PyQt6.QtWidgets")
    return qt.QApplication.instance() or qt.QApplication([])


def _pump(qapp, widget, rounds=200):
    """Let the widget repaint until its tiles are in (the widget polls while pending)."""
    import time
    for _ in range(rounds):
        qapp.processEvents()
        if widget.last_stats[2] == 0 and not widget.viewer.has_results() and widget.last_stats[0]:
            return
        time.sleep(0.01)


def test_the_qt_widget_offscreen(qapp, stores, image):
    from micro_glancer.qt import ViewerWidget
    w = ViewerWidget(mode="offscreen")
    w.resize(240, 180)
    w.open(stores["0.5"])
    w.show()
    qapp.processEvents()                  # shown: the screen's pixel ratio is known now
    w.viewer.set_contrast(0.0, 65535.0)
    w.viewer.set_camera(0.0, 0.0, 1.0)
    changes = []
    w.viewChanged.connect(lambda: changes.append(1))
    w.update()
    _pump(qapp, w)
    pix = w.grab().toImage()
    r = w.devicePixelRatioF()
    assert pix.width() == round(240 * r)
    val, _ = _sample(image, round(240 * r), round(180 * r), (0.0, 0.0, 1.0))
    assert abs(pix.pixelColor(10, 20).red() - _grey(val, 0.0, 65535.0)[20, 10]) <= 1
    w.viewer.fit()
    w._changed()
    assert changes
    w.close()


def test_the_qt_widget_steps_through_z(qapp, stores):
    from micro_glancer.qt import ViewerWidget
    w = ViewerWidget(mode="offscreen")
    w.resize(200, 160)
    w.open(stores["stack"])
    w.show()
    qapp.processEvents()
    assert w.step(0, 1) and w.viewer.position[0][1] == 4.0
    assert not w.step(0, 1)                               # the last plane
    assert w.step(0, -4) and w.viewer.position[0][1] == 0.0
    assert not w.step(1, 1)                               # no second plane axis
    w.close()


def test_the_qt_widget_reports_instead_of_crashing(qapp, tmp_path):
    from micro_glancer.qt import ViewerWidget
    w = ViewerWidget(mode="offscreen")
    w.resize(120, 90)
    errors = []
    w.failed.connect(errors.append)
    w.open(tmp_path / "missing.zarr")                    # opened at the first paint
    w.show()
    qapp.processEvents()
    w.repaint()                                           # no second attempt, no crash
    assert len(errors) == 1 and w.error == errors[0]
    w.close()


def test_the_qt_widget_native(qapp, stores):
    from micro_glancer.qt import ViewerWidget, native_possible
    if not native_possible():
        pytest.skip(f"no native windows on {sys.platform} with Qt's "
                    f"{qapp.platformName()!r} platform")
    w = ViewerWidget(mode="native")
    w.resize(320, 240)
    w.open(stores["0.5"])
    w.show()
    _pump(qapp, w)
    assert w.viewer.size[0] >= 320
    assert w.last_stats[0] >= 1 and w.last_stats[2] == 0
    w.close()


def test_native_without_real_windows_is_refused(qapp, monkeypatch):
    # on Qt's offscreen platform the window id is no window: Windows refused it and
    # macOS crashed on it (CI, 2026-11-23); the widget now says so before the core runs
    import micro_glancer.qt as mvqt
    monkeypatch.setattr(mvqt, "native_possible", lambda: False)
    with pytest.raises(ValueError, match="mode='offscreen'"):
        mvqt.ViewerWidget(mode="native")
    monkeypatch.setenv("MICRO_GLANCER_MODE", "")
    assert mvqt.default_mode() == "offscreen"


def test_the_qt_widget_opens_in_the_background(qapp, stores):
    import time

    from micro_glancer import warm_up
    from micro_glancer.qt import ViewerWidget

    assert isinstance(warm_up(), str)                     # the adapter's description
    w = ViewerWidget(mode="offscreen")
    w.resize(160, 120)
    w.show()
    seen, done = [], []
    w.opening.connect(lambda a, b: seen.append((a, b)))
    w.open_in_background(stores["0.4"], done=done.append)   # replaced by the next one
    w.open_in_background(stores["0.5"], done=done.append)
    end = time.monotonic() + 30
    while w.opening_now and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert done == [None]                                 # only the newest is shown
    assert w.viewer.image is not None
    assert w.viewer.layers[0]["path"].replace("\\", "/").endswith(
        stores["0.5"].replace("\\", "/").rstrip("/").split("/")[-1])
    errors = []
    w.open_in_background(stores["0.5"] + "-missing", done=errors.append)
    end = time.monotonic() + 30
    while w.opening_now and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.005)
    assert len(errors) == 1 and errors[0] is not None
    w.close()


def test_a_viewer_dropped_on_another_thread_is_handed_back():
    # Python's cycle collector can free a Qt window (and its viewer) on any thread that
    # happens to allocate (zarr's event loop, CI 2026-11-24): no error then; the viewer's
    # own thread frees it the next time it uses a viewer. Used on another thread: refused.
    import threading

    from micro_glancer._core import PanicException

    held = [Viewer.offscreen(64, 64)]
    t = threading.Thread(target=held.clear)               # the last reference goes there
    t.start()
    t.join()
    w = Viewer.offscreen(64, 64)                           # frees what came back
    w.fit()
    caught = []

    def use():
        try:
            w.fit()
        except BaseException as exc:  # noqa: BLE001 (a PanicException is a BaseException)
            caught.append(exc)

    t = threading.Thread(target=use)
    t.start()
    t.join()
    assert len(caught) == 1 and isinstance(caught[0], PanicException)
    assert "another thread" in str(caught[0])
