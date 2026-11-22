"""micro-viewer's label images (Workplan WP14, task V9 part 2): found in an image's
labels/ group, drawn over the image (colours per id, outlines, a selected id), exact
ids up to 64 bits, following the image's transform, in flat views and planes at an
angle, and in the view state.

The label layer is checked against the same frame with it hidden: where the id is 0
nothing changes, elsewhere the label's colour shows.
"""
import json
import os

import numpy as np
import pytest

from micro_viewer import Viewer, ViewerError

ozp = pytest.importorskip("ome_zarr_pyramid")

H, W = 120, 160
BIG = 2 ** 60 + 7


def _write(path, data, axes, scales, version="0.5"):
    pyr = ozp.Pyramid().from_arrays([data], axis_order=axes, scales=scales, version=version)
    ozp.IO().write_pyramid(pyr, str(path), overwrite=True)
    return str(path)


def _make_labels(image_path, labels, version="0.5"):
    """Write label images `labels` ({name: (data, axes, scales)}) into the image's
    labels/ group, with the NGFF metadata that makes them label images."""
    for name, (data, axes, scales) in labels.items():
        lp = os.path.join(image_path, "labels", name)
        _write(lp, data, axes, scales, version)
        meta = os.path.join(lp, "zarr.json")
        with open(meta) as f:
            doc = json.load(f)
        doc["attributes"]["ome"]["image-label"] = {"version": version,
                                                   "source": {"image": "../../"}}
        with open(meta, "w") as f:
            json.dump(doc, f)
    group = os.path.join(image_path, "labels", "zarr.json")
    with open(group, "w") as f:
        json.dump({"zarr_format": 3, "node_type": "group",
                   "attributes": {"ome": {"version": version, "labels": list(labels)}}}, f)


@pytest.fixture(scope="module")
def cells():
    lab = np.zeros((H, W), "uint16")
    for k, (y0, x0, h, w) in enumerate([(10, 10, 40, 50), (10, 80, 30, 60), (60, 20, 50, 40),
                                        (70, 90, 40, 50)], start=1):
        lab[y0:y0 + h, x0:x0 + w] = k
    return lab


@pytest.fixture(scope="module")
def stores(tmp_path_factory, cells):
    tmp = tmp_path_factory.mktemp("labels")
    yy, xx = np.mgrid[0:H, 0:W]
    img = ((yy * 37 + xx * 61) % 4000).astype("uint16")
    path = _write(tmp / "img.ome.zarr", img, "yx", [[1.0, 1.0]])
    big = np.zeros((H, W), "uint64")
    big[20:60, 30:90] = BIG
    big[70:100, 100:140] = 3
    _make_labels(path, {"cells": (cells, "yx", [[1.0, 1.0]]),
                        "big": (big, "yx", [[1.0, 1.0]])})
    # a volume with a label volume
    zz, yy, xx = np.mgrid[0:30, 0:H, 0:W]
    vol = ((zz * 700 + yy * 37 + xx * 61) % 4000).astype("uint16")
    vpath = _write(tmp / "vol.ome.zarr", vol, "zyx", [[2.0, 1.0, 1.0]])
    vlab = np.zeros((30, H, W), "uint32")
    vlab[5:25, 20:100, 30:120] = 7
    vlab[10:20, 60:110, 100:150] = 9
    _make_labels(vpath, {"nuclei": (vlab, "zyx", [[2.0, 1.0, 1.0]])})
    return {"img": path, "vol": vpath, "vlab": vlab}


def _frame(v):
    v.settle(30.0)
    data, w, h, _ = v.render_rgba()
    return np.frombuffer(data, np.uint8).reshape(h, w, 4).astype(int)


def _with_and_without(v, layer):
    on = _frame(v)
    v.set_layer(layer, visible=False)
    off = _frame(v)
    v.set_layer(layer, visible=True)
    return on, off


def test_label_images_are_found_and_drawn_over_the_image(stores, cells):
    v = Viewer.offscreen(W, H)
    v.open(stores["img"])
    v.set_contrast(0.0, 4000.0)
    v.look_at((W - 1) / 2, (H - 1) / 2, 1.0)
    assert v.label_names() == ["cells", "big"]
    k = v.add_label("cells", layer=0)
    assert k == 1 and v.current == 1
    lay = v.layers[1]
    assert lay["label"] and lay["follows"] == 0 and lay["opacity"] == pytest.approx(0.6)
    on, off = _with_and_without(v, 1)
    changed = (on[..., :3] != off[..., :3]).any(axis=-1)
    assert not changed[cells == 0].any()                 # id 0: the image as it is
    assert changed[cells > 0].mean() > 0.99
    # one colour per id: the label's own pixels over a uniform patch look alike
    colours = {}
    for k in range(1, 5):
        rows, cols = np.nonzero(cells == k)
        colours[k] = tuple((on[rows, cols, :3] - 0.4 * off[rows, cols, :3]).mean(axis=0).round())
    assert len(set(colours.values())) == 4
    with pytest.raises(ViewerError, match="no label image"):
        v.add_label("nonsense", layer=0)


def test_outlines_and_the_selected_id(stores, cells):
    v = Viewer.offscreen(W, H)
    v.open(stores["img"])
    v.look_at((W - 1) / 2, (H - 1) / 2, 1.0)
    v.add_label("cells", layer=0)
    v.set_labels(outline=True)
    on, off = _with_and_without(v, 1)
    changed = (on[..., :3] != off[..., :3]).any(axis=-1)
    inner = np.zeros_like(cells, bool)
    inner[15:45, 15:55] = True                           # well inside label 1
    assert not changed[inner].any()
    assert changed[10, 10:60].mean() > 0.9               # its top edge
    v.set_labels(outline=False, selected=3, only_selected=True)
    on, off = _with_and_without(v, 1)
    changed = (on[..., :3] != off[..., :3]).any(axis=-1)
    assert changed[cells == 3].mean() > 0.99 and not changed[(cells != 3)].any()
    assert v.layers[1]["selected"] == 3
    v.set_labels(selected=-1, only_selected=False)
    assert v.layers[1]["selected"] is None
    with pytest.raises(ViewerError, match="not a label image"):
        v.set_labels(layer=0, outline=True)


def test_ids_are_exact_to_64_bits(stores):
    v = Viewer.offscreen(W, H)
    v.open(stores["img"])
    v.look_at((W - 1) / 2, (H - 1) / 2, 1.0)
    v.add_label("big", layer=0)
    v.settle(30.0)
    p = v.pick(60.0, 40.0)                               # inside the BIG region
    entry = p["layers"][1]["values"][0]
    assert entry["id"] == BIG
    assert v.pick(120.0, 85.0)["layers"][1]["values"][0]["id"] == 3
    on, off = _with_and_without(v, 1)
    assert (on[40, 60, :3] != off[40, 60, :3]).any()     # drawn


def test_labels_follow_their_image_and_open_directly(stores):
    v = Viewer.offscreen(W, H)
    v.open(stores["img"])
    v.add_label("cells", layer=0)
    v.set_transform([[1.0, 0.0, 12.0], [0.0, 1.0, -5.0]], layer=0)
    assert np.allclose(v.transform(layer=1), [[1.0, 0.0, 12.0], [0.0, 1.0, -5.0]])
    v.remove(0)                                          # the label stands on its own now
    assert v.layers[0]["follows"] is None
    w = Viewer.offscreen(W, H)
    w.open(os.path.join(stores["img"], "labels", "cells"))
    assert w.layers[0]["label"]


def test_a_label_volume_on_a_plane_at_an_angle(stores):
    vlab = stores["vlab"]
    v = Viewer.offscreen(220, 160)
    v.open(stores["vol"])
    v.add_label("nuclei", layer=0)
    v.look_at_3d(80.0, 60.0, 30.0, 0.8)
    v.turn(25.0, 30.0)
    assert v.layers[1]["mode"] == "volume"
    on, off = _with_and_without(v, 1)
    changed = (on[..., :3] != off[..., :3]).any(axis=-1)
    # the nearest voxel's id under every pixel
    w, h = v.size
    c = np.array(v.center_3d)
    right, down, _ = (np.array(r) for r in v.orientation)
    X, Y = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
    world = c + v.camera[2] * (X[..., None] * right + Y[..., None] * down)
    idx = np.floor(world / np.array([1.0, 1.0, 2.0]) + 0.5).astype(int)
    inside = ((idx >= 0) & (idx < np.array([W, H, 30]))).all(axis=-1)
    ids = np.where(inside, vlab[np.clip(idx[..., 2], 0, 29), np.clip(idx[..., 1], 0, H - 1),
                                np.clip(idx[..., 0], 0, W - 1)], 0)
    assert (ids > 0).sum() > 500
    assert changed[ids > 0].mean() > 0.98 and changed[ids == 0].mean() < 0.02
    p = v.pick(110.0, 80.0)
    want = ids[80, 110]
    assert p["layers"][1]["values"][0]["id"] == want


def test_label_settings_in_the_view_state(stores):
    a = Viewer.offscreen(W, H)
    a.open(stores["img"])
    a.add_label("cells", layer=0)
    a.set_labels(outline=True, selected=2, seed=5)
    state = a.view_state()
    doc = json.loads(state)
    assert doc["layers"][1]["labels"] == {"outline": True, "selected": 2,
                                          "only_selected": False, "seed": 5}
    assert doc["layers"][1]["follows"] == 0
    b = Viewer.offscreen(W, H)
    b.set_view_state(state)
    lay = b.layers[1]
    assert (lay["label"], lay["outline"], lay["selected"], lay["seed"], lay["follows"]) == (
        True, True, 2, 5, 0)
