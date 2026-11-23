"""micro-glancer's plates (Workplan WP14, task V9 part 2): an HCS plate opens as one image
(its wells on a grid, coarse levels zoomed out), the well under the pointer, switching
fields, and the navigation tree (plate, wells, fields, label images). A plate opens with
every field of each well (see test_positions.py: at their positions, here on a small
grid, as these fields carry none); these tests show one field, as asked for with
`set_field`.

Values are checked through `pick` against the plate laid out with NumPy: cell pitch =
the largest field + 4 % (at least 2 pixels), positions without a well 0.
"""
import json
import os

import numpy as np
import pytest

from micro_glancer import Viewer, ViewerError, tree

ozp = pytest.importorskip("ome_zarr_pyramid")

FH, FW = 40, 50                     # a field's size
PY, PX = FH + 2, FW + 2             # the cell pitch at level 0 (4 % < 2 pixels here)
WELLS = {"A/1": (0, 0, 2), "A/3": (0, 2, 1), "B/2": (1, 1, 1)}   # row, column, fields


def _field(k, f):
    yy, xx = np.mgrid[0:FH, 0:FW]
    return (1000 * (k + 1) + 100 * f + (yy * 3 + xx) % 97).astype("uint16")


def _group(path, attrs):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": attrs}, f)


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("plates") / "plate.ome.zarr")
    wells = []
    for k, (path, (r, c, nf)) in enumerate(WELLS.items()):
        for f in range(nf):
            data = _field(k, f)
            pyr = ozp.Pyramid().from_arrays([data, data[::2, ::2]], axis_order="yx",
                                            scales=[[1.0, 1.0], [2.0, 2.0]], version="0.5")
            ozp.IO().write_pyramid(pyr, os.path.join(root, path, str(f)), overwrite=True)
        _group(os.path.join(root, path), {"ome": {"version": "0.5", "well": {
            "images": [{"path": str(f)} for f in range(nf)]}}})
        wells.append({"path": path, "rowIndex": r, "columnIndex": c})
    _group(os.path.join(root, "A"), {})
    _group(os.path.join(root, "B"), {})
    _group(root, {"ome": {"version": "0.5", "plate": {
        "name": "test plate", "rows": [{"name": "A"}, {"name": "B"}],
        "columns": [{"name": "1"}, {"name": "2"}, {"name": "3"}], "wells": wells}}})
    # a label image on field 0 of A/1
    lab = np.zeros((FH, FW), "uint16")
    lab[5:20, 5:30] = 1
    lp = os.path.join(root, "A/1/0/labels/cells")
    ozp.IO().write_pyramid(ozp.Pyramid().from_arrays(
        [lab], axis_order="yx", scales=[[1.0, 1.0]], version="0.5"), lp, overwrite=True)
    with open(os.path.join(lp, "zarr.json")) as f:
        doc = json.load(f)
    doc["attributes"]["ome"]["image-label"] = {"version": "0.5"}
    with open(os.path.join(lp, "zarr.json"), "w") as f:
        json.dump(doc, f)
    _group(os.path.join(root, "A/1/0/labels"), {"ome": {"version": "0.5", "labels": ["cells"]}})
    return root


def _expected(field=0, level=0):
    """The plate at `level` as the viewer lays it out."""
    py, px = (PY, PX) if level == 0 else (PY // 2, PX // 2)
    out = np.zeros((2 * py, 3 * px), "uint16")
    for k, (path, (r, c, nf)) in enumerate(WELLS.items()):
        if field < nf:
            d = _field(k, field)[:: 2 ** level, :: 2 ** level]
            out[r * py:r * py + d.shape[0], c * px:c * px + d.shape[1]] = d
    return out


def _check(v, want, scale):
    """Every 7th screen pixel: the picked value is the expected value of its pixel."""
    w, h = v.size
    n = 0
    for sy in range(3, h, 7):
        for sx in range(3, w, 7):
            p = v.pick(sx + 0.5, sy + 0.5)
            if p["pixel"] is None or p["values"][0]["value"] is None:
                continue
            x, y = (int(c / scale) for c in p["pixel"])
            assert p["values"][0]["value"] == want[y, x], (sx, sy, p["pixel"])
            n += 1
    assert n > 100


def test_a_plate_opens_as_one_image(plate):
    v = Viewer.offscreen(3 * PX, 2 * PY)
    v.open(plate)
    assert v.layers[0]["plate"] and v.layers[0]["field"] is None   # every field
    assert [w["gridded"] for w in v.plate()["wells"]] == [True, False, False]
    v.set_field(0)
    lay = v.layers[0]
    assert lay["plate"] and lay["field"] == 0
    info = v.plate()
    assert info["name"] == "test plate" and info["grid"] == (2, 3) and info["fields"] == 2
    assert sorted(w["name"] for w in info["wells"]) == ["A1", "A3", "B2"]
    assert info["skipped"] == []
    v.set_contrast(0.0, 4000.0)
    v.look_at((3 * PX - 1) / 2, (2 * PY - 1) / 2, 1.0)
    v.settle(30.0)
    _check(v, _expected(), 1)
    # the well under the pointer
    p = v.pick(PX * 2 + 10.5, 10.5)
    assert p["well"] == ("A3", "A/3/0")
    assert v.pick(PX + 10.5, 10.5)["well"] is None            # A2: no well there
    assert v.pick(FW + 0.5, 10.5)["well"] is None              # the gap


def test_zoomed_out_a_plate_reads_coarse_levels(plate):
    v = Viewer.offscreen(3 * PX // 2, PY)
    v.open(plate)
    v.set_field(0)
    v.look_at((3 * PX - 1) / 2, (2 * PY - 1) / 2, 2.0)
    v.settle(30.0)
    # the picked level is 1 wherever a value was found
    p = v.pick(10.5, 10.5)
    assert p["values"][0]["level"] == 1
    want = _expected(level=1)
    w, h = v.size
    n = 0
    for sy in range(1, h, 3):
        for sx in range(1, w, 3):
            q = v.pick(sx + 0.5, sy + 0.5)
            if q["pixel"] is None or q["values"][0]["value"] is None:
                continue
            # the level-1 pixel nearest the world point: level 1's pixels are 2 apart,
            # its first centre at 0.5 (ozp's translation for a 2x level)
            wx, wy = q["world"][:2]
            fx, fy = (wx - 0.5) / 2, (wy - 0.5) / 2
            if min(abs(fx % 1 - 0.5), abs(fy % 1 - 0.5)) < 0.05:
                continue                                       # halfway between centres
            x1, y1 = int(np.floor(fx + 0.5)), int(np.floor(fy + 0.5))
            assert q["values"][0]["value"] == want[y1, x1], (sx, sy, wx, wy)
            n += 1
    assert n > 20


def test_switching_fields(plate):
    v = Viewer.offscreen(3 * PX, 2 * PY)
    v.open(plate)
    v.set_contrast(0.0, 4000.0)
    v.set_field(1)
    assert v.plate()["field"] == 1 and v.layers[0]["field"] == 1
    lo, hi = v.channels[0]["lo"], v.channels[0]["hi"]
    assert (lo, hi) == (0.0, 4000.0)                           # the channels stay
    v.look_at((3 * PX - 1) / 2, (2 * PY - 1) / 2, 1.0)
    v.settle(30.0)
    _check(v, _expected(field=1), 1)
    assert v.pick(10.5, 10.5)["well"] == ("A1", "A/1/1")
    with pytest.raises(ViewerError, match="no field 5"):
        v.set_field(5)
    state = json.loads(v.view_state())
    assert state["layers"][0]["field"] == 1
    w = Viewer.offscreen(3 * PX, 2 * PY)
    w.set_view_state(v.view_state())
    assert w.layers[0]["field"] == 1


def test_the_navigation_tree(plate):
    t = tree(plate)
    assert t["kind"] == "plate" and t["name"] == "test plate" and t["grid"] == [2, 3]
    wells = t["children"]
    assert [w["name"] for w in wells] == ["A1", "A3", "B2"]
    a1 = wells[0]
    assert [f["name"] for f in a1["children"]] == ["field 0", "field 1"]
    f0 = a1["children"][0]
    assert f0["kind"] == "image" and f0["field"] == 0
    assert os.path.normpath(f0["path"]) == os.path.normpath(os.path.join(plate, "A/1/0"))
    ft = tree(f0["path"])
    assert ft["kind"] == "image" and [c["name"] for c in ft["children"]] == ["cells"]
    assert ft["children"][0]["kind"] == "label"
    assert tree(ft["children"][0]["path"])["kind"] == "label"
    # a field opens as an image of its own
    v = Viewer.offscreen(FW, FH)
    v.open(f0["path"])
    assert not v.layers[0]["plate"] and v.plate() is None


def test_a_plates_transform_is_not_saved(plate):
    from micro_glancer.save import save_transform
    v = Viewer.offscreen(3 * PX, 2 * PY)
    v.open(plate)
    v.move_layer(5, 0)
    with pytest.raises(ViewerError, match="open a field"):
        save_transform(v)
