"""Image positions (Workplan WP14): every image opens where it sits among its
neighbours, in both ways a file holds positions:

- NGFF 0.4 / 0.5: in the image's own levels (their translations);
- NGFF 0.6: in the `scene` of the group owning the images (a well, a collection), one
  edge per image into the shared system, the image's own levels at 0 (as ozp and eubi
  write them).

A plate shows every field of each well at its position (one field on request); a well
or a bioformats2raw collection opens whole, each image at its position; a single field,
and its label image, open at the field's position. Checked pixel for pixel through
`pick` at world points.
"""
import json
import os

import numpy as np
import pytest

from micro_glancer import Viewer, tree

ozp = pytest.importorskip("ome_zarr_pyramid")

FH, FW = 30, 40
#: field positions in a well (y, x), world units (scale 1): two side by side with a gap
#: of 5, one below, shifted, overlapping nothing
POS = [(0.0, 0.0), (0.0, FW + 5.0), (FH + 3.0, 10.0)]
#: the wells: path, row, column, how many fields (positions from POS); the second well's
#: fields sit 1000 units away, so only their positions relative to each other count
WELLS = {"A/1": (0, 0, 3), "B/2": (1, 1, 2)}
OFFSET = {"A/1": (0.0, 0.0), "B/2": (1000.0, -500.0)}


def _field(k, f):
    yy, xx = np.mgrid[0:FH, 0:FW]
    return (1000 * (k + 1) + 100 * f + (yy * 3 + xx) % 97).astype("uint16")


def _doc(path, attrs):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as fh:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": attrs}, fh)


def _attrs(path):
    with open(os.path.join(path, "zarr.json")) as fh:
        return json.load(fh)["attributes"]


def _set_attrs(path, attrs):
    with open(os.path.join(path, "zarr.json")) as fh:
        doc = json.load(fh)
    doc["attributes"] = attrs
    with open(os.path.join(path, "zarr.json"), "w") as fh:
        json.dump(doc, fh)


def _image(path, data, version, at=None):
    """One image of two levels; `at` (y, x): its translation (0.5), none (0.6)."""
    pyr = ozp.Pyramid().from_arrays([data, data[::2, ::2]], axis_order="yx",
                                    scales=[[1.0, 1.0], [2.0, 2.0]], version=version)
    if at is not None:
        for p in pyr.meta.resolution_paths:
            pyr.meta.update_translation(p, list(at))
    ozp.IO().write_pyramid(pyr, path, overwrite=True)


def _scene(system, edges):
    """A 0.6 scene: shared system `system` (y, x), position edges {path: (y, x)}."""
    return {"coordinateSystems": [{"name": system, "axes": [
                {"name": "y", "type": "space"}, {"name": "x", "type": "space"}]}],
            "coordinateTransformations": [
                {"type": "translation", "translation": list(at),
                 "input": {"name": "physical", "path": p}, "output": {"name": system}}
                for p, at in edges.items()]}


def _where(well, f):
    oy, ox = OFFSET[well]
    return (POS[f][0] + oy, POS[f][1] + ox)


def _write_plate(root, version):
    wells = []
    for k, (path, (r, c, nf)) in enumerate(WELLS.items()):
        for f in range(nf):
            fp = os.path.join(root, path, str(f))
            _image(fp, _field(k, f), version, None if version == "0.6" else _where(path, f))
            if version == "0.6":
                # ozp's mark: the scene is on the well
                a = _attrs(fp)
                a["eubi:scene"] = ".."
                _set_attrs(fp, a)
        well = {"version": version, "well": {"images": [{"path": str(f)} for f in range(nf)]}}
        if version == "0.6":
            well["scene"] = _scene(f"well-{path}", {str(f): _where(path, f) for f in range(nf)})
        _doc(os.path.join(root, path), {"ome": well})
        wells.append({"path": path, "rowIndex": r, "columnIndex": c})
    _doc(os.path.join(root, "A"), {})
    _doc(os.path.join(root, "B"), {})
    _doc(root, {"ome": {"version": version, "plate": {
        "name": "positions", "rows": [{"name": "A"}, {"name": "B"}],
        "columns": [{"name": "1"}, {"name": "2"}], "wells": wells}}})
    # a label image on field 2 of A/1: a square at its pixels 5..15
    lab = np.zeros((FH, FW), "uint16")
    lab[5:15, 5:15] = 7
    lp = os.path.join(root, "A/1/2/labels/cells")
    ozp.IO().write_pyramid(ozp.Pyramid().from_arrays(
        [lab], axis_order="yx", scales=[[1.0, 1.0]], version=version), lp, overwrite=True)
    a = _attrs(lp)
    a["ome"]["image-label"] = {"version": version}
    if version != "0.6":
        # 0.4 / 0.5: a label image carries its image's translation itself
        for ds in a["ome"]["multiscales"][0]["datasets"]:
            for t in ds["coordinateTransformations"]:
                if t["type"] == "translation":
                    t["translation"] = list(_where("A/1", 2))
            if not any(t["type"] == "translation" for t in ds["coordinateTransformations"]):
                ds["coordinateTransformations"].append(
                    {"type": "translation", "translation": list(_where("A/1", 2))})
    _set_attrs(lp, a)
    _doc(os.path.join(root, "A/1/2/labels"), {"ome": {"version": version, "labels": ["cells"]}})


@pytest.fixture(scope="module", params=["0.5", "0.6"])
def plate(request, tmp_path_factory):
    root = str(tmp_path_factory.mktemp(f"pos{request.param}") / "plate.ome.zarr")
    _write_plate(root, request.param)
    return root


def _screen(v, cx, cy, zoom, wx, wy):
    """The screen point of world point (wx, wy) after look_at(cx, cy, zoom)."""
    w, h = v.size
    return (wx - cx) * zoom + w / 2, (wy - cy) * zoom + h / 2


def _value_at(v, cx, cy, wx, wy):
    sx, sy = _screen(v, cx, cy, 1.0, wx, wy)
    return v.pick(sx, sy)


def test_a_well_opens_with_every_field_at_its_position(plate):
    # the well alone: its fields at their positions, in world units
    well = os.path.join(plate, "A", "1")
    v = Viewer.offscreen(200, 200)
    v.open(well)
    m = v.members()
    assert m["names"] == ["field 0", "field 1", "field 2"] and not m["gridded"]
    assert m["skipped"] == []
    cx, cy = 40.0, 30.0
    v.look_at(cx, cy, 1.0)
    v.settle(30.0)
    for f in range(3):
        y0, x0 = POS[f]
        for (dy, dx) in ((0, 0), (7, 11), (FH - 1, FW - 1)):
            q = _value_at(v, cx, cy, x0 + dx, y0 + dy)
            assert q["member"] == (f"field {f}", str(f)), (f, dy, dx, q["member"])
            assert q["values"][0]["value"] == _field(0, f)[dy, dx], (f, dy, dx)
    # the gap between fields 0 and 1 is empty
    q = _value_at(v, cx, cy, FW + 2.0, 5.0)
    assert q["member"] is None
    # the well's own tree lists its fields
    t = tree(well)
    assert t["kind"] == "well" and [c["name"] for c in t["children"]] == [
        "field 0", "field 1", "field 2"]


def test_a_field_and_its_label_open_at_the_fields_position(plate):
    field = os.path.join(plate, "A", "1", "2")
    y0, x0 = POS[2]
    v = Viewer.offscreen(120, 120)
    v.open(field)
    v.add(os.path.join(field, "labels", "cells"))
    cx, cy = x0 + 20.0, y0 + 15.0
    v.look_at(cx, cy, 1.0)
    v.settle(30.0)
    q = _value_at(v, cx, cy, x0 + 8, y0 + 9)
    image, label = q["layers"][0], q["layers"][1]
    assert image["values"][0]["value"] == _field(0, 2)[9, 8]
    assert label["values"][0]["id"] == 7                     # the label's square
    q = _value_at(v, cx, cy, x0 + 20, y0 + 20)
    assert q["layers"][1]["values"][0]["id"] == 0


def _plate_pixels(v):
    """Every 2nd screen pixel of the view: {level-0 plate pixel: (field path, value)} where
    a field of well A1 is under it."""
    w, h = v.size
    seen = {}
    for sy in range(0, h, 2):
        for sx in range(0, w, 2):
            q = v.pick(sx + 0.5, sy + 0.5)
            if q["well"] and q["well"][0] == "A1" and q["values"][0]["value"] is not None:
                seen[tuple(q["pixel"])] = (q["well"][1], q["values"][0]["value"])
    return seen


def test_a_plate_shows_every_field_at_its_position(plate):
    v = Viewer.offscreen(400, 300)
    v.open(plate)
    info = v.plate()
    assert info["field"] is None and v.layers[0]["field"] is None
    assert info["skipped"] == [] and not any(w["gridded"] for w in info["wells"])
    assert v.members() is None                               # a plate is not a group
    v.set_contrast(0.0, 4000.0)
    v.show_well("A1")
    v.settle(30.0)
    seen = _plate_pixels(v)
    assert {path for path, _ in seen.values()} == {"A/1/0", "A/1/1", "A/1/2"}
    # every pixel shows its field's own pixel at its offset from the field's corner: the
    # fields' corners are field 0's plus their positions (integers here: no rounding)
    f0 = [p for p, (path, _) in seen.items() if path == "A/1/0"]
    x0, y0 = min(p[0] for p in f0), min(p[1] for p in f0)
    bad = []
    for (px, py), (path, val) in seen.items():
        f = int(path.split("/")[-1])
        dy, dx = py - y0 - int(POS[f][0]), px - x0 - int(POS[f][1])
        if not (0 <= dy < FH and 0 <= dx < FW) or val != _field(0, f)[dy, dx]:
            bad.append(((px, py), path, val))
    assert bad == [] and len(seen) > 200, bad[:5]


def test_one_field_on_request_and_back(plate):
    v = Viewer.offscreen(200, 200)
    v.open(plate)
    v.set_field(1)
    assert v.plate()["field"] == 1
    state = json.loads(v.view_state())
    assert state["layers"][0]["field"] == 1
    v.set_field(None)
    assert v.plate()["field"] is None
    state = json.loads(v.view_state())
    assert state["layers"][0]["field"] == -1
    w = Viewer.offscreen(200, 200)
    w.set_view_state(v.view_state())
    assert w.plate()["field"] is None


@pytest.fixture(scope="module", params=["0.5", "0.6"])
def collection(request, tmp_path_factory):
    """A bioformats2raw collection of three tiles (series), each at its position."""
    version = request.param
    root = str(tmp_path_factory.mktemp(f"coll{version}") / "tiles.ome.zarr")
    for s in range(3):
        _image(os.path.join(root, str(s)), _field(5, s), version,
               None if version == "0.6" else POS[s])
    attrs = {"ome": {"version": version, "bioformats2raw.layout": 3}}
    if version == "0.6":
        attrs["ome"]["scene"] = _scene("mosaic", {str(s): POS[s] for s in range(3)})
    _doc(root, attrs)
    return root


def test_a_collection_opens_with_every_series_at_its_position(collection):
    v = Viewer.offscreen(200, 200)
    v.open(collection)
    m = v.members()
    assert m["names"] == ["series 0", "series 1", "series 2"] and not m["gridded"]
    cx, cy = 40.0, 30.0
    v.look_at(cx, cy, 1.0)
    v.settle(30.0)
    for s in range(3):
        y0, x0 = POS[s]
        q = _value_at(v, cx, cy, x0 + 3, y0 + 4)
        assert q["member"] == (f"series {s}", str(s))
        assert q["values"][0]["value"] == _field(5, s)[4, 3]
    # one series alone opens at its position too
    w = Viewer.offscreen(100, 100)
    w.open(os.path.join(collection, "1"))
    y0, x0 = POS[1]
    w.look_at(x0 + 20, y0 + 15, 1.0)
    w.settle(30.0)
    q = _value_at(w, x0 + 20, y0 + 15, x0 + 6, y0 + 2)
    assert q["values"][0]["value"] == _field(5, 1)[2, 6]


def test_fields_without_positions_lie_on_a_grid(tmp_path):
    # a well whose fields all sit at 0: side by side in their order
    root = str(tmp_path / "w.ome.zarr")
    for f in range(2):
        _image(os.path.join(root, str(f)), _field(0, f), "0.5")
    _doc(root, {"ome": {"version": "0.5", "well": {"images": [{"path": "0"}, {"path": "1"}]}}})
    v = Viewer.offscreen(200, 100)
    v.open(root)
    m = v.members()
    assert m["gridded"]
    (a, b) = m["rects"]
    assert a[0] == b[0] and b[1] >= a[3]                      # one row, the second right of the first
