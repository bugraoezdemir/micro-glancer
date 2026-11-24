"""micro-glancer's app window (micro_glancer.app): opening images, the layers, planes and
channel panels, the toolbar driving the viewer, and label images (Labels button,
double-click selection, the label controls)."""
import json
import os
import time

import numpy as np
import pytest

qt = pytest.importorskip("PyQt6.QtWidgets")
ozp = pytest.importorskip("ome_zarr_pyramid")
from PyQt6.QtCore import Qt  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return qt.QApplication.instance() or qt.QApplication([])


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    out = {}
    zz, yy, xx = np.mgrid[0:20, 0:60, 0:80]
    vol = ((zz * 700 + yy * 30 + xx * 20) % 20000).astype("uint16")
    for name, data, axes, scales in (
            ("vol", vol, "zyx", [[2.0, 1.0, 1.0]]),
            ("flat", vol[5], "yx", [[1.0, 1.0]])):
        pyr = ozp.Pyramid().from_arrays([data], axis_order=axes, scales=scales, version="0.5")
        path = str(tmp / f"{name}.ome.zarr")
        ozp.IO().write_pyramid(pyr, path, overwrite=True)
        out[name] = path
    return out


def _pump(qapp, seconds=0.6):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        qapp.processEvents()
        time.sleep(0.01)


def test_the_app_window(qapp, images, monkeypatch):
    monkeypatch.setenv("MICRO_GLANCER_MODE", "offscreen")
    from micro_glancer.app import MainWindow
    win = MainWindow([images["vol"], images["flat"]])
    win.show()
    _pump(qapp)
    v = win.view.viewer
    assert len(v.layers) == 2
    assert win.layer_panel.list.count() == 2
    assert win.layer_panel.list.currentRow() == v.current == 1
    # the planes: the volume's z as a slider (current layer 0)
    win.layer_panel.list.setCurrentRow(0)
    _pump(qapp, 0.2)
    assert v.current == 0
    assert "z" in win.plane_panel._rows
    slider, _ = win.plane_panel._rows["z"]
    slider.setValue(7)
    assert v.position[0][1] == pytest.approx(7.0)
    # the toolbar
    win.linked.trigger()
    assert v.linked
    win.mode.setCurrentText("volume")
    assert v.mip and v.projection == "volume"
    win.density.setValue(50)
    assert v.density == pytest.approx(10 ** -1.5, rel=1e-6)
    win.shading.setChecked(False)
    assert not v.shading
    win.smooth.setChecked(True)
    assert v.layers[0]["interpolation"] == "linear"
    win.orient["xz"].trigger()
    assert np.allclose(v.orientation[1], [0, 0, 1])
    # what the keys change shows in the toolbar
    v.linked = False
    win.view._changed()
    _pump(qapp, 0.1)
    assert not win.linked.isChecked()
    # the channel panel follows (its histogram reads the whole volume with a projection)
    _pump(qapp, 0.5)
    assert win.channel_panel.hist._hist is not None
    # layers: hide one, change its opacity, remove it
    win.layer_panel.list.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert not v.layers[1]["visible"]
    win.layer_panel.opacity.setValue(40)
    assert v.layers[0]["opacity"] == pytest.approx(0.4)
    win.layer_panel.list.setCurrentRow(1)
    win.layer_panel.remove.click()
    assert len(v.layers) == 1
    _pump(qapp, 0.1)
    assert "layer 1/" not in win.status.text()           # one layer left: no layer count
    win.close()


def test_the_add_and_open_buttons(qapp, images, monkeypatch):
    """The toolbar's Add layer (it calls the dialog with add=True: once killed the app)."""
    monkeypatch.setenv("MICRO_GLANCER_MODE", "offscreen")
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QFileDialog
    from micro_glancer.app import MainWindow
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: images["flat"]))
    win = MainWindow([images["vol"]])
    win.show()
    _pump(qapp)
    actions = {a.text(): a for a in win.findChildren(QAction)}
    actions["Add layer"].trigger()
    _pump(qapp, 0.3)
    assert len(win.view.viewer.layers) == 2
    actions["Open"].trigger()                            # replaces the layers
    _pump(qapp, 0.3)
    assert len(win.view.viewer.layers) == 1
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(images["flat"]) + "-missing"))
    actions["Add layer"].trigger()                       # an error: reported, no crash
    _pump(qapp, 0.2)
    assert "missing" in win.status.text() or win.view.viewer.error is None
    win.close()


@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    """A flat image with one label image "cells" (id 5 over its middle)."""
    tmp = tmp_path_factory.mktemp("app_labels")
    yy, xx = np.mgrid[0:60, 0:80]
    img = ((yy * 30 + xx * 20) % 4000).astype("uint16")
    path = str(tmp / "img.ome.zarr")
    ozp.IO().write_pyramid(ozp.Pyramid().from_arrays(
        [img], axis_order="yx", scales=[[1.0, 1.0]], version="0.5"), path, overwrite=True)
    lab = np.zeros((60, 80), "uint32")
    lab[15:45, 20:60] = 5
    lp = os.path.join(path, "labels", "cells")
    ozp.IO().write_pyramid(ozp.Pyramid().from_arrays(
        [lab], axis_order="yx", scales=[[1.0, 1.0]], version="0.5"), lp, overwrite=True)
    meta = os.path.join(lp, "zarr.json")
    with open(meta) as f:
        doc = json.load(f)
    doc["attributes"]["ome"]["image-label"] = {"version": "0.5",
                                               "source": {"image": "../../"}}
    with open(meta, "w") as f:
        json.dump(doc, f)
    with open(os.path.join(path, "labels", "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": "group",
                   "attributes": {"ome": {"version": "0.5", "labels": ["cells"]}}}, f)
    return path


def test_label_images_in_the_app(qapp, labelled, monkeypatch):
    monkeypatch.setenv("MICRO_GLANCER_MODE", "offscreen")
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QKeyEvent, QMouseEvent
    from PyQt6.QtWidgets import QMenu
    from micro_glancer.app import MainWindow
    chosen = []

    def first_action(menu, *a, **k):                     # the menu without a click
        acts = [x for x in menu.actions() if x.isEnabled()]
        chosen.append([x.text() for x in menu.actions()])
        if acts:
            acts[0].trigger()
    monkeypatch.setattr(QMenu, "exec", first_action)
    win = MainWindow([labelled])
    win.show()
    _pump(qapp)
    v = win.view.viewer
    win.layer_panel.labels.click()
    _pump(qapp, 0.4)
    assert chosen == [["cells"]]
    assert len(v.layers) == 2 and v.layers[1]["label"] and v.current == 1
    assert win.layer_panel.list.item(1).text().strip() == "labels: cells"
    # the channel panel shows the label controls for a label layer
    win.channel_panel.refresh()
    assert win.channel_panel.labels.isVisibleTo(win.channel_panel)
    assert not win.channel_panel.images.isVisibleTo(win.channel_panel)
    win.channel_panel.outline.setChecked(True)
    assert v.layers[1]["outline"]
    # double-click in the middle selects id 5; on the background clears it
    view = win.view
    r = view.devicePixelRatioF()

    def dbl(x, y):
        e = QMouseEvent(QEvent.Type.MouseButtonDblClick, QPointF(x / r, y / r),
                        QPointF(x / r, y / r), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        view.mouseDoubleClickEvent(e)
    w, h = v.size
    # the ids come from the label tiles in the cache: wait for the view to be sharp (a
    # slow runner may not have read them in a fixed pause, CI macOS 2026-11-24)
    v.settle(30.0, False)
    dbl(w / 2, h / 2)
    assert v.layers[1]["selected"] == 5
    win.channel_panel.refresh()
    assert win.channel_panel.selected.text() == "selected id: 5"
    view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                 Qt.KeyboardModifier.NoModifier))
    assert v.layers[1]["selected"] is None
    view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_O,
                                 Qt.KeyboardModifier.NoModifier))
    assert not v.layers[1]["outline"]
    seed = v.layers[1]["seed"]
    win.channel_panel.recolour.click()
    assert v.layers[1]["seed"] == seed + 1
    # the image layer: the image controls again; its one label image is open already
    win.layer_panel.list.setCurrentRow(0)
    win.channel_panel.refresh()
    assert win.channel_panel.images.isVisibleTo(win.channel_panel)
    win.layer_panel.labels.click()
    assert chosen[-1] == ["(all open)"] and len(v.layers) == 2
    win.close()


from test_plates import plate  # noqa: E402,F401 (the plate fixture)


def test_plates_in_the_app(qapp, plate, monkeypatch):  # noqa: F811
    monkeypatch.setenv("MICRO_GLANCER_MODE", "offscreen")
    from micro_glancer.app import MainWindow
    win = MainWindow([plate])
    win.show()
    _pump(qapp)
    v = win.view.viewer
    assert v.layers[0]["plate"]
    # the Contents panel: plate > wells > fields
    top = win.contents.view.topLevelItem(0)
    assert top.text(0) == "plate: test plate" and top.childCount() == 3
    a3 = top.child(1)
    assert a3.text(0) == "A3"
    # a well: shown on the plate (the camera zooms onto it)
    scale = v.camera[2]
    win.contents.view.itemActivated.emit(a3, 0)
    _pump(qapp, 0.2)
    assert v.camera[2] < scale and len(v.layers) == 1
    assert "well A3" in win.status.text()
    # the field control in the Planes panel
    spin = win.plane_panel._rows["field"][0]
    assert spin.maximum() == 1 and spin.value() == -1 and spin.text() == "all"
    spin.setValue(1)
    assert v.plate()["field"] == 1
    spin.setValue(-1)                                  # every field again
    assert v.plate()["field"] is None
    # a field's label images are read when it is opened up; a label adds over its field
    a1 = top.child(0)
    f0 = a1.child(0)
    f0.setExpanded(True)
    assert f0.childCount() == 1 and f0.child(0).text(0) == "label: cells"
    win.contents.view.itemActivated.emit(f0, 0)       # the field opens on its own
    _pump(qapp, 0.2)
    assert not v.layers[0]["plate"] and len(v.layers) == 1
    win.contents.view.itemActivated.emit(f0.child(0), 0)
    _pump(qapp, 0.2)
    assert len(v.layers) == 2 and v.layers[1]["label"]
    win.close()
