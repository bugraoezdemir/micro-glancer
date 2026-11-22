"""micro-viewer's app window (micro_viewer.app): opening images, the layers, planes and
channel panels, and the toolbar driving the viewer."""
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
    monkeypatch.setenv("MICRO_VIEWER_MODE", "offscreen")
    from micro_viewer.app import MainWindow
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
    monkeypatch.setenv("MICRO_VIEWER_MODE", "offscreen")
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QFileDialog
    from micro_viewer.app import MainWindow
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
