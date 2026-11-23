"""micro-glancer's app: a window around the viewer widget.

``micro-glancer [PATH ...]`` (or ``python -m micro_glancer``) opens the images as layers.

- Toolbar: open an image, add one as a layer, save the current layer's transform, fit;
  look along z (XY), y (XZ) or x (YZ); linked views; the projection (maximum intensity,
  volume or surfaces), the volume's density, shading, smooth (trilinear) sampling.
- Left: the layers (shown or not, the current one, its opacity; remove it, put it back
  where its image places it).
- Right: the current layer's channels with the histogram (micro_glancer.panel): contrast,
  surface level and opacity curve adjusted on it.
- Bottom: the current layer's planes (z, t, ...) as sliders.
- Status bar: the current layer, its plane, level and transform, the values under the
  pointer, errors.

Folders and .zip / .ozx files dropped on the window open as layers. The keys of the
viewer widget work too (Help > Keys).
"""
from __future__ import annotations

import functools
import math
import os
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog,
                             QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                             QMenu, QMessageBox, QPushButton, QSlider, QSpinBox, QToolBar,
                             QVBoxLayout, QWidget)

from . import ERRORS
from .contents import ContentsPanel, open_node
from .panel import ChannelPanel
from .qt import ViewerWidget

KEYS = """Mouse: drag pans, the wheel zooms at the pointer.
F fits the image; Page Up / Page Down move through z (with Shift: t).
Shift + wheel turns the view; [ and ] turn it by 90 degrees; 0 turns it back.
Volumes: Alt + drag turns the plane through the volume; 1, 2, 3 look along z, y, x;
I switches smooth (trilinear) sampling.
M shows the projection; P cycles it (maximum intensity, volume, surfaces);
Shift + P switches shading; - and + make the volume thinner or denser.
L switches linked views (dragging, the wheel and Page Up / Down act in the view under
the pointer).
Layers: Tab picks the current layer; Ctrl + drag moves it, Ctrl + wheel turns it
(with Shift finely), Ctrl + arrows nudge it, Ctrl + Alt + drag tilts a volume,
Ctrl + 0 puts it back, Ctrl + S saves its transform into its image.
Label images (Layers > Labels adds one): double-click selects the id under the pointer,
Esc clears it, O switches outlines.

A plate opens whole: its wells on a grid, every field of each well at its position (the
Planes panel's "field" shows one field instead, "all" every field). A well, or a
collection of series (tiles, views, positions), opens with each image at its position. The Contents panel lists the opened store's wells, fields and label images:
double-click a well to show it, a field or image to open it, a label image to add it."""


def _guarded(method):
    """A slot that shows its error in the status bar instead of raising (PyQt6 ends
    the process when a slot raises)."""
    @functools.wraps(method)
    def run(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except ERRORS as exc:  # noqa: BLE001 (a Rust panic too)
            self.note(str(exc))
    return run


def _turn_and_shift(m):
    """A 2 x 3 transform as words: its turn (degrees) and shift."""
    angle = math.degrees(math.atan2(m[1][0], m[0][0]))
    return f"turned {angle:+.2f} deg, moved ({m[0][2]:+.2f}, {m[1][2]:+.2f})"


class LayerPanel(QWidget):
    """The layers: shown or not, the current one, its opacity; remove, reset."""

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.list = QListWidget()
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.remove = QPushButton("Remove")
        self.reset = QPushButton("Put back")
        self.labels = QPushButton("Labels")
        self.labels.setToolTip("add one of the current image's label images")
        self.reset.setToolTip("the current layer back where its image places it")
        row = QHBoxLayout()
        row.addWidget(QLabel("opacity"))
        row.addWidget(self.opacity, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.labels)
        buttons.addWidget(self.reset)
        buttons.addWidget(self.remove)
        lay = QVBoxLayout(self)
        lay.addWidget(self.list, 1)
        lay.addLayout(row)
        lay.addLayout(buttons)
        self._filling = False
        self.list.currentRowChanged.connect(self._picked)
        self.list.itemChanged.connect(self._ticked)
        self.opacity.valueChanged.connect(self._opacity)
        self.remove.clicked.connect(self._remove)
        self.labels.clicked.connect(self._label_menu)
        self.reset.clicked.connect(self._reset)

    def refresh(self):
        v = self.window.view.viewer
        layers = v.layers
        self._filling = True
        names = [("    " if l["follows"] is not None else "")
                 + ("labels: " if l["label"] else "")
                 + (os.path.basename(l["path"].rstrip("/\\")) or l["path"]) for l in layers]
        if [self.list.item(k).text() for k in range(self.list.count())] != names:
            self.list.clear()
            for name, l in zip(names, layers):
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setToolTip(l["path"])
                self.list.addItem(item)
        for k, l in enumerate(layers):
            self.list.item(k).setCheckState(
                Qt.CheckState.Checked if l["visible"] else Qt.CheckState.Unchecked)
        if layers:
            self.list.setCurrentRow(v.current)
            self.opacity.setValue(round(layers[v.current]["opacity"] * 100))
        self._filling = False

    def _do(self, step):
        if self._filling:
            return
        try:
            step(self.window.view.viewer)
        except ERRORS as exc:  # noqa: BLE001 (see _guarded)
            self.window.note(str(exc))
            return
        self.window.view._changed()

    def _picked(self, row):
        if row >= 0:
            self._do(lambda v: setattr(v, "current", row))

    def _ticked(self, item):
        row = self.list.row(item)
        on = item.checkState() == Qt.CheckState.Checked
        self._do(lambda v: v.set_layer(row, visible=on))

    def _opacity(self, value):
        self._do(lambda v: v.set_layer(v.current, opacity=value / 100))

    def _remove(self):
        self._do(lambda v: v.remove(v.current) if v.layers else None)

    def _label_menu(self):
        """The current image's label images to add (those not open yet)."""
        try:
            v = self.window.view.viewer
            if not v.layers:
                return
            layer = v.current
            if v.layers[layer]["follows"] is not None:
                layer = v.layers[layer]["follows"]      # a label layer: its image
            names = v.label_names(layer=layer)
            layers = v.layers
        except ERRORS as exc:  # noqa: BLE001 (see _guarded)
            self.window.note(str(exc))
            return
        open_ = {os.path.basename(l["path"].rstrip("/\\")) for l in layers
                 if l["follows"] == layer}
        menu = QMenu(self)
        if not names:
            menu.addAction("(no label images)").setEnabled(False)
        elif all(n in open_ for n in names):
            menu.addAction("(all open)").setEnabled(False)
        names = [n for n in names if n not in open_]
        for name in names:
            menu.addAction(name, lambda n=name, k=layer: self._do(
                lambda v: v.add_label(n, layer=k)))
        menu.exec(self.labels.mapToGlobal(self.labels.rect().bottomLeft()))

    def _reset(self):
        self._do(lambda v: v.reset_transform() if v.layers else None)


class PlanePanel(QWidget):
    """The current layer's plane axes (z, t, ...) as sliders."""

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(6, 2, 6, 2)
        self._rows = {}                 # axis -> (slider, label)
        self._axes = None

    def refresh(self):
        v = self.window.view.viewer
        pos = v.position if v.layers else []
        axes = [(n, s) for n, _, s in pos]
        plate = v.plate() if v.layers and v.layers[v.current]["plate"] else None
        if plate:
            axes = axes + [("field", plate["fields"])]
        if axes != self._axes:
            while self.lay.count():
                item = self.lay.takeAt(0)
                if item.layout():
                    while item.layout().count():
                        w = item.layout().takeAt(0).widget()
                        if w:
                            w.deleteLater()
                elif item.widget():
                    item.widget().deleteLater()
            self._rows = {}
            for name, size in axes:
                row = QHBoxLayout()
                if name == "field":
                    spin = QSpinBox()
                    # -1: every field of each well at its position
                    spin.setRange(-1, max(size - 1, 0))
                    spin.setSpecialValueText("all")
                    spin.setToolTip("the field shown in every well (all: every field at "
                                    "its position)")
                    spin.valueChanged.connect(self._field)
                    row.addWidget(QLabel("field"))
                    row.addWidget(spin)
                    row.addStretch(1)
                    self.lay.addLayout(row)
                    self._rows[name] = (spin, None)
                    continue
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, max(size - 1, 0))
                label = QLabel()
                label.setMinimumWidth(70)
                slider.valueChanged.connect(lambda val, n=name: self._moved(n, val))
                row.addWidget(QLabel(name))
                row.addWidget(slider, 1)
                row.addWidget(label)
                self.lay.addLayout(row)
                self._rows[name] = (slider, label)
            self._axes = axes
        for name, index, size in pos:
            slider, label = self._rows[name]
            slider.blockSignals(True)
            slider.setValue(round(index))
            slider.blockSignals(False)
            label.setText(f"{round(index)} / {size - 1}")
        if plate:
            spin = self._rows["field"][0]
            spin.blockSignals(True)
            spin.setValue(-1 if plate["field"] is None else plate["field"])
            spin.blockSignals(False)

    def _field(self, value):
        try:
            self.window.view.viewer.set_field(None if int(value) < 0 else int(value))
        except ERRORS as exc:  # noqa: BLE001 (see _guarded)
            self.window.note(str(exc))
            return
        self.window.view._changed()

    def _moved(self, name, value):
        try:
            self.window.view.viewer.set_position(name, float(value))
        except ERRORS as exc:  # noqa: BLE001 (see _guarded)
            self.window.note(str(exc))
            return
        self.window.view._changed()


class MainWindow(QMainWindow):
    """micro-glancer's window (see the module's notes)."""

    def __init__(self, paths=()):
        super().__init__()
        self.setWindowTitle("micro-glancer")
        self.resize(1600, 900)
        self.setAcceptDrops(True)
        self.view = ViewerWidget()
        self.setCentralWidget(self.view)
        self._pick = ""
        self._note = ""
        # docks
        self.layer_panel = LayerPanel(self)
        self.channel_panel = ChannelPanel(self.view)
        self.plane_panel = PlanePanel(self)
        self.contents = ContentsPanel()
        self.contents.activated.connect(self._open_node)
        for title, widget, area in (
                ("Contents", self.contents, Qt.DockWidgetArea.LeftDockWidgetArea),
                ("Layers", self.layer_panel, Qt.DockWidgetArea.LeftDockWidgetArea),
                ("Channels", self.channel_panel, Qt.DockWidgetArea.RightDockWidgetArea),
                ("Planes", self.plane_panel, Qt.DockWidgetArea.BottomDockWidgetArea)):
            dock = QDockWidget(title, self)
            dock.setObjectName(title)
            dock.setWidget(widget)
            dock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.addDockWidget(area, dock)
        self._toolbar()
        self._menu()
        self.status = QLabel()
        self.statusBar().addWidget(self.status, 1)
        # what follows the view
        self.view.viewChanged.connect(self._view_changed)
        self.view.picked.connect(self._picked)
        self.view.saved.connect(self.note)
        self.view.failed.connect(self.note)
        timer = QTimer(self)
        timer.timeout.connect(self._show)
        timer.start(250)
        self._paths = [str(p) for p in paths]

    # ------------------------------------------------------------------ building
    def _action(self, bar, text, slot, tip="", checkable=False, key=None):
        act = QAction(text, self)
        act.setToolTip(tip or text)
        act.setCheckable(checkable)
        if key:
            act.setShortcut(QKeySequence(key))
        act.triggered.connect(slot)
        bar.addAction(act)
        return act

    def _toolbar(self):
        bar = QToolBar("View")
        bar.setObjectName("view")
        self.addToolBar(bar)
        self._action(bar, "Open", self.open_dialog, "open an image (a folder, .zip or .ozx)")
        self._action(bar, "Add layer", lambda: self.open_dialog(add=True),
                     "add an image as a layer on top")
        self._action(bar, "Save transform", lambda: self.view.save_transform(),
                     "save the current layer's transform into its image")
        self._action(bar, "Fit", self._fit, "show everything (F)")
        bar.addSeparator()
        group = QActionGroup(self)
        self.orient = {}
        for name, tip in (("XY", "look along z (1)"), ("XZ", "look along y (2)"),
                          ("YZ", "look along x (3)")):
            act = self._action(bar, name, lambda _=False, n=name.lower(): self._orient(n),
                               tip, checkable=True)
            group.addAction(act)
            self.orient[name.lower()] = act
        self.orient["xy"].setChecked(True)
        bar.addSeparator()
        self.linked = self._action(bar, "Linked", self._linked,
                                   "the plane, YZ, XZ and the projection in quarters (L)",
                                   checkable=True)
        self.projected = self._action(bar, "Projection", self._projection,
                                      "the projection instead of the plane (M)",
                                      checkable=True)
        self.mode = QComboBox()
        self.mode.addItems(["mip", "volume", "iso"])
        self.mode.setToolTip("what the projection shows (P)")
        self.mode.currentTextChanged.connect(self._mode)
        bar.addWidget(self.mode)
        bar.addWidget(QLabel(" density "))
        self.density = QSlider(Qt.Orientation.Horizontal)
        self.density.setRange(0, 100)          # log scale, 0.001 .. 1
        self.density.setFixedWidth(120)
        self.density.setToolTip("the volume's opacity per voxel at full brightness (- / +)")
        self.density.valueChanged.connect(self._density)
        bar.addWidget(self.density)
        self.shading = QCheckBox("shading")
        self.shading.toggled.connect(self._shading)
        bar.addWidget(self.shading)
        self.smooth = QCheckBox("smooth")
        self.smooth.setToolTip("trilinear sampling between voxels for the current layer (I)")
        self.smooth.toggled.connect(self._smooth)
        bar.addWidget(self.smooth)

    def _menu(self):
        files = self.menuBar().addMenu("&File")
        self._action(files, "&Open...", self.open_dialog, key="Ctrl+O")
        self._action(files, "&Add layer...", lambda: self.open_dialog(add=True),
                     key="Ctrl+Shift+O")
        self._action(files, "Open &zip / ozx...", lambda: self.open_dialog(zipped=True))
        files.addSeparator()
        self._action(files, "&Quit", self.close, key="Ctrl+Q")
        help_menu = self.menuBar().addMenu("&Help")
        self._action(help_menu, "&Keys", lambda: QMessageBox.information(self, "Keys", KEYS))

    # ------------------------------------------------------------------ opening
    def open_paths(self, paths, add=False):
        for k, path in enumerate(paths):
            if k == 0 and not add:
                self.view.open(path)
                self.contents.show_tree(path)
            else:
                self.view.add(path)
        self._first_colours()
        self.setWindowTitle("micro-glancer: " + "  +  ".join(
            os.path.basename(str(p).rstrip("/\\")) for p in paths))
        self._view_changed()

    @_guarded
    def _open_node(self, node):
        """A node of the Contents panel (see contents.open_node)."""
        done = open_node(self.view.viewer, node)
        self.view._changed()
        self.note(done)

    def _first_colours(self):
        """Two one-channel images: green and magenta, to align them."""
        v = self.view.viewer
        if len(v.layers) == 2 and all(len(l["channels"]) == 1 for l in v.layers):
            v.set_channel(0, color=(0.0, 1.0, 0.0), layer=0)
            v.set_channel(0, color=(1.0, 0.0, 1.0), layer=1)

    @_guarded
    def open_dialog(self, add=False, zipped=False):
        if zipped:
            path, _ = QFileDialog.getOpenFileName(self, "Open an OME-Zarr zip", "",
                                                  "OME-Zarr zip (*.zip *.ozx)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Open an OME-Zarr image")
        if path:
            self.open_paths([path], add=add)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            try:
                self.open_paths(paths, add=bool(self.view.viewer.layers))
            except ERRORS as exc:  # noqa: BLE001 (see _guarded)
                self.note(str(exc))

    def showEvent(self, event):
        super().showEvent(event)
        if self._paths:
            paths, self._paths = self._paths, []
            QTimer.singleShot(0, functools.partial(self._open_later, paths))

    @_guarded
    def _open_later(self, paths):
        self.open_paths(paths)
        self.view.setFocus()

    # ------------------------------------------------------------------ toolbar slots
    def _apply(self, step):
        try:
            step(self.view.viewer)
        except ERRORS as exc:  # noqa: BLE001 (see _guarded)
            self.note(str(exc))
            return
        self.view._changed()

    def _fit(self):
        self._apply(lambda v: v.fit())

    def _orient(self, name):
        self._apply(lambda v: v.set_orientation(name))

    def _linked(self, on):
        self._apply(lambda v: setattr(v, "linked", on))

    def _projection(self, on):
        self._apply(lambda v: setattr(v, "mip", on))

    def _mode(self, name):
        def step(v):
            v.projection = name
            v.mip = True
        self._apply(step)

    def _density(self, value):
        self._apply(lambda v: setattr(v, "density", 10 ** (value / 100 * 3 - 3)))

    def _shading(self, on):
        self._apply(lambda v: setattr(v, "shading", on))

    def _smooth(self, on):
        def step(v):
            if v.layers:
                v.set_interpolation("linear" if on else "nearest")
        self._apply(step)

    # ------------------------------------------------------------------ following the view
    @_guarded
    def _view_changed(self):
        v = self.view.viewer
        self.layer_panel.refresh()
        self.plane_panel.refresh()
        # the toolbar shows what the keys changed
        for widget, value in ((self.linked, v.linked), (self.projected, v.mip)):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        for w, setter in ((self.mode, lambda: self.mode.setCurrentText(v.projection)),
                          (self.density, lambda: self.density.setValue(
                              round((math.log10(v.density) + 3) / 3 * 100))),
                          (self.shading, lambda: self.shading.setChecked(v.shading)),
                          (self.smooth, lambda: self.smooth.setChecked(
                              bool(v.layers) and v.layers[v.current]["interpolation"]
                              == "linear"))):
            w.blockSignals(True)
            setter()
            w.blockSignals(False)
        o = [[round(c, 6) for c in row] for row in v.orientation]
        named = {"xy": ([1, 0, 0], [0, 1, 0]), "xz": ([1, 0, 0], [0, 0, 1]),
                 "yz": ([0, 0, 1], [0, 1, 0])}
        for name, act in self.orient.items():
            act.blockSignals(True)
            act.setChecked(tuple(o[:2]) == tuple(named[name]))
            act.blockSignals(False)
        self._show()

    def note(self, text):
        self._note = text
        self._show()

    @_guarded
    def _picked(self, p):
        try:
            if p["pixel"] is None:
                self._pick = ""
            else:
                names = [c["label"] for c in self.view.viewer.channels]
                vals = "  ".join(
                    f"label id {e['id']}" if e.get("id") is not None
                    else f"{names[e['channel']]}: {e['value']:g}" if e["value"] is not None
                    else f"{names[e['channel']]}: -" for e in p["values"])
                where = (f"x={p['voxel'][0]} y={p['voxel'][1]} z={p['voxel'][2]}"
                         if p.get("voxel") else f"x={p['pixel'][0]} y={p['pixel'][1]}")
                if p.get("well"):
                    where = f"well {p['well'][0]} ({p['well'][1]})  " + where
                elif p.get("member"):
                    where = f"{p['member'][0]}  " + where
                self._pick = f"{where}  {vals}"
        except ERRORS as exc:  # noqa: BLE001 (see _guarded)
            self._pick = str(exc)
        self._show()

    def _show(self):
        try:
            v = self.view.viewer
        except ERRORS as exc:  # noqa: BLE001 (see _guarded)
            self.status.setText(str(exc))
            return
        layers = v.layers
        if not layers:
            self.status.setText(self._note or "File > Open (or drop an image here)")
            return
        cur = layers[v.current]
        pos = "  ".join(f"{n}={round(i)}/{s}" for n, i, s in cur["position"])
        level = self.view.last_stats[3]
        pending = self.view.last_stats[2]
        err = f"  error: {v.error}" if v.error else ""
        which = f"layer {v.current + 1}/{len(layers)}  " if len(layers) > 1 else ""
        view = f"view {v.rotation:.0f} deg  " if v.rotation else ""
        if v.mip:
            view += f"projection {v.projection}  "
        self.status.setText(f"{which}{pos}  level {level}  pending {pending}  {view}"
                            f"{_turn_and_shift(cur['transform'])}  {self._pick}{err}"
                            f"  {self._note}")


def main(argv=None) -> int:
    """``micro-glancer [PATH ...]``: the window, opening the images as layers."""
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    win = MainWindow(argv[1:])
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
