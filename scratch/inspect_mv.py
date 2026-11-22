"""A scratch copy of eubi's Inspect page on micro-viewer (Workplan WP14, task V7).

``InspectPageMV`` is eubi's ``InspectPage`` with its Viewer tab drawn by micro-viewer:
``MicroZarrViewer`` stands in for eubi's ``ZarrViewer`` with the members the page uses
(signals ``frame_ready``, ``dataset_loaded``, ``status_changed``; ``channel_panel``;
``load_dataset``, ``path``, ``axes``, ``shape``, ``channels``, ``level_idx``,
``level_paths``, ``orientation``, ``z``, ``fov_center``, ``image_widget``). eubi's own ``ChannelPanel`` drives the channels. The metadata tab,
the sidebar and the save buttons are eubi's, unchanged. Nothing in eubi is modified.

Not in this copy: XZ / YZ orientations (V4), the Auto contrast buttons (they report
that they are not wired), the field-of-view combo and the reload button.

usage (eubi's venv; micro-viewer from a wheel on PYTHONPATH):
    python scratch/inspect_mv.py [IMAGE]
"""
from __future__ import annotations

import functools
import os
import sys

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QSizePolicy, QSlider,
                             QSplitter, QVBoxLayout, QWidget)

from eubi_bridge.qt_gui.pages.inspect_page import InspectPage, pyramid_cache
from eubi_bridge.qt_gui.widgets.channel_panel import ChannelPanel
from micro_viewer.qt import ViewerWidget


def _safe(slot):
    """A slot that reports its error on the status line: PyQt6 ends the whole process
    when a slot raises, and eubi must not go down with a viewer error."""
    @functools.wraps(slot)
    def run(self, *args):
        try:
            return slot(self, *args)
        except Exception as exc:  # noqa: BLE001
            self.status_changed.emit(f"micro-viewer: {exc}")
    return run


def _hex(rgb) -> str:
    return "#" + "".join(f"{round(max(0.0, min(1.0, c)) * 255):02x}" for c in rgb)


def _rgb(text: str):
    t = text.lstrip("#")
    return tuple(int(t[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


class MicroZarrViewer(QWidget):
    """eubi's ZarrViewer, as far as the Inspect page uses it, on micro-viewer."""

    frame_ready = pyqtSignal(object, float, int)
    dataset_loaded = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None, **_ignored):
        super().__init__(parent)
        # as eubi's ZarrViewer: the viewer takes the tab's height, the save row below not
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._path: str | None = None
        self._pyr = None
        self._level_paths: list[str] = []
        self._sliders: list[tuple[str, QSlider, QLabel]] = []
        self.view = ViewerWidget()
        self.channel_panel = ChannelPanel()
        self.channel_panel.channels_changed.connect(self._apply_channels)
        self.channel_panel.auto_requested.connect(
            lambda i: self.status_changed.emit("Auto contrast is not wired to micro-viewer yet"))
        self._planes = QGroupBox("Plane")
        self._planes_layout = QVBoxLayout(self._planes)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self._planes)
        side_layout.addWidget(self.channel_panel, 1)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.view)
        split.addWidget(side)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)
        self.view.viewChanged.connect(self._frame)
        # the core paints without Python; report what the last frame showed now and then
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._frame)
        self._tick.start(250)

    # -- what the Inspect page reads ------------------------------------------------
    @property
    def path(self):
        return self._path

    @property
    def axes(self) -> str:
        """The axes as one lowercase string ("czyx"), as eubi's ZarrViewer gives them."""
        return "".join(self.view.viewer.image["axes"]).lower() if self._path else ""

    @property
    def shape(self) -> list:
        return list(self.view.viewer.image["levels"][0]["shape"]) if self._path else []

    @property
    def channels(self) -> list:
        """The channel dicts as eubi's panel holds them (what "Save Channels" writes)."""
        return self.channel_panel.channel_data()

    @property
    def level_idx(self) -> int:
        return self.view.viewer.level if self._path else 0

    @property
    def level_paths(self) -> list[str]:
        return self._level_paths

    @property
    def orientation(self) -> str:
        return "XY"

    @property
    def z(self) -> int:
        for name, i, _ in (self.view.viewer.position if self._path else []):
            if name == "z":
                return round(i)
        return 0

    @property
    def fov_center(self):
        """(y, x) in level-0 pixels at the view's centre."""
        if not self._path:
            return (0.0, 0.0)
        v = self.view.viewer
        w, h = v.size
        p = v.pick(w / 2, h / 2)
        if p["pixel"] is not None:
            return (float(p["pixel"][1]), float(p["pixel"][0]))
        return (p["world"][1], p["world"][0])

    @property
    def image_widget(self):
        return self

    def set_frame(self, *_args):
        """eubi's image widget takes frames from Python; micro-viewer draws its own."""

    # -- loading ---------------------------------------------------------------------
    def load_dataset(self, zarr_path: str):
        try:
            pyr = pyramid_cache.get(zarr_path)       # eubi's metadata tab reads this
            self.view.open(zarr_path)
            v = self.view.viewer
        except Exception as exc:                     # noqa: BLE001 (the page shows it)
            self.status_changed.emit(f"Error loading: {exc}")
            return
        self._path, self._pyr = zarr_path, pyr
        self._level_paths = [lv["path"] for lv in v.image["levels"]]
        self.channel_panel.set_channels([
            {"index": i, "label": c["label"], "color": _hex(c["color"]),
             "visible": c["visible"], "intensityMin": c["lo"], "intensityMax": c["hi"]}
            for i, c in enumerate(v.channels)])
        self._build_sliders()
        self.status_changed.emit(zarr_path)
        self.dataset_loaded.emit(pyr)
        self._frame()

    def _build_sliders(self):
        while self._planes_layout.count():
            item = self._planes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._sliders = []
        for name, index, size in self.view.viewer.position:
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            label = QLabel()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, max(0, size - 1))
            slider.setValue(round(index))
            slider.valueChanged.connect(lambda val, n=name: self._move(n, val))
            lay.addWidget(QLabel(f"{name.upper()}:"))
            lay.addWidget(slider, 1)
            lay.addWidget(label)
            self._planes_layout.addWidget(row)
            self._sliders.append((name, slider, label))
        self._planes.setVisible(bool(self._sliders))
        self._label_sliders()

    def _label_sliders(self):
        pos = {n: (i, s) for n, i, s in self.view.viewer.position}
        for name, _, label in self._sliders:
            i, s = pos[name]
            label.setText(f"{round(i)} / {s - 1}")

    @_safe
    def _move(self, name: str, value: int):
        self.view.viewer.set_position(name, float(value))
        self._label_sliders()
        self.view._changed()

    @_safe
    def _apply_channels(self, channels: list):
        v = self.view.viewer
        for ch in channels:
            i = ch["index"]
            cur = v.channels[i]
            lo = ch.get("intensityMin")
            hi = ch.get("intensityMax")
            v.set_channel(i, color=_rgb(ch.get("color", "#ffffff")),
                          lo=cur["lo"] if lo is None else float(lo),
                          hi=cur["hi"] if hi is None else float(hi),
                          visible=bool(ch.get("visible", True)),
                          label=str(ch.get("label", cur["label"])))
        self.view._changed()

    @_safe
    def _frame(self):
        if self._path:
            self.frame_ready.emit(None, 0.0, 0)

    def closeEvent(self, event):
        self._tick.stop()
        super().closeEvent(event)


class InspectPageMV(InspectPage):
    """eubi's Inspect page with micro-viewer in its Viewer tab."""

    def _build_viewer_tab(self):
        import eubi_bridge.qt_gui.pages.inspect_page as page
        real = page.ZarrViewer
        page.ZarrViewer = MicroZarrViewer        # the tab is built around it
        try:
            super()._build_viewer_tab()
        finally:
            page.ZarrViewer = real

    def _on_frame_ready(self, rgb, elapsed_ms: float, generation: int):
        try:
            self._update_status_bar(elapsed_ms)
        except Exception as exc:  # noqa: BLE001 (a slot: see _safe)
            self._status_bar.setText(f"micro-viewer: {exc}")


def main(argv=None) -> int:
    from PyQt6.QtWidgets import QApplication
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    page = InspectPageMV()
    page.resize(1400, 900)
    page.setWindowTitle("eubi Inspect on micro-viewer (scratch)")
    page.show()
    if len(argv) > 1:
        page._load_zarr(os.path.abspath(argv[1]))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
