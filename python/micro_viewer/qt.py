"""The Qt widget hosting micro-viewer's core.

Two hosting modes, chosen automatically (``MICRO_VIEWER_MODE`` overrides):

- ``native``: the core draws straight into the widget's own native window (Windows,
  macOS). Qt does not paint the widget; frames never pass through Python.
- ``offscreen``: the core draws into a texture, the frame comes back as RGBA8 into a
  buffer the widget keeps, and Qt paints it (Linux / Wayland, remote desktops, and
  wherever native fails).

Tiles are read in the background; while some are on their way the widget repaints at
about 60 Hz, and not at all when nothing changes.

Input: drag to pan, wheel to zoom at the pointer, F to fit, Page Up / Page Down to move
through the first plane axis (z), with Shift the second (t). HiDPI scaling is passed to
the core in physical pixels.

Signals: ``viewChanged()`` after the camera, plane or channels change;
``picked(dict)`` with what lies under the pointer (``Viewer.pick``) as it moves;
``failed(str)`` when the core cannot draw (the message; also in ``error``). PyQt6 ends
the whole process when a Qt override raises, so the widget's overrides never do: the
host shows ``failed`` instead.
"""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QWidget

from ._core import Viewer

#: zoom per wheel notch
ZOOM_STEP = 1.25
#: repaint interval while tiles are on their way (ms)
POLL_MS = 15


def default_mode() -> str:
    mode = os.environ.get("MICRO_VIEWER_MODE")
    if mode in ("native", "offscreen"):
        return mode
    # Linux needs the X11 display for a native surface, which the widget does not obtain
    # from Qt yet; Wayland always needs the off-screen mode
    return "native" if sys.platform in ("win32", "darwin") else "offscreen"


class ViewerWidget(QWidget):
    """A widget showing an OME-Zarr image, drawn by micro-viewer's core."""

    viewChanged = pyqtSignal()
    picked = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, parent=None, *, mode: str | None = None, software: bool = False,
                 no_vsync: bool = False, atlas_mib: int = 0):
        super().__init__(parent)
        self.mode = mode or default_mode()
        self._software = software
        self._no_vsync = no_vsync
        self._atlas_mib = atlas_mib
        self._viewer: Viewer | None = None
        self._pending: str | None = None
        self._drag = None
        self._frame: bytearray | None = None
        self.last_stats = (0, 0, 0, 0, 0)
        #: the last error the core reported while drawing, or None
        self.error: str | None = None
        self._poll = QTimer(self)
        self._poll.setSingleShot(True)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self.update)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(64, 64)
        if self.mode == "native":
            # the core owns this widget's native window and draws into it
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
            self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    # ---------------------------------------------------------------- the core
    def _pixels(self):
        r = self.devicePixelRatioF()
        return max(1, round(self.width() * r)), max(1, round(self.height() * r)), r

    @property
    def viewer(self) -> Viewer:
        """The core, made when first needed (the native window must exist first)."""
        if self._viewer is None:
            w, h, _ = self._pixels()
            if self.mode == "native":
                self._viewer = Viewer.for_window(int(self.winId()), w, h,
                                                 software=self._software,
                                                 no_vsync=self._no_vsync,
                                                 atlas_mib=self._atlas_mib)
            else:
                self._viewer = Viewer.offscreen(w, h, software=self._software,
                                                atlas_mib=self._atlas_mib)
            if self._pending is not None:
                path, self._pending = self._pending, None   # tried once, even if it fails
                self._viewer.open(path)
        return self._viewer

    def open(self, path) -> None:
        """Open the OME-Zarr image at ``path`` and fit it."""
        if self._viewer is None:
            self._pending = str(path)
        else:
            self._viewer.open(str(path))
        self._changed()

    def _changed(self):
        self.update()
        self.viewChanged.emit()

    # ---------------------------------------------------------------- painting
    def paintEngine(self):
        # native: Qt must not paint over the core's frames
        return None if self.mode == "native" else super().paintEngine()

    def _fail(self, exc: Exception):
        msg = str(exc)
        if msg != self.error:
            self.error = msg
            self.failed.emit(msg)

    def paintEvent(self, _event):
        try:
            self._paint()
        except Exception as exc:  # noqa: BLE001 (see the module's notes)
            self._fail(exc)
            if self.mode != "native":
                painter = QPainter(self)
                painter.fillRect(self.rect(), QColor(20, 20, 20))
                painter.end()

    def _paint(self):
        v = self.viewer
        if self.mode == "native":
            self.last_stats = v.render()
        else:
            w, h = v.size
            if self._frame is None or len(self._frame) != 4 * w * h:
                self._frame = bytearray(4 * w * h)  # kept: the core writes each frame here
            self.last_stats = v.render_into(self._frame)
            image = QImage(self._frame, w, h, 4 * w, QImage.Format.Format_RGBA8888)
            image.setDevicePixelRatio(self.devicePixelRatioF())
            painter = QPainter(self)
            painter.drawImage(0, 0, image)
            painter.end()
        # tiles on their way: paint again soon (the timer is single-shot)
        if (self.last_stats[2] > 0 or v.has_results()) and not self._poll.isActive():
            self._poll.start()

    def resizeEvent(self, event):
        if self._viewer is not None:
            w, h, _ = self._pixels()
            try:
                self._viewer.resize(w, h)
            except Exception as exc:  # noqa: BLE001 (see the module's notes)
                self._fail(exc)
        super().resizeEvent(event)

    # ---------------------------------------------------------------- input
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event):
        p = event.position()
        r = self.devicePixelRatioF()
        if self._drag is not None:
            self.viewer.pan((p.x() - self._drag.x()) * r, (p.y() - self._drag.y()) * r)
            self._drag = p
            self._changed()
        elif self._viewer is not None and self.receivers(self.picked) > 0:
            try:
                self.picked.emit(self._viewer.pick(p.x() * r, p.y() * r))
            except Exception as exc:  # noqa: BLE001 (see the module's notes)
                self._fail(exc)

    def mouseReleaseEvent(self, _event):
        self._drag = None

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        r = self.devicePixelRatioF()
        p = event.position()
        self.viewer.zoom(ZOOM_STEP ** steps, p.x() * r, p.y() * r)
        self._changed()

    def step(self, axis: int, delta: int) -> bool:
        """Move the plane by ``delta`` on plane axis number ``axis`` (0: z, 1: t, ...)."""
        pos = self.viewer.position
        if axis >= len(pos):
            return False
        name, index, size = pos[axis]
        new = min(max(round(index) + delta, 0), size - 1)
        if new == index:
            return False
        self.viewer.set_position(name, float(new))
        self._changed()
        return True

    def keyPressEvent(self, event):
        key = event.key()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_F:
            self.viewer.fit()
            self._changed()
        elif key in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            self.step(1 if shift else 0, 1 if key == Qt.Key.Key_PageUp else -1)
        else:
            super().keyPressEvent(event)


def main(argv=None) -> int:
    """``python -m micro_viewer.qt PATH``: a window showing one image, with the plane
    position, the level and the values under the pointer in the status bar."""
    from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow

    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    if len(argv) < 2:
        print("usage: python -m micro_viewer.qt PATH")
        return 2
    win = QMainWindow()
    widget = ViewerWidget()
    win.setCentralWidget(widget)
    win.resize(1280, 800)
    win.setWindowTitle(f"micro-viewer: {argv[1]}")
    status = QLabel()
    win.statusBar().addWidget(status, 1)
    state = {"pick": ""}

    def show():
        v = widget.viewer
        pos = "  ".join(f"{n}={round(i)}/{s}" for n, i, s in v.position)
        visible, loaded, pending, level, missing = widget.last_stats
        err = f"  read error: {v.error}" if v.error else ""
        status.setText(f"{pos}  level {level}  pending {pending}  {state['pick']}{err}")

    def picked(p):
        if p["pixel"] is None:
            state["pick"] = ""
        else:
            names = [c["label"] for c in widget.viewer.channels]
            vals = "  ".join(
                f"{names[e['channel']]}: {e['value']:g}" if e["value"] is not None
                else f"{names[e['channel']]}: -" for e in p["values"])
            state["pick"] = f"x={p['pixel'][0]} y={p['pixel'][1]}  {vals}"
        show()

    widget.viewChanged.connect(show)
    widget.picked.connect(picked)
    widget.open(argv[1])
    win.show()
    timer = QTimer(win)
    timer.timeout.connect(show)
    timer.start(250)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
