"""The Qt widget hosting micro-viewer's core.

Two hosting modes, chosen automatically (``MICRO_VIEWER_MODE`` overrides):

- ``native``: the core draws straight into the widget's own native window (Windows,
  macOS). Qt does not paint the widget; frames never pass through Python.
- ``offscreen``: the core draws into a texture, the frame comes back as RGBA8 and Qt
  paints it (Linux / Wayland, remote desktops, and wherever native fails).

Input (drag to pan, wheel to zoom at the pointer, F to fit) and HiDPI scaling are passed
to the core in physical pixels.
"""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QWidget

from ._core import Viewer

#: zoom per wheel notch
ZOOM_STEP = 1.25


def default_mode() -> str:
    mode = os.environ.get("MICRO_VIEWER_MODE")
    if mode in ("native", "offscreen"):
        return mode
    # Linux needs the X11 display for a native surface, which the spike does not obtain
    # from Qt yet; Wayland always needs the off-screen mode
    return "native" if sys.platform in ("win32", "darwin") else "offscreen"


class ViewerWidget(QWidget):
    """A widget showing one OME-Zarr plane, drawn by micro-viewer's core."""

    def __init__(self, parent=None, *, mode: str | None = None, software: bool = False,
                 no_vsync: bool = False):
        super().__init__(parent)
        self.mode = mode or default_mode()
        self._software = software
        self._no_vsync = no_vsync
        self._viewer: Viewer | None = None
        self._pending: tuple[str, int] | None = None
        self._drag = None
        self._frame: bytearray | None = None
        self.last_stats = (0, 0)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
                                                 no_vsync=self._no_vsync)
            else:
                self._viewer = Viewer.offscreen(w, h, software=self._software)
            if self._pending is not None:
                self._viewer.open(*self._pending)
                self._pending = None
        return self._viewer

    def open(self, path, level: int = 0) -> None:
        """Open level ``level`` of the OME-Zarr image at ``path`` and fit it."""
        if self._viewer is None:
            self._pending = (str(path), level)
        else:
            self._viewer.open(str(path), level)
        self.update()

    # ---------------------------------------------------------------- painting
    def paintEngine(self):
        # native: Qt must not paint over the core's frames
        return None if self.mode == "native" else super().paintEngine()

    def paintEvent(self, _event):
        v = self.viewer
        if self.mode == "native":
            self.last_stats = v.render()
            return
        w, h = v.size
        if self._frame is None or len(self._frame) != 4 * w * h:
            self._frame = bytearray(4 * w * h)  # kept: the core writes each frame here
        self.last_stats = v.render_into(self._frame)
        image = QImage(self._frame, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        image.setDevicePixelRatio(self.devicePixelRatioF())
        painter = QPainter(self)
        painter.drawImage(0, 0, image)
        painter.end()

    def resizeEvent(self, event):
        if self._viewer is not None:
            w, h, _ = self._pixels()
            self._viewer.resize(w, h)
        super().resizeEvent(event)

    # ---------------------------------------------------------------- input
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        p = event.position()
        r = self.devicePixelRatioF()
        self.viewer.pan((p.x() - self._drag.x()) * r, (p.y() - self._drag.y()) * r)
        self._drag = p
        self.update()

    def mouseReleaseEvent(self, _event):
        self._drag = None

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        r = self.devicePixelRatioF()
        p = event.position()
        self.viewer.zoom(ZOOM_STEP ** steps, p.x() * r, p.y() * r)
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F:
            self.viewer.fit()
            self.update()
        else:
            super().keyPressEvent(event)


def main(argv=None) -> int:
    """``python -m micro_viewer.qt PATH [LEVEL]``: a window showing one image."""
    from PyQt6.QtWidgets import QApplication

    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    if len(argv) < 2:
        print("usage: python -m micro_viewer.qt PATH [LEVEL]")
        return 2
    w = ViewerWidget()
    w.resize(1280, 800)
    w.setWindowTitle(f"micro-viewer: {argv[1]}")
    w.open(argv[1], int(argv[2]) if len(argv) > 2 else 0)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
