"""The Qt widget hosting micro-glancer's core.

Two hosting modes, chosen automatically (``MICRO_GLANCER_MODE`` overrides):

- ``native``: the core draws straight into the widget's own native window (Windows,
  macOS). Qt does not paint the widget; frames never pass through Python.
- ``offscreen``: the core draws into a texture, the frame comes back as RGBA8 into a
  buffer the widget keeps, and Qt paints it (Linux / Wayland, remote desktops, and
  wherever native fails).

Tiles are read in the background; while some are on their way the widget repaints at
about 60 Hz, and not at all when nothing changes.

Input (the view, and the current layer of several):

- drag: pan; wheel: zoom at the pointer; F: fit;
- Shift + wheel: turn the view about the pointer (5 degrees a notch); [ and ]: turn the
  view by 90 degrees; 0: turn it back;
- volumes: Alt + drag turns the view's plane through the volume (left-right about the
  screen's vertical, up-down about its horizontal); 1, 2, 3: look along z (XY), y (XZ)
  or x (YZ); Page Up / Page Down move the plane by one voxel along the view; I switches
  trilinear sampling on and off; Ctrl + Alt + drag tilts the current layer; M shows
  the projection instead of the plane; P shows it and cycles what it shows (maximum
  intensity, volume, surfaces); Shift + P switches shading; - and + make the volume
  thinner or denser;
- L: linked views (the main view, its YZ and XZ companions and the projection in
  quarters, a crosshair where the planes cross); dragging, the wheel and Page Up /
  Page Down act in the view under the pointer;
- Ctrl + drag: move the current layer; Ctrl + wheel: turn it about its centre (1 degree
  a notch, with Shift 0.1); Ctrl + arrow keys: move it by one screen pixel; Ctrl + 0:
  put it back where its image places it; Ctrl + S: save its transform into its image
  (``micro_glancer.save``);
- Tab: the next layer becomes the current one;
- label images: double-click selects the id under the pointer (highlighted; again on
  the background clears it), Esc clears it, O switches outlines;
- Page Up / Page Down: the current layer's first plane axis (z; for a volume the
  plane moves along the view), with Shift the next one (t).

HiDPI scaling is passed to the core in physical pixels.

Signals: ``viewChanged()`` after the camera, a plane, a layer or the channels change;
``picked(dict)`` with what lies under the pointer (``Viewer.pick``) as it moves;
``saved(str)`` after a transform was saved (what was written); ``failed(str)`` when the
core cannot draw or a step fails (the message; also in ``error``). PyQt6 ends the whole
process when a Qt override raises, so the widget's overrides never do: the host shows
``failed`` instead.
"""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QWidget

from . import ERRORS
from ._core import OpenJob, Viewer

_warming = False


def warm_up_in_background(software: bool = False) -> None:
    """Load the GPU driver on a thread of its own, once per process: the first viewer
    otherwise takes about a second longer (the driver's loading). Every
    :class:`ViewerWidget` starts it when it is made; a host may call it earlier.

    Background work here runs on the core's own threads, never Python's: Python's cycle
    collector then runs on the window's thread only (on another thread it could drop a
    widget and its viewer, which belongs to the window's thread)."""
    global _warming
    if _warming:
        return
    _warming = True
    from . import warm_up
    try:
        warm_up(software, background=True)
    except ERRORS:  # noqa: BLE001 (the viewer reports it when it is made)
        pass

#: zoom per wheel notch
ZOOM_STEP = 1.25
#: degrees per wheel notch: the view, a layer, a layer finely
TURN_VIEW, TURN_LAYER, TURN_FINE = 5.0, 1.0, 0.1
#: degrees per screen pixel of an Alt + drag
TURN_DRAG = 0.3
#: repaint interval while tiles are on their way (ms)
POLL_MS = 15


def native_possible() -> bool:
    """Whether the core can draw straight into a widget's native window here.

    A native surface needs a real window: Qt's "windows" and "cocoa" platforms give one
    (not "offscreen" or "minimal", as tests and CI run Qt; there the window id is no
    window, which Windows refuses and macOS crashes on). Linux needs the X11 display
    for a native surface, which the widget does not obtain from Qt yet; Wayland always
    needs the off-screen mode.
    """
    if sys.platform not in ("win32", "darwin"):
        return False
    from PyQt6.QtGui import QGuiApplication
    platform = QGuiApplication.platformName() if QGuiApplication.instance() else ""
    return not platform or platform in ("windows", "cocoa")


def default_mode() -> str:
    mode = os.environ.get("MICRO_GLANCER_MODE")
    if mode in ("native", "offscreen"):
        return mode
    return "native" if native_possible() else "offscreen"


class ViewerWidget(QWidget):
    """A widget showing an OME-Zarr image, drawn by micro-glancer's core."""

    viewChanged = pyqtSignal()
    picked = pyqtSignal(dict)
    saved = pyqtSignal(str)
    failed = pyqtSignal(str)
    #: (images opened, images to open) while :meth:`open_in_background` reads
    opening = pyqtSignal(int, int)

    def __init__(self, parent=None, *, mode: str | None = None, software: bool = False,
                 no_vsync: bool = False, atlas_mib: int = 0):
        super().__init__(parent)
        self.mode = mode or default_mode()
        if self.mode == "native" and not native_possible():
            from PyQt6.QtGui import QGuiApplication
            raise ValueError(
                f"micro-glancer cannot draw into a native window on Qt's "
                f"{QGuiApplication.platformName()!r} platform ({sys.platform}); "
                f"use mode='offscreen' (or unset MICRO_GLANCER_MODE)")
        self._software = software
        self._no_vsync = no_vsync
        self._atlas_mib = atlas_mib
        self._viewer: Viewer | None = None
        self._pending: list[str] = []
        self._drag_layer = False
        self._drag_turn = False
        self._mouse = (0.0, 0.0)
        self._drag = None
        self._frame: bytearray | None = None
        self.last_stats = (0, 0, 0, 0, 0)
        #: the last error the core reported while drawing, or None
        self.error: str | None = None
        self._poll = QTimer(self)
        self._poll.setSingleShot(True)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self.update)
        self._open_job = None                   # the newest open_in_background's job
        self._open_done = None
        self._open_timer = QTimer(self)          # its progress and end, every 30 ms
        self._open_timer.setInterval(30)
        self._open_timer.timeout.connect(self._open_progress)
        warm_up_in_background(software)
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
            if self._pending:
                paths, self._pending = self._pending, []   # tried once, even if it fails
                self._viewer.open(paths[0])
                for p in paths[1:]:
                    self._viewer.add(p)
        return self._viewer

    def open(self, path) -> None:
        """Open the OME-Zarr image at ``path`` as the only layer and fit it."""
        if self._viewer is None:
            self._pending = [str(path)]
        else:
            self._viewer.open(str(path))
        self._changed()

    def open_in_background(self, path, done=None, field=None) -> None:
        """Open the OME-Zarr image at ``path`` as the only layer without making the
        window wait: its metadata (a plate: every field's) is read on a background thread,
        ``opening`` reports progress meanwhile, then it is shown and fitted, and ``done``
        is called with None, or with the error, on the window's thread. A newer call
        replaces an older one still reading. ``field``: one field of every well of a plate
        (None: every field at its position)."""
        job = OpenJob(str(path), field)
        self._open_job, self._open_done = job, done
        job.start()                             # the core's own thread; this one polls
        self._open_timer.start()

    @property
    def opening_now(self) -> bool:
        """Whether an :meth:`open_in_background` is still reading."""
        return self._open_job is not None

    def _open_progress(self):
        """Every 30 ms while an open reads: its progress, and once read, show it."""
        job = self._open_job
        if job is None:
            self._open_timer.stop()
            return
        if not job.done:
            self.opening.emit(*job.progress)
            return
        self._open_job = None
        self._open_timer.stop()
        done, self._open_done = self._open_done, None
        error = None
        try:
            self.viewer.open_job(job)               # raises the reading's error, if any
        except ERRORS as exc:  # noqa: BLE001 (handed to `done`)
            # without its traceback: that holds this frame, whose `error` holds it back,
            # a cycle only Python's collector frees, on whichever thread it then runs
            # (where it would drop this widget and its viewer)
            error = exc.with_traceback(None)
        self._changed()
        if done is not None:
            done(error)

    def add(self, path) -> None:
        """Add the OME-Zarr image at ``path`` as a new layer on top."""
        if self._viewer is None:
            self._pending.append(str(path))
        else:
            self._viewer.add(str(path))
        self._changed()

    def save_transform(self) -> str | None:
        """Save the current layer's transform into its image (``saved`` or ``failed``)."""
        from .save import save_transform
        try:
            done = save_transform(self.viewer)
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self._fail(exc, force=True)
            return None
        self.saved.emit(done)
        self._changed()
        return done

    def _changed(self):
        self.update()
        self.viewChanged.emit()

    # ---------------------------------------------------------------- painting
    def paintEngine(self):
        # native: Qt must not paint over the core's frames
        return None if self.mode == "native" else super().paintEngine()

    def _fail(self, exc: Exception, force: bool = False):
        msg = str(exc)
        if force or msg != self.error:
            self.error = msg
            self.failed.emit(msg)

    def paintEvent(self, _event):
        try:
            self._paint()
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
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
        # tiles on their way, or reading deferred while the view moves: paint again soon
        # (the timer is single-shot)
        if v.wants_frame() and not self._poll.isActive():
            self._poll.start()

    def resizeEvent(self, event):
        if self._viewer is not None:
            w, h, _ = self._pixels()
            try:
                self._viewer.resize(w, h)
            except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
                self._fail(exc)
        super().resizeEvent(event)

    # ---------------------------------------------------------------- input
    def _guard(self, step) -> None:
        """Run an input step; report its error instead of raising (see the notes)."""
        try:
            step()
        except ERRORS as exc:  # noqa: BLE001
            self._fail(exc, force=True)
            return
        self._changed()

    def _safe(handler):  # noqa: N805 (a decorator in the class body)
        """An event handler that reports its error on `failed` instead of raising."""
        def run(self, *args):
            try:
                return handler(self, *args)
            except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
                self._fail(exc, force=True)
        run.__name__ = handler.__name__
        run.__doc__ = handler.__doc__
        return run

    @_safe
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()
            mods = event.modifiers()
            ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
            alt = bool(mods & Qt.KeyboardModifier.AltModifier)
            has = bool(self._viewer is not None and self._viewer.layers)
            self._drag_layer = ctrl and has
            self._drag_turn = alt and has

    @_safe
    def mouseMoveEvent(self, event):
        p = event.position()
        r = self.devicePixelRatioF()
        self._mouse = (p.x() * r, p.y() * r)
        if self._drag is not None:
            dx, dy = (p.x() - self._drag.x()) * r, (p.y() - self._drag.y()) * r
            self._drag = p
            px, py = p.x() * r, p.y() * r
            if self._drag_turn and self._drag_layer:
                self._guard(lambda: self.viewer.turn_layer(dx * TURN_DRAG, -dy * TURN_DRAG))
            elif self._drag_turn:
                self._guard(lambda: self.viewer.turn(dx * TURN_DRAG, -dy * TURN_DRAG))
            elif self._drag_layer:
                self._guard(lambda: self.viewer.move_layer(dx, dy, x=px, y=py))
            else:
                self.viewer.pan(dx, dy, px, py)
                self._changed()
        elif self._viewer is not None and self.receivers(self.picked) > 0:
            try:
                self.picked.emit(self._viewer.pick(p.x() * r, p.y() * r))
            except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
                self._fail(exc)

    def mouseReleaseEvent(self, _event):
        self._drag = None

    @_safe
    def mouseDoubleClickEvent(self, event):
        """Select the label id under the pointer (the topmost shown label layer)."""
        v = self.viewer
        r = self.devicePixelRatioF()
        p = v.pick(event.position().x() * r, event.position().y() * r)
        layers = v.layers
        for entry in reversed(p["layers"]):
            k = entry["layer"]
            if not (layers[k]["label"] and layers[k]["visible"]):
                continue
            ident = entry["values"][0]["id"] if entry["values"] else None
            v.set_labels(layer=k, selected=ident if ident else -1)
            self._changed()
            return

    @_safe
    def wheelEvent(self, event):
        d = event.angleDelta()
        steps = (d.y() or d.x()) / 120.0                 # Shift may turn the wheel sideways
        if not steps:
            return
        r = self.devicePixelRatioF()
        p = event.position()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if ctrl:
            turn = TURN_FINE if shift else TURN_LAYER
            self._guard(lambda: self.viewer.rotate_layer(turn * steps))
        elif shift:
            self.viewer.rotate(TURN_VIEW * steps, p.x() * r, p.y() * r)
            self._changed()
        else:
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

    def focusNextPrevChild(self, _next):
        return False                                     # Tab picks the layer here

    @_safe
    def keyPressEvent(self, event):
        key, K = event.key(), Qt.Key
        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        v = self.viewer
        nudge = {K.Key_Left: (-1, 0), K.Key_Right: (1, 0), K.Key_Up: (0, -1),
                 K.Key_Down: (0, 1)}
        if key == K.Key_F:
            v.fit()
            self._changed()
        elif key in (K.Key_PageUp, K.Key_PageDown):
            delta = 1 if key == K.Key_PageUp else -1
            volume = bool(v.layers) and v.layers[v.current]["volume"]
            if volume and not shift:
                mx, my = self._mouse
                self._guard(lambda: v.step_depth(delta, mx, my))
            elif volume:
                # the first plane axis other than z (t)
                others = [k for k, (n, _, _) in enumerate(v.position) if n != "z"]
                if others:
                    self.step(others[0], delta)
            else:
                self.step(1 if shift else 0, delta)
        elif key in (K.Key_1, K.Key_2, K.Key_3) and not ctrl:
            name = {K.Key_1: "xy", K.Key_2: "xz", K.Key_3: "yz"}[key]
            self._guard(lambda: v.set_orientation(name))
        elif key == K.Key_Escape and v.layers:
            for k, lay in enumerate(v.layers):
                if lay["label"] and lay["selected"] is not None:
                    v.set_labels(layer=k, selected=-1)
            self._changed()
        elif key == K.Key_O and v.layers and v.layers[v.current]["label"]:
            v.set_labels(outline=not v.layers[v.current]["outline"])
            self._changed()
        elif key == K.Key_L:
            v.linked = not v.linked
            self._changed()
        elif key == K.Key_M:
            v.mip = not v.mip
            self._changed()
        elif key == K.Key_P and shift:
            v.shading = not v.shading
            self._changed()
        elif key == K.Key_P:
            if v.mip:
                order = ["mip", "volume", "iso"]
                v.projection = order[(order.index(v.projection) + 1) % len(order)]
            v.mip = True
            self._changed()
        elif key in (K.Key_Minus, K.Key_Plus, K.Key_Equal):
            factor = 1 / 1.5 if key == K.Key_Minus else 1.5
            self._guard(lambda: setattr(v, "density", min(1.0, max(1e-4, v.density * factor))))
        elif key == K.Key_I and v.layers:
            cur = v.layers[v.current]["interpolation"]
            self._guard(lambda: v.set_interpolation("nearest" if cur == "linear" else "linear"))
        elif key == K.Key_Tab and v.layers:
            v.current = (v.current + 1) % len(v.layers)
            self._changed()
        elif ctrl and key == K.Key_S:
            self.save_transform()
        elif ctrl and key == K.Key_0:
            self._guard(lambda: v.reset_transform())
        elif ctrl and key in nudge:
            self._guard(lambda: v.move_layer(*nudge[key]))
        elif key == K.Key_0:
            v.set_rotation(0.0)
            self._changed()
        elif key in (K.Key_BracketLeft, K.Key_BracketRight):
            v.rotate(-90.0 if key == K.Key_BracketLeft else 90.0)
            self._changed()
        else:
            super().keyPressEvent(event)


def main(argv=None) -> int:
    """``python -m micro_glancer.qt PATH [PATH ...]``: micro-glancer's app window
    (micro_glancer.app)."""
    from .app import main as app_main
    return app_main(argv)


if __name__ == "__main__":
    sys.exit(main())
