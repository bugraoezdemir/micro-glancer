"""View controls for micro-glancer's Qt widget (Workplan V14): where the view cuts the
data, at which angle, and how the dataset is oriented.

- ``DepthSlider`` moves the cutting plane along the viewing direction (the arrow keys' z),
  not along the image's z: "Depth" runs from where the plane first touches the visible
  layers to where it last does, one step a voxel of the current layer.
- ``AngleControls`` turns the view about its own axes by exact amounts: Tilt about the
  screen's horizontal axis, Turn about its vertical axis, Roll about the viewing
  direction. Each turn starts from the current view (no absolute Euler angles: their
  order and gimbal lock make them unpredictable). A readout says where the view stands
  against the dataset's axes.
- ``CompassBall`` shows the dataset's own axes as seen from the view (x red, y green, z
  blue; an arrow pointing away is dashed and faint), and turns the view by dragging that
  never mixes axes: on the outer ring it rolls, inside it tilts or turns, whichever way
  the drag starts. Shift snaps to 15 degrees. A double-click on an arrow looks along it.
- ``ViewBar`` puts the three side by side, for the strip below the canvas.

They follow the ``ViewerWidget`` they are given (its ``viewChanged`` signal) and report
errors on its ``failed`` signal instead of raising (PyQt6 ends the process when a slot
raises).
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
                             QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget)

from . import ERRORS

#: the dataset's axes on the compass
AXIS_COLOURS = {"x": QColor(235, 80, 80), "y": QColor(80, 205, 100), "z": QColor(90, 150, 255)}
#: steps offered for the angle buttons, in degrees
ANGLE_STEPS = (1.0, 5.0, 15.0, 45.0, 90.0)
#: degrees per screen pixel of a drag on the compass
DRAG_DEGREES = 0.5
#: Shift while dragging on the compass snaps to multiples of this
SNAP = 15.0


def _dot(a, b) -> float:
    return sum(p * q for p, q in zip(a, b))


def _has_volume(v) -> bool:
    return any(lay["volume"] and lay["visible"] for lay in v.layers)


class DepthSlider(QWidget):
    """The cutting plane's depth along the viewing direction (see the module's notes)."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self._range = None                      # (first, last, step) of the last refresh
        self._filling = False
        self.label = QLabel("Depth")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip("Moves the cutting plane along the viewing direction "
                               "(as the Up / Down keys); the z slider moves along the "
                               "image's own z")
        self.readout = QLabel("")
        self.readout.setMinimumWidth(110)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.readout)
        self.slider.valueChanged.connect(self._moved)
        view.viewChanged.connect(self.refresh)

    def refresh(self) -> None:
        if self.view._viewer is None:
            return
        try:
            v = self.view.viewer
            r = v.depth_range if v.layers and _has_volume(v) else None
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            return
        if r is None or r[3] <= 0 or r[1] - r[0] < r[3]:
            self._range = None
            self.setEnabled(False)
            self.readout.setText("")
            return
        lo, hi, now, step = r
        # positions at voxel centres: one per plane along the direction
        last = max(1, round((hi - lo) / step) - 1)
        k = min(max(round((now - lo) / step - 0.5), 0), last)
        self._range = (lo, hi, step)
        self.setEnabled(True)
        self._filling = True
        self.slider.setRange(0, last)
        self.slider.setValue(k)
        self._filling = False
        self.readout.setText(f"depth {k} of {last}")

    def _moved(self, k: int) -> None:
        if self._filling or self._range is None:
            return
        lo, _, step = self._range
        try:
            self.view.viewer.set_depth(lo + (k + 0.5) * step)
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            return
        self.view._changed()


class AngleControls(QWidget):
    """Turning the view about its own axes by exact amounts (see the module's notes)."""

    #: (name, what it turns about, the viewer call)
    ROWS = (
        ("Tilt", "the screen's horizontal axis", lambda v, d: v.turn(0.0, d)),
        ("Turn", "the screen's vertical axis", lambda v, d: v.turn(d, 0.0)),
        ("Roll", "the viewing direction", lambda v, d: v.rotate(d)),
    )

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)
        grid.addWidget(QLabel("Step"), 0, 0)
        self.step = QComboBox()
        for s in ANGLE_STEPS:
            self.step.addItem(f"{s:g}°", s)
        self.step.setCurrentIndex(1)
        self.step.setToolTip("How far one press of - or + turns")
        grid.addWidget(self.step, 0, 1, 1, 3)
        self.fields = {}
        for r, (name, about, call) in enumerate(self.ROWS, start=1):
            grid.addWidget(QLabel(name), r, 0)
            minus, plus = QToolButton(), QToolButton()
            minus.setText("−")
            plus.setText("+")
            for b, sign in ((minus, -1.0), (plus, 1.0)):
                b.setToolTip(f"{name}: turn about {about} by the step")
                b.setAutoRepeat(True)
                b.clicked.connect(lambda _=False, c=call, s=sign: self._turn(c, s * self._step()))
            field = QDoubleSpinBox()
            field.setRange(-360.0, 360.0)
            field.setDecimals(1)
            field.setSuffix("°")
            field.setToolTip(f"{name}: type an angle and press Enter to turn about {about} "
                             "by exactly that much")
            field.setKeyboardTracking(False)
            field.editingFinished.connect(lambda f=field, c=call: self._exact(f, c))
            grid.addWidget(minus, r, 1)
            grid.addWidget(field, r, 2)
            grid.addWidget(plus, r, 3)
            self.fields[name] = field
        self.readout = QLabel("")
        self.readout.setStyleSheet("color: #999;")
        grid.addWidget(self.readout, len(self.ROWS) + 1, 0, 1, 4)
        view.viewChanged.connect(self.refresh)

    def _step(self) -> float:
        return float(self.step.currentData())

    def _turn(self, call, degrees: float) -> None:
        if degrees == 0.0:
            return
        try:
            call(self.view.viewer, degrees)
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            return
        self.view._changed()

    def _exact(self, field, call) -> None:
        degrees = field.value()
        field.blockSignals(True)
        field.setValue(0.0)
        field.blockSignals(False)
        self._turn(call, degrees)

    def refresh(self) -> None:
        """Where the view stands against the dataset: the viewing direction's angle from
        each of the current layer's axes."""
        if self.view._viewer is None:
            return
        try:
            v = self.view.viewer
            if not v.layers:
                self.readout.setText("")
                return
            normal = v.orientation[2]
            axes = v.axes()
        except ERRORS:  # noqa: BLE001 (the view reports it)
            return
        parts = []
        for name, a in zip("xyz", axes):
            c = max(-1.0, min(1.0, abs(_dot(a, normal))))
            parts.append(f"{math.degrees(math.acos(c)):.0f}° from {name}")
        self.readout.setText("Looking " + ", ".join(parts))


class CompassBall(QWidget):
    """The dataset's axes as seen from the view, and turning by constrained drags (see
    the module's notes)."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.setFixedSize(150, 150)
        self.setMouseTracking(True)
        self.setToolTip("The dataset's axes as you see them (x red, y green, z blue; "
                        "dashed: pointing away). Drag on the ring to roll, inside to tilt "
                        "or turn; Shift snaps to 15°. Double-click an arrow to look along it.")
        self._drag = None          # {"mode", "start", "last", "applied"}
        self._tips = {}            # axis name -> screen point of its arrow tip
        view.viewChanged.connect(self.update)

    # ---------------------------------------------------------------- geometry
    def _centre_radius(self):
        r = min(self.width(), self.height()) / 2.0 - 4.0
        return QPointF(self.width() / 2.0, self.height() / 2.0), r

    def _projected(self):
        """[(name, screen x, screen y, depth)] of the current layer's axes, unit length."""
        if self.view._viewer is None:
            return []
        v = self.view.viewer
        if not v.layers:
            return []
        right, down, normal = v.orientation
        out = []
        for name, a in zip("xyz", v.axes()):
            out.append((name, _dot(a, right), _dot(a, down), _dot(a, normal)))
        return out

    # ---------------------------------------------------------------- painting
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, r = self._centre_radius()
        p.setPen(QPen(QColor(90, 90, 90), 1.0))
        p.setBrush(QColor(32, 32, 36))
        p.drawEllipse(c, r, r)
        # the roll ring, and the tilt / turn axes
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(110, 110, 120), 6.0))
        p.drawEllipse(c, r - 5.0, r - 5.0)
        p.setPen(QPen(QColor(70, 70, 78), 1.0, Qt.PenStyle.DotLine))
        p.drawLine(QPointF(c.x() - r * 0.8, c.y()), QPointF(c.x() + r * 0.8, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - r * 0.8), QPointF(c.x(), c.y() + r * 0.8))
        try:
            axes = self._projected()
        except ERRORS:  # noqa: BLE001 (the view reports it)
            axes = []
        self._tips = {}
        length = r * 0.72
        # farther arrows first, so nearer ones lie on top
        for name, sx, sy, sz in sorted(axes, key=lambda t: -t[3]):
            tip = QPointF(c.x() + sx * length, c.y() + sy * length)
            self._tips[name] = tip
            colour = QColor(AXIS_COLOURS[name])
            away = sz > 0.02
            if away:
                colour.setAlpha(120)
            pen = QPen(colour, 2.5, Qt.PenStyle.DashLine if away else Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.drawLine(c, tip)
            # the head, unless the arrow points (nearly) straight at or away from us
            flat = math.hypot(sx, sy)
            if flat > 0.15:
                ux, uy = sx / flat, sy / flat
                h = 8.0
                left = QPointF(tip.x() - ux * h - uy * h * 0.5, tip.y() - uy * h + ux * h * 0.5)
                right_ = QPointF(tip.x() - ux * h + uy * h * 0.5, tip.y() - uy * h - ux * h * 0.5)
                p.setBrush(colour)
                p.setPen(QPen(colour, 1.0))
                p.drawPolygon(QPolygonF([tip, left, right_]))
                p.setBrush(Qt.BrushStyle.NoBrush)
            else:
                # along the viewing direction: a dot (towards) or a ring (away)
                p.setPen(QPen(colour, 2.0))
                p.setBrush(Qt.BrushStyle.NoBrush if away else colour)
                p.drawEllipse(tip, 5.0, 5.0)
                p.setBrush(Qt.BrushStyle.NoBrush)
            font = QFont(p.font())
            font.setBold(True)
            p.setFont(font)
            p.setPen(colour)
            p.drawText(QPointF(tip.x() + 5.0, tip.y() - 5.0), name)
        if self._drag and self._drag.get("mode"):
            p.setPen(QColor(230, 230, 230))
            text = f"{self._drag['mode']} {self._drag['applied']:+.0f}°"
            p.drawText(QRectF(0, self.height() - 20, self.width(), 18),
                       int(Qt.AlignmentFlag.AlignCenter), text)

    # ---------------------------------------------------------------- dragging
    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        c, r = self._centre_radius()
        pos = e.position()
        on_ring = math.hypot(pos.x() - c.x(), pos.y() - c.y()) > r * 0.8
        self._drag = {"mode": "Roll" if on_ring else None, "start": pos, "last": pos,
                      "total": 0.0, "applied": 0.0}

    def mouseMoveEvent(self, e):
        d = self._drag
        if not d:
            return
        pos = e.position()
        c, _ = self._centre_radius()
        if d["mode"] is None:
            dx, dy = pos.x() - d["start"].x(), pos.y() - d["start"].y()
            if math.hypot(dx, dy) < 4.0:
                return
            # the axis is chosen once, by the way the drag starts, and kept
            d["mode"] = "Turn" if abs(dx) >= abs(dy) else "Tilt"
        if d["mode"] == "Roll":
            a0 = math.atan2(d["last"].y() - c.y(), d["last"].x() - c.x())
            a1 = math.atan2(pos.y() - c.y(), pos.x() - c.x())
            step = math.degrees(a1 - a0)
            step = (step + 180.0) % 360.0 - 180.0
        elif d["mode"] == "Turn":
            step = (pos.x() - d["last"].x()) * DRAG_DEGREES
        else:
            step = (pos.y() - d["last"].y()) * DRAG_DEGREES
        d["last"] = pos
        d["total"] += step
        want = d["total"]
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            want = round(want / SNAP) * SNAP
        delta = want - d["applied"]
        if abs(delta) < 1e-9:
            return
        call = dict((name, f) for name, _, f in AngleControls.ROWS)[d["mode"]]
        try:
            call(self.view.viewer, delta)
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            self._drag = None
            return
        d["applied"] = want
        self.view._changed()

    def mouseReleaseEvent(self, _e):
        self._drag = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        """Look along the dataset's axis whose arrow tip is nearest (its other two axes
        then lie across the screen)."""
        pos = e.position()
        best = min(self._tips.items(),
                   key=lambda kv: math.hypot(kv[1].x() - pos.x(), kv[1].y() - pos.y()),
                   default=None)
        if best is None or math.hypot(best[1].x() - pos.x(), best[1].y() - pos.y()) > 18.0:
            return
        try:
            v = self.view.viewer
            ax, ay, az = (tuple(a) for a in v.axes())
            right, down = {"z": (ax, ay), "y": (ax, az), "x": (az, ay)}[best[0]]
            v.set_orientation(right, down)
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            return
        self._drag = None
        self.view._changed()


class ViewBar(QWidget):
    """The compass, the angle controls and the Depth slider in one strip, for below the
    canvas."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.compass = CompassBall(view)
        self.angles = AngleControls(view)
        self.depth = DepthSlider(view)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self.angles)
        right.addStretch(1)
        right.addWidget(self.depth)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.addWidget(self.compass)
        lay.addLayout(right, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        view.viewChanged.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """Shown for a volume, hidden otherwise; never makes the viewer (the native
        window must exist first)."""
        has = False
        if self.view._viewer is not None:
            try:
                v = self.view.viewer
                has = bool(v.layers) and _has_volume(v)
            except ERRORS:  # noqa: BLE001 (the view reports it)
                has = False
        if self.isHidden() == has:              # only when it changes
            self.setVisible(has)
