"""A channel panel for micro-glancer's Qt widget: the channel's histogram with its contrast
limits, isosurface level and opacity curve, all adjusted by dragging.

``HistogramWidget`` shows one channel's histogram (counts on a log scale, in the
channel's colour) of what is in view, or of the whole plane or volume (always the whole
volume while a projection is shown):

- the contrast limits are the two solid lines: drag them;
- with the isosurface projection shown, the level is the dashed line: drag it (it
  starts at the middle of the contrast window);
- with the volume projection shown, the opacity curve runs over the contrast window:
  drag its points, double-click to add one, right-click a point to remove it.

``ChannelPanel`` puts a channel choice, visibility, colour map, "whole image" and an
Auto contrast button around it, for the current layer of a ``ViewerWidget``. For a label
image it shows the label controls instead: outlines, only the selected id, the selected
id (double-click one in the view), new colours. Both report
errors on the widget's ``failed`` signal instead of raising (PyQt6 ends the process when a
slot raises).
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from . import COLOUR_MAPS, ERRORS

#: pixels within which a handle is caught
GRAB = 6
#: the histogram's bins
BINS = 256


class HistogramWidget(QWidget):
    """One channel's histogram with its contrast limits, isosurface level and opacity
    curve (see the module's notes)."""

    changed = pyqtSignal()

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view                    # the ViewerWidget
        self.channel = 0
        self.whole = False
        self._hist = None                   # the histogram dict, or None
        self._drag = None                   # "lo", "hi", "iso" or ("point", k)
        self.setMinimumSize(220, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ data
    def _channel(self):
        v = self.view.viewer
        chans = v.channels
        return chans[self.channel] if self.channel < len(chans) else None

    def refresh(self) -> None:
        """Read the histogram again (what is in view, or the whole plane / volume)."""
        try:
            v = self.view.viewer
            if not v.layers:
                self._hist = None
            else:
                # with a projection shown the plane does not matter: the whole volume
                volume = v.layers[v.current]["volume"]
                whole = self.whole or (volume and self._mode() is not None)
                self._hist = v.histogram(self.channel, bins=BINS, whole=whole)
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self._hist = None
            self.view._fail(exc, force=True)
        self.update()

    def _range(self):
        """The values the widget spans: the data's and the contrast window's."""
        ch = self._channel()
        lo, hi = (self._hist["lo"], self._hist["hi"]) if self._hist else (0.0, 1.0)
        if ch is not None:
            lo, hi = min(lo, ch["lo"]), max(hi, ch["hi"])
            if ch.get("iso") is not None:
                lo, hi = min(lo, ch["iso"]), max(hi, ch["iso"])
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _x(self, value: float) -> float:
        lo, hi = self._range()
        return (value - lo) / (hi - lo) * (self.width() - 1)

    def _value(self, x: float) -> float:
        lo, hi = self._range()
        return lo + x / max(self.width() - 1, 1) * (hi - lo)

    def _mode(self) -> str | None:
        """The projection shown ("volume", "iso", ...), or None."""
        v = self.view.viewer
        return v.projection if (v.mip or v.linked) else None

    def _iso(self, ch) -> float:
        return ch["iso"] if ch.get("iso") is not None else ch["lo"] + (ch["hi"] - ch["lo"]) / 2

    def _curve(self, ch):
        """The opacity curve's points (x, y in 0..1); a straight rise by default."""
        return [tuple(p) for p in ch["opacity"]] if ch.get("opacity") else [(0.0, 0.0),
                                                                              (1.0, 1.0)]

    def _curve_xy(self, ch, pt):
        """A curve point on the widget: x over the contrast window, y up."""
        lo, hi = ch["lo"], ch["hi"]
        h = self.height() - 1
        return self._x(lo + pt[0] * (hi - lo)), h - pt[1] * h

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _event):
        try:
            self._paint()
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)

    def _paint(self):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 24, 26))
        ch = self._channel()
        w, h = self.width(), self.height()
        if ch is None:
            p.end()
            return
        r, g, b = (round(c * 255) for c in ch["color"])
        colour = QColor(r, g, b)
        if self._hist:
            counts = self._hist["counts"]
            top = math.log1p(max(counts)) or 1.0
            edges = self._hist["edges"]
            fill = QColor(colour)
            fill.setAlpha(170)
            for k, n in enumerate(counts):
                if not n:
                    continue
                x0, x1 = self._x(edges[k]), self._x(edges[k + 1])
                bar = math.log1p(n) / top * (h - 4)
                p.fillRect(QRectF(x0, h - bar, max(x1 - x0, 1.0), bar), fill)
        # outside the contrast window: dimmed
        xl, xh = self._x(ch["lo"]), self._x(ch["hi"])
        shade = QColor(0, 0, 0, 120)
        p.fillRect(QRectF(0, 0, max(xl, 0), h), shade)
        p.fillRect(QRectF(xh, 0, max(w - xh, 0), h), shade)
        p.setPen(QPen(QColor(235, 235, 235), 2))
        for x in (xl, xh):
            p.drawLine(QPointF(x, 0), QPointF(x, h))
        mode = self._mode()
        if mode == "iso":
            pen = QPen(QColor(255, 210, 80), 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            xi = self._x(self._iso(ch))
            p.drawLine(QPointF(xi, 0), QPointF(xi, h))
        if mode == "volume":
            pts = [self._curve_xy(ch, pt) for pt in self._curve(ch)]
            path = QPainterPath(QPointF(0, pts[0][1]))
            for x, y in pts:
                path.lineTo(QPointF(x, y))
            path.lineTo(QPointF(w, pts[-1][1]))
            p.setPen(QPen(QColor(120, 220, 255), 2))
            p.drawPath(path)
            p.setBrush(QColor(120, 220, 255))
            for x, y in pts:
                p.drawRect(QRectF(x - 3, y - 3, 6, 6))
        p.end()

    # ------------------------------------------------------------------ dragging
    def _handle_at(self, pos):
        ch = self._channel()
        if ch is None:
            return None
        x, y = pos.x(), pos.y()
        mode = self._mode()
        if mode == "volume":
            for k, pt in enumerate(self._curve(ch)):
                px, py = self._curve_xy(ch, pt)
                if abs(px - x) <= GRAB and abs(py - y) <= GRAB:
                    return ("point", k)
        if mode == "iso" and abs(self._x(self._iso(ch)) - x) <= GRAB:
            return "iso"
        d_lo, d_hi = abs(self._x(ch["lo"]) - x), abs(self._x(ch["hi"]) - x)
        if min(d_lo, d_hi) <= GRAB:
            return "lo" if d_lo <= d_hi else "hi"
        return None

    def _apply(self, **kw):
        try:
            self.view.viewer.set_channel(self.channel, **kw)
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            return
        self.changed.emit()
        self.view._changed()
        self.update()

    def mousePressEvent(self, event):
        hit = self._handle_at(event.position())
        if event.button() == Qt.MouseButton.RightButton:
            if isinstance(hit, tuple):
                ch = self._channel()
                pts = self._curve(ch)
                if len(pts) > 2:
                    del pts[hit[1]]
                    self._apply(opacity=pts)
            return
        self._drag = hit

    def mouseDoubleClickEvent(self, event):
        ch = self._channel()
        if ch is None or self._mode() != "volume":
            return
        lo, hi = ch["lo"], ch["hi"]
        fx = (self._value(event.position().x()) - lo) / max(hi - lo, 1e-30)
        fy = 1.0 - event.position().y() / max(self.height() - 1, 1)
        fx, fy = min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
        pts = sorted(self._curve(ch) + [(fx, fy)])
        self._apply(opacity=pts)

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        ch = self._channel()
        value = self._value(event.position().x())
        if self._drag == "lo":
            self._apply(lo=min(value, ch["hi"] - 1e-6 * max(abs(ch["hi"]), 1.0)))
        elif self._drag == "hi":
            self._apply(hi=max(value, ch["lo"] + 1e-6 * max(abs(ch["lo"]), 1.0)))
        elif self._drag == "iso":
            self._apply(iso=value)
        else:
            k = self._drag[1]
            pts = self._curve(ch)
            lo, hi = ch["lo"], ch["hi"]
            fx = (value - lo) / max(hi - lo, 1e-30)
            fy = 1.0 - event.position().y() / max(self.height() - 1, 1)
            left = pts[k - 1][0] if k > 0 else 0.0
            right = pts[k + 1][0] if k + 1 < len(pts) else 1.0
            pts[k] = (min(max(fx, left), right), min(max(fy, 0.0), 1.0))
            self._apply(opacity=pts)

    def mouseReleaseEvent(self, _event):
        self._drag = None


class ChannelPanel(QWidget):
    """The current layer's channels: a choice of channel, its visibility and colour map,
    its histogram (see HistogramWidget), "whole image" and Auto contrast."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.pick = QComboBox()
        self.shown = QCheckBox("shown")
        self.lut = QComboBox()
        self.lut.addItems(["colour"] + list(COLOUR_MAPS))
        self.hist = HistogramWidget(view)
        self.whole = QCheckBox("whole image")
        self.auto = QPushButton("Auto")
        self.info = QLabel()
        # label images
        self.labels = QWidget()
        self.outline = QCheckBox("outlines")
        self.only = QCheckBox("only the selected id")
        self.selected = QLabel()
        self.clear = QPushButton("Clear")
        self.recolour = QPushButton("New colours")
        lab = QVBoxLayout(self.labels)
        lab.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(self.outline)
        row.addWidget(self.only)
        lab.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.selected, 1)
        row.addWidget(self.clear)
        lab.addLayout(row)
        lab.addWidget(self.recolour)
        lab.addWidget(QLabel("double-click a label in the view to select it"))
        lab.addStretch(1)
        top = QHBoxLayout()
        top.addWidget(self.pick, 1)
        top.addWidget(self.shown)
        top.addWidget(self.lut)
        bottom = QHBoxLayout()
        bottom.addWidget(self.whole)
        bottom.addStretch(1)
        bottom.addWidget(self.auto)
        self.images = QWidget()
        img = QVBoxLayout(self.images)
        img.setContentsMargins(0, 0, 0, 0)
        img.addLayout(top)
        img.addWidget(self.hist, 1)
        img.addWidget(self.info)
        img.addLayout(bottom)
        lay = QVBoxLayout(self)
        lay.addWidget(self.images, 1)
        lay.addWidget(self.labels, 1)
        self.labels.hide()
        self.outline.toggled.connect(lambda on: self._labels(outline=on))
        self.only.toggled.connect(lambda on: self._labels(only_selected=on))
        self.clear.clicked.connect(lambda: self._labels(selected=-1))
        self.recolour.clicked.connect(self._recolour)
        self._filling = False
        self.pick.currentIndexChanged.connect(self._picked)
        self.shown.toggled.connect(lambda on: self._set(visible=on))
        self.lut.currentTextChanged.connect(
            lambda t: self._set(lut="" if t == "colour" else t))
        self.whole.toggled.connect(self._whole)
        self.auto.clicked.connect(self._auto)
        self.hist.changed.connect(self._show_info)
        # the histogram follows the view, a little behind it (reading takes a moment)
        self._later = QTimer(self)
        self._later.setSingleShot(True)
        self._later.setInterval(250)
        self._later.timeout.connect(self.refresh)
        view.viewChanged.connect(self._later.start)

    def refresh(self) -> None:
        """Channels and histogram again from the viewer's current layer."""
        try:
            v = self.view.viewer
            chans = v.channels
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            return
        layer = v.layers[v.current] if v.layers else None
        is_label = bool(layer and layer["label"])
        self.images.setVisible(not is_label)
        self.labels.setVisible(is_label)
        if is_label:
            self._filling = True
            self.outline.setChecked(layer["outline"])
            self.only.setChecked(layer["only_selected"])
            sel = layer["selected"]
            self.selected.setText("selected id: " + ("none" if sel is None else str(sel)))
            self._filling = False
            return
        self._filling = True
        names = [c["label"] for c in chans]
        if [self.pick.itemText(k) for k in range(self.pick.count())] != names:
            keep = self.pick.currentIndex()
            self.pick.clear()
            self.pick.addItems(names)
            self.pick.setCurrentIndex(min(max(keep, 0), len(names) - 1))
        k = max(self.pick.currentIndex(), 0)
        if k < len(chans):
            ch = chans[k]
            self.shown.setChecked(ch["visible"])
            self.lut.setCurrentText(ch["lut"] or "colour")
        self._filling = False
        self.hist.channel = k
        self.hist.refresh()
        self._show_info()

    def _show_info(self):
        chans = self.view.viewer.channels
        k = self.hist.channel
        if k >= len(chans):
            self.info.setText("")
            return
        ch = chans[k]
        text = f"contrast {ch['lo']:g} .. {ch['hi']:g}"
        if ch.get("iso") is not None:
            text += f"   surface {ch['iso']:g}"
        h = self.hist._hist
        if h:
            text += f"   data {h['lo']:g} .. {h['hi']:g} (level {h['level']})"
        self.info.setText(text)

    def _picked(self, _k):
        if not self._filling:
            self.refresh()

    def _set(self, **kw):
        if self._filling:
            return
        try:
            self.view.viewer.set_channel(self.hist.channel, **kw)
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            return
        self.view._changed()
        self.hist.update()

    def _labels(self, **kw):
        if self._filling:
            return
        try:
            self.view.viewer.set_labels(**kw)
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            return
        self.view._changed()
        self.refresh()

    def _recolour(self):
        try:
            v = self.view.viewer
            v.set_labels(seed=(v.layers[v.current]["seed"] + 1) % (2 ** 32))
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            return
        self.view._changed()

    def _whole(self, on):
        self.hist.whole = on
        self.hist.refresh()
        self._show_info()

    def _auto(self):
        try:
            self.view.viewer.auto_contrast(self.hist.channel)
        except ERRORS as exc:  # noqa: BLE001 (see the module's notes)
            self.view._fail(exc, force=True)
            return
        self.view._changed()
        self.refresh()
