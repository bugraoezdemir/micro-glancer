"""Smoke test of the scratch Inspect page on micro-viewer (off-screen Qt, no window).

usage: QT_QPA_PLATFORM=offscreen MICRO_VIEWER_MODE=offscreen python scratch/smoke_inspect_mv.py IMAGE
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])
from inspect_mv import InspectPageMV  # noqa: E402


def pump(seconds=1.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)


page = InspectPageMV()
page.resize(1200, 800)
page.show()
pump(0.3)
page._load_zarr(os.path.abspath(sys.argv[1]))
pump(1.0)
mv = page._viewer
v = mv.view.viewer
print("loaded:", mv.path is not None, "| pyr for the metadata tab:", page._pyr is not None)
print("status bar:", page._status_bar.text())
print("channels in eubi's panel:", [(c["label"], c["color"], c["intensityMin"], c["intensityMax"])
                                    for c in mv.channel_panel.channel_data()])
print("sliders:", [(n, s.value(), s.maximum()) for n, s, _ in mv._sliders])
img = mv.view.grab().toImage()
lit = sum(1 for y in range(0, img.height(), 7) for x in range(0, img.width(), 7)
          if img.pixelColor(x, y).value() > 40)
print("view", img.width(), "x", img.height(), "lit samples:", lit)

# eubi's panel drives the core: recolour channel 0 and hide channel 1
rows = mv.channel_panel.channel_data()
rows[0]["color"] = "#ff0000"
rows[1]["visible"] = False
mv._apply_channels(rows)
print("core channels after the panel:", [(c["color"], c["visible"]) for c in v.channels])

# eubi's Auto button sets the contrast from the view
before = (v.channels[0]["lo"], v.channels[0]["hi"])
mv.channel_panel.auto_requested.emit(0)
pump(0.2)
print("auto contrast, channel 0:", before, "->", (v.channels[0]["lo"], v.channels[0]["hi"]),
      "| panel:", (mv.channel_panel.channel_data()[0]["intensityMin"],
                   mv.channel_panel.channel_data()[0]["intensityMax"]))

# the z slider moves the plane
name, slider, _ = mv._sliders[0]
slider.setValue(10)
pump(0.5)
print("position after the slider:", v.position, "| status:", page._status_bar.text())
page.close()
pump(0.2)
print("ok")
