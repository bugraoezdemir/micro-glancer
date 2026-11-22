"""Spike gate (Workplan WP14, task V3): frame times for cached pan and zoom on a 4K target.

An 8192 x 8192 uint16 OME-Zarr level (written once with ozp). At 1:1 a 3840 x 2160
view needs ~135 tiles (fits the 256-slot atlas), so after the first frame every frame
draws from the GPU cache. Measured:

- native: the Qt widget in native mode, 3840 x 2160 physical pixels, presented without
  vsync, N frames each panned or zoomed, the GPU waited for at the end;
- offscreen: the core's off-screen frames read back as RGBA8 (what Qt paints in the
  off-screen mode, Wayland / remote);
- each on the GPU and on the software adapter.

usage: python bench/spike_bench.py SCRATCH_DIR
"""
import os
import sys
import time

import numpy as np

N = 240


def image(path):
    if os.path.exists(path):
        return path
    import ome_zarr_pyramid as ozp
    yy, xx = np.mgrid[0:8192, 0:8192]
    data = ((np.sin(yy / 97.0) + np.cos(xx / 131.0)) * 16000 + 32768
            + (yy * 7 + xx * 3) % 512).astype("uint16")
    pyr = ozp.Pyramid().from_arrays([data], axis_order="yx", scales=[[1.0, 1.0]], version="0.5")
    ozp.IO().write_pyramid(pyr, path, overwrite=True)
    return path


def frames(render, v, finish):
    """First frame (cold tiles), then N cached frames alternating pan and zoom."""
    v.set_camera(2000.0, 3000.0, 1.0)
    t0 = time.perf_counter()
    render()
    finish()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(N):
        if i % 2:
            v.pan(3.0 if (i // 2) % 2 else -3.0, 2.0)
        else:
            f = 1.01 if (i // 2) % 2 else 1 / 1.01
            v.zoom(f, 1920.0, 1080.0)
        render()
    finish()
    per = (time.perf_counter() - t0) / N
    return first, per


def offscreen(path, software, w=3840, h=2160):
    from micro_viewer import Viewer
    v = Viewer.offscreen(w, h, software=software)
    v.open(path)
    frame = bytearray(4 * w * h)
    first, per = frames(lambda: v.render_into(frame), v, v.finish)
    return v.adapter, first, per


def native(path, software, w=3840, h=2160):
    from PyQt6.QtWidgets import QApplication
    from micro_viewer.qt import ViewerWidget
    app = QApplication.instance() or QApplication([])
    widget = ViewerWidget(mode="native", software=software, no_vsync=True)
    r = widget.devicePixelRatioF()
    widget.resize(round(w / r), round(h / r))
    widget.show()
    app.processEvents()
    widget.open(path)
    v = widget.viewer
    app.processEvents()
    size = v.size

    def render():
        widget.repaint()

    first, per = frames(render, v, v.finish)
    adapter = v.adapter
    widget.close()
    app.processEvents()
    return adapter, first, per, size


def main():
    path = image(os.path.join(sys.argv[1], "bench8k.ome.zarr"))
    for software in (False, True):
        kind = "software" if software else "GPU"
        adapter, first, per, size = native(path, software)
        print(f"native {kind:8s} {size[0]}x{size[1]}: first frame {first * 1000:7.1f} ms, "
              f"cached {per * 1000:6.2f} ms/frame = {1 / per:7.1f} fps  [{adapter}]", flush=True)
        adapter, first, per = offscreen(path, software)
        print(f"offscreen {kind:8s} 3840x2160: first frame {first * 1000:7.1f} ms, "
              f"cached {per * 1000:6.2f} ms/frame = {1 / per:7.1f} fps  [{adapter}]", flush=True)


if __name__ == "__main__":
    main()
