"""The V7 harness (Workplan WP14): micro-glancer's numbers against the targets
(120 frames per second for cached pan and zoom, 0.5 s from a change to a sharp view).

For each image, on an off-screen 3840 x 2160 target (and the native window for the
cached frames):

- open: time to open (metadata, atlas, contrast) and to the first frame, which never
  waits for tiles (coarser levels or the missing colour stand in);
- sharp after open: time until every tile of the fitted view is read and drawn;
- jump: pan a whole screen at 1:1 to a place never seen, time until sharp;
- plane: one step on the first plane axis (z or t), time until sharp (the planes next
  to the current one are read ahead when the cache has room);
- cached: frames per second for pan and zoom over cached tiles (off-screen frames are
  read back to memory; native frames go to the window);
- volumes: the whole volume, then the plane turned by 30 and 20 degrees through it
  (drawn from bricks): time until sharp, cached frames per second, frames per second
  while turning on (bricks coming in), and the time until sharp after that.

usage: python bench/harness.py SCRATCH_DIR [IMAGE ...]
The synthetic 16384 x 16384 uint16 pyramid is written once into SCRATCH_DIR.
"""
import os
import sys
import time

import numpy as np

W, H = 3840, 2160
N = 240


def synthetic(path):
    if os.path.exists(path):
        return path
    import ome_zarr_pyramid as ozp
    rng = np.random.default_rng(7)
    n = 16384
    levels = []
    base = np.empty((n, n), np.uint16)
    for y0 in range(0, n, 2048):                     # in strips: bounded memory
        yy, xx = np.mgrid[y0:y0 + 2048, 0:n]
        smooth = (np.sin(yy / 211.0) + np.cos(xx / 157.0)) * 12000 + 30000
        base[y0:y0 + 2048] = (smooth + rng.normal(0, 900, smooth.shape)).clip(0, 65535)
    levels.append(base)
    while levels[-1].shape[0] > 512:
        levels.append(np.ascontiguousarray(levels[-1][::2, ::2]))
    scales = [[2.0 ** i, 2.0 ** i] for i in range(len(levels))]
    pyr = ozp.Pyramid().from_arrays(levels, axis_order="yx", scales=scales, version="0.5")
    ozp.IO().write_pyramid(pyr, path, overwrite=True)
    return path


def sharp(v, timeout=60.0):
    """Seconds until every tile in view at the frame's level is read, then one frame
    drawn (what is read ahead is not waited for)."""
    t0 = time.perf_counter()
    v.settle(timeout)
    v.render_into(frame_for(v))
    v.finish()
    return time.perf_counter() - t0


_frames = {}


def frame_for(v):
    w, h = v.size
    if (w, h) not in _frames:
        _frames[(w, h)] = bytearray(4 * w * h)
    return _frames[(w, h)]


def cached_fps(v, render):
    for _ in range(3):
        render()
    v.finish()
    c = v.center
    t0 = time.perf_counter()
    for i in range(N):
        if i % 2:
            v.pan(3.0 if (i // 2) % 2 else -3.0, 2.0)
        else:
            v.zoom(1.01 if (i // 2) % 2 else 1 / 1.01, W / 2, H / 2)
        render()
    v.finish()
    per = (time.perf_counter() - t0) / N
    v.look_at(c[0], c[1], v.camera[2])
    return 1 / per


def offscreen(path):
    from micro_glancer import Viewer
    out = {}
    v = Viewer.offscreen(W, H)
    t0 = time.perf_counter()
    v.open(path)
    out["open_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    stats = v.render_into(frame_for(v))
    v.finish()
    out["first_frame_ms"] = (time.perf_counter() - t0) * 1000
    out["first_pending"] = stats[2]
    out["sharp_after_open_s"] = sharp(v)
    info = v.image
    out["levels"] = len(info["levels"])
    out["shape"] = info["levels"][0]["shape"]
    out["channels"] = len(v.channels)
    out["fit_level"] = v.level
    # 1:1 at the centre, settled, then a jump of one screen to the right and down
    px = info["levels"][0]["pixel"]
    c = v.center
    v.look_at(c[0], c[1], px)
    out["one_to_one_s"] = sharp(v)
    b = v.bounds
    v.look_at(min(c[0] + W * px, b[2] - W * px / 2), min(c[1] + H * px / 2, b[3]), px)
    out["jump_s"] = sharp(v)
    pos = v.position
    if pos and pos[0][2] > 1:
        name, i, n = pos[0]
        v.set_position(name, float(i + 1 if i + 1 < n else i - 1))
        out["plane_axis"] = name
        out["plane_step_s"] = sharp(v)
    out["cached_offscreen_fps"] = cached_fps(v, lambda: v.render_into(frame_for(v)))
    out["atlas_slots"] = v.atlas_slots
    if v.layers[0]["volume"]:
        # the whole volume, then the plane turned through it (bricks)
        v.fit()
        sharp(v)
        v.turn(30.0, 20.0)
        out["oblique_sharp_s"] = sharp(v)
        out["oblique_level"] = v.level
        out["oblique_cached_fps"] = cached_fps(v, lambda: v.render_into(frame_for(v)))
        # turning on: new bricks come in while frames go on
        t0 = time.perf_counter()
        for _ in range(60):
            v.turn(0.5, 0.25)
            v.render_into(frame_for(v))
        v.finish()
        out["turning_fps"] = 60 / (time.perf_counter() - t0)
        out["turning_sharp_s"] = sharp(v)
        out["brick_slots"] = v.brick_slots()
        v.set_orientation("xy")
        v.fit()
        sharp(v)
        # the projection (maximum intensity) of the whole volume along z
        v.mip = True
        out["mip_sharp_s"] = sharp(v)
        out["mip_level"] = v.render_into(frame_for(v))[3]
        out["mip_cached_fps"] = cached_fps(v, lambda: v.render_into(frame_for(v)))
        # shaded: the volume composited front to back, then the isosurfaces
        for mode in ("volume", "iso"):
            v.projection = mode
            out[f"{mode}_sharp_s"] = sharp(v)
            out[f"{mode}_cached_fps"] = cached_fps(v, lambda: v.render_into(frame_for(v)))
        v.projection = "mip"
        v.mip = False
        # linked views: three planes and the projection in quarters
        v.linked = True
        out["linked_sharp_s"] = sharp(v)
        out["linked_cached_fps"] = cached_fps(v, lambda: v.render_into(frame_for(v)))
        v.linked = False
    out["adapter"] = v.adapter
    if v.error:
        out["error"] = v.error
    return out


def native(path):
    from PyQt6.QtWidgets import QApplication
    from micro_glancer.qt import ViewerWidget
    app = QApplication.instance() or QApplication([])
    widget = ViewerWidget(mode="native", no_vsync=True)
    r = widget.devicePixelRatioF()
    widget.resize(round(W / r), round(H / r))
    widget.show()
    app.processEvents()
    widget.open(path)
    v = widget.viewer
    app.processEvents()
    v.settle(60.0)
    fps = cached_fps(v, widget.repaint)
    size = v.size
    widget.close()
    app.processEvents()
    return {"cached_native_fps": fps, "native_size": size}


def main():
    scratch = sys.argv[1]
    images = sys.argv[2:] or []
    images = [synthetic(os.path.join(scratch, "synthetic16k.ome.zarr"))] + images
    for path in images:
        name = os.path.basename(path.rstrip("/\\"))
        res = offscreen(path)
        res.update(native(path))
        print(f"\n== {name}  {res['shape']} x{res['channels']} channels, {res['levels']} levels")
        for k, v in res.items():
            if k in ("shape", "channels", "levels"):
                continue
            print(f"   {k:22s} {v:.3f}" if isinstance(v, float) else f"   {k:22s} {v}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
