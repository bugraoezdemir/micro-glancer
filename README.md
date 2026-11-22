# micro-viewer

A viewer for OME-Zarr images that draws on the GPU. The drawing, the tile cache and the
reading are Rust (the crate `micro-viewer-core`); Python gets a small API and a Qt widget
(the package `micro-viewer`). It is in incubation: not released, and the API may change.

## What you can do

- Open an OME-Zarr image (NGFF 0.4, 0.5 or 0.6) from a folder, or from a `.zip` or `.ozx`
  file.
- Pan and zoom smoothly. The viewer picks the resolution level that fits the zoom. While
  finer tiles are read in the background, coarser ones stand in, so the view never waits.
- See every channel at once, each in its own colour, with its own contrast limits and
  gamma. Colours, labels and contrast windows come from the image's omero metadata when
  it has them.
- Move through z and t. The image opens at omero's default z and t. The planes next to
  the current one are read ahead, so stepping is quick.
- Read the values under the pointer, for every channel.
- Save the whole view (image, camera, plane, channels) as JSON and restore it later.

## Trying it

```
python -m micro_viewer.qt PATH
```

Drag to pan, scroll to zoom at the pointer, press F to fit the image, and press Page Up
or Page Down to move through z (with Shift: t). The status bar shows the plane, the level
and the values under the pointer.

From Python:

```python
from micro_viewer import Viewer

v = Viewer.offscreen(1024, 768)
v.open("image.ome.zarr")
v.set_position("z", 10)
v.set_channel(0, color=(0.0, 1.0, 0.0), lo=100, hi=4000, gamma=0.8)
v.settle()                      # wait until the view is sharp
rgba, width, height, stats = v.render_rgba()
print(v.pick(512, 384))         # the values under the centre
state = v.view_state()          # JSON
```

In Qt, `micro_viewer.qt.ViewerWidget` shows an image. It emits `viewChanged` when the
camera, plane or channels change, `picked` with the values under the pointer, and
`failed` when the core cannot draw.

## How it is hosted

On Windows and macOS the core draws straight into the widget's own native window, and
frames never pass through Python. On Linux, and wherever a native window is not
available, the core draws off-screen and Qt paints the frame. A machine without a GPU
uses the system's software renderer. Set `MICRO_VIEWER_MODE=native` or `offscreen` to
choose the mode.

## Tests and numbers

```
pytest tests                  # needs ome_zarr_pyramid to write the test images
python bench/harness.py DIR [IMAGE ...]
```

The harness measures, on a 3840 x 2160 target: the time until the view is sharp after
opening, after a jump and after a step in z or t, and the frames per second of cached pan
and zoom.

`scratch/inspect_mv.py` is eubi's Inspect page with micro-viewer in its Viewer tab, to
try the viewer inside eubi without changing eubi.
