# micro-viewer

A viewer for OME-Zarr images that draws on the GPU. The drawing, the tile cache and the
reading are Rust (the crate `micro-viewer-core`); Python gets a small API and a Qt widget
(the package `micro-viewer`). It is in incubation: not released, and the API may change.

## What you can do

- Open an OME-Zarr image (NGFF 0.4, 0.5 or 0.6) from a folder, or from a `.zip` or `.ozx`
  file.
- Pan and zoom smoothly. The viewer picks the resolution level that fits the zoom. While
  finer tiles are read in the background, coarser ones stand in, so the view never waits.
- See every channel at once, each in its own colour or colour map (viridis, magma,
  inferno, plasma, cividis, turbo, hot, gray), with its own contrast limits and gamma.
  Colours, labels and contrast windows come from the image's omero metadata when it has
  them. Contrast can also be taken from what is in view.
- Move through z and t. The image opens at omero's default z and t. The planes next to
  the current one are read ahead, so stepping is quick.
- Turn the view.
- Cut a volume (an image with z) by a plane at any angle: look along y or x (XZ, YZ),
  or turn the plane freely through the volume. Voxels show as they are (the nearest
  one), or smoothly between voxel centres (trilinear), per layer. Anisotropic voxels
  are placed by their physical size. z is shared by all layers: the plane cuts every
  volume where it is in the world.
- Open several images as layers in one view. They add up, each can be hidden or made
  fainter, and each can be moved and turned by hand, for example to line two images up
  before a registration tool refines it.
- Save a layer's transform into its image, without rewriting any pixel. NGFF 0.4 and
  0.5 images take a shift only. NGFF 0.6 images take the whole transform, kept apart
  from the image's own placement (in a coordinate system called "aligned"). Reopening
  the image shows it where it was saved.
- Read the values under the pointer, for every channel of every layer.
- Save the whole view (camera, layers with their images, transforms, planes and
  channels) as JSON and restore it later.

On slow sources (remote stores, slow codecs, and later other file formats) the viewer
reads coarser levels while the view moves and the finer ones once it stops. To see how
it behaves, set `MICRO_VIEWER_SLOW_READS_MS` to add that many milliseconds to every
tile read.

## Trying it

```
python -m micro_viewer.qt PATH [PATH ...]
```

Drag to pan, scroll to zoom at the pointer, press F to fit, and press Page Up or Page
Down to move through z (with Shift: t). Shift + scroll turns the view; [ and ] turn it
by 90 degrees, 0 turns it back.

For a volume, Alt + drag turns the plane through it, and 1, 2 and 3 look along z, y and
x. Page Up and Page Down move the plane one voxel along the view, and I switches
trilinear sampling on and off.

With several images, Tab picks the current layer. Ctrl + drag moves it, Ctrl + scroll
turns it (1 degree a notch, with Shift 0.1), Ctrl + arrow keys nudge it by a pixel,
Ctrl + 0 puts it back, Ctrl + Alt + drag tilts a volume, and Ctrl + S saves its
transform into its image. Two one-channel
images open in green and magenta. The status bar shows the current layer, its plane,
level and transform, and the values under the pointer.

From Python:

```python
from micro_viewer import Viewer
from micro_viewer.save import save_transform

v = Viewer.offscreen(1024, 768)
v.open("image.ome.zarr")
v.set_position("z", 10)
v.set_channel(0, color=(0.0, 1.0, 0.0), lo=100, hi=4000, gamma=0.8)
v.auto_contrast()               # or: contrast from what is in view
v.settle()                      # wait until the view is sharp
rgba, width, height, stats = v.render_rgba()
print(v.pick(512, 384))         # the values under the centre

v.add("moving.ome.zarr")        # a second layer, now the current one
v.set_channel(0, lut="magma")
v.rotate_layer(12.5)            # turn it about its centre (degrees)
v.move_layer(40, -10)           # move it (screen pixels)
print(v.transform_matrix())     # the NGFF affine over the image's axes
save_transform(v)               # into moving.ome.zarr
state = v.view_state()          # JSON

v.open("stack.ome.zarr")        # a volume
v.set_orientation("xz")         # look along y: x right, z down
v.turn(30, 20)                  # turn the plane through the volume (degrees)
v.set_interpolation("linear")   # trilinear between voxel centres
v.step_depth(5)                 # move the plane 5 voxels along the view
print(v.pick(512, 384)["voxel"])  # the voxel (x, y, z) under the centre
```

In Qt, `micro_viewer.qt.ViewerWidget` shows the layers. It emits `viewChanged` when the
camera, a plane, a layer or the channels change, `picked` with the values under the
pointer, `saved` after a transform was saved, and `failed` when something fails.

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
and zoom. For volumes also: the time until sharp after turning the plane through the
volume, and the frames per second on the turned plane, cached and while turning on.

`scratch/inspect_mv.py` is eubi's Inspect page with micro-viewer in its Viewer tab, to
try the viewer inside eubi without changing eubi.
