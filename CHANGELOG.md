# Changelog

## 0.0.1 (2026-11-24)

The first release: a viewer for OME-Zarr images (NGFF 0.4, 0.5 and 0.6) that draws on
the GPU, with a Rust core and a Qt widget and app for Python. Wheels for Windows, macOS
and Linux, Python 3.11 and newer (one wheel per platform).

- Images from folders, `.zip` and `.ozx` files; smooth pan and zoom over the resolution
  levels, coarser tiles standing in while finer ones are read.
- Channels in colours or colour maps, contrast limits, gamma, auto contrast, and a
  histogram per channel; omero metadata as the starting point.
- Planes at any angle through a volume, linked XY / YZ / XZ views, and the whole volume
  as a maximum intensity projection, a composited volume or surfaces, with shading and
  opacity curves.
- Several images as layers, each moved and turned by hand; the transform saved into the
  image (a translation for NGFF 0.4 / 0.5, a transform to an "aligned" coordinate system
  for 0.6).
- Label images with exact 32 and 64-bit ids, outlines and selection.
- Image positions: plates with every field at its place, wells and collections as
  mosaics, NGFF 0.6 scenes; a navigation tree of plates, collections and label images.
- `micro-glancer` command (`pip install "micro-glancer[qt]"`); without PyQt6 it says what
  to install. Saving a translation into an NGFF 0.4 / 0.5 image needs ome_zarr_pyramid
  (`pip install "micro-glancer[save]"`).
- The Qt widget draws into its own native window on Windows and macOS, and off screen
  everywhere else. Asking for native drawing where Qt has no real windows (its
  `offscreen` or `minimal` platform) raises an error that names `mode='offscreen'`.
