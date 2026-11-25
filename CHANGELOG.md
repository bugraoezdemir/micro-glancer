# Changelog

## 0.0.2 (2026-11-24)

### Changed
- **Stepping through planes no longer winks**: a step showed a coarser stand-in or, after
  a jump, an empty frame until the new plane's tiles arrived. Now:
  - when the decode units a view needs (a sharded array's inner chunks, else its chunks)
    fit half of the 512 MB block budget, tiles are put together from whole units, each
    decoded once and kept. A chunk 96 planes deep then serves all of its planes. On a
    96^3-chunked volume zoomed in, a step became sharp after 2 ms instead of 386 ms.
    Images whose units do not fit keep reading blocks in their own plane, within the same
    budget;
  - every plane change keeps the previous plane's tiles drawn where the new plane has
    none yet, so there is never an empty frame. After a jump (a slider dragged), a coarser
    tile of the new plane replaces them as soon as it arrives;
  - planes are read ahead in the direction of stepping (four ahead, one behind) when the
    cache has room.
- **Plates, wells and collections read their fields on several threads**: a read that
  covers many fields (a whole plate in view, a histogram, the contrast from the view)
  spreads them over the threads; fields that overlap keep their drawing order (the later
  one shows). A histogram of a whole 384-well plate in view (10,640 fields) took 0.25 s
  instead of 1.4 s.
- **The histogram panel no longer stops the window**: it reads on a thread of the core's
  own, and the window keeps drawing meanwhile; a refresh asked for while one is read
  follows when it is done. Before, every pause while panning over a whole plate froze
  the window for about a second (the histogram read every field in view), so the coarse
  tiles that stand in while finer ones load could not be drawn either.
- **The first viewer of a process starts sooner**: every Qt widget loads the GPU driver
  on a thread of its own when it is made, which took about 1.3 s at the first image
  opened (later viewers took 0.08 s).

### Added
- **Opening in the background**: `ViewerWidget.open_in_background(path, done=...)` reads
  an image's metadata (a plate: every field's) on a thread of the core's own while the
  window stays responsive, reports `opening(done, total)` (fields of a plate, members of
  a group), then shows it and calls `done(error)`; a newer call replaces an older one.
  Underneath, `micro_glancer.OpenJob(path)` with `start()` / `done` / `progress` /
  `error` (or `run()` on the calling thread) and `Viewer.open_job(job)` /
  `Viewer.add_job(job)`.
- `Viewer.histogram_job(...)`: the same histogram as `Viewer.histogram`, as a job:
  `start()` reads and counts on a thread of the core's own (`done`, `result()`), or
  `run()` on the calling thread without Python's lock. `HistogramWidget.wait()` waits
  for the newest histogram asked for, and its `read` signal tells when one arrives.
- `micro_glancer.warm_up(software=False, background=False)`: loads the GPU driver and
  makes a device once, so the first viewer need not wait for it.

### Fixed
- **A viewer freed on another thread** no longer raises "Viewer is unsendable, but is
  being dropped on another thread" (an error Python cannot raise, printed instead) and is
  no longer leaked. Python frees objects in reference cycles, such as Qt windows, from
  whichever thread its cycle collector runs on, for example zarr's event-loop thread. Such
  a viewer is now handed back to its own thread, which frees it the next time it uses a
  viewer. A viewer drawing into a native window is let go without freeing, because its
  window may already be gone. Using a viewer on another thread than the one that made it
  raises a `PanicException`, as before. Background work runs on the core's own threads,
  never on Python threads. CI fails on such errors (`PytestUnraisableExceptionWarning` as
  an error).

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
