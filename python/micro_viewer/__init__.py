"""micro-viewer: a GPU viewer for OME-Zarr images on a Rust core (Workplan WP14).

``Viewer`` (the Rust core) draws into a native window or off-screen; the Qt widget is
``micro_viewer.qt.ViewerWidget``. Incubation: the 2D viewer of task V7, not released.
"""
from ._core import Viewer, ViewerError, __version__

__all__ = ["Viewer", "ViewerError", "__version__"]
