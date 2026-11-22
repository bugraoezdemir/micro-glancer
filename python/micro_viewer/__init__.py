"""micro-viewer: a GPU viewer for OME-Zarr images on a Rust core (Workplan WP14).

``Viewer`` (the Rust core) draws into a native window or off-screen; the Qt widget is
``micro_viewer.qt.ViewerWidget``. A view holds layers (images moved and turned by hand
in a shared world); ``micro_viewer.save.save_transform`` writes a layer's transform into
its image. Incubation: tasks V7 and V9, not released.
"""
from ._core import ALIGNED, COLOUR_MAPS, Viewer, ViewerError, __version__

__all__ = ["ALIGNED", "COLOUR_MAPS", "Viewer", "ViewerError", "__version__"]
