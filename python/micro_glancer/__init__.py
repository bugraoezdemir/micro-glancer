"""micro-glancer: a GPU viewer for OME-Zarr images on a Rust core (Workplan WP14).

``Viewer`` (the Rust core) draws into a native window or off-screen; the Qt widget is
``micro_glancer.qt.ViewerWidget``. A view holds layers (images moved and turned by hand
in a shared world); ``micro_glancer.save.save_transform`` writes a layer's transform into
its image. A plate opens as one image (its wells on a grid, one field of each);
``tree(path)`` gives a store's navigation tree (plate, wells, fields, label images).
Incubation: tasks V7 and V9, not released.
"""
import json as _json

from ._core import (ALIGNED, COLOUR_MAPS, PanicException, Viewer, ViewerError, __version__,
                    tree_json as _tree_json)

#: what a Qt host catches around the viewer: errors, and a Rust panic (which Python sees
#: as a BaseException; PyQt6 ends the process when a slot lets one through)
ERRORS = (Exception, PanicException)



def tree(path) -> dict:
    """The navigation tree of the store at ``path`` (a folder, or a place in a .zip /
    .ozx): ``{"kind", "name", "path", "children"}``. A plate ("plate", with "rows",
    "columns", "grid", "fields") lists its wells ("well", with "row" and "column"), each
    well its fields ("image", with "field" and "well"); an image lists its label images
    ("label"); a bioformats2raw collection ("collection") its series. A field's label
    images come from the field's own tree."""
    return _json.loads(_tree_json(str(path)))


__all__ = ["ALIGNED", "COLOUR_MAPS", "ERRORS", "Viewer", "ViewerError", "__version__", "tree"]
