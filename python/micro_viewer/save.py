"""Saving a layer's transform into its image (metadata only; no pixel is rewritten).

What can be saved depends on the image's NGFF version:

- 0.4 and 0.5 hold a translation and a scale per level, so only a translation can be
  saved: it is added to every level's translation (through ome-zarr-pyramid). A turned
  (or scaled) layer is refused, naming the ways out.
- 0.6 holds coordinate systems linked by transforms. The layer's transform is saved as
  a transform from the image's own coordinate system to the ``"aligned"`` one
  (``micro_viewer.ALIGNED``): a translation, or an affine when the layer is turned. The
  image's own placement stays as it was, so a registration tool can read the hand-made
  alignment as its starting point. Saving the identity removes it again. ozp has no
  public call for such a transform yet, so this edits the image's ``zarr.json``
  directly, in the form ozp's 0.6 schemas give (a multiscale-level transform and its
  output coordinate system).

After saving, the layer is read again: it shows the saved placement, and reopening
the image later shows it too.

Images in a zip (``.zip``, ``.ozx``) cannot be changed in place; save needs a folder.
"""
from __future__ import annotations

import json
import os

from ._core import ALIGNED, Viewer, ViewerError

#: how far from the identity a linear part may be and still count as a translation
TOLERANCE = 1e-9


def _is_translation(m, n: int) -> bool:
    return all(abs(m[r][c] - (1.0 if r == c else 0.0)) <= TOLERANCE
               for r in range(n) for c in range(n))


def save_transform(viewer: Viewer, layer: int | None = None, *, system: str = ALIGNED) -> str:
    """Write layer `layer`'s transform (the current layer by default) into its image.
    `system` names the coordinate system it goes to (NGFF 0.6 only). Returns what was
    written, in words. Raises ``ViewerError`` when the image cannot hold it."""
    i = viewer.current if layer is None else layer
    info = viewer.layers[i]
    path = info["path"]
    if info.get("plate"):
        raise ViewerError(
            "a plate is shown as its wells on a grid, which is not stored in any image: "
            "open a field (in the Contents panel) to save its transform")
    if info.get("group"):
        raise ViewerError(
            "a well or a collection is shown as its images at their positions, which no "
            "one image stores: open one of its images (in the Contents panel) to save "
            "its transform")
    if info.get("label") and info.get("follows") is not None:
        raise ViewerError(
            "this label image follows its image: save the image's transform instead")
    if os.path.isfile(path):
        raise ViewerError(
            f"{path} is a zip file, which cannot be changed in place: save the "
            f"transform into an image folder (unzip it, or convert it to a folder first)")
    m = viewer.transform_matrix(i)
    n = len(m)
    if info["version"] in ("0.4", "0.5"):
        if not _is_translation(m, n):
            raise ViewerError(
                f"NGFF {info['version']} holds only a translation per level, and this "
                f"layer is turned or scaled. Turn it back (or reset its transform) to save "
                f"the shift alone, or convert the image to NGFF 0.6, which holds the "
                f"whole transform")
        done = _save_translation(path, [m[a][n] for a in range(n)], info["axes"])
    else:
        done = _save_06(path, m, system)
    viewer.reload(i)
    return done


def _save_translation(path: str, shift, axes) -> str:
    """Add `shift` (per axis, world units) to every level's translation (0.4, 0.5)."""
    import zarr
    import ome_zarr_pyramid as ozp

    if not any(shift):
        return "nothing to save: the layer is where the image places it"
    pyr = ozp.IO().read_pyramid(path, include_labels=False, apply_scene=None)
    for p in pyr.meta.resolution_paths:
        old = pyr.meta.get_translation(p) or [0.0] * len(shift)
        pyr.meta.update_translation(p, [a + b for a, b in zip(old, shift)])
    zf = 2 if pyr.meta.version == "0.4" else 3
    pyr.meta.zarr_group = zarr.open_group(path, mode="r+", zarr_format=zf)
    pyr.meta.save_changes()
    moved = ", ".join(f"{a} {s:+g}" for a, s in zip(axes, shift) if s)
    return f"added a translation of {moved} to every level"


def _save_06(path: str, m, system: str) -> str:
    """The transform as an edge from the image's own coordinate system to `system`
    (NGFF 0.6), replacing one saved before; the identity removes it."""
    meta = os.path.join(path, "zarr.json")
    with open(meta, encoding="utf-8") as f:
        doc = json.load(f)
    try:
        ms = doc["attributes"]["ome"]["multiscales"][0]
        systems = ms["coordinateSystems"]
        own = ms["datasets"][0]["coordinateTransformations"][0]["output"]["name"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ViewerError(f"{meta}: not an NGFF 0.6 image as expected ({exc!r})") from None
    if system == own:
        raise ViewerError(f"the image's own coordinate system is called {own!r}; "
                          f"save into another one")
    source = next((s for s in systems if s.get("name") == own), None)
    if source is None:
        raise ViewerError(f"{meta}: the image's coordinate system {own!r} is not listed")

    def ours(t):
        return ((t.get("output") or {}).get("name") == system
                and (t.get("input") or {}).get("name") == own)

    edges = [t for t in ms.get("coordinateTransformations", []) if not ours(t)]
    systems[:] = [s for s in systems if s.get("name") != system]
    n = len(m)
    identity = _is_translation(m, n) and not any(m[r][n] for r in range(n))
    if identity:
        said = f"removed the transform to {system!r} (the layer is where the image places it)"
    else:
        systems.append({"name": system, "axes": json.loads(json.dumps(source["axes"]))})
        ends = {"input": {"name": own}, "output": {"name": system}}
        if _is_translation(m, n):
            edges.append({"type": "translation",
                          "translation": [m[r][n] for r in range(n)], **ends})
            said = f"saved a translation from {own!r} to {system!r}"
        else:
            edges.append({"type": "affine", "affine": [list(r) for r in m], **ends})
            said = f"saved an affine from {own!r} to {system!r}"
    if edges:
        ms["coordinateTransformations"] = edges
    else:
        ms.pop("coordinateTransformations", None)
    tmp = meta + ".micro-viewer.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, meta)                    # a reader never sees half a file
    return said
