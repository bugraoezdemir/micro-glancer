"""A store's contents as a tree (``micro_viewer.tree``): a plate's wells and fields, an
image's label images, a collection's series.

``ContentsPanel.show_tree(path)`` reads the store's tree; double-clicking (or Enter on)
a node emits ``activated(dict)`` with the node (``kind``, ``name``, ``path`` and, by
kind, ``field`` / ``well`` / ``row`` / ``column``). A field's label images are read when
the field is expanded. The host decides what a node does (``open_node`` is micro-
viewer's own choice: a plate opens whole, a well is shown on its open plate, an image or
field opens, a label image is added over its open image).
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from . import ERRORS, tree

_ROLE = Qt.ItemDataRole.UserRole


def _same(a, b):
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


class ContentsPanel(QWidget):
    """The tree of the store opened last (see the module's notes)."""

    activated = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QTreeWidget()
        self.view.setHeaderHidden(True)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.info)
        self.root = None
        self.view.itemActivated.connect(lambda item, _col: self._activate(item))
        self.view.itemExpanded.connect(self._expanded)

    def show_tree(self, path) -> dict | None:
        """Read and show the tree of the store at `path`; the tree, or None on an error
        (shown in the panel)."""
        self.view.clear()
        try:
            t = tree(path)
        except ERRORS as exc:  # noqa: BLE001 (a host's slot must not raise)
            self.root = None
            self.info.setText(str(exc))
            return None
        self.root = t
        item = self._item(t)
        self.view.addTopLevelItem(item)
        item.setExpanded(True)
        what = {"plate": f"plate: {len(t['children'])} wells, grid {t.get('grid')}, "
                         f"up to {t.get('fields')} fields",
                "collection": f"{len(t['children'])} series"}.get(t["kind"], t["kind"])
        self.info.setText(what)
        return t

    def _item(self, node):
        label = node["name"] if node["kind"] in ("image", "well") else (
            f"{node['kind']}: {node['name']}")
        item = QTreeWidgetItem([label])
        data = {k: v for k, v in node.items() if k != "children"}
        if node["kind"] == "well" and node.get("children"):
            data["first_field"] = node["children"][0]["path"]
        item.setData(0, _ROLE, data)
        item.setToolTip(0, str(node.get("path", "")))
        for child in node.get("children", []):
            item.addChild(self._item(child))
        if node["kind"] == "image" and "field" in node and not node.get("children"):
            # a plate's field: its label images are read when it is opened up
            item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        return item

    def _expanded(self, item):
        node = item.data(0, _ROLE)
        if not node or item.childCount() or "field" not in node:
            return
        try:
            sub = tree(node["path"])
        except ERRORS:  # noqa: BLE001
            sub = {"children": []}
        for child in sub.get("children", []):
            item.addChild(self._item(child))
        if not item.childCount():
            item.setChildIndicatorPolicy(
                QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicatorWhenChildless)

    def _activate(self, item):
        node = item.data(0, _ROLE)
        if node:
            parent = item.parent()
            node = dict(node, parent=parent.data(0, _ROLE) if parent else None)
            self.activated.emit(node)


def open_node(viewer, node) -> str:
    """What micro-viewer does with an activated node on `viewer` (a ``Viewer``); returns
    what it did, in words. A plate opens whole; a well is shown on its plate when the
    plate is open (else the well opens, every field at its position); an image, a
    field or a collection opens (a collection with every series at its position); a
    label image is added over its image when that is open (else it opens alone)."""
    kind, path = node["kind"], node["path"]
    layers = viewer.layers
    if kind == "well":
        for k, lay in enumerate(layers):
            if lay["plate"] and node.get("parent") and _same(lay["path"], node["parent"]["path"]):
                viewer.current = k
                viewer.show_well(node["name"])
                return f"well {node['name']}"
        viewer.open(path)
        viewer.fit()
        return f"well {node['name']}: every field at its position"
    if kind == "label":
        image = os.path.dirname(os.path.dirname(os.path.normpath(path)))
        for k, lay in enumerate(layers):
            if not lay["label"] and _same(lay["path"], image):
                viewer.add_label(node["name"], layer=k)
                return f"label image {node['name']}"
        viewer.open(path)
        return f"label image {node['name']} alone"
    viewer.open(path)
    viewer.fit()
    return f"{kind} {node['name']}"
