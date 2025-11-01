from __future__ import annotations

"""Remote-backed file picker dialog for selecting server-side datasets.

This dialog mimics a standard file dialog but lists entries by issuing
`call.command` requests to the napari-cuda server (fs.listdir). When the user
accepts a selection, callers can request the server to open the dataset
(`napari.zarr.load`).

Usage (from the streaming client launcher):

    dlg = RemoteFileDialog(loop, parent=qt_viewer)
    if dlg.exec_():
        path = dlg.selected_path()
        if path:
            loop.open_remote_dataset(path)

"""

from collections.abc import Mapping

from qtpy import QtCore, QtGui, QtWidgets


class RemoteFileDialog(QtWidgets.QDialog):
    _listing_ready = QtCore.Signal(object)
    """Simple mac-like remote file picker backed by server RPC.

    Parameters
    ----------
    loop : object
        The `ClientStreamLoop` instance (provides `_issue_command` and
        helpers added by client side changes). Required.
    parent : QWidget | None
        Parent widget.
    start_path : str | None
        Initial server path to list (None = server-configured data root).
    only : Sequence[str] | None
        Filename suffixes to show (e.g., (".zarr",)). Directories are always
        included. If None, show all files.
    select_folders : bool
        If True, allow selecting directories (used for "Open Folder...").
    """

    def __init__(
        self,
        loop,
        parent: QtWidgets.QWidget | None = None,
        *,
        start_path: str | None = None,
        only: Sequence[str] | None = (".zarr",),
        select_folders: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select file(s) on server…")
        self.resize(800, 520)
        self._loop = loop
        self._only = tuple(only) if only else None
        self._select_folders = bool(select_folders)
        self._current_path: str | None = start_path
        self._history: list[str] = []
        self._history_index: int = -1
        self._selected: str | None = None

        self._build_ui()
        self._connect_signals()
        self._listing_ready.connect(self._on_listing_ready)

        # Initial listing
        self._request_listdir(self._current_path)

    # ------------------------------- UI ---------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Toolbar: Back, Up, Path line edit
        toolbar = QtWidgets.QHBoxLayout()

        self._btn_back = QtWidgets.QToolButton(self)
        self._btn_back.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack))
        self._btn_back.setEnabled(False)
        toolbar.addWidget(self._btn_back)

        self._btn_up = QtWidgets.QToolButton(self)
        self._btn_up.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp))
        toolbar.addWidget(self._btn_up)

        self._path_edit = QtWidgets.QLineEdit(self)
        self._path_edit.setPlaceholderText("/server/data…")
        toolbar.addWidget(self._path_edit, 1)

        self._btn_go = QtWidgets.QToolButton(self)
        self._btn_go.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOkButton))
        toolbar.addWidget(self._btn_go)

        layout.addLayout(toolbar)

        # File list
        self._view = QtWidgets.QTreeView(self)
        self._view.setRootIsDecorated(False)
        self._view.setAlternatingRowColors(True)
        self._view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._view.setUniformRowHeights(True)
        self._view.doubleClicked.connect(self._on_double_clicked)

        self._model = QtGui.QStandardItemModel(self)
        self._model.setHorizontalHeaderLabels(["Name", "Type", "Size", "Modified"])  # basic columns
        # Sort proxy: directories first, then case-insensitive name
        class _DirFirstProxy(QtCore.QSortFilterProxyModel):
            def lessThan(self, left: QtCore.QModelIndex, right: QtCore.QModelIndex) -> bool:  # type: ignore[override]
                src = self.sourceModel()
                li = src.index(left.row(), 0)
                ri = src.index(right.row(), 0)
                ldir = bool(src.data(li, QtCore.Qt.UserRole + 2))
                rdir = bool(src.data(ri, QtCore.Qt.UserRole + 2))
                if ldir != rdir:
                    return ldir and not rdir
                lname = str(src.data(li) or "").lower()
                rname = str(src.data(ri) or "").lower()
                return lname < rname

        self._proxy = _DirFirstProxy(self)
        self._proxy.setSortCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self._proxy.setSourceModel(self._model)
        self._view.setModel(self._proxy)
        self._view.setSortingEnabled(True)
        self._view.header().setStretchLastSection(True)
        layout.addWidget(self._view, 1)

        # Buttons
        btn_box = QtWidgets.QDialogButtonBox(self)
        self._btn_open = btn_box.addButton("Open", QtWidgets.QDialogButtonBox.AcceptRole)
        self._btn_open.setEnabled(False)
        btn_box.addButton(QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _connect_signals(self) -> None:
        self._view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._btn_back.clicked.connect(self._go_back)
        self._btn_up.clicked.connect(self._go_up)
        self._btn_go.clicked.connect(self._go_to_path)
        self._path_edit.returnPressed.connect(self._go_to_path)

    # ----------------------------- Helpers -------------------------------
    def _clear_model(self) -> None:
        self._model.removeRows(0, self._model.rowCount())

    def _add_row(
        self,
        *,
        name: str,
        path: str,
        is_dir: bool,
        size: int | None = None,
        mtime: float | None = None,
    ) -> None:
        icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_DirIcon if is_dir else QtWidgets.QStyle.SP_FileIcon
        )

        item_name = QtGui.QStandardItem(icon, name)
        item_name.setData(path, QtCore.Qt.UserRole + 1)
        item_name.setData(bool(is_dir), QtCore.Qt.UserRole + 2)
        # Secondary columns (simple strings for now)
        item_type = QtGui.QStandardItem("Folder" if is_dir else "File")
        item_size = QtGui.QStandardItem("" if is_dir or size is None else f"{int(size):,}")
        item_mtime = QtGui.QStandardItem("")
        self._model.appendRow([item_name, item_type, item_size, item_mtime])

    def _select_first_row(self) -> None:
        if self._proxy.rowCount() > 0:
            index = self._proxy.index(0, 0)
            self._view.setCurrentIndex(index)

    def _current_selection(self) -> tuple[str | None, bool]:
        idx = self._view.currentIndex()
        if not idx.isValid():
            return None, False
        src_idx = self._proxy.mapToSource(idx)
        item = self._model.itemFromIndex(src_idx)
        path = str(item.data(QtCore.Qt.UserRole + 1))
        is_dir = bool(item.data(QtCore.Qt.UserRole + 2))
        return path, is_dir

    def _can_open(self, *, is_dir: bool, name: str) -> bool:
        # In files mode: allow opening directories that match suffix (e.g. ".zarr")
        if self._select_folders:
            return is_dir
        lowered = name.lower()
        if is_dir:
            if not self._only:
                return False
            return any(lowered.endswith(sfx.lower()) for sfx in self._only)
        # Regular file path
        if not self._only:
            return True
        return any(lowered.endswith(sfx.lower()) for sfx in self._only)

    # --------------------------- Navigation ------------------------------
    def _push_history(self, path: str | None) -> None:
        if path is None:
            return
        # Trim forward history
        if self._history_index >= 0 and self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]
        self._history.append(path)
        self._history_index = len(self._history) - 1
        self._btn_back.setEnabled(self._history_index > 0)

    def _go_back(self) -> None:
        if self._history_index <= 0:
            return
        self._history_index -= 1
        target = self._history[self._history_index]
        self._btn_back.setEnabled(self._history_index > 0)
        self._request_listdir(target, add_history=False)

    def _go_up(self) -> None:
        path = self._current_path or ""
        if not path:
            return
        # naive parent: server will normalize
        parent = path.rstrip("/")
        parent = parent[: parent.rfind("/")] if "/" in parent else "/"
        if not parent:
            parent = "/"
        self._request_listdir(parent)

    def _go_to_path(self) -> None:
        path = self._path_edit.text().strip() or None
        self._request_listdir(path)

    # ----------------------------- RPC ----------------------------------
    def _request_listdir(self, path: str | None, *, add_history: bool = True) -> None:
        # Guard: loop present
        loop = self._loop
        assert loop is not None, "client loop is required"

        # Attempt to use helper, else fall back to _issue_command
        if hasattr(loop, "list_remote_dir"):
            future = loop.list_remote_dir(path, only=self._only, show_hidden=False)
        else:
            payload: dict[str, object] = {"path": path, "show_hidden": False}
            if self._only is not None:
                payload["only"] = list(self._only)
            future = loop._issue_command("fs.listdir", kwargs=payload, origin="ui")  # type: ignore[attr-defined]

        # UI: indicate loading by disabling controls
        for widget in (self._btn_back, self._btn_up, self._btn_go, self._view, self._path_edit):
            widget.setEnabled(False)
        self._clear_model()

        def _on_done() -> None:
            # Called on the Qt thread
            for widget in (self._btn_back, self._btn_up, self._btn_go, self._view, self._path_edit):
                widget.setEnabled(True)

        def _apply_result(res: Mapping[str, object]) -> None:
            # Populate model
            self._clear_model()
            new_path = str(res.get("path") or path or "")
            self._current_path = new_path
            self._path_edit.setText(new_path)
            entries = res.get("entries")
            rows = entries if isinstance(entries, list) else []
            for entry in rows:  # type: ignore[assignment]
                name = str(entry.get("name") or "")
                full = str(entry.get("path") or name)
                is_dir = bool(entry.get("is_dir") or False)
                if (not is_dir) and self._only:
                    lowered = name.lower()
                    if not any(lowered.endswith(sfx.lower()) for sfx in self._only):
                        continue
                size_val = entry.get("size")
                size_int = None if size_val is None else int(size_val)  # type: ignore[arg-type]
                mtime_val = entry.get("mtime")
                mtime_float = None if mtime_val is None else float(mtime_val)  # type: ignore[arg-type]
                self._add_row(name=name, path=full, is_dir=is_dir, size=size_int, mtime=mtime_float)
            self._select_first_row()
            self._proxy.sort(0)
            # Auto-resize columns for visibility
            self._view.resizeColumnToContents(0)
            if add_history:
                self._push_history(self._current_path)

        def _handle_future_done(fut) -> None:  # type: ignore[no-untyped-def]
            payload = fut.result()
            result = getattr(payload, "result", None)
            assert isinstance(result, Mapping), "invalid listdir result"
            self._listing_ready.emit(result)

        if future is not None:
            future.add_done_callback(_handle_future_done)

    # Invoked on the Qt thread when a listdir result is ready
    def _on_listing_ready(self, res_obj: object) -> None:
        assert isinstance(res_obj, Mapping), "invalid listdir payload"
        res: Mapping[str, object] = res_obj
        # Re-enable controls
        for widget in (self._btn_back, self._btn_up, self._btn_go, self._view, self._path_edit):
            widget.setEnabled(True)

        # Populate model
        self._clear_model()
        new_path = str(res.get("path") or self._current_path or "")
        self._current_path = new_path
        self._path_edit.setText(new_path)
        entries = res.get("entries")
        rows = entries if isinstance(entries, list) else []
        for entry in rows:  # type: ignore[assignment]
            name = str(entry.get("name") or "")  # type: ignore[index]
            full = str(entry.get("path") or name)  # type: ignore[index]
            is_dir = bool(entry.get("is_dir") or False)  # type: ignore[index]
            if (not is_dir) and self._only:
                lowered = name.lower()
                if not any(lowered.endswith(sfx.lower()) for sfx in self._only):
                    continue
            size_val = entry.get("size")  # type: ignore[index]
            size_int = None if size_val is None else int(size_val)  # type: ignore[arg-type]
            mtime_val = entry.get("mtime")  # type: ignore[index]
            mtime_float = None if mtime_val is None else float(mtime_val)  # type: ignore[arg-type]
            self._add_row(name=name, path=full, is_dir=is_dir, size=size_int, mtime=mtime_float)
        self._select_first_row()
        self._proxy.sort(0)
        self._view.resizeColumnToContents(0)

    # ---------------------------- Slots ----------------------------------
    def _on_selection_changed(self) -> None:
        path, is_dir = self._current_selection()
        if path is None:
            self._btn_open.setEnabled(False)
            return
        # Enable Open when selection satisfies mode
        name = self._model.item(self._view.currentIndex().row(), 0).text()
        self._btn_open.setEnabled(self._can_open(is_dir=is_dir, name=name))

    def _on_double_clicked(self, index: QtCore.QModelIndex) -> None:
        src_idx = self._proxy.mapToSource(index)
        item = self._model.itemFromIndex(src_idx)
        path = str(item.data(QtCore.Qt.UserRole + 1))
        is_dir = bool(item.data(QtCore.Qt.UserRole + 2))
        name = item.text()
        if is_dir and self._can_open(is_dir=True, name=name):
            # Treat suffix-matching directories (e.g. ".zarr") as openable datasets
            self._selected = path
            self.accept()
            return
        if is_dir:
            self._request_listdir(path)
            return
        if self._can_open(is_dir=False, name=name):
            self._selected = path
            self.accept()

    def _on_accept(self) -> None:
        path, is_dir = self._current_selection()
        if path is None:
            return
        name = self._model.item(self._view.currentIndex().row(), 0).text()
        if not self._can_open(is_dir=is_dir, name=name):
            return
        self._selected = path
        self.accept()

    # --------------------------- Public API ------------------------------
    def selected_path(self) -> str | None:
        return self._selected


__all__ = ["RemoteFileDialog"]
