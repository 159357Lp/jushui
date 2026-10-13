"""Download center page — HuggingFace search and download."""

import json
import threading
import httpx
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QProgressBar, QHeaderView, QFrame,
    QTabWidget,
)
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QColor

from lmloader.gui.api_client import run_in_thread
from lmloader.gui.i18n import tr
from lmloader.gui.styles import c


class _OllamaSearchWorker(QObject):
    """Search the Ollama library (or list one model's tags) via the backend."""
    finished = Signal(object, object)   # (models | None, tags | None)
    failed = Signal(str)

    def __init__(self, base_url: str, query: str = "", model: str = ""):
        super().__init__()
        self._base = base_url.rstrip("/")
        self._query = query
        self._model = model

    def run(self):
        try:
            if self._model:
                path = "/api/ollama/tags"
                params = {"model": self._model}
            else:
                path = "/api/ollama/search"
                params = {"q": self._query}
            with httpx.Client(timeout=httpx.Timeout(25.0)) as client:
                resp = client.get(f"{self._base}{path}", params=params)
                resp.raise_for_status()
                data = resp.json()
            if self._model:
                self.finished.emit(None, data.get("tags", []))
            else:
                self.finished.emit(data.get("models", []), None)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _SearchWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, query: str, hf_mirror: str):
        super().__init__()
        self._query = query
        self._hf_mirror = hf_mirror

    def run(self):
        try:
            url = f"{self._hf_mirror}/api/models"
            resp = httpx.get(url, params={"search": self._query, "limit": 20, "sort": "downloads", "direction": -1}, timeout=15.0)
            resp.raise_for_status()
            self.finished.emit(resp.json())
        except Exception as e:
            self.error.emit(str(e))


class _RepoSizeWorker(QObject):
    """Fetch each repo's file tree; emit one GGUF file row per quantization."""
    finished = Signal(object)  # [(search_index, repo_id, files | None)]
    failed = Signal(str)

    def __init__(self, repos: list[tuple[int, str]], hf_mirror: str, seq: int):
        super().__init__()
        self._repos = repos          # [(search_index, repo_id)]
        self._mirror = hf_mirror
        self._seq = seq

    def _fetch_one(self, item: tuple[int, str]):
        """Returns (index, repo_id, files) — files is None when the request failed."""
        idx, repo_id = item
        files: list[tuple[str, int]] = []
        try:
            url = f"{self._mirror}/api/models/{repo_id}/tree/main"
            resp = httpx.get(url, params={"recursive": "true"}, timeout=12.0)
            resp.raise_for_status()
            data = resp.json()
            entries = data if isinstance(data, list) else []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") == "file" and entry.get("path", "").lower().endswith(".gguf"):
                    files.append((entry["path"], int(entry.get("size") or 0)))
        except Exception:
            return idx, repo_id, None
        return idx, repo_id, files

    def run(self):
        try:
            with ThreadPoolExecutor(max_workers=5) as pool:
                results = list(pool.map(self._fetch_one, self._repos))
            results.sort(key=lambda x: x[0])
            self.finished.emit(results)
        except Exception as e:
            self.failed.emit(str(e))


class _DownloadWorker(QObject):
    """Downloads via /api/pull and emits progress/result signals."""
    progress_updated = Signal(int, str)   # (percent, detail)
    finished_ok = Signal(str)             # path
    failed = Signal(str)                  # error message
    cancelled = Signal()                  # user pressed Stop

    def __init__(self, api_base_url: str, repo_or_spec: str,
                 endpoint: str = "/api/pull", extra_params: dict | None = None):
        super().__init__()
        self._url = f"{api_base_url}{endpoint}"
        self._spec = repo_or_spec
        self._params = extra_params or {}
        self._stop = threading.Event()
        # Kept so stop_download() can close the live response and unblock the
        # blocking readline() instead of waiting for the next progress event.
        self._resp = None
        self._client = None

    def stop(self):
        """Request cancellation. Safe to call from the GUI thread."""
        self._stop.set()
        resp = self._resp
        if resp is not None:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def run(self):
        try:
            with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
                self._client = client
                params = {"name": self._spec}
                params.update(self._params)
                with client.stream("POST", self._url, params=params) as resp:
                    self._resp = resp
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if self._stop.is_set():
                            self.cancelled.emit()
                            return
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        status = event.get("status", "")
                        if status == "downloading":
                            progress = event.get("progress", 0)
                            speed = event.get("speed", 0)
                            speed_mb = speed / (1024 * 1024) if speed else 0
                            self.progress_updated.emit(
                                int(progress),
                                f"{progress}% — {speed_mb:.1f} MB/s",
                            )
                        elif status == "completed":
                            self.finished_ok.emit(event.get("path", ""))
                            return
                        elif status == "cancelled":
                            self.cancelled.emit()
                            return
                        elif status == "error":
                            self.failed.emit(event.get("error", "unknown"))
                            return
        except Exception as e:  # noqa: BLE001
            # Closing the response mid-stream surfaces as a transport error;
            # that is the expected outcome of a Stop, not a real failure.
            if self._stop.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(e))
        finally:
            self._resp = None
            self._client = None


class DownloadPage(QWidget):
    def __init__(self, api):
        super().__init__()
        self.api = api
        self._threads: list[QThread] = []
        self._dl_widgets: dict = {}
        # worker -> reached a terminal state (completed / failed / cancelled).
        # Kept apart from worker.stopped, which only means "user pressed Stop".
        self._finished_workers: dict = {}
        self._hf_mirror = "https://hf-mirror.com"
        self._search_seq = 0
        self._search_results: list = []
        # row -> {"repo": id, "file": path | None}  (None = repo-level fallback)
        self._row_repos: dict[int, dict] = {}
        # Single source of truth for the Ollama table; rows are always
        # re-derived from it (see _render_ollama_rows).
        self._ollama_models_cache: list[dict] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        title = QLabel(tr("download.title"))
        title.setObjectName("title")
        layout.addWidget(title)

        # ---- manual-fetch card: HF repo search + Ollama pull ------------
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(10)

        card_title = QLabel(tr("download.manual_title"))
        card_title.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {c('title')};")
        card_layout.addWidget(card_title)

        def _field_row(label_text: str, placeholder: str, *buttons) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setSpacing(8)
            label = QLabel(label_text)
            label.setFixedWidth(104)
            label.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
            row.addWidget(label)
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            row.addWidget(edit, 1)
            for btn in buttons:
                row.addWidget(btn)
            return row, edit

        search_btn = QPushButton(tr("download.search"))
        search_btn.setObjectName("primary")
        search_btn.clicked.connect(self._search)

        direct_btn = QPushButton(tr("download.direct_download"))
        direct_btn.clicked.connect(self._direct_download)

        # ---- HuggingFace repo row ---------------------------------------
        search_row, self.search_input = _field_row(
            tr("download.hf_label"), tr("download.search_placeholder"),
            search_btn, direct_btn,
        )
        card_layout.addLayout(search_row)

        # ---- Ollama registry row ----------------------------------------
        ollama_btn = QPushButton(tr("download.ollama_pull"))
        ollama_btn.setObjectName("primary")
        ollama_btn.clicked.connect(self._ollama_pull_builtin)

        ollama_cli_btn = QPushButton(tr("download.ollama_cli"))
        ollama_cli_btn.setToolTip(tr("download.ollama_cli_tip"))
        ollama_cli_btn.clicked.connect(self._ollama_pull_cli)

        ollama_browse_btn = QPushButton(tr("download.ollama_browse"))
        ollama_browse_btn.setToolTip(tr("download.ollama_browse_tip"))
        ollama_browse_btn.clicked.connect(self._ollama_search)

        ollama_row, self.ollama_input = _field_row(
            tr("download.ollama_label"), tr("download.ollama_placeholder"),
            ollama_browse_btn, ollama_btn, ollama_cli_btn,
        )
        card_layout.addLayout(ollama_row)

        layout.addWidget(card)

        # Two result tables behind a switcher: HuggingFace repos and the
        # Ollama library are different shapes (repo/author vs tags/context),
        # so they get separate column layouts instead of one overloaded grid.
        self.tabs = QTabWidget()
        self.tabs.setObjectName("tabs")

        hf_tab = QWidget()
        hf_lay = QVBoxLayout(hf_tab)
        hf_lay.setContentsMargins(0, 8, 0, 0)
        hf_lay.addWidget(self._build_hf_table())
        self.tabs.addTab(hf_tab, "HuggingFace")

        ollama_tab = QWidget()
        ollama_lay = QVBoxLayout(ollama_tab)
        ollama_lay.setContentsMargins(0, 8, 0, 0)
        ollama_lay.addWidget(self._build_ollama_table())
        self.tabs.addTab(ollama_tab, "Ollama")

        layout.addWidget(self.tabs, 1)

        self.progress_area = QVBoxLayout()
        layout.addLayout(self.progress_area)

        # Bulk housekeeping: individual rows clear themselves, this clears
        # every finished row at once.
        self.clear_btn = QPushButton(tr("download.clear_all_done"))
        self.clear_btn.setToolTip(tr("download.clear_all_done_tip"))
        self.clear_btn.clicked.connect(self.clear_finished_tasks)
        self.clear_btn_row = QHBoxLayout()
        self.clear_btn_row.setContentsMargins(0, 0, 0, 0)
        self.clear_btn_row.addStretch(1)
        self.clear_btn_row.addWidget(self.clear_btn, 0)
        layout.addLayout(self.clear_btn_row)

        self.status_label = QLabel(tr("download.status_default"))
        self.status_label.setObjectName("subtitle")
        layout.addWidget(self.status_label)

    def _build_hf_table(self) -> QTableWidget:
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            tr("download.col_model"), tr("download.col_author"), tr("download.col_size"),
            tr("download.col_downloads"), tr("download.col_action"),
        ])
        hdr = self.results_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)      # model fills space
        hdr.setSectionResizeMode(1, QHeaderView.Fixed)        # author
        hdr.setSectionResizeMode(2, QHeaderView.Fixed)        # size
        hdr.setSectionResizeMode(3, QHeaderView.Fixed)        # downloads
        hdr.setSectionResizeMode(4, QHeaderView.Fixed)        # action
        self.results_table.setColumnWidth(1, 150)
        self.results_table.setColumnWidth(2, 110)
        self.results_table.setColumnWidth(3, 100)
        self.results_table.setColumnWidth(4, 84)
        self.results_table.setWordWrap(False)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.verticalHeader().setVisible(False)
        # Lightweight action column: a clickable text item instead of one
        # QPushButton widget per row (hundreds of widgets froze the UI).
        self.results_table.cellClicked.connect(self._on_cell_clicked)
        self.results_table.setMouseTracking(True)
        self.results_table.cellEntered.connect(self._on_cell_entered)
        return self.results_table

    def _build_ollama_table(self) -> QTableWidget:
        """Model / tag table. Rows carry their kind in Qt.UserRole+1 so one
        click handler can tell a model row (expand tags) from a tag row
        (pull it) without a parallel data structure."""
        self.ollama_table = QTableWidget()
        self.ollama_table.setColumnCount(5)
        self.ollama_table.setHorizontalHeaderLabels([
            tr("download.ollama_col_model"), tr("download.ollama_col_info"),
            tr("download.col_size"), tr("download.ollama_col_stat"),
            tr("download.col_action"),
        ])
        hdr = self.ollama_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for col, width in ((1, 150), (2, 110), (3, 100), (4, 84)):
            hdr.setSectionResizeMode(col, QHeaderView.Fixed)
            self.ollama_table.setColumnWidth(col, width)
        self.ollama_table.setWordWrap(False)
        self.ollama_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ollama_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.ollama_table.verticalHeader().setVisible(False)
        self.ollama_table.setMouseTracking(True)
        self.ollama_table.cellClicked.connect(self._on_ollama_clicked)
        self.ollama_table.cellDoubleClicked.connect(self._on_ollama_double_clicked)
        self.ollama_table.cellEntered.connect(self._on_ollama_entered)
        return self.ollama_table

    def _search(self):  # FIXED: G1
        query = self.search_input.text().strip()
        if not query:
            return

        if "/" in query and ":" in query.split("/")[-1]:
            self._start_download(query)
            return

        self.status_label.setText(tr("download.searching").format(query=query))

        worker = _SearchWorker(query, self._hf_mirror)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_search_results)
        worker.error.connect(self._on_search_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread._worker = worker
        thread.start()
        self._threads.append(thread)

    def _on_search_results(self, results: list):
        self._search_seq += 1
        self._search_results = results
        # row -> {"repo": id, "file": path | None}  (None = repo-level fallback)
        self._row_repos = {}

        # Placeholder rows while file trees are being fetched.
        self.results_table.setRowCount(len(results))
        repos: list[tuple[int, str]] = []
        for row, model in enumerate(results):
            model_id = model.get("id", "")
            parts = model_id.split("/")
            name = parts[-1] if len(parts) > 1 else model_id
            author = parts[0] if len(parts) > 1 else ""

            name_item = QTableWidgetItem(name)
            name_item.setToolTip(model_id)
            self.results_table.setItem(row, 0, name_item)
            self.results_table.setItem(row, 1, QTableWidgetItem(author))
            self.results_table.setItem(row, 2, QTableWidgetItem(tr("download.size_loading")))

            downloads = model.get("downloads", 0)
            dl_item = QTableWidgetItem(f"{downloads:,}")
            dl_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.results_table.setItem(row, 3, dl_item)

            self.results_table.setItem(row, 4, self._action_item(enabled=True))
            self._row_repos[row] = {"repo": model_id, "file": None}

            repos.append((row, model_id))

        self.status_label.setText(tr("download.found").format(count=len(results)))

        # Fetch GGUF file lists, then expand one row per GGUF file.
        worker = _RepoSizeWorker(repos, self._hf_mirror, self._search_seq)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda payload, s=self._search_seq: self._on_repo_sizes(s, payload))
        worker.failed.connect(self._on_search_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread._worker = worker
        thread.start()
        self._threads.append(thread)

    def _on_repo_sizes(self, seq: int, payload: list):
        if seq != self._search_seq:
            return
        files_by_idx = {idx: files for idx, _repo, files in payload}

        # Expand: one table row per GGUF file; repos without GGUF keep one row.
        # entry: (model_text, author, size_text, downloads|None, file_path|None, repo_id)
        expanded: list[tuple[str, str, str, int | None, str | None, str]] = []
        for idx, model in enumerate(self._search_results):
            model_id = model.get("id", "")
            parts = model_id.split("/")
            repo_name = parts[-1] if len(parts) > 1 else model_id
            author = parts[0] if len(parts) > 1 else ""
            downloads = model.get("downloads", 0)
            files = files_by_idx.get(idx)

            if files:
                first = True
                for path, size in sorted(files, key=lambda x: x[1]):
                    expanded.append((
                        path.split("/")[-1],
                        author if first else "",
                        self._fmt_file_size(size),
                        downloads if first else None,
                        path,
                        model_id,
                    ))
                    first = False
            else:
                # [] = genuinely no GGUF; None = fetch failed (allow fallback)
                size_text = tr("download.size_no_gguf") if files == [] else "-"
                expanded.append((repo_name, author, size_text, downloads, None, model_id))

        self._row_repos = {}
        table = self.results_table
        table.setUpdatesEnabled(False)
        table.setRowCount(len(expanded))
        n_files = 0
        for r, (text, author, size_text, downloads, file_path, repo_id) in enumerate(expanded):
            name_item = QTableWidgetItem(text)
            name_item.setToolTip(f"{repo_id}/{file_path}" if file_path else repo_id)
            table.setItem(r, 0, name_item)
            table.setItem(r, 1, QTableWidgetItem(author))

            size_item = QTableWidgetItem(size_text)
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(r, 2, size_item)

            if downloads is None:
                table.setItem(r, 3, QTableWidgetItem(""))
            else:
                dl_item = QTableWidgetItem(f"{downloads:,}")
                dl_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(r, 3, dl_item)

            self._row_repos[r] = {"repo": repo_id, "file": file_path}
            if file_path is not None:
                n_files += 1

            no_gguf = size_text == tr("download.size_no_gguf")
            table.setItem(r, 4, self._action_item(enabled=not no_gguf))

        table.setUpdatesEnabled(True)

        self.status_label.setText(
            tr("download.found_files").format(repos=len(self._search_results), files=n_files)
            if n_files else tr("download.found").format(count=len(self._search_results))
        )

    def _action_item(self, enabled: bool) -> QTableWidgetItem:
        item = QTableWidgetItem(tr("download.download") if enabled else "—")
        item.setTextAlignment(Qt.AlignCenter)
        if enabled:
            item.setForeground(QColor(c("green")))
            f = item.font()
            f.setBold(True)
            item.setFont(f)
            item.setData(Qt.UserRole, "download")
        else:
            item.setForeground(QColor(c("muted")))
        return item

    def _on_cell_clicked(self, row: int, col: int):
        if col != 4:
            return
        item = self.results_table.item(row, 4)
        if item is None or item.data(Qt.UserRole) != "download":
            return
        self._download_row(row)

    def _on_cell_entered(self, row: int, col: int):
        if col != 4:
            self.results_table.unsetCursor()
            return
        item = self.results_table.item(row, 4)
        if item is not None and item.data(Qt.UserRole) == "download":
            self.results_table.setCursor(Qt.PointingHandCursor)
        else:
            self.results_table.unsetCursor()

    def _fmt_file_size(self, size_bytes: int) -> str:
        if size_bytes >= 1024**3:
            return f"{size_bytes / 1024**3:.2f} GB"
        return f"{size_bytes / 1024**2:.0f} MB"

    def _download_row(self, row: int):
        info = self._row_repos.get(row)
        if not info:
            return
        repo_id = info["repo"]
        file_path = info["file"]
        if file_path is None:
            # Repo-level row (fetch failed): let the backend auto-resolve.
            self._start_download(repo_id)
            return
        url = f"{self._hf_mirror}/{repo_id}/resolve/main/{file_path}"
        self._start_download(url)

    def _on_search_error(self, err: str):
        self.status_label.setText(tr("download.search_failed").format(err=err))

    def _direct_download(self):
        spec = self.search_input.text().strip()
        if spec:
            self._start_download(spec)

    def _ollama_pull_builtin(self):
        spec = self.ollama_input.text().strip()
        if spec:
            self._start_download(spec, endpoint="/api/ollama/pull")

    def _ollama_pull_cli(self):
        spec = self.ollama_input.text().strip()
        if spec:
            self._start_download(spec, endpoint="/api/ollama/pull", extra_params={"mode": "cli"})

    # --- Ollama library browsing -----------------------------------------

    # UserRole holds the action ("expand" | "pull"), UserRole+1 the row kind
    # ("model" | "tag"). Keeping both on the item means no parallel row->model
    # dict to fall out of sync when rows are inserted or removed.
    _ROW_ACTION = Qt.UserRole
    _ROW_KIND = Qt.UserRole + 1

    def _ollama_search(self):
        query = self.ollama_input.text().strip()
        if not query:
            return
        self.tabs.setCurrentIndex(1)
        self.status_label.setText(tr("download.ollama_searching").format(query=query))
        worker = _OllamaSearchWorker(self.api.base_url, query=query)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda m, t: self._on_ollama_search_done(query, m, t))
        worker.failed.connect(self._on_ollama_search_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread._worker = worker
        worker._thread = thread
        thread.start()
        self._threads.append(thread)

    def _on_ollama_search_done(self, query: str, models, tags):
        if tags is not None:                       # tags response
            self._on_ollama_tags_done(query, tags)
            return
        self._ollama_models_cache = [
            {**m, "tags": [], "expanded": False} for m in models
        ]
        self._render_ollama_rows()
        self.status_label.setText(
            tr("download.ollama_found").format(count=len(models)))

    def _render_ollama_rows(self):
        """Rebuild the table from the model cache.

        The cache is the single source of truth: search replaces it, and
        expanding a model only mutates its 'tags'/'expanded' keys. Rows are
        always derived, so there is no row->model index to desynchronise.
        """
        rows: list[tuple[str, dict]] = []
        for m in self._ollama_models_cache:
            rows.append(("model", m))
            if m.get("expanded"):
                for t in m.get("tags", []):
                    rows.append(("tag", t))

        table = self.ollama_table
        table.setUpdatesEnabled(False)
        table.setRowCount(len(rows))
        for r, (kind, payload) in enumerate(rows):
            if kind == "model":
                name_item = QTableWidgetItem(payload["name"])
                name_item.setData(self._ROW_KIND, "model")
                f = name_item.font()
                f.setBold(True)
                name_item.setFont(f)
                name_item.setToolTip(payload.get("description") or payload["name"])
                table.setItem(r, 0, name_item)

                caps = payload.get("capabilities") or []
                meta = " · ".join(x for x in ([payload.get("params")] + caps) if x)
                table.setItem(r, 1, QTableWidgetItem(meta))

                size_item = QTableWidgetItem("")
                size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(r, 2, size_item)

                dl = payload.get("downloads", 0)
                dl_item = QTableWidgetItem(f"{dl:,}" if dl else "")
                dl_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(r, 3, dl_item)

                action = QTableWidgetItem(
                    tr("download.ollama_collapse") if payload.get("expanded")
                    else tr("download.ollama_expand_short"))
                action.setTextAlignment(Qt.AlignCenter)
                action.setData(self._ROW_ACTION, "toggle")
                action.setData(self._ROW_KIND, "model")
                action.setForeground(QColor(c("accent")))
                table.setItem(r, 4, action)
            else:
                ref = payload["ref"]
                tag_item = QTableWidgetItem(f"    {ref}")
                tag_item.setData(self._ROW_KIND, "tag")
                table.setItem(r, 0, tag_item)

                bits = []
                if payload.get("context"):
                    bits.append(payload["context"])
                if payload.get("modality") == "vision":
                    bits.append("vision")
                table.setItem(r, 1, QTableWidgetItem(" · ".join(bits)))

                size_txt = (self._fmt_file_size(payload["size_bytes"])
                            if payload.get("size_bytes") else "—")
                size_item = QTableWidgetItem(size_txt)
                size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(r, 2, size_item)

                upd = payload.get("updated") or ""
                ctx_item = QTableWidgetItem(upd)
                ctx_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(r, 3, ctx_item)

                action = QTableWidgetItem(tr("download.download"))
                action.setTextAlignment(Qt.AlignCenter)
                action.setData(self._ROW_ACTION, "pull")
                action.setData(self._ROW_KIND, "tag")
                af = action.font()
                af.setBold(True)
                action.setFont(af)
                action.setForeground(QColor(c("green")))
                table.setItem(r, 4, action)
        table.setUpdatesEnabled(True)

    def _on_ollama_search_failed(self, err: str):
        self.status_label.setText(tr("download.ollama_search_failed").format(err=err))

    def _on_ollama_clicked(self, row: int, col: int):
        if col != 4:
            return
        item = self.ollama_table.item(row, 4)
        if item is None:
            return
        action = item.data(self._ROW_ACTION)
        if action == "pull":
            spec_item = self.ollama_table.item(row, 0)
            if spec_item is not None:
                spec = spec_item.text().strip()
                self.ollama_input.setText(spec)
                self._start_download(spec, endpoint="/api/ollama/pull")
        elif action == "toggle":
            name_item = self.ollama_table.item(row, 0)
            if name_item is not None:
                self._toggle_ollama_model(name_item.text())

    def _on_ollama_double_clicked(self, row: int, col: int):
        name_item = self.ollama_table.item(row, 0)
        if name_item is not None and name_item.data(self._ROW_KIND) == "model":
            self._toggle_ollama_model(name_item.text())

    def _on_ollama_entered(self, row: int, col: int):
        if col != 4:
            self.ollama_table.unsetCursor()
            return
        item = self.ollama_table.item(row, 4)
        if item is not None and item.data(self._ROW_ACTION) in ("toggle", "pull"):
            self.ollama_table.setCursor(Qt.PointingHandCursor)
        else:
            self.ollama_table.unsetCursor()

    def _find_cached(self, model: str):
        for m in self._ollama_models_cache:
            if m["name"] == model:
                return m
        return None

    def _toggle_ollama_model(self, model: str):
        """Expand a model (fetch tags on first expand) or collapse it."""
        entry = self._find_cached(model)
        if entry is None:
            return
        if entry.get("expanded"):
            entry["expanded"] = False
            self._render_ollama_rows()
            return
        if entry.get("tags"):
            entry["expanded"] = True
            self._render_ollama_rows()
            return
        self._expand_ollama_model(model)

    def _expand_ollama_model(self, model: str):
        self.status_label.setText(
            tr("download.ollama_tags_loading").format(model=model))
        worker = _OllamaSearchWorker(self.api.base_url, model=model)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(
            lambda _m, t, name=model: self._on_ollama_tags_done(name, t))
        worker.failed.connect(self._on_ollama_search_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread._worker = worker
        worker._thread = thread
        thread.start()
        self._threads.append(thread)

    def _on_ollama_tags_done(self, model: str, tags):
        entry = self._find_cached(model)
        if entry is not None:
            entry["tags"] = tags
            entry["expanded"] = True
        self._render_ollama_rows()
        self.status_label.setText(
            tr("download.ollama_expanded").format(model=model, count=len(tags)))

    def _start_download(self, repo_or_spec: str, endpoint: str = "/api/pull",
                        extra_params: dict | None = None):
        # Each task lives in its own container so it can be removed as one
        # unit later; adding three loose items per task makes precise removal
        # impossible without shifting every following row.
        container = QWidget()
        row = QVBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 6)
        row.setSpacing(4)

        label = QLabel(tr("download.downloading").format(spec=repo_or_spec))
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        detail = QLabel(tr("download.starting"))

        # One button, two roles: Stop while running, Clear once terminal.
        # A finished row that cannot be removed piles up at the bottom of the
        # page forever, which is what the user reported.
        action_btn = QPushButton(tr("download.ollama_stop"))
        action_btn.setObjectName("danger")
        action_btn.setFixedWidth(72)
        action_btn.setToolTip(tr("download.stop_tip"))

        detail_row = QHBoxLayout()
        detail_row.setContentsMargins(0, 0, 0, 0)
        detail_row.addWidget(detail, 1)
        detail_row.addWidget(action_btn, 0, Qt.AlignRight)

        row.addWidget(label)
        row.addWidget(bar)
        row.addLayout(detail_row)

        self.progress_area.addWidget(container)

        worker = _DownloadWorker(self.api.base_url, repo_or_spec, endpoint, extra_params)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        widgets = [label, bar, detail, action_btn, container]
        self._dl_widgets[worker] = widgets
        # Bound-method callbacks run on the main thread (queued connections).
        worker.progress_updated.connect(self._on_dl_progress)
        worker.finished_ok.connect(self._on_dl_done)
        worker.failed.connect(self._on_dl_failed)
        worker.cancelled.connect(self._on_dl_cancelled)

        worker.finished_ok.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)

        action_btn.clicked.connect(
            lambda _=False, w=worker, b=action_btn: self._on_task_action(w, b))

        def _release_thread():  # keep wrappers alive until thread exits (crash fix)
            worker._thread = None
            thread._worker = None

        thread.finished.connect(_release_thread)
        thread.finished.connect(thread.deleteLater)

        # Mutual refs prevent GC from deleting the running C++ QThread.
        worker._thread = thread
        thread._worker = worker
        thread.start()

    def _on_task_action(self, worker, btn: QPushButton):
        """Dispatch on the button's current role.

        The role is read from the button's own text, NOT from worker.stopped:
        `stopped` only means "the user pressed Stop", so a task that finished
        on its own still reports False and a Clear click would fall through
        to the stop branch and do nothing.
        """
        if btn.text() in (tr("download.ollama_stop"), tr("download.stopping")):
            self._stop_download(worker, btn)
        else:
            self._remove_task_row(worker, btn)

    def _remove_task_row(self, worker, btn: QPushButton):
        """Delete a finished task's whole row from the progress area.

        Each task owns one container QWidget holding its label / bar / detail
        / button, so removing the container removes exactly that row. Removing
        the individual layout items instead would either leave invisible gaps
        (removeWidget keeps the QLayoutItem) or, once takeAt() is involved,
        shift every following row.
        """
        widgets = self._dl_widgets.pop(worker, None)
        self._finished_workers.pop(worker, None)
        if not widgets:
            return
        container = widgets[4] if len(widgets) > 4 else None
        if container is None:
            return
        self.progress_area.removeWidget(container)
        container.setParent(None)
        container.deleteLater()

    def clear_finished_tasks(self):
        """Remove every task row that reached a terminal state.

        Tracked by the page: worker.stopped only means "the user pressed
        Stop", so a task that completed or failed on its own must be tracked
        separately or bulk-clear would leave those rows behind.
        """
        for worker in list(self._dl_widgets):
            if self._finished_workers.get(worker):
                self._remove_task_row(worker, None)

    def _stop_download(self, worker, btn: QPushButton):
        """Stop button handler (main thread)."""
        if btn is not None:
            btn.setEnabled(False)
            btn.setText(tr("download.stopping"))
        worker.stop()
        # Tell the backend too, so the in-flight .part is removed and a CLI
        # pull gets its subprocess terminated. Fire-and-forget: the local
        # stream is already closing, this only needs to land eventually.
        mode = worker._params.get("mode", "builtin")
        try:
            httpx.post(
                f"{self.api.base_url}/api/ollama/pull/cancel",
                params={"name": worker._spec, "mode": mode},
                timeout=5.0,
                headers=self.api._headers(),
            )
        except Exception:  # noqa: BLE001
            # The backend may not be reachable (already shutting down) — the
            # local stream close above still stops the UI-side download.
            pass

    def _on_dl_progress(self, progress: int, detail_text: str):  # main thread
        worker = self.sender()
        widgets = self._dl_widgets.get(worker)
        if widgets:
            _, bar, detail = widgets[0], widgets[1], widgets[2]
            bar.setValue(int(progress))
            detail.setText(detail_text)

    def _mark_finished(self, worker, widgets, tone: str) -> None:
        """Terminal state: the button stops being 'Stop' and becomes 'Clear'.

        A failed or cancelled row is exactly as removable as a successful one
        — leaving it enabled-but-dead was the bug the user hit.

        The worker is passed in rather than read from self.sender(): the
        latter is None whenever this is called outside a Qt signal (tests,
        programmatic retries), which would silently skip the bookkeeping.
        """
        if worker is not None:
            self._finished_workers[worker] = True
        btn = widgets[3] if len(widgets) > 3 else None
        if btn is None:
            return
        btn.setText(tr("download.clear_task"))
        btn.setToolTip(tr("download.clear_task_tip"))
        btn.setObjectName("" if tone == "green" else "danger")
        btn.setEnabled(True)

    def _on_dl_done(self, path: str):  # main thread
        worker = self.sender()
        widgets = self._dl_widgets.get(worker)
        if widgets:
            label, bar, detail = widgets[0], widgets[1], widgets[2]
            bar.setValue(100)
            detail.setText(tr("download.completed").format(path=path))
            label.setStyleSheet(f"color: {c('green')};")
            self._mark_finished(worker, widgets, "green")

    def _on_dl_failed(self, err: str):  # main thread
        worker = self.sender()
        widgets = self._dl_widgets.get(worker)
        if widgets:
            label, detail = widgets[0], widgets[2]
            detail.setText(tr("download.error").format(err=err))
            label.setStyleSheet(f"color: {c('red')};")
            self._mark_finished(worker, widgets, "red")

    def _on_dl_cancelled(self):  # main thread
        worker = self.sender()
        widgets = self._dl_widgets.get(worker)
        if widgets:
            label, detail = widgets[0], widgets[2]
            detail.setText(tr("download.cancelled"))
            label.setStyleSheet(f"color: {c('muted')};")
            self._mark_finished(worker, widgets, "muted")

    def on_navigate(self):
        pass
