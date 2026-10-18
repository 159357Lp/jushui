"""Settings page — configure engine paths, ports, defaults."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QCheckBox, QPushButton, QLabel,
    QGroupBox, QFileDialog, QMessageBox, QComboBox, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt, Signal
import yaml
from pathlib import Path

from lmloader.gui.api_client import LMLoaderAPI
from lmloader.gui.i18n import tr, set_language
from lmloader.gui.styles import c
from lmloader.gui.theme_picker import CustomThemeRow


def _get_config_path() -> Path:  # FIXED: G4
    for candidate in [
        Path("config.yaml"),
        Path.home() / ".lmloader" / "config.yaml",
    ]:
        if candidate.exists():
            return candidate
    return Path("config.yaml")


# The two built-in slots always own the first rows of the version table, so their
# widgets survive every rescan and the update/rollback code can keep addressing
# them by attribute name.
_SLOT_ROWS = (("primary", "settings.engine_row_primary"),
              ("ternary", "settings.engine_row_ternary"))


class SettingsPage(QWidget):
    theme_changed = Signal(str)

    def __init__(self, api: LMLoaderAPI):
        super().__init__()
        self.api = api
        self._config_path = _get_config_path()  # FIXED: G4
        self._threads: list = []
        self._custom_actions: dict[str, tuple] = {}
        self._custom_targets: dict[str, dict] = {}
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        title = QLabel(tr("settings.title"))
        title.setObjectName("title")
        layout.addWidget(title)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(14)

        # ① Engines: three equal-width cards, config only. Version/update
        # controls used to be embedded in the two left cards, which made the
        # page a stack of forms of different heights.
        engines = QHBoxLayout()
        engines.setSpacing(12)
        # AlignTop: without it every card is stretched to the tallest one and the
        # two short cards become empty boxes.
        engines.addWidget(self._build_engine_group(tr("settings.primary_engine"), "primary"), 1, Qt.AlignTop)
        engines.addWidget(self._build_engine_group(tr("settings.ternary_engine"), "ternary"), 1, Qt.AlignTop)
        engines.addWidget(self._build_pluggable_group(), 1, Qt.AlignTop)
        scroll_layout.addLayout(engines)

        # ② Versions: one row per engine, slots first, then custom registrations
        # and anything else discovery found.
        scroll_layout.addWidget(self._build_version_group())

        # ③④ Server + appearance side by side.
        row2 = QHBoxLayout()
        row2.setSpacing(12)
        row2.addWidget(self._build_server_group(), 1, Qt.AlignTop)
        row2.addWidget(self._build_appearance_group(), 1, Qt.AlignTop)
        scroll_layout.addLayout(row2)

        # ⑤⑥ Paths + downloads side by side.
        row3 = QHBoxLayout()
        row3.setSpacing(12)
        row3.addWidget(self._build_paths_group(), 1, Qt.AlignTop)
        row3.addWidget(self._build_download_group(), 1, Qt.AlignTop)
        scroll_layout.addLayout(row3)

        scroll_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        save_btn = QPushButton(tr("settings.save"))
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._save_settings)
        btn_bar.addWidget(save_btn)
        reset_btn = QPushButton(tr("settings.reset"))
        reset_btn.clicked.connect(self._load_settings)
        btn_bar.addWidget(reset_btn)
        layout.addLayout(btn_bar)

    def _build_engine_group(self, title: str, slot: str) -> QGroupBox:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        path_edit = QLineEdit()
        path_edit.setPlaceholderText(tr("settings.binary_placeholder"))
        browse_btn = QPushButton(tr("settings.browse"))
        browse_btn.clicked.connect(lambda: self._browse_file(path_edit))

        path_row = QHBoxLayout()
        path_row.addWidget(path_edit, 1)
        path_row.addWidget(browse_btn)
        path_label_key = (
            "settings.ternary_path" if slot == "ternary" else "settings.binary_path"
        )
        form.addRow(tr(path_label_key), path_row)
        setattr(self, f"{slot}_path", path_edit)

        flash_check = QCheckBox(tr("settings.enable_flash"))
        flash_check.setChecked(True)
        form.addRow("", flash_check)
        setattr(self, f"{slot}_flash", flash_check)

        cuda_lbl = QLabel("...")
        cuda_lbl.setObjectName("subtitle")
        cuda_lbl.setWordWrap(True)
        form.addRow(cuda_lbl)
        setattr(self, f"{slot}_cuda_lbl", cuda_lbl)

        # Per-slot startup timeout. Lives on the slot (not on defaults) because
        # that is where LifecycleManager._wait_for_health reads it.
        timeout_spin = QSpinBox()
        timeout_spin.setRange(10, 600)
        timeout_spin.setValue(120)
        timeout_spin.setSuffix("s")
        timeout_spin.setToolTip(tr("settings.startup_timeout_tip"))
        form.addRow(tr("settings.startup_timeout"), timeout_spin)
        setattr(self, f"{slot}_timeout", timeout_spin)

        return group

    def _build_pluggable_group(self) -> QGroupBox:
        """Third engine card: search dirs + custom engine registration."""
        group = QGroupBox(tr("settings.pluggable_engine"))
        form = QFormLayout(group)
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        tip = QLabel(tr("settings.pluggable_tip"))
        tip.setObjectName("subtitle")
        tip.setWordWrap(True)
        form.addRow(tip)

        self.scan_dirs_edit = QLineEdit()
        self.scan_dirs_edit.setPlaceholderText("D:/engines;D:/rwkv")
        self.save_dirs_btn = QPushButton(tr("settings.save_dirs"))
        self.save_dirs_btn.clicked.connect(self._save_scan_dirs)
        dirs_row = QHBoxLayout()
        dirs_row.addWidget(self.scan_dirs_edit, 1)
        dirs_row.addWidget(self.save_dirs_btn)
        form.addRow(tr("settings.engine_scan_dirs"), dirs_row)

        self.rescan_btn = QPushButton(tr("settings.rescan"))
        self.rescan_btn.clicked.connect(self._refresh_engines)
        rescan_row = QHBoxLayout()
        rescan_row.addWidget(self.rescan_btn)
        rescan_row.addStretch()
        form.addRow("", rescan_row)

        self.custom_name_edit = QLineEdit()
        self.custom_name_edit.setPlaceholderText(tr("settings.engine_name_ph"))
        form.addRow(tr("settings.engine_name"), self.custom_name_edit)

        self.custom_kind_combo = QComboBox()
        form.addRow(tr("settings.engine_type"), self.custom_kind_combo)

        self.custom_path_edit = QLineEdit()
        self.custom_path_edit.setPlaceholderText("D:/engines/rwkv-server.exe")
        browse_btn = QPushButton(tr("settings.browse"))
        browse_btn.clicked.connect(self._browse_custom_engine)
        path_row = QHBoxLayout()
        path_row.addWidget(self.custom_path_edit, 1)
        path_row.addWidget(browse_btn)
        form.addRow(tr("settings.engine_path"), path_row)

        self.add_engine_btn = QPushButton(tr("settings.add_engine"))
        self.add_engine_btn.setObjectName("primary")
        self.add_engine_btn.clicked.connect(self._add_custom_engine)
        add_row = QHBoxLayout()
        add_row.addWidget(self.add_engine_btn)
        add_row.addStretch()
        form.addRow("", add_row)

        return group

    def _build_version_group(self) -> QGroupBox:
        """One row per engine: version, what it can run, and its update controls."""
        group = QGroupBox(tr("settings.section_versions"))
        vbox = QVBoxLayout(group)
        vbox.setSpacing(8)

        tip = QLabel(tr("settings.versions_tip"))
        tip.setObjectName("subtitle")
        tip.setWordWrap(True)
        vbox.addWidget(tip)

        table = QTableWidget(len(_SLOT_ROWS), 6)
        table.setHorizontalHeaderLabels([
            tr("settings.col_engine"), tr("settings.col_kind"), tr("settings.col_path"),
            tr("settings.engine_formats"), tr("settings.col_version"), tr("settings.col_action"),
        ])
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Interactive)
        header.setSectionResizeMode(5, QHeaderView.Interactive)
        table.verticalHeader().setVisible(False)
        # Cell widgets (multi-line version labels, button rows) do not drive row
        # height, so the default has to leave room for them.
        table.verticalHeader().setDefaultSectionSize(72)
        table.setColumnWidth(4, 300)
        table.setColumnWidth(5, 330)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setMaximumHeight(460)
        self.version_table = table
        vbox.addWidget(table)

        for row, (slot, name_key) in enumerate(_SLOT_ROWS):
            self._set_cell(row, 0, tr(name_key))
            self._set_cell(row, 1, "llama")

            ver_lbl = QLabel("...")
            ver_lbl.setWordWrap(True)
            ver_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            latest_lbl = QLabel("-")
            latest_lbl.setObjectName("subtitle")
            latest_lbl.setWordWrap(True)
            setattr(self, f"{slot}_ver_lbl", ver_lbl)
            setattr(self, f"{slot}_latest_lbl", latest_lbl)
            cell = QWidget()
            cell_box = QVBoxLayout(cell)
            cell_box.setContentsMargins(4, 2, 4, 2)
            cell_box.setSpacing(2)
            cell_box.addWidget(ver_lbl)
            cell_box.addWidget(latest_lbl)
            table.setCellWidget(row, 4, cell)

            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(2, 2, 2, 2)
            btn_row.setSpacing(6)
            check_btn = QPushButton(tr("settings.check_update"))
            check_btn.clicked.connect(lambda _=False, s=slot: self._check_engine_update(s))
            setattr(self, f"{slot}_check_btn", check_btn)
            btn_row.addWidget(check_btn)

            upd_btn = QPushButton(tr("settings.update_engine"))
            upd_btn.setObjectName("primary")
            upd_btn.setEnabled(False)
            upd_btn.clicked.connect(lambda _=False, s=slot: self._update_engine(s))
            setattr(self, f"{slot}_update_btn", upd_btn)
            btn_row.addWidget(upd_btn)

            rollback_btn = QPushButton(tr("settings.rollback_engine"))
            rollback_btn.setEnabled(False)
            rollback_btn.setToolTip(tr("settings.rollback_tip"))
            rollback_btn.clicked.connect(lambda _=False, s=slot: self._rollback_engine(s))
            setattr(self, f"{slot}_rollback_btn", rollback_btn)
            btn_row.addWidget(rollback_btn)
            cell = QWidget()
            cell.setLayout(btn_row)
            table.setCellWidget(row, 5, cell)

        return group

    def _set_cell(self, row: int, col: int, text: str):
        item = QTableWidgetItem(text)
        item.setToolTip(text)
        self.version_table.setItem(row, col, item)

    def _reflow_rows(self):
        """Grow rows to whatever the version cell actually wraps into.

        Two Qt quirks stack up here: a row never expands because a cell widget
        got taller, and QTableView hands an index widget a rect a few pixels
        shorter than the row — that gap is measured off Qt instead of guessed.
        """
        table = self.version_table
        chrome = 0
        for row in range(table.rowCount()):
            for col in (4, 5):
                cell = table.cellWidget(row, col)
                if cell is not None and cell.height() > 0:
                    chrome = max(chrome, table.rowHeight(row) - cell.height())
        for row in range(table.rowCount()):
            height = table.verticalHeader().defaultSectionSize()
            for col in (4, 5):
                cell = table.cellWidget(row, col)
                box = cell.layout() if cell is not None else None
                if box is None:
                    continue
                margins = box.contentsMargins()
                inner_w = max(60, table.columnWidth(col) - margins.left() - margins.right() - 8)
                labels = cell.findChildren(QLabel)
                if labels:
                    needed = margins.top() + margins.bottom()
                    needed += box.spacing() * (len(labels) - 1)
                    for lbl in labels:
                        lbl.setFixedWidth(inner_w)
                        # QLabel caches its hint, so heightForWidth is asked before
                        # the new text has been laid out. QFontMetrics at the width
                        # the label will really get is the only number that agrees
                        # with what ends up on screen; +8 covers the document margin.
                        needed += lbl.fontMetrics().boundingRect(
                            0, 0, inner_w, 10000, Qt.TextWordWrap,
                            lbl.text()).height() + 8
                else:
                    needed = box.totalSizeHint().height()
                height = max(height, needed + chrome)
            table.setRowHeight(row, height)

    def _check_engine_update(self, slot: str):
        from lmloader.gui.api_client import run_in_thread

        check_btn = getattr(self, f"{slot}_check_btn")
        upd_btn = getattr(self, f"{slot}_update_btn")
        latest_lbl = getattr(self, f"{slot}_latest_lbl")
        check_btn.setEnabled(False)
        check_btn.setText(tr("settings.checking"))

        def fetch():
            return self.api.check_engine_update(slot)

        def on_result(data):
            check_btn.setEnabled(True)
            check_btn.setText(tr("settings.check_update"))
            ver = data.get("local", {})
            latest = data.get("latest", {})
            getattr(self, f"{slot}_ver_lbl").setText(ver.get("raw", "unknown"))
            if data.get("has_update"):
                size_mb = (latest.get("size") or 0) // 1048576
                latest_lbl.setText(
                    f'{latest.get("tag", "?")} · {latest.get("asset_name", "")}'
                    + (f" · {size_mb} MB" if size_mb else "")
                )
                upd_btn.setEnabled(True)
            else:
                latest_lbl.setText(tr("settings.no_update"))
                upd_btn.setEnabled(False)
            self._reflow_rows()

        def on_error(err):
            check_btn.setEnabled(True)
            check_btn.setText(tr("settings.check_update"))
            latest_lbl.setText(tr("settings.update_error").format(err=err))
            self._reflow_rows()

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _update_engine(self, slot: str):
        from lmloader.gui.api_client import run_in_thread

        reply = QMessageBox.question(
            self, tr("settings.update_confirm_title"), tr("settings.update_confirm_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        upd_btn = getattr(self, f"{slot}_update_btn")
        upd_btn.setEnabled(False)
        upd_btn.setText(tr("settings.updating"))

        def fetch():
            return self.api.update_engine(slot)

        def on_result(data):
            upd_btn.setText(tr("settings.update_engine"))
            upd_btn.setEnabled(False)
            ver = data.get("local", {})
            getattr(self, f"{slot}_ver_lbl").setText(ver.get("raw", "unknown"))
            getattr(self, f"{slot}_latest_lbl").setText(
                tr("settings.updated_ok").format(tag=data.get("result", {}).get("tag", ""))
            )
            self._reflow_rows()
            # path field may have changed (slot moved into its own directory)
            installed = data.get("result", {}).get("installed")
            if installed:
                path_edit = getattr(self, f"{slot}_path", None)
                if path_edit is not None:
                    path_edit.setText(installed)
                self._set_cell(self._slot_row(slot), 2, installed)
            QMessageBox.information(self, tr("settings.updated_title"), tr("settings.updated_msg"))

        def on_error(err):
            upd_btn.setText(tr("settings.update_engine"))
            upd_btn.setEnabled(True)
            QMessageBox.critical(self, tr("settings.error_title"), tr("settings.update_error").format(err=err))

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _rollback_engine(self, slot: str):
        from lmloader.gui.api_client import run_in_thread

        reply = QMessageBox.question(
            self, tr("settings.rollback_confirm_title"),
            tr("settings.rollback_confirm_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        rb_btn = getattr(self, f"{slot}_rollback_btn")
        rb_btn.setEnabled(False)
        rb_btn.setText(tr("settings.rolling_back"))

        def fetch():
            return self.api.rollback_engine(slot)

        def on_result(data):
            rb_btn.setText(tr("settings.rollback_engine"))
            rb_btn.setEnabled(True)
            # Refresh version label.
            ver_lbl = getattr(self, f"{slot}_ver_lbl", None)
            if ver_lbl is not None:
                ver_lbl.setText(data.get("to", "unknown"))
            # Path may have changed.
            path_edit = getattr(self, f"{slot}_path", None)
            if path_edit is not None:
                path_edit.setText(str(Path(data.get("backup_dir", "")).parent / "llama-server.exe")
                                  if data.get("backup_dir") else path_edit.text())
            QMessageBox.information(
                self, tr("settings.rolled_back_title"),
                tr("settings.rolled_back_msg").format(
                    frm=data.get("from", "?"), to=data.get("to", "?"),
                ),
            )
            # Re-check status so rollback button reflects remaining backups.
            self._load_engine_status(slot)

        def on_error(err):
            rb_btn.setText(tr("settings.rollback_engine"))
            rb_btn.setEnabled(True)
            QMessageBox.critical(self, tr("settings.error_title"), str(err))

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def on_navigate(self):
        self._load_engine_status("primary")
        self._load_engine_status("ternary")
        self._load_download_config()
        self._refresh_engines()

    def _load_engine_status(self, slot: str):
        from lmloader.gui.api_client import run_in_thread

        def fetch():
            return self.api.engine_update_status(slot)

        def on_result(data):
            ver = data.get("local", {})
            lbl = getattr(self, f"{slot}_ver_lbl", None)
            if lbl is not None:
                if data.get("installed"):
                    lbl.setText(ver.get("raw", "unknown"))
                else:
                    fb = data.get("fallback")
                    if fb:
                        fb_ver = (fb.get("local") or {}).get("raw", "unknown")
                        lbl.setText(tr("settings.engine_fallback").format(ver=fb_ver))
                    else:
                        lbl.setText(tr("settings.engine_not_installed"))
            latest_lbl = getattr(self, f"{slot}_latest_lbl", None)
            if latest_lbl is not None and not data.get("bundled"):
                latest_lbl.setText(tr("settings.engine_not_bundled"))

            # Enable rollback button if backups exist.
            rb_btn = getattr(self, f"{slot}_rollback_btn", None)
            if rb_btn is not None:
                backups = data.get("backups", [])
                rb_btn.setEnabled(bool(backups))
                if backups:
                    rb_btn.setToolTip(tr("settings.rollback_tip").format(ver=backups[0].get("version", "?")))
                else:
                    rb_btn.setToolTip(tr("settings.rollback_no_backup"))

            # Show CUDA runtime status.
            cuda_lbl = getattr(self, f"{slot}_cuda_lbl", None)
            if cuda_lbl is not None:
                if data.get("cuda_runtime_available"):
                    cuda_lbl.setText(tr("settings.cuda_ok"))
                else:
                    cuda_lbl.setText(tr("settings.cuda_missing"))
            self._reflow_rows()

        def on_error(_err):
            lbl = getattr(self, f"{slot}_ver_lbl", None)
            if lbl is not None:
                lbl.setText("...")

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _refresh_engines(self):
        """Re-read discovery + registrations and rebuild the non-slot rows."""
        from lmloader.gui.api_client import run_in_thread

        self.rescan_btn.setEnabled(False)
        self.rescan_btn.setText(tr("settings.scanning"))

        def fetch():
            return self.api.engine_discover(), self.api.list_custom_engines(), self.api.engine_kinds()

        def on_result(data):
            discovered, customs, kinds = data
            self.rescan_btn.setEnabled(True)
            self.rescan_btn.setText(tr("settings.rescan"))
            self.scan_dirs_edit.setText(";".join(discovered.get("scan_dirs", [])))
            if kinds and self.custom_kind_combo.count() == 0:
                for k in kinds:
                    self.custom_kind_combo.addItem(k["label"], k["kind"])
            self._fill_engine_rows(discovered.get("discovered", []), customs)

        def on_error(err):
            self.rescan_btn.setEnabled(True)
            self.rescan_btn.setText(tr("settings.rescan"))
            QMessageBox.critical(self, tr("settings.error_title"), str(err))

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _fill_engine_rows(self, discovered: list[dict], customs: list[dict]):
        while self.version_table.rowCount() > len(_SLOT_ROWS):
            self.version_table.removeRow(len(_SLOT_ROWS))
        self._custom_actions.clear()
        self._custom_targets.clear()

        formats_by_name: dict[str, str] = {}
        slot_row = {slot: i for i, (slot, _) in enumerate(_SLOT_ROWS)}
        for eng in discovered:
            name = eng.get("name", "")
            formats_by_name[name] = ", ".join(eng.get("formats", []))
            if eng.get("source") == "config" and name in slot_row:
                self._set_cell(slot_row[name], 2, eng.get("binary_path", ""))
                self._set_cell(slot_row[name], 3, ", ".join(eng.get("formats", [])))

        for entry in customs:
            self._append_custom_row(entry, formats_by_name.get(entry.get("name", ""), ""))

        for eng in discovered:
            if eng.get("source") != "scan":
                continue
            row = self.version_table.rowCount()
            self.version_table.insertRow(row)
            self._set_cell(row, 0, eng.get("name", ""))
            self._set_cell(row, 1, eng.get("kind", ""))
            self._set_cell(row, 2, eng.get("binary_path", ""))
            self._set_cell(row, 3, ", ".join(eng.get("formats", [])))
            self._set_cell(row, 4, tr("settings.auto_discovered"))
            self._set_cell(row, 5, tr("settings.action_none"))

        if not customs and not any(e.get("source") == "scan" for e in discovered):
            row = self.version_table.rowCount()
            self.version_table.insertRow(row)
            self._set_cell(row, 0, tr("settings.no_discovered"))
        self._reflow_rows()

    def _append_custom_row(self, entry: dict, formats: str):
        table = self.version_table
        name = entry.get("name", "")
        history = entry.get("history") or []
        row = table.rowCount()
        table.insertRow(row)
        self._set_cell(row, 0, name)
        self._set_cell(row, 1, entry.get("kind", ""))
        self._set_cell(row, 2, entry.get("binary_path", ""))
        self._set_cell(row, 3, formats)

        ver_lbl = QLabel(tr("settings.manual_registered"))
        ver_lbl.setObjectName("subtitle")
        hist_lbl = QLabel(
            tr("settings.custom_history").format(n=len(history)) if history
            else tr("settings.custom_no_history")
        )
        hist_lbl.setObjectName("subtitle")
        hist_lbl.setWordWrap(True)
        cell = QWidget()
        cell_box = QVBoxLayout(cell)
        cell_box.setContentsMargins(4, 2, 4, 2)
        cell_box.setSpacing(2)
        cell_box.addWidget(ver_lbl)
        cell_box.addWidget(hist_lbl)
        table.setCellWidget(row, 4, cell)

        swap_btn = QPushButton(tr("settings.swap_binary"))
        swap_btn.clicked.connect(lambda _=False, n=name: self._swap_custom_binary(n))
        rb_btn = QPushButton(tr("settings.rollback_engine"))
        rb_btn.setEnabled(bool(history))
        rb_btn.clicked.connect(lambda _=False, n=name: self._rollback_custom(n))
        if history:
            self._custom_targets[name] = history[0]
            rb_btn.setToolTip(
                tr("settings.rollback_tip").format(ver=history[0].get("version", "?"))
                + f"\n{history[0].get('binary_path', '')}"
            )
        else:
            rb_btn.setToolTip(tr("settings.custom_no_history"))
        self._custom_actions[name] = (swap_btn, rb_btn)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(2, 2, 2, 2)
        btn_row.setSpacing(6)
        btn_row.addWidget(swap_btn)
        btn_row.addWidget(rb_btn)
        cell = QWidget()
        cell.setLayout(btn_row)
        table.setCellWidget(row, 5, cell)

    def _slot_row(self, slot: str) -> int:
        for i, (name, _) in enumerate(_SLOT_ROWS):
            if name == slot:
                return i
        raise KeyError(slot)

    def _swap_custom_binary(self, name: str):
        from lmloader.gui.api_client import run_in_thread

        path, _ = QFileDialog.getOpenFileName(
            self, tr("settings.swap_binary_title"), "",
            "Engine binaries (*.exe);;All files (*)",
        )
        if not path:
            return
        reply = QMessageBox.question(
            self, tr("settings.swap_confirm_title"),
            tr("settings.custom_swap_confirm_msg").format(name=name, to=path),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        swap_btn, rb_btn = self._custom_actions[name]
        swap_btn.setEnabled(False)
        rb_btn.setEnabled(False)
        swap_btn.setText(tr("settings.working"))

        def fetch():
            return self.api.update_custom_engine(name, path)

        def on_result(data):
            swap_btn.setText(tr("settings.swap_binary"))
            if data.get("status") == "already_current":
                QMessageBox.information(
                    self, tr("settings.added_title"),
                    tr("settings.custom_already_current").format(name=name),
                )
            else:
                QMessageBox.information(
                    self, tr("settings.custom_swapped_title"),
                    tr("settings.custom_swapped_msg").format(name=name),
                )
            self._refresh_engines()

        def on_error(err):
            swap_btn.setText(tr("settings.swap_binary"))
            swap_btn.setEnabled(True)
            rb_btn.setEnabled(True)
            QMessageBox.critical(self, tr("settings.error_title"), str(err))

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _rollback_custom(self, name: str):
        from lmloader.gui.api_client import run_in_thread

        target = self._custom_targets.get(name) or {}
        to = target.get("binary_path", "?")
        reply = QMessageBox.question(
            self, tr("settings.rollback_confirm_title"),
            tr("settings.rollback_custom_confirm_msg").format(name=name, to=to),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        swap_btn, rb_btn = self._custom_actions[name]
        swap_btn.setEnabled(False)
        rb_btn.setEnabled(False)
        rb_btn.setText(tr("settings.rolling_back"))

        def fetch():
            return self.api.rollback_custom_engine(name)

        def on_result(data):
            rb_btn.setText(tr("settings.rollback_engine"))
            QMessageBox.information(
                self, tr("settings.rolled_back_title"),
                tr("settings.custom_rolled_back_msg").format(
                    name=name, to=data.get("binary_path", "?"),
                ),
            )
            self._refresh_engines()

        def on_error(err):
            rb_btn.setText(tr("settings.rollback_engine"))
            swap_btn.setEnabled(True)
            rb_btn.setEnabled(True)
            QMessageBox.critical(self, tr("settings.error_title"), str(err))

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _browse_custom_engine(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("settings.browse"), "",
            "Engine binaries (*.exe);;All files (*)",
        )
        if path:
            self.custom_path_edit.setText(path)

    def _save_scan_dirs(self):
        from lmloader.gui.api_client import run_in_thread

        dirs = [d.strip() for d in self.scan_dirs_edit.text().replace("；", ";").split(";") if d.strip()]
        self.save_dirs_btn.setEnabled(False)

        def fetch():
            return self.api.set_engine_scan_dirs(dirs)

        def on_result(_data):
            self.save_dirs_btn.setEnabled(True)
            self._refresh_engines()

        def on_error(err):
            self.save_dirs_btn.setEnabled(True)
            QMessageBox.critical(self, tr("settings.error_title"), str(err))

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _add_custom_engine(self):
        from lmloader.gui.api_client import run_in_thread

        name = self.custom_name_edit.text().strip()
        kind = self.custom_kind_combo.currentData()
        path = self.custom_path_edit.text().strip()
        if not name or not path:
            QMessageBox.warning(self, tr("settings.error_title"), tr("settings.add_engine_missing"))
            return
        self.add_engine_btn.setEnabled(False)

        def fetch():
            return self.api.add_custom_engine(name, kind, path)

        def on_result(_data):
            self.add_engine_btn.setEnabled(True)
            self.custom_name_edit.clear()
            self.custom_path_edit.clear()
            QMessageBox.information(self, tr("settings.added_title"), tr("settings.added_msg"))
            self._refresh_engines()

        def on_error(err):
            self.add_engine_btn.setEnabled(True)
            QMessageBox.critical(self, tr("settings.error_title"), str(err))

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _build_server_group(self) -> QGroupBox:
        group = QGroupBox(tr("settings.server"))
        form = QFormLayout(group)

        port_start = QSpinBox()
        port_start.setRange(1024, 65535)
        port_start.setValue(8100)
        form.addRow(tr("settings.port_start"), port_start)
        self.port_start = port_start

        port_end = QSpinBox()
        port_end.setRange(1024, 65535)
        port_end.setValue(8200)
        form.addRow(tr("settings.port_end"), port_end)
        self.port_end = port_end

        auto_unload = QCheckBox(tr("settings.auto_unload"))
        auto_unload.setChecked(True)
        form.addRow("", auto_unload)
        self.auto_unload = auto_unload

        return group

    def _build_appearance_group(self) -> QGroupBox:
        group = QGroupBox(tr("settings.appearance"))
        form = QFormLayout(group)

        lang_combo = QComboBox()
        lang_combo.addItem(tr("settings.lang_zh"), "zh")
        lang_combo.addItem(tr("settings.lang_en"), "en")
        form.addRow(tr("settings.language"), lang_combo)
        self.language_combo = lang_combo

        theme_combo = QComboBox()
        theme_combo.addItem(tr("settings.theme_dark"), "dark")
        theme_combo.addItem(tr("settings.theme_light"), "light")
        theme_combo.addItem(tr("settings.theme_custom"), "custom")
        form.addRow(tr("settings.theme"), theme_combo)
        self.theme_combo = theme_combo

        # The picker writes ui.custom_theme and switches immediately; keeping the
        # combo in sync is what stops 保存设置 from quietly dropping back to dark.
        self.theme_row = CustomThemeRow()
        self.theme_row.theme_applied.connect(self._on_custom_theme)
        form.addRow(tr("settings.custom_theme"), self.theme_row)

        return group

    def _on_custom_theme(self, theme: str):
        index = self.theme_combo.findData(theme)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        self.theme_changed.emit(theme)

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox(tr("settings.paths"))
        form = QFormLayout(group)

        model_dir = QLineEdit()
        model_dir.setPlaceholderText(tr("settings.model_dir_placeholder"))
        browse_btn = QPushButton(tr("settings.browse"))
        browse_btn.clicked.connect(lambda: self._browse_dir(model_dir))
        row = QHBoxLayout()
        row.addWidget(model_dir, 1)
        row.addWidget(browse_btn)
        form.addRow(tr("settings.model_directory"), row)
        self.model_dir = model_dir

        cuda_path = QLineEdit()
        cuda_path.setPlaceholderText(tr("settings.cuda_placeholder"))
        form.addRow(tr("settings.cuda_path"), cuda_path)
        self.cuda_path = cuda_path

        return group

    def _build_download_group(self) -> QGroupBox:
        group = QGroupBox(tr("settings.download"))
        form = QFormLayout(group)

        proxy = QLineEdit()
        proxy.setPlaceholderText(tr("settings.proxy_placeholder"))
        form.addRow(tr("settings.proxy"), proxy)
        self.dl_proxy = proxy

        ollama_dir = QLineEdit()
        ollama_dir.setPlaceholderText(tr("settings.ollama_dir_placeholder"))
        browse_btn = QPushButton(tr("settings.browse"))
        browse_btn.clicked.connect(lambda: self._browse_dir(ollama_dir))
        row = QHBoxLayout()
        row.addWidget(ollama_dir, 1)
        row.addWidget(browse_btn)
        form.addRow(tr("settings.ollama_dir_label"), row)
        self.dl_ollama_dir = ollama_dir
        # The auto-detected path is shown separately and is NOT written into
        # the editable box: it used to be loaded as the field's text, so simply
        # opening this page and pressing save would freeze the probe result
        # into config.yaml as if the user had typed it.
        self.dl_ollama_detected = QLabel("")
        self.dl_ollama_detected.setObjectName("subtitle")
        self.dl_ollama_detected.setWordWrap(True)
        form.addRow("", self.dl_ollama_detected)

        # Download settings are saved by the page-level 保存设置 button now; this
        # label is where that save reports the outcome of the second write.
        self.dl_status = QLabel(tr("settings.download_saved_hint"))
        self.dl_status.setObjectName("subtitle")
        self.dl_status.setWordWrap(True)
        form.addRow("", self.dl_status)
        return group

    def _load_download_config(self):
        try:
            cfg = self.api.get_download_config()
            # Only the SAVED value goes into the editable field. The detected
            # path is read-only context — copying it into the field would make
            # an unrelated save silently persist it.
            self.dl_proxy.setText(cfg.get("proxy", "") or "")
            self.dl_ollama_dir.setText(cfg.get("ollama_dir", "") or "")
            detected = cfg.get("detected_ollama_root", "") or ""
            saved = cfg.get("ollama_dir", "") or ""
            if detected and not saved:
                self.dl_ollama_detected.setText(
                    tr("settings.ollama_detected_empty").format(path=detected)
                )
            elif detected and saved and detected.lower() == saved.lower():
                self.dl_ollama_detected.setText(
                    tr("settings.ollama_detected_current").format(path=detected)
                )
            else:
                self.dl_ollama_detected.setText("")
        except Exception:  # noqa: BLE001
            pass

    def _browse_file(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, tr("settings.browse_file_title"), "", "Executables (*.exe);;All Files (*)")
        if path:
            line_edit.setText(path)

    def _browse_dir(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, tr("settings.browse_dir_title"))
        if path:
            line_edit.setText(path)

    def _load_settings(self):
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            return

        engines = config.get("engines", {})

        for slot in ("primary", "ternary"):
            slot_config = engines.get(slot, {})
            defaults = slot_config.get("defaults", {})

            path_edit = getattr(self, f"{slot}_path", None)
            if path_edit:
                path_edit.setText(slot_config.get("binary_path", ""))
                self._set_cell(self._slot_row(slot), 2, slot_config.get("binary_path", ""))

            gpu = getattr(self, f"{slot}_gpu_layers", None)
            if gpu:
                gpu.setValue(defaults.get("n_gpu_layers", 999))

            ctx = getattr(self, f"{slot}_ctx", None)
            if ctx:
                ctx.setValue(defaults.get("ctx_size", 65536))

            flash = getattr(self, f"{slot}_flash", None)
            if flash:
                flash.setChecked(defaults.get("flash_attn", True))

            threads = getattr(self, f"{slot}_threads", None)
            if threads:
                threads.setValue(defaults.get("threads", 8))

            # Slot-level (not defaults) — matches the engine's read path.
            timeout = getattr(self, f"{slot}_timeout", None)
            if timeout:
                timeout.setValue(slot_config.get("startup_timeout", 120))

        port_range = config.get("port_range", {})
        self.port_start.setValue(port_range.get("start", 8100))
        self.port_end.setValue(port_range.get("end", 8200))

        router_config = config.get("router", {})
        self.auto_unload.setChecked(router_config.get("auto_unload_on_switch", True))

        model_dirs = config.get("model_dirs", [])  # FIXED: G4
        if model_dirs:
            self.model_dir.setText(model_dirs[0])

        self.cuda_path.setText(config.get("cuda_path", ""))

        ui_cfg = config.get("ui", {})
        lang = ui_cfg.get("language", "zh")
        idx = self.language_combo.findData(lang)
        if idx >= 0:
            self.language_combo.setCurrentIndex(idx)
        theme = ui_cfg.get("theme", "dark")
        t_idx = self.theme_combo.findData(theme)
        if t_idx >= 0:
            self.theme_combo.setCurrentIndex(t_idx)

    def _save_settings(self):
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            config = {}

        if "engines" not in config:
            config["engines"] = {}

        for slot in ("primary", "ternary"):
            if slot not in config["engines"]:
                config["engines"][slot] = {"defaults": {}}
            if "defaults" not in config["engines"][slot]:
                config["engines"][slot]["defaults"] = {}

            slot_cfg = config["engines"][slot]
            defaults = slot_cfg["defaults"]

            path_edit = getattr(self, f"{slot}_path", None)
            if path_edit and path_edit.text():
                slot_cfg["binary_path"] = path_edit.text()
                self._set_cell(self._slot_row(slot), 2, path_edit.text())

            gpu = getattr(self, f"{slot}_gpu_layers", None)
            if gpu:
                defaults["n_gpu_layers"] = gpu.value()

            ctx = getattr(self, f"{slot}_ctx", None)
            if ctx:
                defaults["ctx_size"] = ctx.value()

            flash = getattr(self, f"{slot}_flash", None)
            if flash:
                defaults["flash_attn"] = flash.isChecked()

            threads = getattr(self, f"{slot}_threads", None)
            if threads:
                defaults["threads"] = threads.value()

            timeout = getattr(self, f"{slot}_timeout", None)
            if timeout:
                slot_cfg["startup_timeout"] = timeout.value()

        config["port_range"] = {
            "start": self.port_start.value(),
            "end": self.port_end.value(),
        }

        if "router" not in config:
            config["router"] = {}
        config["router"]["auto_unload_on_switch"] = self.auto_unload.isChecked()

        if self.model_dir.text():
            config["model_dirs"] = [self.model_dir.text()]  # FIXED: G4

        if self.cuda_path.text():
            config["cuda_path"] = self.cuda_path.text()

        # Merge instead of replacing: ui.* also holds sidebar_collapsed and
        # system_prompt, which a wholesale overwrite would silently delete.
        ui = config.get("ui") or {}
        ui["language"] = self.language_combo.currentData() or "zh"
        ui["theme"] = self.theme_combo.currentData() or "dark"
        config["ui"] = ui
        set_language(config["ui"]["language"])

        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            # Emit only once the file holds the new value: the window resolves a
            # "custom" theme by reading ui.custom_theme back off disk, and a save
            # that failed midway would repaint from the palette it just rejected.
            self.theme_changed.emit(config["ui"]["theme"])
        except Exception as e:
            QMessageBox.critical(self, tr("settings.error_title"), tr("settings.save_error").format(err=e))
            return

        # The download config lives behind its own endpoint, so it is a second
        # write — reported in its own group rather than merged into this dialog.
        try:
            self.api.set_download_config({
                "proxy": self.dl_proxy.text().strip(),
                "ollama_dir": self.dl_ollama_dir.text().strip(),
            })
            self.dl_status.setText(tr("settings.download_saved"))
            self.dl_status.setStyleSheet(f"color: {c('green')};")
            self._load_download_config()
        except Exception as e:  # noqa: BLE001
            self.dl_status.setText(tr("settings.download_save_failed").format(err=str(e)))
            self.dl_status.setStyleSheet(f"color: {c('red')};")

        QMessageBox.information(self, tr("settings.saved_title"), tr("settings.saved_msg"))
