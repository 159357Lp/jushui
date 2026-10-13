"""Settings page — configure engine paths, ports, defaults."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QCheckBox, QPushButton, QLabel,
    QGroupBox, QFileDialog, QMessageBox, QComboBox, QScrollArea, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt, Signal
import yaml
from pathlib import Path

from lmloader.gui.api_client import LMLoaderAPI
from lmloader.gui.i18n import tr, set_language
from lmloader.gui.styles import c


def _get_config_path() -> Path:  # FIXED: G4
    for candidate in [
        Path("config.yaml"),
        Path.home() / ".lmloader" / "config.yaml",
    ]:
        if candidate.exists():
            return candidate
    return Path("config.yaml")


class SettingsPage(QWidget):
    theme_changed = Signal(str)

    def __init__(self, api: LMLoaderAPI):
        super().__init__()
        self.api = api
        self._config_path = _get_config_path()  # FIXED: G4
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
        scroll_layout.setSpacing(16)

        # Two-column layout: engine management on the left, server / paths /
        # network settings on the right. Cuts the page's scroll height roughly
        # in half; the discovered-engine table stays full width below.
        left_col = QVBoxLayout()
        left_col.setSpacing(16)
        right_col = QVBoxLayout()
        right_col.setSpacing(16)

        left_col.addWidget(
            self._build_engine_group(tr("settings.primary_engine"), "primary", with_update=True)
        )
        left_col.addWidget(
            self._build_engine_group(tr("settings.ternary_engine"), "ternary", with_update=True)
        )

        right_col.addWidget(self._build_server_group())
        right_col.addWidget(self._build_paths_group())
        right_col.addWidget(self._build_download_group())

        columns = QHBoxLayout()
        columns.setSpacing(16)
        columns.addLayout(left_col, 1)
        columns.addLayout(right_col, 1)
        scroll_layout.addLayout(columns)

        scroll_layout.addWidget(self._build_extra_engines_group())

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

    def _build_engine_group(self, title: str, slot: str, with_update: bool = False) -> QGroupBox:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setSpacing(8)

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

        if with_update:
            if not getattr(self, "_threads", None):
                self._threads: list = []
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            form.addRow(sep)

            title_key = (
                "settings.update_title_official" if slot == "primary"
                else "settings.update_title_prism"
            )
            sub = QLabel(tr(title_key))
            f = sub.font()
            f.setBold(True)
            sub.setFont(f)
            form.addRow(sub)

            ver_lbl = QLabel("...")
            ver_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            form.addRow(tr("settings.engine_current"), ver_lbl)
            setattr(self, f"{slot}_ver_lbl", ver_lbl)

            latest_lbl = QLabel("-")
            latest_lbl.setWordWrap(True)
            form.addRow(tr("settings.engine_latest"), latest_lbl)
            setattr(self, f"{slot}_latest_lbl", latest_lbl)

            btn_row = QHBoxLayout()
            check_btn = QPushButton(tr("settings.check_update"))
            check_btn.clicked.connect(lambda _=False, s=slot: self._check_engine_update(s))
            btn_row.addWidget(check_btn)
            setattr(self, f"{slot}_check_btn", check_btn)

            upd_btn = QPushButton(tr("settings.update_engine"))
            upd_btn.setObjectName("primary")
            upd_btn.setEnabled(False)
            upd_btn.clicked.connect(lambda _=False, s=slot: self._update_engine(s))
            btn_row.addWidget(upd_btn)
            setattr(self, f"{slot}_update_btn", upd_btn)

            rollback_btn = QPushButton(tr("settings.rollback_engine"))
            rollback_btn.setEnabled(False)
            rollback_btn.setToolTip(tr("settings.rollback_tip"))
            rollback_btn.clicked.connect(lambda _=False, s=slot: self._rollback_engine(s))
            btn_row.addWidget(rollback_btn)
            setattr(self, f"{slot}_rollback_btn", rollback_btn)
            btn_row.addStretch()
            form.addRow("", btn_row)

            tip_key = (
                "settings.update_tip_official" if slot == "primary"
                else "settings.update_tip_prism"
            )
            tip = QLabel(tr(tip_key))
            tip.setObjectName("subtitle")
            tip.setWordWrap(True)
            form.addRow("", tip)

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

        def on_error(err):
            check_btn.setEnabled(True)
            check_btn.setText(tr("settings.check_update"))
            latest_lbl.setText(tr("settings.update_error").format(err=err))

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
            # path field may have changed (slot moved into its own directory)
            installed = data.get("result", {}).get("installed")
            if installed:
                path_edit = getattr(self, f"{slot}_path", None)
                if path_edit is not None:
                    path_edit.setText(installed)
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

        def on_error(_err):
            lbl = getattr(self, f"{slot}_ver_lbl", None)
            if lbl is not None:
                lbl.setText("...")

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

        # engine kinds for the custom-engine combo
        try:
            import httpx as _httpx
            kinds = _httpx.get(f"{self.api.base_url}/api/engine/kinds", timeout=10.0).json().get("kinds", [])
            self.custom_kind_combo.clear()
            for k in kinds:
                self.custom_kind_combo.addItem(k["label"], k["kind"])
        except Exception:
            pass

        self._rescan_engines()

    def _build_extra_engines_group(self) -> QGroupBox:
        """Other-format engines: search dirs + rescan + custom engines."""
        group = QGroupBox(tr("settings.extra_engines"))
        vbox = QVBoxLayout(group)
        vbox.setSpacing(8)

        tip = QLabel(tr("settings.extra_engines_tip"))
        tip.setObjectName("subtitle")
        tip.setWordWrap(True)
        vbox.addWidget(tip)

        dir_row = QHBoxLayout()
        self.scan_dirs_edit = QLineEdit()
        self.scan_dirs_edit.setPlaceholderText("D:/engines;D:/rwkv")
        dir_row.addWidget(QLabel(tr("settings.engine_scan_dirs")))
        dir_row.addWidget(self.scan_dirs_edit, 1)
        vbox.addLayout(dir_row)

        btn_row = QHBoxLayout()
        self.save_dirs_btn = QPushButton(tr("settings.save_dirs"))
        self.save_dirs_btn.clicked.connect(self._save_scan_dirs)
        btn_row.addWidget(self.save_dirs_btn)
        self.rescan_btn = QPushButton(tr("settings.rescan"))
        self.rescan_btn.clicked.connect(self._rescan_engines)
        btn_row.addWidget(self.rescan_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self.discovered_table = QTableWidget(0, 4)
        self.discovered_table.setHorizontalHeaderLabels(
            [tr("settings.engine_name"), tr("settings.engine_type"),
             tr("settings.engine_path"), tr("settings.engine_formats")]
        )
        self.discovered_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.discovered_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.discovered_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.discovered_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.discovered_table.verticalHeader().setVisible(False)
        self.discovered_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.discovered_table.setMaximumHeight(160)
        vbox.addWidget(self.discovered_table)

        custom_row = QHBoxLayout()
        self.custom_name_edit = QLineEdit()
        self.custom_name_edit.setPlaceholderText(tr("settings.engine_name_ph"))
        custom_row.addWidget(QLabel(tr("settings.engine_name")))
        custom_row.addWidget(self.custom_name_edit, 1)
        vbox.addLayout(custom_row)

        custom_row2 = QHBoxLayout()
        self.custom_kind_combo = QComboBox()
        custom_row2.addWidget(QLabel(tr("settings.engine_type")))
        custom_row2.addWidget(self.custom_kind_combo, 1)
        self.custom_path_edit = QLineEdit()
        self.custom_path_edit.setPlaceholderText("D:/engines/rwkv-server.exe")
        custom_row2.addWidget(QLabel(tr("settings.engine_path")))
        custom_row2.addWidget(self.custom_path_edit, 3)
        browse_btn = QPushButton(tr("settings.browse"))
        browse_btn.clicked.connect(self._browse_custom_engine)
        custom_row2.addWidget(browse_btn)
        vbox.addLayout(custom_row2)

        add_row = QHBoxLayout()
        self.add_engine_btn = QPushButton(tr("settings.add_engine"))
        self.add_engine_btn.setObjectName("primary")
        self.add_engine_btn.clicked.connect(self._add_custom_engine)
        add_row.addWidget(self.add_engine_btn)
        add_row.addStretch()
        vbox.addLayout(add_row)

        self._threads: list = []
        return group

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
            self._rescan_engines()

        def on_error(err):
            self.save_dirs_btn.setEnabled(True)
            QMessageBox.critical(self, tr("settings.error_title"), str(err))

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _rescan_engines(self):
        from lmloader.gui.api_client import run_in_thread

        self.rescan_btn.setEnabled(False)
        self.rescan_btn.setText(tr("settings.scanning"))

        def fetch():
            return self.api.engine_discover()

        def on_result(data):
            self.rescan_btn.setEnabled(True)
            self.rescan_btn.setText(tr("settings.rescan"))
            scan_dirs = data.get("scan_dirs", [])
            self.scan_dirs_edit.setText(";".join(scan_dirs))
            self._fill_discovered(data.get("discovered", []))

        def on_error(err):
            self.rescan_btn.setEnabled(True)
            self.rescan_btn.setText(tr("settings.rescan"))

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _fill_discovered(self, discovered: list[dict]):
        self.discovered_table.setRowCount(0)
        if not discovered:
            self.discovered_table.setRowCount(1)
            item = QTableWidgetItem(tr("settings.no_discovered"))
            item.setTextAlignment(Qt.AlignCenter)
            self.discovered_table.setItem(0, 0, item)
            return
        for eng in discovered:
            row = self.discovered_table.rowCount()
            self.discovered_table.insertRow(row)
            self.discovered_table.setItem(row, 0, QTableWidgetItem(eng.get("name", "")))
            self.discovered_table.setItem(row, 1, QTableWidgetItem(eng.get("kind", "")))
            self.discovered_table.setItem(row, 2, QTableWidgetItem(eng.get("binary_path", "")))
            self.discovered_table.setItem(row, 3, QTableWidgetItem(", ".join(eng.get("formats", []))))

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
            self._rescan_engines()

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

        lang_combo = QComboBox()
        lang_combo.addItem(tr("settings.lang_zh"), "zh")
        lang_combo.addItem(tr("settings.lang_en"), "en")
        form.addRow(tr("settings.language"), lang_combo)
        self.language_combo = lang_combo

        theme_combo = QComboBox()
        theme_combo.addItem(tr("settings.theme_dark"), "dark")
        theme_combo.addItem(tr("settings.theme_light"), "light")
        form.addRow(tr("settings.theme"), theme_combo)
        self.theme_combo = theme_combo

        return group

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

        save_btn = QPushButton(tr("settings.download_save"))
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._save_download_config)
        form.addRow("", save_btn)
        self.dl_status = QLabel("")
        self.dl_status.setObjectName("subtitle")
        form.addRow("", self.dl_status)
        return group

    def _save_download_config(self):
        try:
            self.api.set_download_config({
                "proxy": self.dl_proxy.text().strip(),
                "ollama_dir": self.dl_ollama_dir.text().strip(),
            })
            self.dl_status.setText(tr("settings.download_saved"))
            self.dl_status.setStyleSheet(f"color: {c('green')};")
            # Re-read so the detected-path hint reflects what was just saved.
            self._load_download_config()
        except Exception as e:  # noqa: BLE001
            self.dl_status.setText(tr("settings.download_save_failed").format(err=str(e)))
            self.dl_status.setStyleSheet(f"color: {c('red')};")

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

        config["ui"] = {
            "language": self.language_combo.currentData() or "zh",
            "theme": self.theme_combo.currentData() or "dark",
        }
        set_language(config["ui"]["language"])
        self.theme_changed.emit(config["ui"]["theme"])

        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            QMessageBox.information(self, tr("settings.saved_title"), tr("settings.saved_msg"))
        except Exception as e:
            QMessageBox.critical(self, tr("settings.error_title"), tr("settings.save_error").format(err=e))
