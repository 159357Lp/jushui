"""Local Server page — LM Studio style: server status, loaded models,
OpenAI-compatible endpoints, and a terminal-style runtime log."""

import datetime
import html as html_lib

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QMessageBox, QGroupBox, QFrame,
    QPlainTextEdit,
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QGuiApplication, QFont, QTextCursor

from lmloader.gui.api_client import LMLoaderAPI, run_in_thread
from lmloader.gui.i18n import tr
from lmloader.gui.styles import c
from lmloader.gui.icons import eye_icon


def _level_style(level: str) -> tuple[str, str]:
    styles = {
        "info": (c("accent"), tr("logs.level_info")),
        "warning": (c("warn"), tr("logs.level_warning")),
        "error": (c("red"), tr("logs.level_error")),
    }
    return styles.get(level, (c("accent"), tr("logs.level_info")))


def _source_color(source: str) -> str:
    # Terminal-style ANSI hues per log source, adapted to the panel background.
    dark = c("log_bg").lower() != "#ffffff"
    palette = {
        "http": ("#79c0ff", "#0969da"),       # blue  — inbound API traffic
        "engine": ("#56d364", "#1a7f37"),     # green — llama-server output
        "server": ("#d2a8ff", "#8250df"),     # purple — startup banner
        "lifecycle": ("#e3b341", "#9a6700"),  # amber — load/unload events
        "timing": ("#2dd4bf", "#0f766e"),     # teal — tg t/s, MTP acceptance
    }
    pair = palette.get(source)
    if not pair:
        return c("muted")
    return pair[0] if dark else pair[1]

_ENDPOINTS = [
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/embeddings"),
    ("POST", "/v1/responses"),
]


def _fmt_gb(size_bytes: float) -> str:
    if not size_bytes:
        return "-"
    return f"{size_bytes / (1024**3):.2f} GB"


class LogsPage(QWidget):
    def __init__(self, api: LMLoaderAPI):
        super().__init__()
        self.api = api
        self._threads: list = []
        self._last_id = 0
        self._paused = False
        self._event_count = 0
        self._refreshing = False
        self._refresh_thread = None
        self._loaded_signature = None
        self._init_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_server)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        title = QLabel(tr("logs.title"))
        title.setObjectName("title")
        layout.addWidget(title)

        # ---- server panel (LM Studio Local Server style) ------------------
        self.server_group = QGroupBox(tr("logs.server_panel"))
        sv = QVBoxLayout(self.server_group)
        sv.setContentsMargins(12, 14, 12, 12)
        sv.setSpacing(8)

        # status line
        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {c('dim')}; font-size: 16px;")
        status_row.addWidget(self.status_dot)
        self.status_label = QLabel(tr("logs.server_stopped"))
        self.status_label.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {c('muted')};")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        self.refresh_btn = QPushButton(tr("logs.refresh"))
        self.refresh_btn.clicked.connect(self._refresh_server)
        status_row.addWidget(self.refresh_btn)
        sv.addLayout(status_row)

        # loaded models list
        self.loaded_box = QVBoxLayout()
        self.loaded_box.setSpacing(4)
        self.loaded_container = QWidget()
        self.loaded_container.setLayout(self.loaded_box)
        sv.addWidget(self.loaded_container)

        # endpoints
        ep_label = QLabel(tr("logs.endpoints_title"))
        ep_label.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
        sv.addWidget(ep_label)
        self.ep_box = QVBoxLayout()
        self.ep_box.setSpacing(2)
        self.ep_container = QWidget()
        self.ep_container.setLayout(self.ep_box)
        sv.addWidget(self.ep_container)

        layout.addWidget(self.server_group)

        # ---- terminal-style log panel -------------------------------------
        log_group = QGroupBox(tr("logs.log_panel"))
        lg = QVBoxLayout(log_group)
        lg.setContentsMargins(12, 14, 12, 12)
        lg.setSpacing(8)

        toolbar = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItem(tr("logs.filter_all"), "all")
        self.filter_combo.addItem(tr("logs.level_info"), "info")
        self.filter_combo.addItem(tr("logs.level_warning"), "warning")
        self.filter_combo.addItem(tr("logs.level_error"), "error")
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self.filter_combo)
        toolbar.addStretch()
        self.pause_btn = QPushButton(tr("logs.pause"))
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause_toggled)
        toolbar.addWidget(self.pause_btn)
        self.copy_all_btn = QPushButton(tr("logs.copy_all"))
        self.copy_all_btn.clicked.connect(self._copy_all)
        toolbar.addWidget(self.copy_all_btn)
        self.clear_btn = QPushButton(tr("logs.clear"))
        self.clear_btn.setObjectName("danger")
        self.clear_btn.clicked.connect(self._clear)
        toolbar.addWidget(self.clear_btn)
        lg.addLayout(toolbar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setFont(QFont("Consolas", 10))
        self.log_view.setStyleSheet(self._terminal_style())
        lg.addWidget(self.log_view, 1)

        self.count_label = QLabel("")
        self.count_label.setObjectName("subtitle")
        lg.addWidget(self.count_label)

        layout.addWidget(log_group, 1)

    @staticmethod
    def _terminal_style() -> str:
        return (
            f"QPlainTextEdit {{ background-color: {c('log_bg')}; color: {c('log_fg')};"
            f" border: 1px solid {c('border')}; border-radius: 10px;"
            " font-family: Consolas, 'Courier New', monospace; }"
        )

    # ---------------------------------------------------------------- server

    def on_navigate(self):
        self._timer.start(2000)
        self._refresh_server()

    def on_leave(self):
        self._timer.stop()

    def retheme(self):
        """Rebuild server panel + log rows after a theme switch."""
        self.log_view.setStyleSheet(self._terminal_style())
        self.log_view.clear()
        self._last_id = 0
        self._event_count = 0
        self._loaded_signature = None
        if self._timer.isActive():
            self._refresh_server()

    def _refresh_server(self):
        if self._refreshing:
            return
        self._refreshing = True
        level = self._current_level()
        since = self._last_id

        def fetch():
            loaded = self.api.get_loaded_models()
            events = self.api.get_events(since_id=since, level=level)
            return loaded, events

        def on_result(result):
            self._refreshing = False
            self._refresh_thread = None
            if level != self._current_level():
                self._last_id = 0
                QTimer.singleShot(0, self._refresh_server)
                return
            loaded, events = result
            signature = tuple(
                (
                    item.get("model", {}).get("name"),
                    item.get("port"),
                    item.get("pid"),
                    item.get("engine_slot"),
                )
                for item in loaded
            )
            if signature != self._loaded_signature:
                self._render_server(loaded)
                self._loaded_signature = signature

            evs = events.get("events", [])
            sb = self.log_view.verticalScrollBar()
            stick = sb.value() >= sb.maximum() - 4  # user is tailing
            if evs:
                rows = [self._format_event(ev) for ev in evs]
                self.log_view.appendHtml("<br>".join(rows))
                self._event_count += len(evs)
            if evs:
                self._last_id = events.get("last_id", self._last_id)
            if not self._paused and evs and stick:
                self._scroll_to_end()
            self.count_label.setText(tr("logs.event_count").format(count=self._event_count))

        def on_error(err):
            self._refreshing = False
            self._refresh_thread = None
            self.count_label.setText(tr("logs.error_prefix").format(err=err))

        self._refresh_thread = run_in_thread(fetch, on_result, on_error)

    def _render_server(self, loaded: list[dict]):
        # clear previous rows
        while self.loaded_box.count():
            item = self.loaded_box.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        while self.ep_box.count():
            item = self.ep_box.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not loaded:
            self.status_dot.setStyleSheet(f"color: {c('dim')}; font-size: 16px;")
            self.status_label.setText(tr("logs.server_stopped"))
            self.status_label.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {c('muted')};")

            empty = QLabel(tr("logs.no_models_loaded"))
            empty.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
            self.loaded_box.addWidget(empty)

            ep_empty = QLabel(tr("logs.endpoints_empty"))
            ep_empty.setStyleSheet(f"color: {c('dim')}; font-size: 12px;")
            self.ep_box.addWidget(ep_empty)
            return

        self.status_dot.setStyleSheet(f"color: {c('green')}; font-size: 16px;")
        self.status_label.setText(tr("logs.server_running"))
        self.status_label.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {c('green')};")

        for item in loaded:
            self.loaded_box.addWidget(self._build_loaded_row(item))

        for item in loaded:
            self.ep_box.addWidget(self._build_api_base_row(item))
            self.ep_box.addWidget(self._build_endpoint_row(item))

    def _copy_feedback(self, button: QPushButton, text: str):
        QGuiApplication.clipboard().setText(text)
        old = button.text()
        button.setText(tr("logs.copied"))
        button.setEnabled(False)
        QTimer.singleShot(1200, lambda: (button.setText(old), button.setEnabled(True)))

    def _build_api_base_row(self, item: dict) -> QFrame:
        """Prominent API base URL — one click copies the whole base."""
        port = item.get("port", "?")
        base = f"http://127.0.0.1:{port}/v1"

        row = QFrame()
        row.setObjectName("apiBase")
        row.setCursor(Qt.PointingHandCursor)
        row.setToolTip(tr("logs.click_to_copy"))
        row.setStyleSheet(
            f"#apiBase {{ background-color: {c('accent_soft')}; border: 1px solid {c('accent_border')};"
            " border-radius: 10px; }"
            f"#apiBase:hover {{ background-color: {c('accent_hover')}; }}"
        )
        lay = QHBoxLayout(row)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(10)

        tag = QLabel(tr("logs.api_base"))
        tag.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
        lay.addWidget(tag, 0, Qt.AlignVCenter)

        url = QLabel(base)
        url.setStyleSheet(
            f"color: {c('accent')}; font-size: 14px; font-weight: 700;"
            "font-family: Consolas, monospace;"
        )
        lay.addWidget(url, 1, Qt.AlignVCenter)

        copy = QPushButton(tr("logs.copy"))
        copy.setFixedWidth(60)
        copy.clicked.connect(lambda checked=False, b=base: self._copy_feedback(copy, b))
        lay.addWidget(copy, 0, Qt.AlignVCenter)

        def _row_copy(_ev=None):
            self._copy_feedback(copy, base)

        row.mousePressEvent = _row_copy
        return row

    def _build_loaded_row(self, item: dict) -> QFrame:
        model = item.get("model", {})
        name = model.get("name", "?")
        size = _fmt_gb(model.get("size", 0))
        port = item.get("port", "?")
        slot = item.get("engine_slot", "?")
        vision = item.get("mmproj")

        row = QFrame()
        row.setObjectName("serverRow")
        row.setStyleSheet(
            f"#serverRow {{ background-color: {c('surface_hover')}; border: 1px solid {c('border')};"
            " border-radius: 10px; }"
        )
        lay = QHBoxLayout(row)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(10)

        ready = QLabel("READY")
        ready.setStyleSheet(
            f"color: {c('green')}; background: {c('green_bg')}; border-radius: 8px;"
            "padding: 2px 8px; font-size: 11px; font-weight: 700;"
        )
        lay.addWidget(ready, 0, Qt.AlignVCenter)

        name_lbl = QLabel(self._model_row_text(name, item.get("metrics")))
        name_lbl.setTextFormat(Qt.RichText)
        name_lbl.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {c('title')};")
        lay.addWidget(name_lbl, 1, Qt.AlignVCenter)

        meta = QLabel(f"{slot} :{port}")
        meta.setStyleSheet(f"color: {c('accent')}; font-size: 12px;")
        lay.addWidget(meta, 0, Qt.AlignVCenter)

        if vision:
            v = QPushButton()
            v.setIcon(eye_icon(c("green")))
            v.setFixedSize(22, 20)
            v.setIconSize(QSize(16, 16))
            v.setToolTip(tr("logs.vision_on"))
            v.setStyleSheet("background: transparent; border: none;")
            lay.addWidget(v, 0, Qt.AlignVCenter)
            vtxt = QLabel(tr("logs.vision"))
            vtxt.setStyleSheet(f"color: {c('green')}; font-size: 11px;")
            lay.addWidget(vtxt, 0, Qt.AlignVCenter)

        size_lbl = QLabel(size)
        size_lbl.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
        lay.addWidget(size_lbl, 0, Qt.AlignVCenter)

        eject = QPushButton(tr("logs.eject"))
        eject.setObjectName("danger")
        eject.clicked.connect(lambda checked=False, n=name: self._unload(n))
        lay.addWidget(eject, 0, Qt.AlignVCenter)

        return row

    def _model_row_text(self, name: str, metrics: dict | None) -> str:
        """Model name with a live tg / MTP-acceptance stats sub-line."""
        import html as _h
        title = _h.escape(name)
        if not metrics:
            return title
        bits = []
        tg = metrics.get("tg")
        if tg:
            bits.append(tr("logs.metric_tg").format(tg=f"{float(tg):.1f}"))
        acc = metrics.get("acceptance")
        if acc is not None:
            bits.append(tr("logs.metric_mtp").format(pct=f"{float(acc) * 100:.0f}"))
        if not bits:
            return title
        sub = " · ".join(bits)
        return (
            f"{title}<br>"
            f'<span style="font-size:11px;font-weight:500;color:{_source_color("timing")};">'
            f"⚡ {sub}</span>"
        )

    def _build_endpoint_row(self, item: dict) -> QFrame:
        port = item.get("port", "?")
        base = f"http://127.0.0.1:{port}"
        row = QFrame()
        row.setCursor(Qt.PointingHandCursor)
        row.setToolTip(tr("logs.click_to_copy"))
        lay = QHBoxLayout(row)
        lay.setContentsMargins(10, 2, 10, 2)
        lay.setSpacing(10)

        for method, path in _ENDPOINTS:
            m = QLabel(method)
            m.setFixedWidth(44)
            m.setStyleSheet(
                f"color: {c('accent')}; font-weight: 700; font-size: 11px;"
                "font-family: Consolas, monospace;"
            )
            lay.addWidget(m, 0, Qt.AlignVCenter)
            url = QLabel(f"{base}{path}")
            url.setStyleSheet(
                f"color: {c('muted')}; font-size: 11px; font-family: Consolas, monospace;"
            )
            lay.addWidget(url, 1, Qt.AlignVCenter)

        copy = QPushButton(tr("logs.copy"))
        copy.setFixedWidth(52)
        full = f"{base}{_ENDPOINTS[1][1]}"
        copy.clicked.connect(lambda checked=False, b=full: self._copy_feedback(copy, b))
        lay.addWidget(copy, 0, Qt.AlignVCenter)

        def _row_copy(_ev=None):
            self._copy_feedback(copy, full)

        row.mousePressEvent = _row_copy
        return row

    def _unload(self, model_name: str):
        def do_unload():
            return self.api.unload_model(model_name)

        def on_result(_r):
            self._refresh_server()

        def on_error(err):
            self.count_label.setText(tr("logs.error_prefix").format(err=err))

        t = run_in_thread(do_unload, on_result, on_error)
        self._threads.append(t)

    # ---------------------------------------------------------------- logs

    def _current_level(self) -> str:
        return self.filter_combo.currentData()

    def _format_event(self, ev: dict) -> str:
        level = ev.get("level", "info")
        ts = datetime.datetime.fromtimestamp(ev.get("ts", 0)).strftime("%H:%M:%S")
        source = ev.get("source", "")
        message = html_lib.escape(str(ev.get("message", "")))
        # preserve JSON indentation / blank lines
        message = message.replace("  ", "&nbsp;&nbsp;").replace("\n", "<br>")

        level_color, level_text = _level_style(level)
        src_color = _source_color(source)
        if level == "error":
            msg_color = c("red")
        elif source == "timing":
            msg_color = _source_color("timing")
        else:
            msg_color = c("log_fg")
        prefix = "⚡ " if source == "timing" else ""
        row = (
            f'<span style="color:{c("muted")};">{ts}</span>&nbsp;&nbsp;'
            f'<span style="color:{level_color};font-weight:600;">[{level_text}]</span>&nbsp;&nbsp;'
            f'<span style="color:{src_color};">{html_lib.escape(source)}</span>&nbsp;&nbsp;'
            f'<span style="color:{msg_color};font-weight:{700 if source == "timing" else 400};">'
            f'{prefix}{message}</span>'
        )
        return row

    def _scroll_to_end(self):
        self.log_view.moveCursor(QTextCursor.End)
        sb = self.log_view.verticalScrollBar()

        def _roll():
            sb.setValue(sb.maximum())

        QTimer.singleShot(0, _roll)
        QTimer.singleShot(150, _roll)

    # ---------------------------------------------------------------- actions

    def _on_filter_changed(self, _idx: int):
        self.log_view.clear()
        self._last_id = 0
        self._event_count = 0
        self._refresh_server()

    def _on_pause_toggled(self, checked: bool):
        self._paused = checked
        self.pause_btn.setText(tr("logs.resume") if checked else tr("logs.pause"))

    def _copy_all(self):
        """Copy the whole terminal buffer as plain text."""
        text = self.log_view.toPlainText()
        if text.strip():
            QGuiApplication.clipboard().setText(text)
            self.copy_all_btn.setText(tr("logs.copied"))
            QTimer.singleShot(1200, lambda: self.copy_all_btn.setText(tr("logs.copy_all")))

    def _clear(self):
        reply = QMessageBox.question(
            self, tr("logs.clear_confirm_title"), tr("logs.clear_confirm_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        def do_clear():
            return self.api.clear_events()

        def on_result(_r):
            self.log_view.clear()
            self._last_id = 0
            self._event_count = 0
            self.count_label.setText(tr("logs.cleared"))

        def on_error(err):
            self.count_label.setText(tr("logs.error_prefix").format(err=err))

        t = run_in_thread(do_clear, on_result, on_error)
        self._threads.append(t)
