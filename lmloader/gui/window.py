"""Main window — collapsible sidebar navigation + stacked pages."""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QPushButton, QLabel, QSizePolicy,
    QSystemTrayIcon, QMenu, QApplication, QMessageBox,
)
from PySide6.QtCore import Qt, QSize, QPointF, QRectF, QTimer
from PySide6.QtGui import (
    QIcon, QAction, QColor, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
)

from lmloader import __version__
from lmloader.gui.api_client import LMLoaderAPI, run_in_thread
from lmloader.gui.i18n import tr, get_language
from lmloader.gui.pages.chat import ChatPage
from lmloader.gui.pages.models import ModelsPage
from lmloader.gui.pages.download import DownloadPage
from lmloader.gui.pages.settings import SettingsPage
from lmloader.gui.pages.monitor import MonitorPage
from lmloader.gui.pages.logs import LogsPage
from lmloader.gui.styles import apply_theme, c as theme_color
from lmloader.gui.icons import app_icon
from lmloader.gui.ui_config import load_ui_setting, save_ui_setting


SIDEBAR_WIDTH_EXPANDED = 200
SIDEBAR_WIDTH_COLLAPSED = 52


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("掬水")
        self.setMinimumSize(1000, 680)
        self.resize(1200, 800)

        # Same icon as the desktop shortcut, so the taskbar entry and the
        # .lnk look like the same app.
        _icon = app_icon()
        if not _icon.isNull():
            self.setWindowIcon(_icon)

        self.api = LMLoaderAPI()
        self._connection_check_running = False
        self._collapsed = bool(load_ui_setting("sidebar_collapsed", False))

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = self._build_sidebar()
        self._apply_sidebar_style()
        main_layout.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 1)

        self.pages = {}
        self._init_pages()
        self.settings_page.theme_changed.connect(self._on_theme_changed)
        self._navigate(0)
        self._apply_sidebar_state()
        self._setup_tray()

        # ---- ConfigGuardian: poll for pending config changes ----
        self._guardian_polling = False
        self._guardian_dialog_active = False
        self._guardian_timer = QTimer(self)
        self._guardian_timer.setInterval(1000)
        self._guardian_timer.timeout.connect(self._poll_guardian)
        self._guardian_timer.start()

        # ---- Connection watchdog ----
        # The one-shot check at startup froze on a still-booting backend (the
        # GUI opens before serve is ready), leaving the badge stuck at 未连接
        # even after the backend comes up. Re-probe periodically so the status
        # self-heals on boot and flips back to 未连接 if the backend dies.
        # Cheap now that httpx Client is pooled, and it runs off-thread.
        self._conn_timer = QTimer(self)
        self._conn_timer.setInterval(2000)
        self._conn_timer.timeout.connect(self._check_connection)
        self._conn_timer.start()

    def _on_theme_changed(self, theme: str):
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, theme)
        self._apply_sidebar_style()
        # Rebuild pages that embed theme colors in inline QSS.
        for page in (self.models_page, self.download_page):
            if hasattr(page, "_refresh"):
                page._refresh()
        if hasattr(self.logs_page, "retheme"):
            self.logs_page.retheme()
        if (self.stack.currentWidget() is self.monitor_page
                and hasattr(self.monitor_page, "on_navigate")):
            self.monitor_page.on_navigate()
        self._check_connection()

    def _sidebar_qss(self) -> str:
        """Sidebar stylesheet built from theme tokens so a theme switch
        recolors the rail instead of leaving the old hard-coded navy."""
        return f"""
            QWidget#sidebar {{
                background-color: {theme_color("sidebar")};
                border: none;
            }}
            #sidebar QLabel {{
                color: {theme_color("sidebar_fg")};
                background: transparent;
            }}
            #sidebar QPushButton {{
                background-color: transparent;
                color: {theme_color("sidebar_fg")};
                border: none;
                border-radius: 10px;
                padding: 10px 14px;
                text-align: left;
                font-size: 14px;
            }}
            #sidebar QPushButton:hover {{
                background-color: {theme_color("sidebar_hover")};
                color: #ffffff;
            }}
            #sidebar QPushButton[active="true"] {{
                background-color: {theme_color("sidebar_active")};
                color: #ffffff;
                border: none;
            }}
            #sidebar QPushButton[rail="true"] {{
                padding: 10px 0;
                text-align: center;
                font-size: 15px;
            }}
            #sidebar QPushButton#sidebarToggle {{
                background-color: transparent;
                color: {theme_color("sidebar_fg")};
                border: none;
                border-radius: 10px;
                font-size: 17px;
                text-align: left;
                padding: 4px 14px;
            }}
            #sidebar QPushButton#sidebarToggle:hover {{
                background-color: {theme_color("sidebar_hover")};
                color: #ffffff;
            }}
            #sidebar QPushButton#sidebarToggle[rail="true"] {{
                padding: 4px 0;
                text-align: center;
            }}
        """

    def _apply_sidebar_style(self):
        self.sidebar.setStyleSheet(self._sidebar_qss())
        self.logo_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {theme_color('accent')};"
            " padding: 8px 12px 2px 12px;"
        )

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH_EXPANDED)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(2)

        # Collapse / expand toggle (hamburger when expanded, » when collapsed).
        self.toggle_btn = QPushButton("\u2630")  # ☰
        self.toggle_btn.setObjectName("sidebarToggle")
        self.toggle_btn.setFixedSize(36, 36)
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.clicked.connect(self._toggle_sidebar)
        layout.addWidget(self.toggle_btn)

        self.logo_label = QLabel("掬水")  # color set by _apply_sidebar_style
        layout.addWidget(self.logo_label)

        self.ver_label = QLabel(f"v{__version__}")
        self.ver_label.setObjectName("subtitle")
        self.ver_label.setStyleSheet("padding: 0 12px 10px 12px;")
        layout.addWidget(self.ver_label)

        layout.addSpacing(6)

        # (i18n key, collapsed glyph for Chinese, glyph for English)
        zh = get_language() != "en"
        nav_items = [
            ("nav.chat",     "聊", "Ch",  "chat"),
            ("nav.models",   "模", "Mo",  "models"),
            ("nav.download", "下", "Dl",  "download"),
            ("nav.monitor",  "控", "Mn",  "monitor"),
            ("nav.logs",     "服", "Srv", "logs"),
            ("nav.settings", "设", "Set", "settings"),
        ]

        self.nav_buttons: list[QPushButton] = []
        self._nav_full: list[str] = []
        self._nav_glyph: list[str] = []
        for i, (key, glyph_zh, glyph_en, _slot) in enumerate(nav_items):
            full = tr(key)
            glyph = glyph_zh if zh else glyph_en
            btn = QPushButton(full)
            btn.setCheckable(True)
            btn.setProperty("active", False)
            btn.setProperty("rail", False)
            btn.setFixedHeight(42)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(full)
            btn.clicked.connect(lambda checked, idx=i: self._navigate(idx))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)
            self._nav_full.append(full)
            self._nav_glyph.append(glyph)

        layout.addStretch()

        # Connection indicator: colored dot + text (text hidden when collapsed).
        conn_row = QHBoxLayout()
        conn_row.setContentsMargins(12, 6, 12, 4)
        conn_row.setSpacing(8)
        self.conn_dot = QLabel("\u25CF")  # ●
        self.conn_dot.setObjectName("connDot")
        self.conn_dot.setFixedWidth(12)
        conn_row.addWidget(self.conn_dot)
        self.conn_label = QLabel(tr("status.disconnected"))
        self.conn_label.setObjectName("subtitle")
        conn_row.addWidget(self.conn_label)
        conn_row.addStretch()
        layout.addLayout(conn_row)

        return sidebar

    def _toggle_sidebar(self):
        self._collapsed = not self._collapsed
        self._apply_sidebar_state()
        try:
            save_ui_setting("sidebar_collapsed", self._collapsed)
        except Exception:
            # A read-only config.yaml must not break collapsing the rail.
            pass

    def _apply_sidebar_state(self):
        collapsed = self._collapsed
        self.sidebar.setFixedWidth(
            SIDEBAR_WIDTH_COLLAPSED if collapsed else SIDEBAR_WIDTH_EXPANDED
        )
        margins = (6, 10, 6, 10) if collapsed else (8, 10, 8, 10)
        self.sidebar.layout().setContentsMargins(*margins)

        self.logo_label.setVisible(not collapsed)
        self.ver_label.setVisible(not collapsed)
        self.conn_label.setVisible(not collapsed)

        self.toggle_btn.setText("L" if collapsed else "\u2630")  # ☰
        self.toggle_btn.setToolTip(tr("nav.expand_tip") if collapsed else tr("nav.collapse_tip"))
        self.toggle_btn.setStyleSheet(
            "QPushButton { background-color: %s; color: white; border: none;"
            " border-radius: 10px; font-size: 18px; font-weight: 800; padding: 0; }"
            " QPushButton:hover { background-color: %s; }"
            % (theme_color("accent"), theme_color("accent_hover"))
            if collapsed else ""
        )
        self.toggle_btn.setProperty("rail", collapsed)
        self.toggle_btn.style().unpolish(self.toggle_btn)
        self.toggle_btn.style().polish(self.toggle_btn)

        for i, btn in enumerate(self.nav_buttons):
            btn.setText("" if collapsed else self._nav_full[i])
            btn.setIcon(self._make_nav_icon(i, btn.property("active")) if collapsed else QIcon())
            btn.setIconSize(QSize(20, 20))
            btn.setProperty("rail", collapsed)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        self._check_connection()

    def _make_nav_icon(self, index: int, active: bool) -> QIcon:
        color = QColor("#ffffff" if active else theme_color("sidebar_fg"))
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(color, 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)

        if index == 0:
            painter.drawRoundedRect(QRectF(3.5, 4.0, 17, 13), 4, 4)
            tail = QPainterPath(QPointF(8, 17))
            tail.lineTo(6, 21)
            tail.lineTo(12, 17)
            painter.drawPath(tail)
            painter.drawLine(QPointF(7, 9), QPointF(17, 9))
            painter.drawLine(QPointF(7, 12), QPointF(14, 12))
        elif index == 1:
            painter.drawPolygon(QPolygonF([
                QPointF(12, 2.5), QPointF(20.5, 7), QPointF(20.5, 17),
                QPointF(12, 21.5), QPointF(3.5, 17), QPointF(3.5, 7),
            ]))
            painter.drawLine(QPointF(3.5, 7), QPointF(12, 12))
            painter.drawLine(QPointF(20.5, 7), QPointF(12, 12))
            painter.drawLine(QPointF(12, 12), QPointF(12, 21.5))
        elif index == 2:
            painter.drawLine(QPointF(12, 3), QPointF(12, 15))
            painter.drawLine(QPointF(7.5, 10.5), QPointF(12, 15))
            painter.drawLine(QPointF(16.5, 10.5), QPointF(12, 15))
            painter.drawRoundedRect(QRectF(4, 16, 16, 4.5), 1.5, 1.5)
            painter.drawLine(QPointF(8, 18), QPointF(16, 18))
        elif index == 3:
            painter.drawLine(QPointF(4, 3), QPointF(4, 20))
            painter.drawLine(QPointF(4, 20), QPointF(21, 20))
            painter.drawPolyline(QPolygonF([
                QPointF(6, 15), QPointF(10, 11), QPointF(13, 13),
                QPointF(18, 7), QPointF(20, 9),
            ]))
        elif index == 4:
            for y in (4, 14):
                painter.drawRoundedRect(QRectF(3.5, y, 17, 6), 2, 2)
                painter.drawEllipse(QPointF(7, y + 3), 0.8, 0.8)
                painter.drawLine(QPointF(11, y + 3), QPointF(17, y + 3))
        else:
            painter.drawEllipse(QPointF(12, 12), 4, 4)
            painter.drawEllipse(QPointF(12, 12), 8, 8)
            for angle in range(0, 360, 45):
                painter.save()
                painter.translate(12, 12)
                painter.rotate(angle)
                painter.drawLine(QPointF(0, -11), QPointF(0, -8))
                painter.restore()

        painter.end()
        return QIcon(pixmap)

    def _init_pages(self):
        self.chat_page = ChatPage(self.api)
        self.stack.addWidget(self.chat_page)
        self.pages["chat"] = self.chat_page

        self.models_page = ModelsPage(self.api)
        self.stack.addWidget(self.models_page)
        self.pages["models"] = self.models_page

        self.download_page = DownloadPage(self.api)
        self.stack.addWidget(self.download_page)
        self.pages["download"] = self.download_page

        self.monitor_page = MonitorPage(self.api)
        self.stack.addWidget(self.monitor_page)
        self.pages["monitor"] = self.monitor_page

        self.logs_page = LogsPage(self.api)
        self.stack.addWidget(self.logs_page)
        self.pages["logs"] = self.logs_page

        self.settings_page = SettingsPage(self.api)
        self.stack.addWidget(self.settings_page)
        self.pages["settings"] = self.settings_page

    def _navigate(self, index: int):
        previous_page = self.stack.currentWidget()
        if previous_page is not None and hasattr(previous_page, "on_leave"):
            previous_page.on_leave()

        for i, btn in enumerate(self.nav_buttons):
            btn.setProperty("active", i == index)
            if self._collapsed:
                btn.setIcon(self._make_nav_icon(i, i == index))
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        previous_page = self.stack.currentWidget()

        self.stack.setCurrentIndex(index)

        # Give the page we just left a chance to stop background work (the
        # model page polls the loaded-model set, so a hidden page would keep
        # issuing requests for nothing).
        if previous_page is not None and hasattr(previous_page, "on_leave"):
            previous_page.on_leave()

        current_page = self.stack.currentWidget()
        if hasattr(current_page, "on_navigate"):
            current_page.on_navigate()

        self._check_connection()

    def _check_connection(self):
        if self._connection_check_running:
            return
        self._connection_check_running = True

        def on_result(connected):
            self._connection_check_running = False
            self._update_connection_status(connected)

        def on_error(_err):
            self._connection_check_running = False
            self._update_connection_status(False)

        run_in_thread(self.api.health, on_result, on_error)

    def _update_connection_status(self, connected: bool):
        was_connected = getattr(self, "_was_connected", False)
        self._was_connected = connected
        if connected:
            self.conn_label.setText(tr("status.connected"))
            dot_color = theme_color("green")
        else:
            self.conn_label.setText(tr("status.disconnected"))
            dot_color = theme_color("red")
        self.conn_label.setStyleSheet(f"color: {dot_color};")
        if hasattr(self, "conn_dot"):
            self.conn_dot.setStyleSheet(f"color: {dot_color}; font-size: 12px;")
            self.conn_dot.setToolTip(self.conn_label.text())
        # On the offline→online edge, refresh whatever page is showing so data
        # that was fetched during the outage (e.g. the chat model list) repopulates.
        if connected and not was_connected:
            current = self.stack.currentWidget()
            if current is not None and hasattr(current, "on_navigate"):
                current.on_navigate()

    # ---- ConfigGuardian: poll & confirm config changes ----

    def _poll_guardian(self):
        """Poll backend for pending config-change requests; show approval dialog."""
        if self._guardian_polling or self._guardian_dialog_active:
            return
        self._guardian_polling = True

        def fetch():
            return self.api.get_pending_config_changes()

        def on_result(pending):
            self._guardian_polling = False
            if pending and not self._guardian_dialog_active:
                self._show_guardian_dialog(pending[0])

        def on_error(_err):
            self._guardian_polling = False

        run_in_thread(fetch, on_result, on_error)

    def _show_guardian_dialog(self, request: dict):
        """Show a confirmation dialog for a pending config change."""
        self._guardian_dialog_active = True
        request_id = request.get("request_id", "")
        description = request.get("description", "配置变更")
        endpoint = request.get("endpoint", "")

        box = QMessageBox(self)
        box.setWindowTitle("配置变更审批")
        box.setText("有命令请求修改配置，是否允许？")
        box.setInformativeText(
            f"操作：{description}\n"
            f"接口：{endpoint}"
        )
        box.setIcon(QMessageBox.Question)
        allow_btn = box.addButton("允许", QMessageBox.YesRole)
        deny_btn = box.addButton("拒绝", QMessageBox.NoRole)
        box.setDefaultButton(deny_btn)
        box.exec()

        approved = box.clickedButton() is allow_btn

        def do_approve():
            self.api.approve_config_change(request_id)

        def do_reject():
            self.api.reject_config_change(request_id)

        def on_ok(_data):
            self._guardian_dialog_active = False

        def on_err(_err):
            self._guardian_dialog_active = False

        if approved:
            run_in_thread(do_approve, on_ok, on_err)
        else:
            run_in_thread(do_reject, on_ok, on_err)

    def _setup_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setToolTip("掬水")

        # Was `icon = QIcon()` — always null, so the `if not icon.isNull()`
        # branch never fired and the blue-dot fallback below was the only path
        # ever taken. Load the real application icon and keep the fallback for
        # installs that ship without the asset.
        icon = app_icon()
        if not icon.isNull():
            self.tray_icon.setIcon(icon)
        else:
            from PySide6.QtGui import QPixmap, QPainter, QColor
            pixmap = QPixmap(32, 32)
            pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pixmap)
            painter.setBrush(QColor(theme_color("accent")))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(4, 4, 24, 24)
            painter.end()
            self.tray_icon.setIcon(QIcon(pixmap))

        tray_menu = QMenu()
        show_action = QAction(tr("tray.show"), self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)

        quit_action = QAction(tr("tray.quit"), self)
        quit_action.triggered.connect(QApplication.quit)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()

    def closeEvent(self, event):
        # Closing the window used to silently minimize to tray, so users who
        # clicked X believed the app (and its loaded models) had exited while
        # it kept running and holding VRAM. Ask explicitly instead.
        box = QMessageBox(self)
        box.setWindowTitle(tr("quit.title"))
        box.setText(tr("quit.message"))
        box.setInformativeText(tr("quit.hint"))
        box.setIcon(QMessageBox.Question)
        tray_btn = box.addButton(tr("quit.to_tray"), QMessageBox.AcceptRole)
        quit_btn = box.addButton(tr("quit.exit"), QMessageBox.DestructiveRole)
        box.setDefaultButton(tray_btn)
        box.exec()

        if box.clickedButton() is quit_btn:
            event.accept()
            self.tray_icon.hide()
            QApplication.quit()
        else:
            event.ignore()
            self.hide()
            self.tray_icon.showMessage(
                "掬水",
                tr("tray.minimized"),
                QSystemTrayIcon.Information,
                2000,
            )
