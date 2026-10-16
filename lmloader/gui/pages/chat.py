"""Chat workbench page — conversations with streaming."""

import os
import re
import base64
import time
import asyncio
import threading
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QTextEdit, QPushButton, QLabel, QComboBox, QSplitter,
    QMenu, QMessageBox, QGraphicsOpacityEffect, QFrame, QScrollArea,
    QFileDialog, QApplication,
)
from PySide6.QtCore import Qt, QThread, QEvent, QTimer, QSize
from PySide6.QtGui import QFont, QAction, QCursor, QFontMetrics, QIcon

from lmloader.gui.api_client import (
    LMLoaderAPI,
    ChatStreamWorker,
    run_in_thread,
    DEFAULT_MAX_TOKENS,
)
from lmloader.gui.icons import copy_icon, download_icon
from lmloader.gui.i18n import tr
from lmloader.gui.repetition import detect_repetition_loop
from lmloader.gui.styles import c
from lmloader.gui.ui_config import load_ui_setting, save_ui_setting
from lmloader.infra.chat_db import chat_db


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tga"}
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".py", ".js",
    ".ts", ".html", ".css", ".csv", ".log", ".ini", ".toml", ".xml",
}


def _run_async_in_thread(async_fn):
    """Run an async function in a background thread."""
    def _wrapper():
        asyncio.run(async_fn())
    threading.Thread(target=_wrapper, daemon=True).start()


class ConversationRow(QWidget):
    def __init__(self, title: str, on_select, on_delete, parent=None):
        super().__init__(parent)
        self._on_select = on_select
        self._title = title
        self._hovered = False
        self.setMinimumHeight(40)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 6, 4)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.title_label.setMinimumWidth(0)
        self.title_label.setToolTip(title)
        layout.addWidget(self.title_label, 1)

        self.delete_button = QPushButton("×")
        self.delete_button.setObjectName("conversationDelete")
        self.delete_button.setFixedSize(26, 26)
        self.delete_button.setCursor(Qt.PointingHandCursor)
        self.delete_button.setToolTip(tr("chat.delete"))
        self.delete_button.setAccessibleName(f"{tr('chat.delete')}: {title}")
        self._delete_opacity = QGraphicsOpacityEffect(self.delete_button)
        self._delete_opacity.setOpacity(0)
        self.delete_button.setGraphicsEffect(self._delete_opacity)
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(on_delete)
        layout.addWidget(self.delete_button)

        self.setStyleSheet(f"""
            ConversationRow {{
                background-color: transparent;
                border-radius: 10px;
            }}
            QLabel {{
                color: {c('muted')};
                background: transparent;
            }}
            QPushButton#conversationDelete {{
                padding: 0;
                min-height: 0;
                color: {c('muted')};
                background: transparent;
                border: none;
                border-radius: 8px;
                font-size: 18px;
            }}
            QPushButton#conversationDelete:hover {{
                color: {c('red')};
                background: {c('red_bg')};
            }}
        """)
        self._watched_widgets = [self, self.delete_button]
        for widget in self._watched_widgets:
            widget.installEventFilter(self)

    def set_title(self, title: str):
        self._title = title
        self.title_label.setToolTip(title)
        self.delete_button.setAccessibleName(f"{tr('chat.delete')}: {title}")
        self._update_title()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_title)

    def _update_title(self):
        available_width = self.title_label.width()
        if available_width > 0:
            metrics = QFontMetrics(self.title_label.font())
            self.title_label.setText(
                metrics.elidedText(self._title, Qt.ElideRight, available_width)
            )

    def set_active(self, active: bool):
        self._active = active
        self._apply_style()

    def _set_hovered(self, hovered: bool):
        if self._hovered == hovered:
            return
        self._hovered = hovered
        self._delete_opacity.setOpacity(1 if hovered else 0)
        self.delete_button.setEnabled(hovered)
        self._apply_style()

    def _apply_style(self):
        background = (
            c("accent_surface") if getattr(self, "_active", False)
            else c("surface_hover") if self._hovered else "transparent"
        )
        title_color = c("accent") if getattr(self, "_active", False) else c("muted")
        title_weight = "600" if getattr(self, "_active", False) else "400"
        self.setStyleSheet(f"""
            ConversationRow {{
                background-color: {background};
                border-radius: 10px;
            }}
            QLabel {{
                color: {title_color};
                background: transparent;
                font-weight: {title_weight};
            }}
            QPushButton#conversationDelete {{
                padding: 0;
                min-height: 0;
                color: {c('muted')};
                background: transparent;
                border: none;
                border-radius: 8px;
                font-size: 18px;
            }}
            QPushButton#conversationDelete:hover {{
                color: {c('red')};
                background: {c('red_bg')};
            }}
        """)

    def eventFilter(self, watched, event):
        if watched in self._watched_widgets:
            if event.type() == QEvent.Enter:
                self._set_hovered(True)
            elif event.type() == QEvent.Leave:
                QTimer.singleShot(0, self._update_hover_from_cursor)
        return super().eventFilter(watched, event)

    def _update_hover_from_cursor(self):
        self._set_hovered(self.rect().contains(self.mapFromGlobal(QCursor.pos())))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._on_select()
        super().mousePressEvent(event)


class ChatPage(QWidget):
    def __init__(self, api: LMLoaderAPI):
        super().__init__()
        self.api = api
        self.conversations: list[dict] = []
        self.current_conv_id: int | None = None
        self._stream_thread: QThread | None = None
        self._stream_worker: ChatStreamWorker | None = None
        self._retired_stream_threads: set[QThread] = set()
        # Repetition-loop guard state, reset on every send (see
        # _check_repetition_guard). _repetition_tripped latches so the guard
        # fires at most once per turn.
        self._repetition_tripped = False
        self._repetition_info: dict | None = None
        self._is_streaming = False
        self._stream_model_name: str | None = None
        self._stream_started_at = 0.0
        self._stream_generation = 0
        self._stats_refreshing = False
        self._stats_final_pending = False
        self._db_ready = False
        self._deleting_conversations: set[int] = set()
        # Loaded models whose engine is a Decision 3.0 package: they answer
        # /v1/systemone questions, not chat completions.
        self._decision_models: set[str] = set()
        # Model dropdown state: name -> loaded, plus guards so repopulating or
        # restoring a conversation never retriggers an auto-load.
        self._model_loaded_map: dict[str, bool] = {}
        self._programmatic_change = False
        self._model_switching = False
        self._message_rows: list[QWidget] = []
        self._rendered_conv: dict | None = None
        self._attachments: list[dict] = []   # {"name", "path", "text"}
        # System prompt panel state. _prompt_syncing guards the programmatic
        # setText() path (switching conversations) from re-triggering a save.
        self._prompt_syncing = False
        self._init_ui()
        self._stream_render_timer = QTimer(self)
        self._stream_render_timer.setInterval(100)
        self._stream_render_timer.timeout.connect(self._render_stream_update)
        self._resize_render_timer = QTimer(self)
        self._resize_render_timer.setSingleShot(True)
        self._resize_render_timer.setInterval(120)
        self._resize_render_timer.timeout.connect(
            lambda: self._render_messages(preserve_scroll=True)
        )
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(750)
        self._stats_timer.timeout.connect(self._poll_generation_stats)
        self._prompt_save_timer = QTimer(self)
        self._prompt_save_timer.setSingleShot(True)
        self._prompt_save_timer.setInterval(600)
        self._prompt_save_timer.timeout.connect(self._persist_prompt)
        self._load_models()
        self._init_chat_db()

    def _init_chat_db(self):
        from lmloader.gui.api_client import TaskWorker
        from PySide6.QtCore import QThread

        def _init_db():
            async def _do_init():
                await chat_db.init()
                return True
            return asyncio.run(_do_init())

        worker = TaskWorker(_init_db)
        self._db_thread = QThread()
        worker.moveToThread(self._db_thread)
        self._db_thread.started.connect(worker.run)
        # Bound-method callbacks run in the main thread via queued connections.
        worker.finished.connect(self._on_db_ready)
        worker.error.connect(self._on_db_error)
        worker.finished.connect(self._db_thread.quit)
        worker.error.connect(self._db_thread.quit)
        self._db_thread.finished.connect(self._db_thread.deleteLater)
        self._db_thread._worker = worker
        self._db_thread.start()

    def _on_db_ready(self, result):  # main thread
        self._db_ready = True
        asyncio.run(self._load_conversations())

    def _on_db_error(self, err):  # main thread
        self._db_ready = False

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_chat_area())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 800])

        main_layout.addWidget(splitter)

    def _build_sidebar(self) -> QWidget:
        # NOTE: objectName must NOT be "sidebar" — the main window's dark rail
        # owns that id and the global #sidebar rule would paint this panel dark.
        sidebar = QWidget()
        sidebar.setObjectName("chatSidebar")
        sidebar.setMaximumWidth(320)
        sidebar.setMinimumWidth(210)
        sidebar.setStyleSheet(
            f"QWidget#chatSidebar {{ background-color: {c('surface')};"
            f" border-right: 1px solid {c('border')}; }}"
        )
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        heading = QLabel(tr("chat.history"))
        heading.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {c('title')};")
        header.addWidget(heading, 1)
        self.export_btn = QPushButton("⤓")
        self.export_btn.setFixedSize(30, 30)
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_btn.setToolTip(tr("chat.save_conv"))
        self.export_btn.clicked.connect(self._save_conversation)
        self.export_btn.setStyleSheet(self._icon_btn_qss())
        header.addWidget(self.export_btn)
        layout.addLayout(header)

        new_btn = QPushButton(tr("chat.new_chat"))
        new_btn.setObjectName("primary")
        new_btn.clicked.connect(self._new_conversation)
        new_btn.setCursor(Qt.PointingHandCursor)
        new_btn.setMinimumHeight(38)
        new_btn.setStyleSheet(
            f"QPushButton#primary {{ background-color: {c('accent')}; border-color: {c('accent')};"
            f" color: white; font-weight: 600; }}"
            f" QPushButton#primary:hover {{ background-color: {c('accent_hover')}; }}"
        )
        layout.addWidget(new_btn)

        self.conv_list = QListWidget()
        self.conv_list.setStyleSheet(f"""
            QListWidget {{
                background-color: transparent;
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                padding: 0;
                border: none;
                background: transparent;
            }}
        """)
        self.conv_list.setSpacing(2)
        self.conv_list.currentRowChanged.connect(self._switch_conversation)
        self.conv_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self._show_conv_menu)
        layout.addWidget(self.conv_list, 1)

        return sidebar

    def _build_chat_area(self) -> QWidget:
        area = QWidget()
        area.setObjectName("chatArea")
        layout = QVBoxLayout(area)
        layout.setContentsMargins(20, 12, 20, 14)
        layout.setSpacing(10)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 8)
        self.chat_title = QLabel(tr("chat.new_chat_title"))
        self.chat_title.setMaximumWidth(360)
        self.chat_title.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {c('title')};")
        self._chat_title_full = ""
        self._set_chat_title(tr("chat.new_chat_title"))
        top_bar.addWidget(self.chat_title)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # 系统提示词摘要条：面板收起时也始终看得见模型被交代了什么。
        self.prompt_bar = QFrame()
        self.prompt_bar.setObjectName("chatPromptBar")
        self.prompt_bar.setStyleSheet(
            f"QFrame#chatPromptBar {{ background: {c('accent_soft')};"
            f" border: 1px solid {c('accent_border')}; border-radius: 8px; }}"
            " QFrame#chatPromptBar QLabel { background: transparent; }"
        )
        bar_layout = QHBoxLayout(self.prompt_bar)
        bar_layout.setContentsMargins(10, 5, 8, 5)
        bar_layout.setSpacing(8)
        bar_title = QLabel(tr("chat.prompt_title"))
        bar_title.setStyleSheet(f"color: {c('accent')}; font-size: 12px; font-weight: 600;")
        bar_layout.addWidget(bar_title)
        self.prompt_bar_preview = QLabel("")
        self.prompt_bar_preview.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
        bar_layout.addWidget(self.prompt_bar_preview, 1)
        bar_edit = QPushButton(tr("chat.prompt_edit"))
        bar_edit.setCursor(Qt.PointingHandCursor)
        bar_edit.setStyleSheet(self._mini_action_qss())
        bar_edit.clicked.connect(self._open_prompt_panel)
        bar_layout.addWidget(bar_edit)
        self.prompt_bar.setVisible(False)
        layout.addWidget(self.prompt_bar)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("chatDisplay")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.NoFrame)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chat_scroll.setStyleSheet(
            f"QScrollArea#chatDisplay {{ background-color: {c('bg')}; border: none; }}"
            f" QScrollArea#chatDisplay QWidget#chatMessages {{ background-color: {c('bg')}; }}"
        )
        self.chat_messages = QWidget()
        self.chat_messages.setObjectName("chatMessages")
        self.chat_messages_layout = QVBoxLayout(self.chat_messages)
        self.chat_messages_layout.setContentsMargins(0, 12, 0, 16)
        self.chat_messages_layout.setSpacing(14)
        self.chat_scroll.setWidget(self.chat_messages)
        layout.addWidget(self.chat_scroll, 1)

        composer = QWidget()
        composer.setObjectName("chatComposer")
        composer.setStyleSheet(f"""
            QWidget#chatComposer {{
                background-color: {c('surface')};
                border: 1px solid {c('border')};
                border-radius: 12px;
            }}
        """)
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(10, 8, 10, 9)
        composer_layout.setSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        self._thinking_on = True

        # Model switcher sits right above the input so you can change model
        # mid-chat. It lists every model; picking an unloaded one loads it.
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("chatModelCombo")
        self.model_combo.setMinimumWidth(200)
        self.model_combo.setMaximumWidth(340)
        self.model_combo.setCursor(Qt.PointingHandCursor)
        self.model_combo.setToolTip(tr("chat.model_tip"))
        self.model_combo.setStyleSheet(self._model_combo_qss())
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        toolbar.addWidget(self.model_combo)

        refresh_btn = QPushButton("⟳")
        refresh_btn.setObjectName("chatModelRefresh")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setToolTip(tr("chat.refresh"))
        refresh_btn.clicked.connect(self._load_models)
        refresh_btn.setStyleSheet(self._chip_btn_qss())
        toolbar.addWidget(refresh_btn)

        self.attach_btn = QPushButton(tr("chat.attach"))
        self.attach_btn.setCursor(Qt.PointingHandCursor)
        self.attach_btn.setToolTip(tr("chat.attach_tip"))
        self.attach_btn.clicked.connect(self._pick_attachment)
        self.attach_btn.setStyleSheet(self._chip_btn_qss())
        toolbar.addWidget(self.attach_btn)

        self.thinking_btn = QPushButton(tr("chat.thinking"))
        self.thinking_btn.setCheckable(True)
        self.thinking_btn.setChecked(True)
        self.thinking_btn.setCursor(Qt.PointingHandCursor)
        self.thinking_btn.toggled.connect(self._on_thinking_toggled)
        toolbar.addWidget(self.thinking_btn)

        self.prompt_btn = QPushButton(tr("chat.prompt"))
        self.prompt_btn.setCheckable(True)
        self.prompt_btn.setCursor(Qt.PointingHandCursor)
        self.prompt_btn.setToolTip(tr("chat.prompt_tip"))
        self.prompt_btn.setStyleSheet(self._prompt_chip_qss())
        self.prompt_btn.toggled.connect(self._on_prompt_panel_toggled)
        toolbar.addWidget(self.prompt_btn)
        toolbar.addStretch()
        input_hint = QLabel(tr("chat.input_hint"))
        input_hint.setStyleSheet(f"color: {c('muted')}; font-size: 11px;")
        toolbar.addWidget(input_hint)
        composer_layout.addLayout(toolbar)

        # Attached-file chips (hidden until something is attached).
        self.attach_bar = QWidget()
        self.attach_bar.setObjectName("chatAttachBar")
        self.attach_layout = QHBoxLayout(self.attach_bar)
        self.attach_layout.setContentsMargins(0, 2, 0, 0)
        self.attach_layout.setSpacing(6)
        self.attach_layout.addStretch()
        self.attach_bar.setVisible(False)
        composer_layout.addWidget(self.attach_bar)
        self._update_thinking_btn()

        self._prompt_panel = self._build_prompt_panel()
        composer_layout.addWidget(self._prompt_panel)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(10)

        self.input_box = QTextEdit()
        self.input_box.setObjectName("chat_input")
        self.input_box.setPlaceholderText(tr("chat.input_placeholder"))
        self.input_box.setMaximumHeight(72)
        self.input_box.setMinimumHeight(44)
        self.input_box.setStyleSheet(
            f"QTextEdit#chat_input {{ background-color: transparent; border: none;"
            f" border-radius: 0; padding: 7px; color: {c('fg')}; }}"
        )
        self.input_box.installEventFilter(self)
        bottom.addWidget(self.input_box, 1)

        send_btn = QPushButton(tr("chat.send"))
        send_btn.setObjectName("primary")
        send_btn.setMinimumSize(78, 36)
        send_btn.setCursor(Qt.PointingHandCursor)
        send_btn.setStyleSheet(
            f"QPushButton#primary {{ background-color: {c('accent')}; border-color: {c('accent')};"
            f" color: white; font-weight: 600; }}"
            f" QPushButton#primary:hover {{ background-color: {c('accent_hover')}; }}"
            # Streaming state: the same button becomes the stop control. Without
            # it a runaway generation can only be ended by closing the app.
            f" QPushButton#primary[danger='true'] {{ background-color: {c('red')};"
            f" border-color: {c('red')}; color: white; font-weight: 600; }}"
            f" QPushButton#primary[danger='true']:hover {{ border-color: {c('red')}; }}"
        )
        send_btn.clicked.connect(self._on_send_or_stop)
        self.send_btn = send_btn
        bottom.addWidget(send_btn)

        composer_layout.addLayout(bottom)
        layout.addWidget(composer)

        self._toast_label = QLabel("")
        self._toast_label.setObjectName("chatToast")
        self._toast_label.setAlignment(Qt.AlignCenter)
        self._toast_label.setWordWrap(True)
        self._toast_label.setVisible(False)
        self._toast_label.setStyleSheet(
            f"QLabel#chatToast {{ background: {c('green_bg')}; color: {c('green')};"
            f" border: 1px solid {c('green')}; border-radius: 10px;"
            " padding: 6px 10px; font-size: 11px; }}"
        )
        layout.addWidget(self._toast_label)
        return area

    def _on_thinking_toggled(self, checked: bool):
        self._thinking_on = checked
        self._update_thinking_btn()

    def _update_send_button_state(self):
        if not hasattr(self, "send_btn"):
            return
        current_is_deleting = (
            self.current_conv_id is not None
            and 0 <= self.current_conv_id < len(self.conversations)
            and id(self.conversations[self.current_conv_id]) in self._deleting_conversations
        )
        # Role swap by streaming state, mirroring the download page's task rows:
        # idle -> "send" (accent), streaming -> "stop" (danger). Disabled only
        # while a conversation is being deleted.
        if self._is_streaming:
            self.send_btn.setText(tr("chat.stop"))
            self.send_btn.setProperty("danger", "true")
            self.send_btn.setEnabled(True)
            self.send_btn.setToolTip(tr("chat.stop_tip"))
        else:
            self.send_btn.setText(tr("chat.send"))
            self.send_btn.setProperty("danger", "false")
            self.send_btn.setEnabled(not current_is_deleting)
            self.send_btn.setToolTip("")
        # Re-polish so the QSS property selector above takes effect.
        self.send_btn.style().unpolish(self.send_btn)
        self.send_btn.style().polish(self.send_btn)

    def _on_send_or_stop(self):
        """The send button doubles as the stop button while streaming."""
        if self._is_streaming:
            self._stop_stream()
        else:
            self._send_message()

    def _stop_stream(self):
        """Cancel the in-flight generation at the user's request."""
        worker = self._stream_worker
        if worker is not None:
            worker.stop()
        self._on_stream_finished(truncated=True, stopped=True)

    def _update_thinking_btn(self):
        btn = self.thinking_btn
        btn.setText(tr("chat.thinking_on") if self._thinking_on else tr("chat.thinking_off"))
        if self._thinking_on:
            btn.setStyleSheet(
                f"QPushButton {{ color: {c('accent')}; background: {c('accent_soft')};"
                f" border: 1px solid {c('accent_border')}; border-radius: 12px;"
                " padding: 3px 14px; font-size: 12px; font-weight: 600; }"
            )
            btn.setToolTip(tr("chat.thinking_on_tip"))
        else:
            btn.setStyleSheet(
                f"QPushButton {{ color: {c('muted')}; background: transparent;"
                f" border: 1px solid {c('border')}; border-radius: 12px;"
                " padding: 3px 14px; font-size: 12px; }"
            )
            btn.setToolTip(tr("chat.thinking_off_tip"))

    # ------------------------------------------------------------ system prompt

    @staticmethod
    def _prompt_chip_qss() -> str:
        return (
            f"QPushButton {{ color: {c('muted')}; background: transparent;"
            f" border: 1px solid {c('chip_border')}; border-radius: 12px;"
            " padding: 3px 12px; font-size: 12px; }"
            f" QPushButton:hover {{ color: {c('accent')}; background: {c('accent_soft')};"
            f" border-color: {c('accent_border')}; }}"
            f" QPushButton:checked {{ color: {c('accent')}; background: {c('accent_soft')};"
            f" border-color: {c('accent_border')}; }}"
            f' QPushButton[has="true"] {{ color: {c("green")};'
            f" border-color: {c('green')}66; }}"
        )

    def _prompt_templates(self) -> list[tuple[str, str]]:
        return [
            (tr("chat.prompt_tpl1_name"), tr("chat.prompt_tpl1_body")),
            (tr("chat.prompt_tpl2_name"), tr("chat.prompt_tpl2_body")),
            (tr("chat.prompt_tpl3_name"), tr("chat.prompt_tpl3_body")),
        ]

    def _build_prompt_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("chatPromptPanel")
        panel.setStyleSheet(
            f"QFrame#chatPromptPanel {{ background: {c('bg')};"
            f" border: 1px solid {c('accent_border')}; border-radius: 10px; }}"
            " QFrame#chatPromptPanel QLabel { background: transparent; }"
        )
        box = QVBoxLayout(panel)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(8)
        title = QLabel(tr("chat.prompt_title"))
        title.setStyleSheet(f"color: {c('accent')}; font-size: 12px; font-weight: 600;")
        head.addWidget(title)

        self.prompt_scope = QComboBox()
        self.prompt_scope.setCursor(Qt.PointingHandCursor)
        self.prompt_scope.setToolTip(tr("chat.prompt_scope_tip"))
        self.prompt_scope.addItem(tr("chat.prompt_scope_conv"), "conv")
        self.prompt_scope.addItem(tr("chat.prompt_scope_global"), "global")
        self.prompt_scope.currentIndexChanged.connect(self._on_prompt_scope_changed)
        head.addWidget(self.prompt_scope)

        self.prompt_tpl = QComboBox()
        self.prompt_tpl.setCursor(Qt.PointingHandCursor)
        self.prompt_tpl.setToolTip(tr("chat.prompt_tpl_tip"))
        self.prompt_tpl.addItem(tr("chat.prompt_template"), "")
        for name, body in self._prompt_templates():
            self.prompt_tpl.addItem(name, body)
        self.prompt_tpl.activated.connect(self._on_template_chosen)
        head.addWidget(self.prompt_tpl)

        head.addStretch()
        clear_btn = QPushButton(tr("chat.prompt_clear"))
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setStyleSheet(self._mini_action_qss())
        clear_btn.clicked.connect(self._clear_prompt)
        head.addWidget(clear_btn)
        close_btn = QPushButton(tr("chat.prompt_collapse"))
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(self._mini_action_qss())
        close_btn.clicked.connect(lambda: self.prompt_btn.setChecked(False))
        head.addWidget(close_btn)
        box.addLayout(head)

        self.prompt_box = QTextEdit()
        self.prompt_box.setObjectName("chatPromptBox")
        self.prompt_box.setPlaceholderText(tr("chat.prompt_placeholder"))
        self.prompt_box.setMinimumHeight(76)
        self.prompt_box.setMaximumHeight(150)
        self.prompt_box.setStyleSheet(
            f"QTextEdit#chatPromptBox {{ background: {c('surface')};"
            f" border: 1px solid {c('border')}; border-radius: 8px;"
            f" color: {c('fg')}; font-size: 13px; padding: 6px 8px; }}"
            f" QTextEdit#chatPromptBox:focus {{ border-color: {c('accent')}; }}"
        )
        self.prompt_box.textChanged.connect(self._on_prompt_text_changed)
        box.addWidget(self.prompt_box)

        foot = QHBoxLayout()
        foot.setSpacing(10)
        self.prompt_count = QLabel("")
        self.prompt_count.setStyleSheet(f"color: {c('fg')}; font-size: 11px;")
        foot.addWidget(self.prompt_count)
        self.prompt_state = QLabel("")
        self.prompt_state.setStyleSheet(f"color: {c('muted')}; font-size: 11px;")
        foot.addWidget(self.prompt_state)
        foot.addStretch()
        hint = QLabel(tr("chat.prompt_inject_hint"))
        hint.setStyleSheet(f"color: {c('dim')}; font-size: 11px;")
        foot.addWidget(hint)
        box.addLayout(foot)

        panel.setVisible(False)
        return panel

    def _current_conv(self) -> dict | None:
        if self.current_conv_id is None:
            return None
        if not (0 <= self.current_conv_id < len(self.conversations)):
            return None
        return self.conversations[self.current_conv_id]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count for the footer: CJK ≈ 1 token/char, ASCII ≈ 4 chars/token."""
        if not text:
            return 0
        cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
        other = len(text) - cjk
        return cjk + max(1, other // 4) if other else cjk

    def _open_prompt_panel(self):
        self.prompt_btn.setChecked(True)
        self.prompt_box.setFocus()

    def _on_prompt_panel_toggled(self, checked: bool):
        if not hasattr(self, "_prompt_panel"):
            return
        self._prompt_panel.setVisible(checked)
        if checked:
            self._apply_scope_from_conv()
        self._refresh_prompt_ui()

    def _on_template_chosen(self, index: int):
        body = self.prompt_tpl.itemData(index)
        if body is None:
            return
        self.prompt_box.setPlainText(body)
        self.prompt_tpl.setCurrentIndex(0)

    def _clear_prompt(self):
        self.prompt_box.clear()
        self.prompt_box.setFocus()

    def _on_prompt_text_changed(self):
        if self._prompt_syncing:
            return
        conv = self._current_conv()
        if conv is not None:
            conv["system_prompt"] = self.prompt_box.toPlainText()
        self._refresh_prompt_ui()
        self._prompt_save_timer.start()

    def _on_prompt_scope_changed(self, _index: int):
        if self._prompt_syncing:
            return
        text = self.prompt_box.toPlainText()
        mode = self.prompt_scope.currentData()
        try:
            if mode == "global":
                save_ui_setting("system_prompt", text)
            elif str(load_ui_setting("system_prompt", "") or "") == text:
                # Only un-set the global default when it is this same text, so
                # switching back never clobbers an unrelated default.
                save_ui_setting("system_prompt", "")
        except Exception as exc:
            self._prompt_state_error(str(exc))
            return
        self._refresh_prompt_ui()

    def _prompt_state_error(self, err: str):
        self.prompt_state.setText(tr("chat.prompt_save_fail").format(err=err))
        self.prompt_state.setStyleSheet(f"color: {c('warn')}; font-size: 11px;")

    def _apply_scope_from_conv(self):
        conv = self._current_conv()
        text = (conv or {}).get("system_prompt", "") if conv else ""
        glob = str(load_ui_setting("system_prompt", "") or "")
        idx = 1 if (text and glob and text.strip() == glob.strip()) else 0
        self._prompt_syncing = True
        self.prompt_scope.setCurrentIndex(idx)
        self._prompt_syncing = False

    def _apply_prompt_to_ui(self):
        """Reflect the current conversation's prompt into the panel and bar."""
        if not hasattr(self, "prompt_box"):
            return
        conv = self._current_conv()
        text = (conv or {}).get("system_prompt", "") if conv else ""
        self._prompt_syncing = True
        self.prompt_box.setPlainText(text)
        self._prompt_syncing = False
        self._apply_scope_from_conv()
        self._refresh_prompt_ui()

    def _refresh_prompt_ui(self):
        conv = self._current_conv()
        text = (conv or {}).get("system_prompt", "") if conv else ""
        if not self._prompt_syncing:
            text = self.prompt_box.toPlainText()
        has = bool(text.strip())
        mode = self.prompt_scope.currentData() or "conv"

        self.prompt_btn.setText(
            f"{tr('chat.prompt')} · {tr('chat.prompt_scope_global') if mode == 'global' else tr('chat.prompt_scope_conv')}"
            if has else tr("chat.prompt")
        )
        self.prompt_btn.setProperty("has", has)
        self.prompt_btn.style().unpolish(self.prompt_btn)
        self.prompt_btn.style().polish(self.prompt_btn)

        chars = len(text)
        self.prompt_count.setText(
            tr("chat.prompt_count").format(chars=chars, tok=self._estimate_tokens(text))
        )
        if not has:
            self.prompt_state.setText(tr("chat.prompt_state_none"))
            self.prompt_state.setStyleSheet(f"color: {c('muted')}; font-size: 11px;")
        elif mode == "global":
            self.prompt_state.setText(tr("chat.prompt_state_global"))
            self.prompt_state.setStyleSheet(f"color: {c('green')}; font-size: 11px;")
        else:
            self.prompt_state.setText(tr("chat.prompt_state_conv"))
            self.prompt_state.setStyleSheet(f"color: {c('muted')}; font-size: 11px;")

        self.prompt_bar.setVisible(has)
        if has:
            first = text.strip().splitlines()[0]
            avail = max(60, self.prompt_bar_preview.width() or self.prompt_bar.width() - 190)
            fm = self.prompt_bar_preview.fontMetrics()
            self.prompt_bar_preview.setText(
                fm.elidedText(first, Qt.TextElideMode.ElideRight, avail)
            )

    def _persist_prompt(self):
        conv = self._current_conv()
        if conv is None:
            return
        db_id = conv.get("db_id")
        if db_id is None or not self._db_ready:
            return  # stored by create_conversation on the first send
        text = conv.get("system_prompt", "")
        _run_async_in_thread(lambda: chat_db.update_system_prompt(db_id, text))

    # ------------------------------------------------------------ attachments

    @staticmethod
    def _chip_btn_qss() -> str:
        return (
            f"QPushButton {{ color: {c('muted')}; background: transparent;"
            f" border: 1px solid {c('chip_border')}; border-radius: 12px;"
            " padding: 3px 12px; font-size: 12px; }"
            f" QPushButton:hover {{ color: {c('accent')}; background: {c('accent_soft')};"
            f" border-color: {c('accent_border')}; }}"
        )

    @staticmethod
    def _model_combo_qss() -> str:
        return (
            f"QComboBox#chatModelCombo {{ color: {c('fg')}; background: {c('surface')};"
            f" border: 1px solid {c('chip_border')}; border-radius: 12px;"
            " padding: 3px 10px; font-size: 12px; }"
            f" QComboBox#chatModelCombo:hover {{ border-color: {c('accent_border')}; }}"
            f" QComboBox#chatModelCombo::drop-down {{ border: none; width: 18px; }}"
            f" QComboBox#chatModelCombo QAbstractItemView {{"
            f" background: {c('surface')}; color: {c('fg')};"
            f" border: 1px solid {c('border')}; selection-background-color: {c('accent_soft')};"
            f" selection-color: {c('fg')}; }}"
        )

    @staticmethod
    def _icon_btn_qss() -> str:
        return (
            f"QPushButton {{ color: {c('muted')}; background: transparent;"
            f" border: 1px solid {c('chip_border')}; border-radius: 8px;"
            " padding: 0; min-height: 0; font-size: 14px; font-weight: 700; }"
            f" QPushButton:hover {{ color: {c('accent')}; background: {c('accent_soft')};"
            f" border-color: {c('accent_border')}; }}"
        )

    @staticmethod
    def _mini_action_qss() -> str:
        return (
            f"QPushButton {{ color: {c('muted')}; background: transparent;"
            " border: none; border-radius: 6px;"
            " padding: 2px 6px; min-height: 0; font-size: 11px; }"
            f" QPushButton:hover {{ color: {c('accent')}; background: {c('accent_soft')}; }}"
        )

    def _pick_attachment(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("chat.attach"), "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp);;"
            "Text files (*.txt *.md *.markdown *.json *.yaml *.yml *.py *.js "
            "*.ts *.html *.css *.csv *.log *.ini *.toml *.xml);;"
            "All files (*)",
        )
        for path in paths:
            self._add_attachment(path)

    # A 200 KB ceiling was arbitrary and blocked any real document — and every
    # image, since even a thumbnail exceeds it. Text is now read whole and the
    # model's own context window is the real bound; this cap only guards
    # against pasting a multi-hundred-megabyte log into the prompt.
    # Measured in CHARACTERS, not bytes: UTF-8 CJK costs 3 bytes per char, so
    # a byte limit would silently cut Chinese text to a third of its length.
    TEXT_PREVIEW_LIMIT = 400_000  # characters

    def _add_attachment(self, path: str):
        name = os.path.basename(path)
        ext = os.path.splitext(name)[1].lower()
        if any(a["path"] == path for a in self._attachments):
            return

        if ext in IMAGE_EXTS:
            entry = self._make_image_attachment(path, name)
            if entry is None:
                return
        else:
            entry = self._make_text_attachment(path, name, ext)
            if entry is None:
                return

        self._attachments.append(entry)
        self._render_attachments()

    def _make_image_attachment(self, path: str, name: str):
        """Read an image and encode it as a data URI for the vision path."""
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except Exception as exc:
            QMessageBox.warning(
                self, tr("chat.attach"),
                tr("chat.attach_read_error").format(err=exc),
            )
            return None
        if not raw:
            return None
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
            ".tga": "image/x-tga",
        }.get(os.path.splitext(name)[1].lower(), "application/octet-stream")
        b64 = base64.b64encode(raw).decode("ascii")
        return {
            "name": name,
            "path": path,
            "kind": "image",
            "text": "",
            "data_uri": f"data:{mime};base64,{b64}",
            "size": len(raw),
        }

    def _make_text_attachment(self, path: str, name: str, ext: str):
        """Read a text file. Oversized files are truncated with a marker."""
        try:
            size = os.path.getsize(path)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read(self.TEXT_PREVIEW_LIMIT + 1)
        except Exception as exc:
            QMessageBox.warning(
                self, tr("chat.attach"),
                tr("chat.attach_read_error").format(err=exc),
            )
            return None
        truncated = False
        if len(text) > self.TEXT_PREVIEW_LIMIT:
            text = text[: self.TEXT_PREVIEW_LIMIT]
            truncated = True
        entry = {
            "name": name,
            "path": path,
            "kind": "text",
            "text": text,
            "data_uri": "",
            "size": size,
        }
        if truncated:
            entry["note"] = tr("chat.attach_truncated").format(
                shown=len(text), total=size)
        elif ext and ext not in TEXT_EXTS:
            # Binary file read as text: say so rather than pretending the
            # garbage is meaningful.
            entry["note"] = tr("chat.attach_binary")
        return entry

    def _remove_attachment(self, attachment: dict):
        self._attachments = [a for a in self._attachments if a is not attachment]
        self._render_attachments()

    def _render_attachments(self):
        # Drop every chip (and the trailing stretch) then rebuild.
        while self.attach_layout.count():
            item = self.attach_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for attachment in self._attachments:
            chip = QFrame()
            chip.setObjectName("attachChip")
            chip.setStyleSheet(
                f"QFrame#attachChip {{ background: {c('chip_bg')};"
                f" border: 1px solid {c('chip_border')}; border-radius: 10px; }}"
            )
            chip_layout = QHBoxLayout(chip)
            chip_layout.setContentsMargins(8, 3, 4, 3)
            chip_layout.setSpacing(6)
            label = QLabel(attachment["name"])
            label.setStyleSheet(f"color: {c('fg')}; font-size: 11px; border: none;")
            kind = attachment.get("kind", "text")
            size_mb = attachment.get("size", 0) / (1024 * 1024)
            tip = attachment["path"]
            if kind == "image":
                tip = f"{tr('chat.attach_image')}\n{tip}"
            elif attachment.get("note"):
                tip = f"{attachment['note']}\n{tip}"
            if size_mb >= 0.01:
                tip += f"\n{size_mb:.2f} MB"
            label.setToolTip(tip)
            label.setMaximumWidth(160)
            chip_layout.addWidget(label)
            remove = QPushButton("×")
            remove.setFixedSize(18, 18)
            remove.setCursor(Qt.PointingHandCursor)
            remove.setToolTip(tr("chat.delete"))
            remove.setStyleSheet(self._mini_action_qss())
            remove.clicked.connect(
                lambda checked=False, a=attachment: self._remove_attachment(a)
            )
            chip_layout.addWidget(remove)
            self.attach_layout.addWidget(chip)
        self.attach_layout.addStretch()
        self.attach_bar.setVisible(bool(self._attachments))

    def _attachment_payload(self) -> str:
        """Render text attachments as fenced blocks appended to the outgoing text.

        Images are NOT included here — they travel as OpenAI ``image_url``
        content parts so the vision encoder actually receives the pixels
        (see _build_user_content).
        """
        parts = []
        for attachment in self._attachments:
            if attachment.get("kind") == "image":
                continue
            note = attachment.get("note", "")
            head = f"**[{tr('chat.attachment')}] {attachment['name']}**"
            if note:
                head += f"\n_{note}_"
            parts.append(f"{head}\n```\n{attachment.get('text', '')}\n```")
        if not parts:
            return ""
        return "\n\n" + "\n\n".join(parts)

    def _image_attachments(self) -> list[dict]:
        return [a for a in self._attachments if a.get("kind") == "image"]

    def _build_user_content(self, text: str) -> object:
        """Build the outgoing user message content.

        Text-only turns stay a plain string (what the engine and the stored
        transcript expect). When images are attached the content becomes the
        OpenAI list form: [text?, image_url parts...]. The backend proxies the
        request body to llama-server verbatim, so no server change is needed.
        """
        images = self._image_attachments()
        if not images:
            return text
        parts: list[dict] = []
        if text:
            parts.append({"type": "text", "text": text})
        for img in images:
            parts.append({
                "type": "image_url",
                "image_url": {"url": img.get("data_uri", "")},
            })
        return parts

    # ------------------------------------------------------- copy / save text

    def _copy_text(self, text: str):
        QApplication.clipboard().setText(text)
        if hasattr(self, "_toast_label"):
            self._toast(tr("chat.copied"))

    def _save_text(self, text: str, default_name: str):
        if not text.strip():
            QMessageBox.information(self, tr("chat.save"), tr("chat.nothing_to_save"))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("chat.save_title"), default_name,
            "Markdown (*.md);;Text (*.txt)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except Exception as exc:
            QMessageBox.warning(
                self, tr("chat.save"), tr("chat.save_failed").format(err=exc)
            )
            return
        self._toast(tr("chat.save_done").format(path=path))

    def _message_action_row(self, text: str, suggested_name: str) -> QWidget:
        """Icon-only copy / download row, shown under assistant bubbles only.

        Text labels ("复制"/"保存") put a visible chrome row on every bubble;
        icons keep the bubble clean. The buttons stay always visible rather
        than hover-only, because a hover-only control cannot be reached by
        keyboard and disappears the moment the pointer moves away.
        """
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(2, 0, 0, 0)
        row_layout.setSpacing(2)
        row_layout.addWidget(self._icon_action(
            copy_icon(c("muted"), 15), tr("chat.copy"),
            lambda: self._copy_text(text)))
        row_layout.addWidget(self._icon_action(
            download_icon(c("muted"), 15), tr("chat.save"),
            lambda: self._save_text(text, suggested_name)))
        row_layout.addStretch()
        return row

    def _icon_action(self, icon: QIcon, tip: str, handler) -> QPushButton:
        """A square icon-only button; the tooltip carries the label."""
        btn = QPushButton()
        btn.setIcon(icon)
        btn.setIconSize(QSize(15, 15))
        btn.setFixedSize(24, 22)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(tip)
        # Transparent by default so the row reads as part of the bubble.
        btn.setStyleSheet(self._icon_action_qss())
        btn.clicked.connect(lambda checked=False: handler())
        return btn

    @staticmethod
    def _icon_action_qss() -> str:
        return (
            "QPushButton { background: transparent; border: none;"
            " border-radius: 6px; padding: 0px; }"
            f"QPushButton:hover {{ background: {c('accent_soft')}; }}"
            f"QPushButton:pressed {{ background: {c('accent_surface')}; }}"
        )

    def _save_conversation(self):
        conv = (
            self.conversations[self.current_conv_id]
            if self.current_conv_id is not None
            and 0 <= self.current_conv_id < len(self.conversations)
            else None
        )
        if conv is None or not conv["messages"]:
            QMessageBox.information(self, tr("chat.save"), tr("chat.nothing_to_save"))
            return
        lines = [f"# {conv['title']}", ""]
        for msg in conv["messages"]:
            who = tr("chat.you") if msg["role"] == "user" else tr("chat.assistant")
            lines.append(f"## {who}")
            lines.append(msg.get("content") or "")
            lines.append("")
        default_name = re.sub(r'[\\/:*?"<>|]', "_", conv["title"])[:60] + ".md"
        path, _ = QFileDialog.getSaveFileName(
            self, tr("chat.save_conv"), default_name,
            "Markdown (*.md);;Text (*.txt)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
        except Exception as exc:
            QMessageBox.warning(
                self, tr("chat.save"), tr("chat.save_failed").format(err=exc)
            )
            return
        self._toast(tr("chat.save_conv_done").format(path=path))

    def _toast(self, message: str):
        self._toast_label.setText(message)
        self._toast_label.setVisible(True)
        QTimer.singleShot(2600, lambda: self._toast_label.setVisible(False))

    def eventFilter(self, obj, event):
        if obj is self.input_box and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key_Return and not event.modifiers():
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def _new_conversation(self):
        conv = {
            "id": int(time.time() * 1000),
            "db_id": None,
            "model_name": "",
            "title": tr("chat.new_chat_title"),
            "messages": [],
            "system_prompt": str(load_ui_setting("system_prompt", "") or ""),
        }
        self.conversations.append(conv)
        row = len(self.conversations) - 1
        self._add_conversation_row(row)
        self.conv_list.setCurrentRow(row)

    async def _load_conversations(self):
        if not self._db_ready:
            return
        convs = await chat_db.list_conversations()
        for c in convs:
            messages = await chat_db.get_messages(c["id"])
            conv = {
                "id": c["id"],
                "db_id": c["id"],
                "model_name": c.get("model_name", ""),
                "title": c["title"],
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                "system_prompt": c.get("system_prompt", ""),
            }
            self.conversations.append(conv)
            self._add_conversation_row(len(self.conversations) - 1)
        if self.conversations:
            self.conv_list.setCurrentRow(0)
            if self.current_conv_id is None:
                self._switch_conversation(0)

    def _add_conversation_row(self, index: int):
        conv = self.conversations[index]
        item = QListWidgetItem()
        row_widget = ConversationRow(
            conv["title"],
            on_select=lambda item=conv: self._select_conversation(item),
            on_delete=lambda checked=False, item=conv: self._delete_conversation_item(item),
        )
        item.setSizeHint(QSize(0, 40))
        self.conv_list.addItem(item)
        self.conv_list.setItemWidget(item, row_widget)
        row_widget.set_active(index == self.current_conv_id)

    def _select_conversation(self, conv: dict):
        index = next(
            (row for row, item in enumerate(self.conversations) if item is conv),
            None,
        )
        if index is not None:
            self.conv_list.setCurrentRow(index)

    def _delete_conversation_item(self, conv: dict):
        index = next(
            (row for row, item in enumerate(self.conversations) if item is conv),
            None,
        )
        if index is not None:
            self._delete_conversation(index)

    def _switch_conversation(self, index):
        if index < 0 or index >= len(self.conversations):
            return
        self.current_conv_id = index
        self._set_chat_title(self.conversations[index]["title"])
        self._update_send_button_state()
        for row in range(self.conv_list.count()):
            widget = self.conv_list.itemWidget(self.conv_list.item(row))
            if isinstance(widget, ConversationRow):
                widget.set_active(row == index)
        self._render_messages()
        self._apply_prompt_to_ui()

    def _show_conv_menu(self, pos):
        item = self.conv_list.itemAt(pos)
        if item is None:
            return
        row = self.conv_list.row(item)
        menu = QMenu(self)
        act_delete = QAction(tr("chat.delete"), self)
        menu.addAction(act_delete)
        chosen = menu.exec(self.conv_list.viewport().mapToGlobal(pos))
        if chosen is act_delete:
            self._delete_conversation(row)

    def _delete_conversation(self, row: int | None = None):
        if row is None:
            row = self.conv_list.currentRow()
        if row is None or row < 0 or row >= len(self.conversations):
            return
        conv = self.conversations[row]
        if id(conv) in self._deleting_conversations:
            return
        if self._is_streaming and getattr(self, "_stream_conv", None) is conv:
            QMessageBox.information(
                self, tr("chat.delete"),
                tr("chat.delete_streaming"),
            )
            return
        reply = QMessageBox.question(
            self, tr("chat.delete"),
            tr("chat.delete_confirm").format(title=conv["title"]),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        db_id = conv.get("db_id")
        if db_id is not None:
            if not self._db_ready:
                QMessageBox.warning(self, tr("chat.delete"), tr("chat.delete_error").format(err="database is not ready"))
                return
            widget = self.conv_list.itemWidget(self.conv_list.item(row))
            if isinstance(widget, ConversationRow):
                widget.delete_button.setEnabled(False)
            self._deleting_conversations.add(id(conv))
            self._update_send_button_state()

            def on_deleted(_result):
                self._deleting_conversations.discard(id(conv))
                current_row = next(
                    (idx for idx, item in enumerate(self.conversations) if item is conv),
                    None,
                )
                if current_row is not None:
                    self._remove_conversation_at(current_row)
                self._update_send_button_state()

            def on_delete_error(err):
                self._deleting_conversations.discard(id(conv))
                current_row = next(
                    (idx for idx, item in enumerate(self.conversations) if item is conv),
                    None,
                )
                if current_row is not None:
                    widget = self.conv_list.itemWidget(self.conv_list.item(current_row))
                    if isinstance(widget, ConversationRow):
                        widget.delete_button.setEnabled(True)
                self._update_send_button_state()
                QMessageBox.warning(
                    self, tr("chat.delete"),
                    tr("chat.delete_error").format(err=err),
                )

            run_in_thread(
                lambda cid=db_id: asyncio.run(chat_db.delete_conversation(cid)),
                on_deleted,
                on_delete_error,
            )
            return

        self._remove_conversation_at(row)

    def _remove_conversation_at(self, row: int):
        current_conv = (
            self.conversations[self.current_conv_id]
            if self.current_conv_id is not None
            and 0 <= self.current_conv_id < len(self.conversations)
            else None
        )
        removed_conv = self.conversations[row]
        self.conv_list.blockSignals(True)
        self.conv_list.takeItem(row)
        self.conversations.pop(row)
        self.conv_list.setCurrentRow(-1)
        self.current_conv_id = None
        self.conv_list.blockSignals(False)

        if not self.conversations:
            self._clear_message_widgets()
            self._set_chat_title(tr("chat.new_chat_title"))
            return

        if current_conv is not None and current_conv is not removed_conv:
            new_row = self.conversations.index(current_conv)
        else:
            new_row = min(row, len(self.conversations) - 1)
        self.conv_list.setCurrentRow(new_row)
        self._switch_conversation(new_row)

    def _set_chat_title(self, title: str):
        self._chat_title_full = title
        self.chat_title.setToolTip(title)
        self._update_chat_title()

    def _update_chat_title(self):
        width = self.chat_title.width()
        if width > 0:
            metrics = QFontMetrics(self.chat_title.font())
            self.chat_title.setText(
                metrics.elidedText(self._chat_title_full, Qt.ElideRight, width)
            )

    @staticmethod
    def _content_to_text(content) -> str:
        """Flatten OpenAI-style list content into displayable text.

        A user turn that carried images is stored as
        [{type: text}, {type: image_url}, ...]. The transcript only shows
        the text plus a marker, so rendering must never assume a str.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            images = 0
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    images += 1
            out = "\n".join(t for t in texts if t)
            if images:
                marker = tr("chat.attached_images").format(n=images)
                out = f"{marker}\n{out}" if out else marker
            return out
        return "" if content is None else str(content)

    def _render_messages(self, preserve_scroll: bool = False):
        scrollbar = self.chat_scroll.verticalScrollBar()
        scroll_position = scrollbar.value()
        self._clear_message_widgets()
        if self.current_conv_id is None or self.current_conv_id >= len(self.conversations):
            self._rendered_conv = None
            self.chat_messages_layout.addStretch(1)
            return
        conv = self.conversations[self.current_conv_id]
        # Restoring the conversation's model must not fire the auto-load handler.
        self._programmatic_change = True
        try:
            model_index = self.model_combo.findData(conv.get("model_name", ""))
            self.model_combo.setCurrentIndex(max(0, model_index))
        finally:
            self._programmatic_change = False
        last_idx = len(conv["messages"]) - 1
        for msg_idx, msg in enumerate(conv["messages"]):
            role = msg["role"]
            content = self._content_to_text(msg["content"])
            if role == "user":
                self._append_message_bubble(tr("chat.you"), content, c("accent_surface"), c("accent"))
            elif role == "assistant":
                reasoning = msg.get("_reasoning")
                shown = content
                if not shown and reasoning:
                    shown = tr("chat.thinking")
                is_open = msg.get("_reasoning_open", False) or (
                    bool(reasoning) and self._is_streaming and msg_idx == last_idx
                )
                self._append_assistant_message(
                    conv.get("model_name") or tr("chat.assistant"),
                    shown, reasoning, msg_idx, is_open,
                    footer=msg.get("_stats", ""),
                )
        self._rendered_conv = conv
        self.chat_messages_layout.addStretch(1)
        QTimer.singleShot(
            0,
            lambda: scrollbar.setValue(
                scroll_position if preserve_scroll else scrollbar.maximum()
            ),
        )

    def _clear_message_widgets(self):
        self._message_rows.clear()
        self._rendered_conv = None
        while self.chat_messages_layout.count():
            item = self.chat_messages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _render_stream_update(self):
        conv = getattr(self, "_stream_conv", None)
        if (
            not self._is_streaming
            or conv is None
            or not self._is_current_conversation(conv)
            or self._rendered_conv is not conv
            or len(self._message_rows) != len(conv["messages"])
        ):
            return

        message = conv["messages"][-1]
        row = self._message_rows[-1]
        body = row.findChild(QLabel, "messageBody")
        if body is not None:
            content = message.get("content") or (
                tr("chat.thinking") if message.get("_reasoning") else ""
            )
            body.setText(self._md_to_html(content))
            body.adjustSize()

        reasoning = message.get("_reasoning", "")
        reasoning_card = row.findChild(QFrame, "reasoningCard")
        reasoning_button = row.findChild(QPushButton, "reasoningToggle")
        reasoning_text = row.findChild(QLabel, "reasoningText")
        if reasoning_card is not None:
            reasoning_card.setVisible(bool(reasoning))
        elif reasoning:
            self._render_messages()
            return
        if reasoning_button is not None:
            reasoning_button.setText(f"▾  {tr('chat.reasoning_title')}")
        if reasoning_text is not None and reasoning:
            reasoning_text.setText(self._md_to_html(reasoning))
            reasoning_text.setVisible(True)
            reasoning_text.adjustSize()

        stats = row.findChild(QLabel, "messageStats")
        if stats is None and message.get("_stats"):
            self._render_messages()
            return
        if stats is not None:
            value = message.get("_stats", "")
            stats.setText(value)
            stats.setVisible(bool(value))
        row.adjustSize()
        self.chat_messages.adjustSize()
        scrollbar = self.chat_scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: scrollbar.setValue(scrollbar.maximum()))

    def _bubble_width(self) -> int:
        # 0.72 keeps long assistant answers inside a comfortable reading
        # measure instead of spanning the full column.
        return max(1, min(680, int(self.chat_scroll.viewport().width() * 0.72)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "chat_title"):
            QTimer.singleShot(0, self._update_chat_title)
        if hasattr(self, "prompt_bar"):
            QTimer.singleShot(0, self._refresh_prompt_ui)
        if hasattr(self, "_resize_render_timer"):
            self._resize_render_timer.start()

    def _on_anchor_clicked(self, url):
        # Links are rendered as "#r<msg_idx>"; QUrl mangles numeric custom
        # schemes (reason://1 -> 0.0.0.1), so the index rides in the fragment.
        frag = url.fragment()
        if not frag.startswith("r"):
            return
        try:
            msg_idx = int(frag[1:])
        except ValueError:
            return
        if self.current_conv_id is None:
            return
        conv = self.conversations[self.current_conv_id]
        if msg_idx < 0 or msg_idx >= len(conv["messages"]):
            return
        msg = conv["messages"][msg_idx]
        if not msg.get("_reasoning"):
            return
        msg["_reasoning_open"] = not msg.get("_reasoning_open", False)
        sb = self.chat_scroll.verticalScrollBar()
        pos = sb.value()
        self._render_messages()
        sb.setValue(pos)

    def _create_message_row(
        self, sender: str, is_user: bool, content_width: int | None = None
    ):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)

        avatar_text = "你" if is_user else (sender[:1].upper() if sender else "AI")
        avatar = QLabel(avatar_text)
        avatar.setFixedSize(30, 30)
        avatar.setAlignment(Qt.AlignCenter)
        avatar_color = c("accent_surface") if is_user else c("accent")
        avatar_text = c("accent") if is_user else "#ffffff"
        avatar.setStyleSheet(
            f"QLabel {{ background-color: {avatar_color}; color: {avatar_text};"
            " border-radius: 10px; font-size: 11px; font-weight: 700; }"
        )

        content = QWidget()
        if content_width is not None:
            content.setFixedWidth(content_width)
        else:
            content.setMaximumWidth(720)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        sender_label = QLabel(sender)
        sender_label.setStyleSheet(
            f"color: {c('muted')}; font-size: 11px; font-weight: 600;"
        )
        content_layout.addWidget(sender_label)

        if is_user:
            row_layout.addStretch(1)
            row_layout.addWidget(content, 0, Qt.AlignRight | Qt.AlignTop)
            row_layout.addWidget(avatar, 0, Qt.AlignTop)
            sender_label.setAlignment(Qt.AlignRight)
        else:
            row_layout.addWidget(avatar, 0, Qt.AlignTop)
            row_layout.addWidget(content, 0, Qt.AlignLeft | Qt.AlignTop)
            row_layout.addStretch(1)
        return row, content_layout

    def _create_message_bubble(
        self, text: str, bg: str, border: str, width: int | None = None
    ):
        bubble = QFrame()
        bubble.setObjectName("messageBubble")
        bubble.setFixedWidth(width or self._bubble_width())
        bubble.setStyleSheet(
            f"QFrame#messageBubble {{ background-color: {bg}; border: 1px solid {border};"
            " border-radius: 12px; }"
        )
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(13, 10, 13, 10)
        bubble_layout.setSpacing(6)
        body = QLabel()
        body.setObjectName("messageBody")
        body.setTextFormat(Qt.RichText)
        body.setText(self._md_to_html(text))
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        body.setStyleSheet(
            f"QLabel {{ color: {c('fg')}; background: transparent; border: none;"
            " font-size: 12px; }"
        )
        bubble_layout.addWidget(body)
        return bubble

    def _append_message_bubble(
        self, sender: str, text: str, bg: str, border: str, footer: str = ""
    ):
        is_user = sender == tr("chat.you")
        bubble_width = (
            self._user_bubble_width(text) if is_user else self._bubble_width()
        )
        row, content_layout = self._create_message_row(
            sender, is_user, bubble_width
        )
        bubble = self._create_message_bubble(text, bg, border, bubble_width)
        content_layout.addWidget(
            bubble, 0, Qt.AlignRight if is_user else Qt.AlignLeft
        )
        if footer:
            stats = QLabel(footer)
            stats.setObjectName("messageStats")
            stats.setWordWrap(True)
            stats.setStyleSheet(f"color: {c('muted')}; font-size: 10px;")
            content_layout.addWidget(stats)
        # No copy/download row on the user's own message: copying what you
        # just typed back out of the transcript is not an action anyone needs,
        # and it doubled the chrome on every bubble.
        self.chat_messages_layout.addWidget(row)
        self._message_rows.append(row)

    @staticmethod
    def _suggest_filename(sender: str, text: str) -> str:
        head = " ".join((text or "").split())[:24] or sender
        return re.sub(r'[\\/:*?"<>|]', "_", head) + ".md"

    def _append_assistant_message(
        self, sender: str, text: str, reasoning: str | None, msg_idx: int,
        reasoning_open: bool, footer: str = "",
    ):
        bubble_width = self._bubble_width()
        row, content_layout = self._create_message_row(
            sender, False, bubble_width
        )
        if reasoning:
            reasoning_card = QFrame()
            reasoning_card.setObjectName("reasoningCard")
            reasoning_card.setStyleSheet(
                f"QFrame#reasoningCard {{ background-color: {c('surface')};"
                f" border: 1px solid {c('border')}; border-radius: 10px; }}"
            )
            reasoning_layout = QVBoxLayout(reasoning_card)
            reasoning_layout.setContentsMargins(10, 7, 10, 7)
            reasoning_layout.setSpacing(5)
            toggle = QPushButton(
                f"{'▾' if reasoning_open else '▸'}  {tr('chat.reasoning_title')}"
            )
            toggle.setObjectName("reasoningToggle")
            toggle.setFlat(True)
            toggle.setCursor(Qt.PointingHandCursor)
            toggle.setStyleSheet(
                f"QPushButton {{ color: {c('muted')}; border: none; text-align: left;"
                " padding: 0; font-size: 11px; font-weight: 600; }"
            )
            toggle.clicked.connect(
                lambda checked=False, index=msg_idx: self._toggle_reasoning(index)
            )
            reasoning_layout.addWidget(toggle)
            if reasoning_open:
                reasoning_text = QLabel(self._md_to_html(reasoning))
                reasoning_text.setObjectName("reasoningText")
                reasoning_text.setTextFormat(Qt.RichText)
                reasoning_text.setWordWrap(True)
                reasoning_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
                reasoning_text.setStyleSheet(
                    f"color: {c('muted')}; font-size: 11px; background: transparent;"
                    " border: none;"
                )
                reasoning_layout.addWidget(reasoning_text)
            content_layout.addWidget(reasoning_card, 0, Qt.AlignLeft)

        bubble = self._create_message_bubble(text, c("surface"), c("border"))
        content_layout.addWidget(bubble, 0, Qt.AlignLeft)
        if footer:
            stats = QLabel(footer)
            stats.setObjectName("messageStats")
            stats.setWordWrap(True)
            stats.setStyleSheet(f"color: {c('muted')}; font-size: 10px;")
            content_layout.addWidget(stats)
        content_layout.addWidget(
            self._message_action_row(text, self._suggest_filename(sender, text)),
            0, Qt.AlignLeft,
        )
        self.chat_messages_layout.addWidget(row)
        self._message_rows.append(row)

    def _user_bubble_width(self, text: str) -> int:
        import re
        plain = re.sub(r"<[^>]+>", "", text)
        plain = plain.replace("**", "").replace("`", "")
        metrics = QFontMetrics(QFont("Microsoft YaHei UI", 12))
        longest_line = max(
            (metrics.horizontalAdvance(line) for line in plain.splitlines()),
            default=0,
        )
        return min(self._bubble_width(), max(180, longest_line + 34))

    def _toggle_reasoning(self, msg_idx: int):
        if self.current_conv_id is None or self.current_conv_id >= len(self.conversations):
            return
        conv = self.conversations[self.current_conv_id]
        if msg_idx < 0 or msg_idx >= len(conv["messages"]):
            return
        msg = conv["messages"][msg_idx]
        msg["_reasoning_open"] = not msg.get("_reasoning_open", False)
        self._render_messages(preserve_scroll=True)

    def _md_to_html(self, text: str) -> str:
        import re
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        code_blocks = []
        def save_code_block(m):
            code_blocks.append(m.group(2))
            return f"\x00CODE{len(code_blocks)-1}\x00"
        text = re.sub(r'```(\w*)\n(.*?)```', save_code_block, text, flags=re.DOTALL)
        text = re.sub(
            r'`([^`]+)`',
            f'<code style="background:{c("btn")};padding:2px 4px;border-radius:3px;">\\1</code>',
            text,
        )
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        text = re.sub(r'^### (.+)$', f'<h4 style="color:{c("accent")};">\\1</h4>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', f'<h3 style="color:{c("accent")};">\\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', f'<h2 style="color:{c("accent")};">\\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^- (.+)$', r'&bull; \1', text, flags=re.MULTILINE)
        text = re.sub(r"\n{2,}", "<br><br>", text)
        text = text.replace("\n", "<br>")
        for i, code in enumerate(code_blocks):
            code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            replacement = (
                f'<pre style="background:{c("surface")};padding:10px;border-radius:6px;'
                f'border:1px solid {c("border")};overflow-x:auto;">'
                f'<code>{code_escaped}</code></pre>'
            )
            text = text.replace(f"\x00CODE{i}\x00", replacement)
        return text

    def _load_models(self):
        """Fetch every model + the loaded set off the main thread.

        Called from the constructor, navigation and the refresh button. A
        synchronous request here froze the window for up to the request timeout
        whenever the backend was still booting, so both fetches run in a worker
        and the combo fills on the result.
        """
        from lmloader.gui.api_client import run_in_thread

        if getattr(self, "_models_loading", False):
            return
        self._models_loading = True

        def fetch():
            return self.api.list_models(), self.api.get_loaded_models()

        def on_result(res):
            self._models_loading = False
            all_models, loaded = res
            self._populate_model_combo(all_models or [], loaded or [])

        def on_error(_err):
            self._models_loading = False
            self._populate_model_combo([], [])

        run_in_thread(fetch, on_result, on_error)

    def _populate_model_combo(self, all_models: list, loaded: list):
        prev = self.model_combo.currentData()
        self._programmatic_change = True
        try:
            self.model_combo.clear()
            self._decision_models = set()
            self._model_loaded_map = {}

            loaded_by_name = {}
            for item in loaded:
                nm = (item.get("model") or {}).get("name")
                if nm:
                    loaded_by_name[nm] = item

            # Loaded models first (with their live port), then the rest.
            ordered = [(nm, True, item) for nm, item in loaded_by_name.items()]
            ordered += [
                (m.get("name"), False, m)
                for m in all_models
                if m.get("name") and m.get("name") not in loaded_by_name
            ]

            for nm, is_loaded, item in ordered:
                self._model_loaded_map[nm] = is_loaded
                if is_loaded:
                    port = item.get("port")
                    if item.get("engine_kind") == "d3":
                        self._decision_models.add(nm)
                        self.model_combo.addItem(
                            f"{nm} (:{port}) · {tr('chat.decision_tag')} ✓", nm
                        )
                    else:
                        self.model_combo.addItem(f"{nm} (:{port}) ✓", nm)
                else:
                    meta = item.get("metadata") or {}
                    if item.get("is_decision_model") or meta.get("decision_type"):
                        self._decision_models.add(nm)
                    self.model_combo.addItem(
                        f"{nm} · {tr('chat.unloaded_suffix')}", nm
                    )

            if not ordered:
                self.model_combo.addItem(tr("chat.no_models_available"), "")

            if prev:
                self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(prev)))
        finally:
            self._programmatic_change = False

    def _on_model_selected(self, _index: int):
        """Auto-load a model the moment it is chosen from the dropdown.

        Programmatic repopulation and an in-flight switch must not retrigger a
        load, so both are guarded. Already-loaded models are a no-op: the combo
        value is simply read at send time.
        """
        if getattr(self, "_programmatic_change", False) or getattr(self, "_model_switching", False):
            return
        name = self.model_combo.currentData()
        if not name or self._model_loaded_map.get(name, True):
            return
        self._switch_to_model(name)

    def _switch_to_model(self, name: str):
        from lmloader.gui.api_client import run_in_thread

        self._model_switching = True
        self.model_combo.setEnabled(False)
        idx = self.model_combo.currentIndex()
        if idx >= 0:
            self.model_combo.setItemText(idx, tr("chat.model_loading"))

        def work():
            # keep_previous=True: never unload the models already resident.
            return self.api.load_model(name, keep_previous=True)

        def on_result(_data):
            self._model_switching = False
            self.model_combo.setEnabled(True)
            self._load_models()
            self._append_message_bubble(
                tr("chat.system"), tr("chat.model_load_ok").format(name=name),
                c("accent_soft"), c("accent_border"),
            )

        def on_error(err):
            self._model_switching = False
            self.model_combo.setEnabled(True)
            self._load_models()
            self._append_message_bubble(
                tr("chat.system"), tr("chat.model_load_fail").format(err=str(err)),
                c("red_bg"), c("red_border"),
            )

        run_in_thread(work, on_result, on_error)

    def _send_message(self):
        if self._is_streaming:
            return

        text = self.input_box.toPlainText().strip()
        payload = self._attachment_payload()
        if not text and not payload:
            return
        text = (text + payload).strip()

        model_name = self.model_combo.currentData()
        if not model_name:
            self._append_message_bubble(tr("chat.system"), tr("chat.no_model_loaded"), c("red_bg"), c("red_border"))
            return

        if model_name in self._decision_models:
            self._append_message_bubble(
                tr("chat.system"),
                tr("chat.decision_model_hint").format(name=model_name),
                c("red_bg"), c("red_border"),
            )
            return

        if self.current_conv_id is None:
            self._new_conversation()

        conv = self.conversations[self.current_conv_id]
        if id(conv) in self._deleting_conversations:
            QMessageBox.information(self, tr("chat.delete"), tr("chat.deleting"))
            return

        if conv.get("db_id") is None and self._db_ready:
            self._persist_and_send(conv, model_name, text)  # FIXED: G6
            return

        self._do_send(conv, text, model_name)

    def _persist_and_send(self, conv: dict, model_name: str, text: str):  # FIXED: G6
        """Create the DB conversation in a worker thread, then send on the main thread."""
        self._is_streaming = True
        self._stream_conv = conv
        self._update_send_button_state()

        def _create():
            async def _do_create():
                try:
                    return await chat_db.create_conversation(
                        conv["title"], model_name, conv.get("system_prompt", "")
                    )
                except Exception:
                    return None
            return asyncio.run(_do_create())

        def on_created(db_id):  # main thread
            self._is_streaming = False
            self._update_send_button_state()
            conv["db_id"] = db_id
            self._do_send(conv, text, model_name)

        def on_error(err):  # main thread
            self._is_streaming = False
            self._update_send_button_state()
            self._do_send(conv, text, model_name)

        from lmloader.gui.api_client import run_in_thread
        run_in_thread(_create, on_created, on_error)

    def _do_send(self, conv: dict, text: str, model_name: str):  # FIXED: G6
        conv["model_name"] = model_name
        # Images ride along as image_url parts; the transcript keeps the plain
        # text so reloading a conversation never re-uploads megabytes of base64.
        has_images = bool(self._image_attachments())
        conv["messages"].append({"role": "user", "content": text})
        if has_images:
            conv["messages"][-1]["content"] = self._build_user_content(text)
        # Attachments are consumed by this message — drop the chips.
        if self._attachments:
            self._attachments = []
            self._render_attachments()
        if len(conv["messages"]) == 1 or (len(conv["messages"]) == 2 and conv["messages"][0]["role"] != "user"):
            conv["title"] = text[:30] + ("..." if len(text) > 30 else "")
            row = self.conv_list.itemWidget(self.conv_list.item(self.current_conv_id))
            if isinstance(row, ConversationRow):
                row.set_title(conv["title"])
                self._set_chat_title(conv["title"])
            if conv.get("db_id") is not None and self._db_ready:
                _run_async_in_thread(lambda: chat_db.update_conversation_title(conv["db_id"], conv["title"]))

        self.input_box.clear()
        self._render_messages()

        if conv.get("db_id") is not None and self._db_ready:
            _run_async_in_thread(lambda: chat_db.add_message(conv["db_id"], "user", text))

        self._is_streaming = True
        self._repetition_tripped = False
        self._repetition_info = None
        # Flips the button into its stop role; no setText("...") here, that
        # would overwrite the label _update_send_button_state just set.
        self._update_send_button_state()

        messages = list(conv["messages"])
        # The prompt lives on the conversation, not in the transcript, so the
        # DB rows stay clean and editing it never rewrites history.
        prompt = conv.get("system_prompt", "").strip()
        if prompt:
            messages.insert(0, {"role": "system", "content": prompt})

        self._stream_worker = ChatStreamWorker(
            self.api, model_name, messages, 0.7, self._thinking_on,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        self._stream_thread = QThread()
        self._stream_worker.moveToThread(self._stream_thread)

        self._stream_conv = conv
        self._stream_full_response = ""
        self._stream_reasoning = ""
        self._stream_model_name = model_name
        self._stream_started_at = time.time()
        self._stream_generation += 1
        self._stats_final_pending = False

        conv["messages"].append({"role": "assistant", "content": ""})
        self._stream_render_timer.start()
        self._stats_timer.start()
        self._render_messages()

        # Bound-method callbacks: queued to the main thread (AutoConnection).
        self._stream_worker.token_received.connect(self._on_stream_token)
        # Keep completion bound to this QObject so Qt queues it to the GUI thread.
        self._stream_worker.finished.connect(self._on_stream_completed)
        self._stream_worker.error.connect(self._on_stream_error)

        self._stream_thread.started.connect(self._stream_worker.run)
        self._stream_worker.finished.connect(self._stream_thread.quit)
        self._stream_worker.error.connect(self._stream_thread.quit)
        self._stream_thread._worker = self._stream_worker
        self._stream_thread.start()

    def _on_stream_token(self, kind: str, token: str):  # main thread
        if self._stream_conv is None:
            return
        msg = self._stream_conv["messages"][-1]
        if kind == "reasoning":
            self._stream_reasoning += token
            msg["_reasoning"] = self._stream_reasoning
        else:
            self._stream_full_response += token
            msg["content"] = self._stream_full_response
        self._check_repetition_guard()

    def _check_repetition_guard(self) -> bool:
        """Trip the repetition guard if the model has fallen into a loop.

        Runs on every token, so it only inspects the tail of the accumulated
        text (see ``lmloader.gui.repetition``). The answer so far is preserved —
        we cut the stream, we do not discard the user's tokens — and a notice is
        appended so the truncated output is never mistaken for a complete one.
        """
        if self._repetition_tripped:
            return True
        # Judge reasoning + content together: the degenerate loop in practice
        # shows up in the chain-of-thought block, and a model stuck there never
        # reaches content at all.
        probe = self._stream_reasoning + self._stream_full_response
        hit = detect_repetition_loop(probe)
        if hit is None:
            return False

        self._repetition_tripped = True
        self._repetition_info = hit
        worker = self._stream_worker
        if worker is not None:
            worker.stop()
        # Finish the turn on the GUI thread: the worker deliberately stays quiet
        # after stop() so this is the single place that resets streaming state.
        self._on_stream_finished(truncated=True, stopped=False)
        return True

    def _on_stream_finished(self, truncated: bool = False, stopped: bool = False):  # main thread
        conv = self._stream_conv or {}
        self._is_streaming = False
        self._stream_render_timer.stop()
        self._stats_timer.stop()
        self._update_send_button_state()
        response = self._stream_full_response
        msg = conv.get("messages", [{}])[-1]
        # Keep the reasoning text (collapsed) so it can be expanded on demand;
        # only the final answer is persisted to the chat DB.
        msg["_reasoning_open"] = False
        notice = ""
        if stopped:
            notice = tr("chat.stopped_notice")
        elif self._repetition_tripped:
            notice = tr("chat.repetition_notice")
        if not response:
            msg["content"] = notice or tr("chat.empty_response")
        elif notice:
            msg["content"] = response + "\n\n" + notice
        self._render_messages()
        self._poll_generation_stats(final=True)
        if response and conv.get("db_id") is not None and self._db_ready:
            _run_async_in_thread(
                lambda r=response, cid=conv.get("db_id"): chat_db.add_message(cid, "assistant", r)
            )
        self._cleanup_stream_thread()
        self._repetition_tripped = False
        self._repetition_info = None

    def _on_stream_completed(self):
        self._on_stream_finished()

    def _on_stream_error(self, err: str):  # main thread
        conv = self._stream_conv or {}
        self._is_streaming = False
        self._stream_render_timer.stop()
        self._stats_timer.stop()
        self._update_send_button_state()
        self.send_btn.setText(tr("chat.send"))
        conv["messages"][-1]["content"] = tr("chat.error_prefix").format(err=err)
        self._render_messages()
        self._cleanup_stream_thread()

    def _poll_generation_stats(self, final: bool = False):
        if self._stats_refreshing:
            self._stats_final_pending = self._stats_final_pending or final
            return
        if not self._is_streaming and not final:
            return
        conv = self._stream_conv
        model_name = self._stream_model_name
        started_at = self._stream_started_at
        generation = self._stream_generation
        if conv is None or not model_name or not started_at:
            return

        self._stats_refreshing = True

        def fetch():
            return self.api.get_loaded_models()

        def on_result(loaded):
            self._stats_refreshing = False
            if generation != self._stream_generation:
                return
            elapsed = max(0.0, time.time() - started_at)
            item = next(
                (entry for entry in loaded
                 if entry.get("model", {}).get("name") == model_name),
                None,
            )
            metrics = (item or {}).get("metrics") or {}
            if metrics.get("ts", 0) < started_at:
                metrics = {}
            msg = conv["messages"][-1]
            if final and not any(
                key in metrics for key in ("prompt_tokens", "gen_tokens", "tg_final")
            ):
                msg.pop("_stats", None)
            else:
                msg["_stats"] = self._format_generation_stats(
                    metrics, elapsed, final, (item or {}).get("ctx_size")
                )
            if self._is_current_conversation(conv):
                if final:
                    self._render_messages()
                else:
                    self._render_stream_update()
            if self._stats_final_pending:
                self._stats_final_pending = False
                QTimer.singleShot(0, lambda: self._poll_generation_stats(final=True))

        def on_error(_err):
            self._stats_refreshing = False
            if generation != self._stream_generation:
                return
            if self._stats_final_pending:
                self._stats_final_pending = False
                QTimer.singleShot(0, lambda: self._poll_generation_stats(final=True))

        run_in_thread(fetch, on_result, on_error)

    def _format_generation_stats(
        self, metrics: dict, elapsed: float, final: bool, context_limit
    ) -> str:
        if not metrics:
            return tr("chat.stats_elapsed").format(elapsed=f"{elapsed:.1f}")
        prompt_count = metrics.get("prompt_tokens")
        generated_count = metrics.get("gen_tokens" if final else "n_gen")
        speed = (
            metrics.get("tg_final") or metrics.get("tg")
            if final else metrics.get("tg_3s") or metrics.get("tg")
        )
        prompt_speed = metrics.get("prompt_tps")
        context_used = (
            prompt_count + generated_count
            if prompt_count is not None and generated_count is not None
            else None
        )
        return tr("chat.stats").format(
            elapsed=f"{elapsed:.1f}",
            prompt=f"{prompt_count:,}" if prompt_count is not None else "-",
            prompt_speed=f"{prompt_speed:.1f}" if prompt_speed is not None else "-",
            generated=f"{generated_count:,}" if generated_count is not None else "-",
            speed=f"{speed:.1f}" if speed is not None else "-",
            context_used=f"{context_used:,}" if context_used is not None else "-",
            context_limit=f"{int(context_limit):,}" if context_limit else "-",
        )

    def _is_current_conversation(self, conv: dict) -> bool:
        return (
            self.current_conv_id is not None
            and 0 <= self.current_conv_id < len(self.conversations)
            and self.conversations[self.current_conv_id] is conv
        )

    def _cleanup_stream_thread(self):
        """Retire the stream QThread without ever destroying a running one.

        ``_on_stream_finished`` is called from the main thread on the stop and
        repetition-guard paths, and on those paths the worker deliberately does
        NOT emit ``finished()`` (see ``ChatStreamWorker.run``) — so
        ``worker.finished -> thread.quit`` never fires and the thread's event
        loop is still sitting in ``exec()``.

        Calling ``deleteLater()`` on a QThread that is still running makes Qt
        destroy the C++ object while the OS thread is alive, which terminates
        the process with 0xC0000409 (the exact failure mode
        ``run_in_thread`` documents in this very module). So: ask the loop to
        quit, and defer the deletion until the thread actually reports finished.

        ``wait()`` is deliberately not used here — the thread can be parked in
        a blocking socket read, and blocking the GUI thread on it would freeze
        the window for as long as the read takes.
        """
        thread = self._stream_thread
        if thread is None:
            self._stream_worker = None
            self._stream_conv = None
            return
        self._stream_thread = None

        if thread.isRunning():
            # Keep a Python reference too: dropping the page's last reference
            # to a running QThread destroys its C++ object and aborts the app.
            self._retired_stream_threads.add(thread)
            thread.finished.connect(self._on_stream_thread_finished)
            thread.quit()
            if thread.isFinished():
                self._release_stream_thread(thread)
        else:
            thread.deleteLater()

        self._stream_worker = None
        self._stream_conv = None

    def _on_stream_thread_finished(self):
        thread = self.sender()
        if isinstance(thread, QThread):
            self._release_stream_thread(thread)

    def _release_stream_thread(self, thread: QThread):
        if thread not in self._retired_stream_threads:
            return
        self._retired_stream_threads.remove(thread)
        thread.deleteLater()

    def on_navigate(self):
        self._load_models()
