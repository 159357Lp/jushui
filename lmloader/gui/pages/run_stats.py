"""速度档案页 —— 每只模型一行，展开看最高/中位/最低三次的完整配置与最近 10 次。

页面自己不算任何数：所有渲染都走 lmloader.engine.run_stats 里的纯函数，
所以"缺项显示成什么""哪条算中位"这两件事在测试里就能判，不必起 Qt。

三态必须分开，这是这个页面的底线：
  * 查不到（后端没起）      -> 明说"读不到"，不显示"还没有记录"
  * 查到了但库里是空的      -> "还没有运行记录"
  * 有记录                  -> 每只模型一行
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from lmloader.engine import run_stats
from lmloader.gui.api_client import LMLoaderAPI, run_in_thread
from lmloader.gui.i18n import tr
from lmloader.gui.styles import c, tint

_MONO = QFont("Consolas", 9)
_MONO.setStyleHint(QFont.Monospace)


def _group_colour(group: str) -> str:
    """卡片里三类行的墨色：已知项正常、无标签的解出项偏淡、未识别的 argv 用告警色，
    让"映射表漏了这个 flag"在页面上是显眼的，而不是混在正常配置里。

    色值在调用时取，不做成模块常量 —— 换主题时 c() 才反映新调色板。
    """
    return {"known": c("fg"), "extra": c("muted"), "other": c("warn"),
            "error": c("red")}.get(group, c("fg"))


def _card_title(kind_label: str, view: dict) -> str:
    if not view:
        return f"{kind_label}  —"
    return f"{kind_label} · {run_stats.speed_text(view)} · {view.get('ts') or '—'}"


class _ConfigCard(QGroupBox):
    """一次运行的完整配置：已知项全列（没传标"该引擎不传"）+ argv 原文。"""

    def __init__(self, title: str, view: dict, parent=None):
        super().__init__(title, parent)
        inner = QVBoxLayout(self)
        cfg = view.get("cfg") if isinstance(view, dict) else None
        env = view.get("env") if isinstance(view, dict) else None

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        rows = [(label, text, _group_colour(group))
                for group, label, text in run_stats.config_lines(cfg)]
        rows += [(label, text, c("muted"))
                 for _group, label, text in run_stats.env_lines(env)]
        for row, (label, text, colour) in enumerate(rows):
            grid.addWidget(self._ink(label, c("muted"), False), row, 0)
            grid.addWidget(self._ink(text, colour, True), row, 1)
        grid.setColumnStretch(1, 1)
        inner.addLayout(grid)

        argv = view.get("args") if isinstance(view, dict) else None
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setFont(_MONO)
        box.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        # 高度不写死：写死的框会把长 argv 藏进滚动条后面，而这一栏的存在理由就是
        # "原文一个字都不少"。给它布局的余量（stretch 1），最少留 3 行。
        box.setMinimumHeight(QFontMetrics(_MONO).lineSpacing() * 3 + 12)
        if isinstance(argv, list) and argv:
            box.setPlainText(" ".join(str(t) for t in argv))
        elif isinstance(argv, dict) and "_parse_error" in argv:
            box.setPlainText(str(argv["_parse_error"]))
        else:
            box.setPlainText(tr("runstats.no_argv"))
        inner.addWidget(box, 1)

    @staticmethod
    def _ink(text: str, colour: str, selectable: bool) -> QLabel:
        label = QLabel(str(text))
        label.setStyleSheet(f"color: {colour};")
        label.setWordWrap(True)
        if selectable:
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label


class RunStatsPage(QWidget):
    def __init__(self, api: LMLoaderAPI):
        super().__init__()
        self.api = api
        self._timer = QTimer(self)
        self._timer.setInterval(3000)
        self._timer.timeout.connect(self._refresh)
        self._busy = False
        self._stale = 0
        self._expanded: set[str] = set()
        self._payload_key = None
        self._init_ui()

    # ------------------------------------------------------------------ 界面骨架

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title = QLabel(tr("runstats.title"))
        title.setObjectName("title")
        layout.addWidget(title)

        self.status_label = QLabel(tr("runstats.waiting"))
        self.status_label.setObjectName("subtitle")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.live_group = QGroupBox(tr("runstats.live"))
        self.live_host = QVBoxLayout(self.live_group)
        self.live_host.setSpacing(2)
        layout.addWidget(self.live_group)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        self.rows_layout = QVBoxLayout(host)
        self.rows_layout.setSpacing(8)
        self.rows_layout.addStretch()
        self.scroll.setWidget(host)
        layout.addWidget(self.scroll, 1)

    def on_navigate(self):
        self._refresh()
        self._timer.start()

    def on_leave(self):
        self._timer.stop()

    def retheme(self):
        self._rebuild(getattr(self, "_last_models", None) or [])

    # ------------------------------------------------------------------ 取数

    def _refresh(self):
        if self._busy:
            return
        self._busy = True

        def fetch():
            return self.api.fetch_run_stats_or_none(), self.api.fetch_loaded_models_or_none()

        def on_result(result):
            self._busy = False
            stats, loaded = result
            if stats is None:
                # 查不到 != 没有记录：后端挂了却显示"还没有运行记录"是假绿灯。
                self.status_label.setText(tr("runstats.unreachable"))
                self._stale += 1
                self._render_live(loaded)
                return
            self._stale = 0
            self._render_live(loaded)
            self._render_archive(stats)

        def on_error(err):
            self._busy = False
            self.status_label.setText(tr("runstats.fetch_failed").replace("{err}", str(err)))

        run_in_thread(fetch, on_result, on_error)

    def _render_live(self, loaded):
        while self.live_host.count():
            item = self.live_host.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        models = loaded or []
        if not models:
            self.live_host.addWidget(self._muted(tr("runstats.live_none")))
            return
        for item in models:
            m = (item or {}).get("metrics") or {}
            name = ((item or {}).get("model") or {}).get("name") or "?"
            tg = m.get("tg_final") or m.get("tg") or m.get("tg_3s")
            bits = [f"{float(tg):.1f} t/s" if tg else tr("runstats.live_idle")]
            if m.get("n_gen"):
                bits.append(f"{m['n_gen']} tok")
            if m.get("prompt_tokens"):
                bits.append(tr("runstats.live_prompt").replace(
                    "{n}", str(m["prompt_tokens"])))
            if m.get("acceptance") is not None:
                bits.append(tr("runstats.live_accept").replace(
                    "{p}", f"{float(m['acceptance']) * 100:.0f}%"))
            # 占用与权重分账是这一行的判别量：长对话变慢时先看占用逼没逼上限，
            # 再看权重是不是全量上了卡。
            cs = (item or {}).get("cache_state")
            if cs:
                bits.append(f"缓存 {run_stats.format_cache(cs)}")
            split = run_stats.weight_split((item or {}).get("memory"))
            if split != run_stats.NOT_PASSED:
                bits.append(f"权重 {split}")
            host = run_stats.format_host((item or {}).get("host_free"))
            if host != run_stats.NOT_PASSED:
                bits.append(f"空闲 {host}")
            row = QLabel(f"{name}    " + " · ".join(bits))
            row.setStyleSheet(f"color: {c('accent')};")
            self.live_host.addWidget(row)

    def _render_archive(self, stats):
        models = stats.get("models") or []
        total = stats.get("total")
        if total in (0, None):
            self.status_label.setText(tr("runstats.empty"))
        else:
            self.status_label.setText(
                tr("runstats.summary")
                .replace("{n}", str(total))
                .replace("{m}", str(len(models)))
            )
        # 只在数据真变了时重建行，否则每 3 秒会把用户正在读的那一行滚回顶部。
        key = [(r.get("model_name"), r.get("count")) for r in models]
        if key != self._payload_key:
            self._payload_key = key
            self._rebuild(models)

    def _rebuild(self, models):
        while self.rows_layout.count() > 1:
            item = self.rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for entry in models or []:
            self.rows_layout.insertWidget(self.rows_layout.count() - 1,
                                          self._build_row(entry))

    # ------------------------------------------------------------------ 单只模型

    def _build_row(self, entry: dict) -> QWidget:
        name = entry.get("model_name") or "?"
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {tint(c('surface'), 0.55)};"
            f" border: 1px solid {tint(c('border'), 0.8)}; border-radius: 8px; }}"
        )
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(6)

        head = QHBoxLayout()
        fold = QPushButton("▾" if name in self._expanded else "▸")
        fold.setFlat(True)
        fold.setFixedWidth(22)
        head.addWidget(fold)

        head.addWidget(self._ink(name, c("fg"), True))
        head.addStretch()
        for kind_key, slot in (("runstats.high", "high"), ("runstats.median", "median"),
                               ("runstats.low", "low")):
            view = entry.get(slot) or {}
            head.addWidget(self._ink(
                f"{tr(kind_key)} {self._fmt_speed(view.get('decode_tps'))}",
                c("muted"), False))
        head.addWidget(self._ink(tr("runstats.count").replace(
            "{n}", str(entry.get("count", 0))), c("dim"), False))
        outer.addLayout(head)

        body = QWidget()
        body.setVisible(name in self._expanded)
        outer.addWidget(body)

        def toggle():
            open_now = not body.isVisible()
            body.setVisible(open_now)
            fold.setText("▾" if open_now else "▸")
            if open_now:
                self._expanded.add(name)
                self._fill_body(body, entry)
            else:
                self._expanded.discard(name)

        fold.clicked.connect(toggle)
        if name in self._expanded:
            self._fill_body(body, entry)
        return frame

    def _fill_body(self, body: QWidget, entry: dict):
        if body.layout() is not None:
            return  # Qt6 对第二套布局不报错也不生效：漏出来的是没有父项的顶层控件
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        cards = QHBoxLayout()
        cards.setSpacing(8)
        for kind_key, slot in (("runstats.high", "high"), ("runstats.median", "median"),
                               ("runstats.low", "low")):
            view = entry.get(slot)
            card = _ConfigCard(_card_title(tr(kind_key), view or {}), view or {})
            card.setMinimumWidth(300)
            cards.addWidget(card, 1)
        lay.addLayout(cards)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        head = self._ink(tr("runstats.recent_title"), c("fg"), False)
        head.setStyleSheet(f"color: {c('fg')}; font-weight: bold;")
        grid.addWidget(head, 0, 0, 1, len(run_stats.TABLE_COLUMNS) + 1)
        for index, text in enumerate(run_stats.TABLE_COLUMNS):
            label = self._ink(text, c("muted"), False)
            label.setStyleSheet(f"color: {c('muted')}; font-weight: bold;")
            grid.addWidget(label, 1, index + 1)
        recent = entry.get("recent") or []
        for r, view in enumerate(recent, start=2):
            for c_index, text in enumerate(run_stats.table_row(view)):
                mono = run_stats.TABLE_COLUMNS[c_index] in run_stats.MONO_COLS
                label = self._ink(text, c("accent") if mono else c("fg"), False)
                if mono:
                    label.setFont(_MONO)
                grid.addWidget(label, r, c_index + 1)
        grid.setColumnStretch(len(run_stats.TABLE_COLUMNS), 1)
        lay.addLayout(grid)

    # ------------------------------------------------------------------ 小工具

    @staticmethod
    def _fmt_speed(value) -> str:
        return f"{float(value):.1f} t/s" if value is not None else "没测到"

    @staticmethod
    def _ink(text: str, colour: str, selectable: bool) -> QLabel:
        label = QLabel(str(text))
        label.setStyleSheet(f"color: {colour};")
        if selectable:
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    @staticmethod
    def _muted(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"color: {c('muted')};")
        return label
