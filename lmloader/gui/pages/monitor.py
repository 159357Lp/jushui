"""Resource monitor page — system stats display."""

import psutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QProgressBar, QGroupBox,
)
from PySide6.QtCore import QTimer, Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath

from lmloader.gui.api_client import run_in_thread
from lmloader.gui.i18n import tr
from lmloader.gui.styles import c


def _fmt_mib(mib: float) -> str:
    if mib >= 1024:
        return f"{mib / 1024:.1f} GB"
    return f"{mib:.0f} MB"


class SegmentBar(QWidget):
    """Horizontal stacked bar, one colored run per memory class.

    QProgressBar can only express a single value; the point of these two cards
    is which class owns which slice of the pool. Widths are scaled against the
    pool's capacity, so the filled extent still reads as the usage percent.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(10)
        self._parts: list[tuple[float, str]] = []
        self._capacity = 0.0

    def set_parts(self, parts: list[tuple[float, str]], capacity: float):
        self._parts = [(float(v), col) for v, col in parts if v and v > 0]
        self._capacity = float(capacity)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        track = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        clip = QPainterPath()
        clip.addRoundedRect(track, 5, 5)
        p.setClipPath(clip)
        p.fillPath(clip, QColor(c("border")))
        if self._capacity <= 0:
            return
        x = track.left()
        for value, color in self._parts:
            w = track.width() * min(1.0, value / self._capacity)
            if w <= 0:
                continue
            p.setBrush(QColor(color))
            p.drawRect(QRectF(x, track.top(), max(1.0, w), track.height()))
            x += w
            if x >= track.right():
                break


class MonitorPage(QWidget):
    def __init__(self, api):
        super().__init__()
        self.api = api
        self._refresh_thread = None
        self._refreshing = False
        self._init_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        title = QLabel(tr("monitor.title"))
        title.setObjectName("title")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(16)

        self.cpu_bar = self._make_stat_card(tr("monitor.cpu"))
        self.ram_bar = self._make_stat_card(tr("monitor.ram"), segmented=True)
        self.swap_bar = self._make_stat_card(tr("monitor.swap"))
        self.gpu_bar = self._make_stat_card(tr("monitor.gpu"))  # FIXED: G3
        self.vram_bar = self._make_stat_card(tr("monitor.vram"), segmented=True)
        self.disk_bar = self._make_stat_card(tr("monitor.disk"))

        cards = [self.cpu_bar, self.ram_bar, self.swap_bar,
                 self.gpu_bar, self.vram_bar, self.disk_bar]
        # Three equal columns and two equal rows. Without the stretches the grid
        # sized each column to its content, so the wide cards ate the row and the
        # empty sixth cell made it worse.
        for col in range(3):
            grid.setColumnStretch(col, 1)
        for row in range(2):
            grid.setRowStretch(row, 1)
        for index, card in enumerate(cards):
            grid.addWidget(card["group"], index // 3, index % 3)

        # The grid takes part of the page's slack (factor 1, shared with the
        # addStretch below) so both rows are stretched to the same height instead
        # of each keeping its own content height — otherwise the row holding the
        # two legend cards sits taller than the row without legends.
        layout.addLayout(grid, 1)

        self.engine_group = QGroupBox(tr("monitor.engine_status"))
        engine_layout = QVBoxLayout(self.engine_group)
        self.engine_status_label = QLabel(tr("monitor.not_connected"))
        self.engine_status_label.setObjectName("subtitle")
        self.engine_status_label.setWordWrap(True)
        engine_layout.addWidget(self.engine_status_label)
        layout.addWidget(self.engine_group)

        self.loaded_group = QGroupBox(tr("monitor.loaded_models"))
        loaded_layout = QVBoxLayout(self.loaded_group)
        self.loaded_status_label = QLabel(tr("monitor.no_models_loaded"))
        self.loaded_status_label.setObjectName("subtitle")
        loaded_layout.addWidget(self.loaded_status_label)
        layout.addWidget(self.loaded_group)

        layout.addStretch()
        self._refresh()

    def _make_stat_card(self, name: str, segmented: bool = False) -> dict:
        group = QGroupBox(name)
        inner = QVBoxLayout(group)

        value_label = QLabel("0%")
        value_label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {c('accent')};")
        value_label.setAlignment(Qt.AlignCenter)
        inner.addWidget(value_label)

        bar = SegmentBar() if segmented else QProgressBar()
        if not segmented:
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
        inner.addWidget(bar)

        legend_label = QLabel("")
        legend_label.setObjectName("subtitle")
        legend_label.setAlignment(Qt.AlignCenter)
        legend_label.setWordWrap(True)
        legend_label.setTextFormat(Qt.TextFormat.RichText)
        legend_label.hide()
        inner.addWidget(legend_label)

        detail_label = QLabel("")
        detail_label.setObjectName("subtitle")
        detail_label.setAlignment(Qt.AlignCenter)
        # Without wrapping, the longest detail string (the GPU name) sets its
        # column's size hint and that column ends up wider than the other two.
        detail_label.setWordWrap(True)
        inner.addWidget(detail_label)

        return {
            "group": group, "value": value_label, "bar": bar,
            "legend": legend_label, "detail": detail_label,
            "segmented": segmented,
        }

    def _set_percent(self, card: dict, pct: float):
        if not card["segmented"]:
            card["bar"].setValue(int(pct))

    def _set_breakdown(self, card: dict, parts: list[tuple[float, str, str]], capacity: float):
        """parts: (MiB, color, label) — legend shows only the non-zero slices."""
        shown = [(v, col, label) for v, col, label in parts if v and v > 0.05]
        card["bar"].set_parts([(v, col) for v, col, _ in shown], capacity)
        if not shown:
            card["legend"].clear()
            card["legend"].hide()
            return
        chips = "  ".join(
            f'<span style="color:{col};">■</span> {label} {_fmt_mib(v)}'
            for v, col, label in shown
        )
        card["legend"].setText(chips)
        card["legend"].show()

    def _clear_breakdown(self, card: dict, _reason: str = ""):
        card["bar"].set_parts([], 0)
        card["legend"].clear()
        card["legend"].hide()

    @staticmethod
    def _class_parts(pool: dict) -> list[tuple[float, str, str]]:
        """Weights / KV / recurrent-state / compute slices from one memory pool."""
        return [
            (pool.get("weights", 0.0), c("accent"), tr("monitor.mem_weights")),
            (pool.get("kv", 0.0), c("green"), tr("monitor.mem_kv")),
            (pool.get("recurrent", 0.0), c("warn"), tr("monitor.mem_state")),
            (pool.get("compute", 0.0) + pool.get("output", 0.0),
             c("mem_compute"), tr("monitor.mem_compute")),
        ]

    def _render_vram(self, used_mib: float, total_mib: float, breakdown: dict):
        pool = (breakdown or {}).get("vram") or {}
        parts = self._class_parts(pool)
        reported = sum(v for v, _, _ in parts)
        # Whatever the GPU says is used but no engine claims: the desktop
        # compositor, other processes, and each engine's CUDA context — that
        # last one is not separable on Windows, where nvidia-smi answers [N/A]
        # for per-process memory.
        parts.append((max(0.0, used_mib - reported), c("mem_other"),
                      tr("monitor.mem_other_proc")))
        self._set_breakdown(self.vram_bar, parts, total_mib)

    def _render_ram(self, used_mib: float, total_mib: float, breakdown: dict):
        pool = (breakdown or {}).get("ram") or {}
        parts = self._class_parts(pool)
        reported = sum(v for v, _, _ in parts)
        engine_rss = float((breakdown or {}).get("engine_rss_mib") or 0.0)
        parts.append((max(0.0, engine_rss - reported), c("mem_other"),
                      tr("monitor.mem_engine_overhead")))
        parts.append((max(0.0, used_mib - engine_rss), c("mem_system"),
                      tr("monitor.mem_system_other")))
        self._set_breakdown(self.ram_bar, parts, total_mib)

    def _render_disk(self, disk: dict):
        """Busy time is the bar; throughput and the installed drive's free space
        are the two lines under it. Rates are deltas between polls, so the first
        sample of a session legitimately reads 0."""
        busy = disk.get("busy_percent", 0)
        self.disk_bar["value"].setText(f"{busy}%")
        self._set_percent(self.disk_bar, busy)
        legend = self.disk_bar["legend"]
        if disk:
            legend.setText(tr("monitor.disk_io").format(
                r=disk.get("read_mb_s", 0), w=disk.get("write_mb_s", 0)))
            legend.show()
        else:
            legend.hide()
        used_gb = disk.get("used", 0) / (1024 ** 3)
        total_gb = disk.get("total", 0) / (1024 ** 3)
        drive = disk.get("drive") or ""
        prefix = f"{drive} " if drive else ""
        self.disk_bar["detail"].setText(f"{prefix}{used_gb:.0f} / {total_gb:.0f} GB")

    @staticmethod
    def _model_memory_suffix(bd: dict | None) -> str:
        """One-line weights/KV account for a loaded model, MiB."""
        if not bd:
            return ""

        def side(pool: str) -> str:
            d = bd.get(pool) or {}
            return (
                f"{tr('monitor.mem_weights')} {d.get('weights', 0):.0f}"
                f" + {tr('monitor.mem_kv')} {d.get('kv', 0):.0f} MB"
            )

        return (
            f"  |  {tr('monitor.mem_vram_short')} {side('vram')}"
            f"  ·  {tr('monitor.mem_ram_short')} {side('ram')}"
        )

    def _refresh(self):
        if self._refreshing:
            return
        self._refreshing = True

        def fetch():
            return self.api.get_monitor(), self.api.get_engine()

        def on_result(result):
            self._refreshing = False
            self._refresh_thread = None
            self._render_data(*result)

        def on_error(_err):
            self._refreshing = False
            self._refresh_thread = None
            self.engine_status_label.setText(tr("monitor.server_not_reachable"))
            self.loaded_status_label.setText("-")

        self._refresh_thread = run_in_thread(fetch, on_result, on_error)

    def _render_data(self, monitor: dict, engine: dict):
        if monitor:
            cpu = monitor.get("cpu", {})
            cpu_pct = cpu.get("percent", psutil.cpu_percent(interval=0))
            self.cpu_bar["value"].setText(f"{cpu_pct}%")
            self._set_percent(self.cpu_bar, cpu_pct)
            cores = cpu.get("cores", psutil.cpu_count())
            self.cpu_bar["detail"].setText(tr("monitor.cores").format(cores=cores))

            mem = monitor.get("memory", {})
            breakdown = monitor.get("memory_breakdown") or {}
            ram_pct = mem.get("percent", 0)
            self.ram_bar["value"].setText(f"{ram_pct}%")
            used_mib = mem.get("used", 0) / (1024 ** 2)
            total_mib = mem.get("total", 0) / (1024 ** 2)
            self.ram_bar["detail"].setText(
                f"{used_mib / 1024:.1f} / {total_mib / 1024:.1f} GB"
            )
            self._render_ram(used_mib, total_mib, breakdown)

            swap = monitor.get("swap", {})
            swap_pct = swap.get("percent", 0)
            self.swap_bar["value"].setText(f"{swap_pct}%")
            self._set_percent(self.swap_bar, swap_pct)
            swap_used = swap.get("used", 0) / (1024**3)
            swap_total = swap.get("total", 0) / (1024**3)
            self.swap_bar["detail"].setText(f"{swap_used:.1f} / {swap_total:.1f} GB")

            gpu_data = monitor.get("gpu", {})
            gpus = gpu_data.get("gpus", [])
            if gpus:
                gpu = gpus[0]
                util = gpu.get("utilization", 0)
                self.gpu_bar["value"].setText(f"{util}%")
                self._set_percent(self.gpu_bar, util)
                mem_used = gpu.get("memory_used", 0)
                mem_total = gpu.get("memory_total", 0)
                temp = gpu.get("temperature", 0)
                name = gpu.get("name", "GPU")
                self.gpu_bar["detail"].setText(f"{name} | {temp}°C")

                if mem_total:
                    vram_pct = round(mem_used / mem_total * 100, 1)
                    self.vram_bar["value"].setText(f"{vram_pct}%")
                    self.vram_bar["detail"].setText(
                        f"{mem_used / 1024:.1f} / {mem_total / 1024:.1f} GB"
                    )
                    self._render_vram(mem_used, mem_total, breakdown)
                else:
                    self._clear_breakdown(self.vram_bar, tr("monitor.no_gpu"))
                    self.vram_bar["value"].setText("N/A")
                    self.vram_bar["detail"].setText(tr("monitor.no_gpu"))
            else:
                self.gpu_bar["value"].setText("N/A")
                self._set_percent(self.gpu_bar, 0)
                self.gpu_bar["detail"].setText(tr("monitor.no_gpu"))
                self.vram_bar["value"].setText("N/A")
                self._clear_breakdown(self.vram_bar, tr("monitor.no_gpu"))
            self._render_disk(monitor.get("disk", {}))
        else:
            cpu_pct = psutil.cpu_percent(interval=0)
            self.cpu_bar["value"].setText(f"{cpu_pct}%")
            self._set_percent(self.cpu_bar, cpu_pct)
            self.cpu_bar["detail"].setText(tr("monitor.cores").format(cores=psutil.cpu_count()))

            mem = psutil.virtual_memory()
            self.ram_bar["value"].setText(f"{mem.percent}%")
            self.ram_bar["detail"].setText(f"{mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB")
            self._render_ram(mem.used / (1024 ** 2), mem.total / (1024 ** 2), {})

            swap = psutil.swap_memory()
            self.swap_bar["value"].setText(f"{swap.percent}%")
            self._set_percent(self.swap_bar, swap.percent)
            self.swap_bar["detail"].setText(f"{swap.used / (1024**3):.1f} / {swap.total / (1024**3):.1f} GB")

            self.gpu_bar["value"].setText("N/A")
            self._set_percent(self.gpu_bar, 0)
            self.gpu_bar["detail"].setText(tr("monitor.server_offline"))

            self.vram_bar["value"].setText("N/A")
            self._clear_breakdown(self.vram_bar, tr("monitor.server_offline"))
            self.vram_bar["detail"].setText(tr("monitor.server_offline"))

            self.disk_bar["value"].setText("N/A")
            self._set_percent(self.disk_bar, 0)
            self.disk_bar["legend"].hide()
            self.disk_bar["detail"].setText(tr("monitor.server_offline"))

        if engine:
            engines = engine.get("engines", {})
            parts = []
            for slot_name, slot_data in engines.items():
                cuda = slot_data.get("cuda_available", False)
                ver = slot_data.get("version", "N/A")
                binary = slot_data.get("binary_path", "")
                status = tr("monitor.ok") if binary and ver != "N/A" else tr("monitor.not_found")
                cuda_str = tr("monitor.cuda") if cuda else tr("monitor.no_cuda")
                parts.append(f"{slot_name}: {status} ({ver}, {cuda_str})")
            self.engine_status_label.setText("\n".join(parts) if parts else tr("monitor.no_engines"))

            loaded = engine.get("loaded_models", [])
            if loaded:
                lines = []
                for item in loaded:
                    model = item.get("model", {})
                    name = model.get("name", "?")
                    port = item.get("port", "?")
                    pid = item.get("pid", "?")
                    slot = item.get("engine_slot", "?")
                    line = f"  {name}  |  port={port}  |  pid={pid}  |  engine={slot}"
                    line += self._model_memory_suffix(item.get("memory"))
                    lines.append(line)
                self.loaded_status_label.setText("\n".join(lines))
            else:
                self.loaded_status_label.setText(tr("monitor.no_models_loaded"))
        else:
            self.engine_status_label.setText(tr("monitor.server_not_reachable"))
            self.loaded_status_label.setText("-")

    def on_navigate(self):
        self._timer.start(2000)
        self._refresh()

    def on_leave(self):
        self._timer.stop()
