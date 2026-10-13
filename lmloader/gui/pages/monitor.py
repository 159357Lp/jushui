"""Resource monitor page — system stats display."""

import psutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QProgressBar, QGroupBox,
)
from PySide6.QtCore import QTimer, Qt

from lmloader.gui.api_client import run_in_thread
from lmloader.gui.i18n import tr
from lmloader.gui.styles import c


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
        self.ram_bar = self._make_stat_card(tr("monitor.ram"))
        self.swap_bar = self._make_stat_card(tr("monitor.swap"))
        self.gpu_bar = self._make_stat_card(tr("monitor.gpu"))  # FIXED: G3
        self.vram_bar = self._make_stat_card(tr("monitor.vram"))

        grid.addWidget(self.cpu_bar["group"], 0, 0)
        grid.addWidget(self.ram_bar["group"], 0, 1)
        grid.addWidget(self.swap_bar["group"], 0, 2)
        grid.addWidget(self.gpu_bar["group"], 1, 0)
        grid.addWidget(self.vram_bar["group"], 1, 1)

        layout.addLayout(grid)

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

    def _make_stat_card(self, name: str) -> dict:
        group = QGroupBox(name)
        inner = QVBoxLayout(group)

        value_label = QLabel("0%")
        value_label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {c('accent')};")
        value_label.setAlignment(Qt.AlignCenter)
        inner.addWidget(value_label)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(10)
        inner.addWidget(bar)

        detail_label = QLabel("")
        detail_label.setObjectName("subtitle")
        detail_label.setAlignment(Qt.AlignCenter)
        inner.addWidget(detail_label)

        return {"group": group, "value": value_label, "bar": bar, "detail": detail_label}

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
            self.cpu_bar["bar"].setValue(cpu_pct)
            cores = cpu.get("cores", psutil.cpu_count())
            self.cpu_bar["detail"].setText(tr("monitor.cores").format(cores=cores))

            mem = monitor.get("memory", {})
            ram_pct = mem.get("percent", 0)
            self.ram_bar["value"].setText(f"{ram_pct}%")
            self.ram_bar["bar"].setValue(ram_pct)
            used_gb = mem.get("used", 0) / (1024**3)
            total_gb = mem.get("total", 0) / (1024**3)
            self.ram_bar["detail"].setText(f"{used_gb:.1f} / {total_gb:.1f} GB")

            swap = monitor.get("swap", {})
            swap_pct = swap.get("percent", 0)
            self.swap_bar["value"].setText(f"{swap_pct}%")
            self.swap_bar["bar"].setValue(swap_pct)
            swap_used = swap.get("used", 0) / (1024**3)
            swap_total = swap.get("total", 0) / (1024**3)
            self.swap_bar["detail"].setText(f"{swap_used:.1f} / {swap_total:.1f} GB")

            gpu_data = monitor.get("gpu", {})
            gpus = gpu_data.get("gpus", [])
            if gpus:
                gpu = gpus[0]
                util = gpu.get("utilization", 0)
                self.gpu_bar["value"].setText(f"{util}%")
                self.gpu_bar["bar"].setValue(int(util))
                mem_used = gpu.get("memory_used", 0)
                mem_total = gpu.get("memory_total", 0)
                temp = gpu.get("temperature", 0)
                name = gpu.get("name", "GPU")
                self.gpu_bar["detail"].setText(f"{name} | {temp}°C")

                if mem_total:
                    vram_pct = round(mem_used / mem_total * 100, 1)
                    self.vram_bar["value"].setText(f"{vram_pct}%")
                    self.vram_bar["bar"].setValue(int(vram_pct))
                    self.vram_bar["detail"].setText(
                        f"{mem_used / 1024:.1f} / {mem_total / 1024:.1f} GB"
                    )
                else:
                    self.vram_bar["value"].setText("N/A")
                    self.vram_bar["bar"].setValue(0)
                    self.vram_bar["detail"].setText(tr("monitor.no_gpu"))
            else:
                self.gpu_bar["value"].setText("N/A")
                self.gpu_bar["bar"].setValue(0)
                self.gpu_bar["detail"].setText(tr("monitor.no_gpu"))
                self.vram_bar["value"].setText("N/A")
                self.vram_bar["bar"].setValue(0)
                self.vram_bar["detail"].setText(tr("monitor.no_gpu"))
        else:
            cpu_pct = psutil.cpu_percent(interval=0)
            self.cpu_bar["value"].setText(f"{cpu_pct}%")
            self.cpu_bar["bar"].setValue(cpu_pct)
            self.cpu_bar["detail"].setText(tr("monitor.cores").format(cores=psutil.cpu_count()))

            mem = psutil.virtual_memory()
            self.ram_bar["value"].setText(f"{mem.percent}%")
            self.ram_bar["bar"].setValue(mem.percent)
            self.ram_bar["detail"].setText(f"{mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB")

            swap = psutil.swap_memory()
            self.swap_bar["value"].setText(f"{swap.percent}%")
            self.swap_bar["bar"].setValue(swap.percent)
            self.swap_bar["detail"].setText(f"{swap.used / (1024**3):.1f} / {swap.total / (1024**3):.1f} GB")

            self.gpu_bar["value"].setText("N/A")
            self.gpu_bar["bar"].setValue(0)
            self.gpu_bar["detail"].setText(tr("monitor.server_offline"))

            self.vram_bar["value"].setText("N/A")
            self.vram_bar["bar"].setValue(0)
            self.vram_bar["detail"].setText(tr("monitor.server_offline"))

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
                    lines.append(f"  {name}  |  port={port}  |  pid={pid}  |  engine={slot}")
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
