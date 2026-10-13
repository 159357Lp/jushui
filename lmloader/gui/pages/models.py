"""Model management page — LM Studio-style aligned card list.

Vision encoders (mmproj / CLIP) are hidden from the list; their load/unload
is controlled through the eye button on the paired text model.
"""

import os
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QMessageBox, QListView,
    QDialog, QFormLayout, QSpinBox, QCheckBox, QComboBox,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QFont, QFontMetrics, QGuiApplication

from lmloader.gui.api_client import LMLoaderAPI, run_in_thread
from lmloader.gui.i18n import tr
from lmloader.gui.styles import c
from lmloader.gui.icons import copy_icon, eye_icon, gear_icon

# How often the model page re-reads the loaded-model set. Short enough that a
# dead engine's badge disappears promptly, long enough to stay cheap (one GET).
POLL_INTERVAL_MS = 3000


def _extract_author(model: dict) -> str:
    meta = model.get("metadata") or {}
    author = meta.get("general.author")
    if author:
        return str(author)
    p = model.get("path", "").replace("\\", "/")
    parts = p.split("/")
    try:
        idx = parts.index("models")
        if idx + 1 < len(parts) - 1:
            return parts[idx + 1]
    except ValueError:
        pass
    return ""


# --------------------------------------------------------------- row geometry
#
# The row must never need more width than the list viewport: when it does, Qt
# grows a horizontal scrollbar and the 删除 button slides out of sight (that
# regression shipped once already). So the widths are declared here, in one
# table, and the layout modes below are DERIVED from it — an earlier version
# used a hard-coded 1120px threshold that was wrong on both sides and silently
# overflowed at the default window size.
#
# The name column is deliberately absent: it is the elastic one. It takes
# whatever the fixed columns leave over and elides, so it never forces the row
# wider than the viewport.

COL_SPACING = 7
COL_MARGIN = 22          # layout left (12) + right (10)

COL_WIDTHS = {
    "copy":    32,
    "cfg":     64,
    "quant":   88,
    "author":  92,
    "params":  46,
    "arch":    62,
    "fmt":     44,
    "size":    62,
    "badge":   62,
    "engine":  76,
    "eye":     34,
    "gear":    32,
    "load":    62,
    "delete":  58,
}

# Columns that get a wider share when a sparser mode frees up room, so the
# visible ones stay readable instead of collapsing to 8-character stubs.
# mode id -> {column: width}
MODE_WIDTH_OVERRIDES = {
    "compact": {"params": 60, "arch": 84},
}

# Densest first. "hidden" lists the optional columns a mode drops; "name_min"
# is the floor the elastic name column may shrink to in that mode.
#
# The name_min values are tuned so each mode's threshold lands between real
# viewports rather than on top of one: compact must fit 976px (a 1200px window
# with the rail expanded — the app's default and most common case) or the row
# would drop straight to minimal and lose params/arch/format there.
LAYOUT_MODES = (
    {"id": "full",    "hidden": (),                                    "name_min": 168},
    {"id": "compact", "hidden": ("author", "fmt"),                     "name_min": 144},
    # name_min 100 leaves headroom for the narrowest supported layout: a 1000px
    # window with the rail expanded leaves a ~774px viewport, and QListWidget's
    # frame plus item spacing eat a few px more than the arithmetic assumes.
    {"id": "minimal", "hidden": ("author", "fmt", "params", "arch"),   "name_min": 100},
)


def column_width(column: str, mode: dict) -> int:
    return MODE_WIDTH_OVERRIDES.get(mode["id"], {}).get(column, COL_WIDTHS[column])


def row_min_width(mode: dict) -> int:
    """Width the row needs when every column is at its declared width.

    Layout spacing is charged per gap, and the row has one gap more than it has
    fixed columns (the gaps that flank the elastic name and the stretch). This
    is a *predictive* figure used only to choose a mode — `_build_card` is the
    source of truth and the test suite asserts the two agree.
    """
    hidden = set(mode["hidden"])
    fixed_cols = [name for name in COL_WIDTHS if name not in hidden]
    fixed = sum(column_width(name, mode) for name in fixed_cols)
    gaps = len(fixed_cols) + 1
    return COL_MARGIN + fixed + COL_SPACING * gaps + mode["name_min"]


def pick_layout_mode(viewport_width: int) -> dict:
    """Densest mode whose predicted minimum width still fits the viewport.

    Derived from COL_WIDTHS rather than a guessed constant, so adding or
    widening a column can never silently reintroduce the overflow that pushed
    删除 off-screen at the default 1200px window.
    """
    for mode in LAYOUT_MODES:
        if row_min_width(mode) <= viewport_width:
            return mode
    return LAYOUT_MODES[-1]


class _ElidedLabel(QLabel):
    """A label that shows as much of its text as it has room for.

    Qt's QLabel does not elide — it either clips mid-glyph or, worse, reports
    its full text width and stretches the whole row. Horizontal
    ``QSizePolicy.Ignored`` makes the layout treat the width as free, and the
    resizeEvent re-elides against the real width.
    """

    def __init__(self, text: str = "", min_width: int = 120, parent=None):
        super().__init__(parent)
        self._full = text
        self.setMinimumWidth(min_width)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setToolTip(text)
        self._apply()

    def setText(self, text: str):  # noqa: N802 - Qt naming
        self._full = text
        self.setToolTip(text)
        self._apply()

    def full_text(self) -> str:
        return self._full

    def _apply(self):
        width = max(self.minimumWidth(), self.width() or self.minimumWidth())
        metrics = QFontMetrics(self.font())
        super().setText(metrics.elidedText(self._full, Qt.ElideRight, width))

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._apply()


class ModelConfigDialog(QDialog):
    """LM Studio style per-model run config editor."""

    def __init__(self, api: LMLoaderAPI, model_name: str, config: dict, parent=None,
                 vram_total_mb: Optional[int] = None):
        super().__init__(parent)
        self.api = api
        self.model_name = model_name
        self._saved = config.get("saved", False)
        self._vram_total_mb = vram_total_mb
        cfg = config.get("config", {})

        # Real model limits parsed from the GGUF header
        model_dict = config.get("model") or {}
        self._model_dict = model_dict
        meta = model_dict.get("metadata") or {}
        arch = model_dict.get("architecture") or meta.get("architecture") or "-"
        quant = model_dict.get("quantization") or meta.get("quantization") or "-"
        params = model_dict.get("parameters") or "-"
        native_ctx = int(meta.get("context_length") or 0)
        block_count = int(meta.get("block_count") or 0)
        self._block_count = block_count

        self.setWindowTitle(tr("models.config_title").format(name=model_name))
        self.setMinimumWidth(480)
        form = QFormLayout(self)
        form.setSpacing(10)

        # Real model limits parsed from the GGUF header
        native_ctx = int(meta.get("context_length") or 0)
        block_count = int(meta.get("block_count") or 0)
        self._block_count = block_count
        self._supports_mtp = bool(meta.get("supports_mtp"))
        cpu_threads = os.cpu_count() or 8
        self._cpu_threads = cpu_threads

        # Model identity / capability line
        bits = [str(arch), str(quant)]
        if params and params != "-":
            bits.append(str(params))
        if block_count:
            bits.append(tr("models.cfg_info_layers").format(n=block_count))
        if self._supports_mtp:
            bits.append(tr("models.cfg_info_mtp"))
        info = QLabel("  ·  ".join(bits))
        info.setObjectName("subtitle")
        form.addRow("", info)

        # ---- section: context & offload ----
        form.addRow("", self._section_label(tr("models.cfg_sec_offload")))

        # Context length — hard upper limit is the model's native trained context
        ctx_row = QWidget()
        ctx_hbox = QHBoxLayout(ctx_row)
        ctx_hbox.setContentsMargins(0, 0, 0, 0)
        self.ctx_spin = QSpinBox()
        ctx_max = native_ctx or 262144
        self.ctx_spin.setRange(512, ctx_max)
        self.ctx_spin.setSingleStep(1024)
        ctx_val = int(cfg.get("ctx_size") or (native_ctx or 4096))
        if native_ctx and not self._saved and ctx_val > native_ctx:
            ctx_val = native_ctx
        self.ctx_spin.setValue(min(ctx_val, ctx_max))
        ctx_hbox.addWidget(self.ctx_spin, 1)
        if native_ctx:
            ctx_hint = QLabel(tr("models.cfg_ctx_native").format(n=native_ctx))
            ctx_hint.setObjectName("subtitle")
            ctx_hbox.addWidget(ctx_hint)
        form.addRow(tr("models.cfg_ctx"), ctx_row)

        # GPU layers — 999 means full offload. The range must NOT be capped at
        # block_count: the vision mmproj and MTP draft head add layers beyond
        # the trunk, and clamping there leaves those on CPU (cuts MTP speed by
        # ~25%). Migrate legacy configs saved at the trunk-only count to 999.
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setRange(0, 999)
        gpu_val = int(cfg.get("n_gpu_layers") or 999)
        if block_count > 0 and gpu_val == block_count:
            gpu_val = 999
        self.gpu_spin.setValue(min(max(gpu_val, 0), 999))
        self.gpu_spin.setSpecialValueText(tr("models.cfg_gpu_cpu"))
        if block_count > 0:
            self.gpu_spin.setToolTip(
                tr("models.cfg_gpu_tip") + f" [trunk layers: {block_count}]"
            )
        else:
            self.gpu_spin.setToolTip(tr("models.cfg_gpu_tip"))
        form.addRow(tr("models.cfg_gpu"), self.gpu_spin)

        # CPU threads — must not exceed logical core count
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, cpu_threads)
        self.threads_spin.setSuffix(tr("models.cfg_threads_suffix").format(n=cpu_threads))
        self.threads_spin.setValue(min(int(cfg.get("threads") or cpu_threads), cpu_threads))
        self.threads_spin.setToolTip(tr("models.cfg_threads_tip"))
        form.addRow(tr("models.cfg_threads"), self.threads_spin)

        # ---- section: advanced ----
        form.addRow("", self._section_label(tr("models.cfg_sec_advanced")))

        # Logical / physical batch sizes (bounded by context and each other)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, self.ctx_spin.value())
        self.batch_spin.setValue(min(int(cfg.get("batch_size") or 2048), self.ctx_spin.value()))
        self.batch_spin.setToolTip(tr("models.cfg_batch_tip"))
        form.addRow(tr("models.cfg_batch"), self.batch_spin)

        self.ubatch_spin = QSpinBox()
        self.ubatch_spin.setRange(8, self.batch_spin.value())
        self.ubatch_spin.setValue(min(int(cfg.get("ubatch_size") or 512), self.batch_spin.value()))
        self.ubatch_spin.setToolTip(tr("models.cfg_ubatch_tip"))
        form.addRow(tr("models.cfg_ubatch"), self.ubatch_spin)

        self.ctx_spin.valueChanged.connect(
            lambda v: self.batch_spin.setMaximum(max(1, v)))
        self.batch_spin.valueChanged.connect(
            lambda v: self.ubatch_spin.setMaximum(max(self.ubatch_spin.minimum(), v)))

        # ---- parallel slots: capped by a VRAM estimate for THIS model ----
        parallel_row = QWidget()
        p_hbox = QHBoxLayout(parallel_row)
        p_hbox.setContentsMargins(0, 0, 0, 0)
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 8)
        self.parallel_spin.setValue(min(int(cfg.get("parallel") or 1), 8))
        self.parallel_spin.setToolTip(tr("models.cfg_parallel_tip"))
        p_hbox.addWidget(self.parallel_spin, 1)
        self.parallel_hint = QLabel("")
        self.parallel_hint.setObjectName("subtitle")
        p_hbox.addWidget(self.parallel_hint)
        form.addRow(tr("models.cfg_parallel"), parallel_row)

        # Recompute the slot cap whenever ctx or KV cache quant changes.
        # Initialize MTP widgets to None now so _sync_parallel_limit() (and
        # _values()) can run safely before the MTP section is built below.
        self.mtp_chk = None
        self.spec_draft_spin = None
        self.ctx_spin.valueChanged.connect(self._sync_parallel_limit)
        self.kv_k_combo = None  # assigned below; signals wired after creation
        self.kv_v_combo = None

        self.flash_chk = QCheckBox()
        self.flash_chk.setChecked(bool(cfg.get("flash_attn", True)))
        form.addRow(tr("models.cfg_flash"), self.flash_chk)

        kv_k_label = tr("models.cfg_kv_k") + " " + tr("models.cfg_experimental")
        self.kv_k_combo = QComboBox()
        for v in ("q8_0", "q4_0", "f16", "q8_1"):
            self.kv_k_combo.addItem(v, v)
        idx = self.kv_k_combo.findData(str(cfg.get("cache_type_k", "q4_0")))
        self.kv_k_combo.setCurrentIndex(max(0, idx))
        self.kv_k_combo.currentIndexChanged.connect(self._sync_parallel_limit)
        form.addRow(kv_k_label, self.kv_k_combo)

        kv_v_label = tr("models.cfg_kv_v") + " " + tr("models.cfg_experimental")
        self.kv_v_combo = QComboBox()
        for v in ("q8_0", "q4_0", "f16", "q8_1"):
            self.kv_v_combo.addItem(v, v)
        idx = self.kv_v_combo.findData(str(cfg.get("cache_type_v", "q4_0")))
        self.kv_v_combo.setCurrentIndex(max(0, idx))
        self.kv_v_combo.currentIndexChanged.connect(self._sync_parallel_limit)
        form.addRow(kv_v_label, self.kv_v_combo)

        self._sync_parallel_limit()

        # MTP speculative decoding — shown ONLY for models that embed MTP layers
        self.mtp_chk = None
        self.spec_draft_spin = None
        if self._supports_mtp:
            mtp_row = QWidget()
            mtp_hbox = QHBoxLayout(mtp_row)
            mtp_hbox.setContentsMargins(0, 0, 0, 0)
            self.mtp_chk = QCheckBox(tr("models.cfg_mtp_enable"))
            self.mtp_chk.setChecked(bool(cfg.get("mtp_enabled", True)))
            mtp_hbox.addWidget(self.mtp_chk)
            mtp_hbox.addWidget(QLabel(tr("models.cfg_mtp_draft")))
            self.spec_draft_spin = QSpinBox()
            self.spec_draft_spin.setRange(0, 16)
            self.spec_draft_spin.setValue(min(int(cfg.get("spec_draft_max") or 1), 16))
            mtp_hbox.addWidget(self.spec_draft_spin)
            mtp_hbox.addStretch()
            form.addRow(tr("models.cfg_mtp"), mtp_row)

            # MTP rollback window (1 + draft-max) must fit the micro-batch, so
            # batch/ubatch need a dynamic floor of draft-max + 2.
            self.mtp_chk.toggled.connect(self._sync_batch_limits)
            self.spec_draft_spin.valueChanged.connect(self._sync_batch_limits)

        self._sync_batch_limits()

        # state note
        note = QLabel(tr("models.cfg_saved_note") if self._saved else tr("models.cfg_unsaved_note"))
        note.setObjectName("subtitle")
        note.setWordWrap(True)
        form.addRow("", note)

        btns = QHBoxLayout()
        self.save_btn = QPushButton(tr("models.cfg_save"))
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self._save)
        btns.addWidget(self.save_btn)

        self.save_load_btn = QPushButton(tr("models.cfg_save_load"))
        self.save_load_btn.clicked.connect(self._save_and_load)
        btns.addWidget(self.save_load_btn)

        self.reset_btn = QPushButton(tr("models.cfg_reset"))
        self.reset_btn.clicked.connect(self._reset)
        btns.addWidget(self.reset_btn)

        cancel = QPushButton(tr("models.cfg_cancel"))
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        btns.addStretch()
        form.addRow("", btns)

    def _sync_batch_limits(self, *_args):
        """Enforce engine batch constraints: ubatch <= batch, and with MTP on,
        both must be >= spec_draft_max + 2 (rollback window 1 + draft-max)."""
        mtp_on = bool(self.mtp_chk and self.mtp_chk.isChecked())
        draft = self.spec_draft_spin.value() if self.spec_draft_spin else 1
        floor = max(3, draft + 2) if mtp_on else 8
        self.ubatch_spin.setMinimum(floor)
        self.batch_spin.setMinimum(floor)
        self.ubatch_spin.setMaximum(max(floor, self.batch_spin.value()))
        if self.batch_spin.value() < self.ubatch_spin.value():
            self.batch_spin.setValue(self.ubatch_spin.value())

    def _sync_parallel_limit(self, *_args):
        """Cap parallel slots by a VRAM estimate for this model + current ctx
        + KV cache quant, and surface the reason (LM Studio style)."""
        try:
            from lmloader.engine.router import estimate_max_parallel
            model = self._model_dict
            est = estimate_max_parallel(model, self._values(), self._vram_total_mb)
        except Exception:
            est = 8
        est = max(1, min(8, est))
        self.parallel_spin.setMaximum(est)
        if self.parallel_spin.value() > est:
            self.parallel_spin.setValue(est)
        if self._vram_total_mb:
            self.parallel_hint.setText(tr("models.cfg_parallel_hint").format(n=est))
        else:
            self.parallel_hint.setText("")

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        f = QFont()
        f.setBold(True)
        lbl.setFont(f)
        lbl.setContentsMargins(0, 6, 0, 0)
        return lbl

    def _values(self) -> dict:
        vals = {
            "ctx_size": self.ctx_spin.value(),
            "n_gpu_layers": self.gpu_spin.value(),
            "threads": self.threads_spin.value(),
            "parallel": self.parallel_spin.value(),
            "flash_attn": self.flash_chk.isChecked(),
            "cache_type_k": self.kv_k_combo.currentData(),
            "cache_type_v": self.kv_v_combo.currentData(),
            "batch_size": self.batch_spin.value(),
            "ubatch_size": self.ubatch_spin.value(),
        }
        if self._supports_mtp and self.mtp_chk is not None:
            vals["mtp_enabled"] = self.mtp_chk.isChecked()
            vals["spec_draft_max"] = self.spec_draft_spin.value()
        else:
            vals["mtp_enabled"] = True
            vals["spec_draft_max"] = 1
        return vals

    def _save(self):
        self.save_btn.setEnabled(False)

        def do_save():
            return self.api.save_model_run_config(self.model_name, self._values())

        def on_result(_r):
            self._saved = True
            self.save_btn.setEnabled(True)
            QMessageBox.information(self, tr("models.cfg_saved_title"), tr("models.cfg_saved_msg"))
            self.accept()

        def on_error(err):
            self.save_btn.setEnabled(True)
            QMessageBox.critical(self, tr("models.error_title"), tr("models.error_prefix").format(err=err))

        t = run_in_thread(do_save, on_result, on_error)
        self._t = t

    def _save_and_load(self):
        self.save_load_btn.setEnabled(False)

        def do_both():
            self.api.save_model_run_config(self.model_name, self._values())
            return self.api.load_model(self.model_name)

        def on_result(_r):
            self.accept()

        def on_error(err):
            self.save_load_btn.setEnabled(True)
            QMessageBox.critical(self, tr("models.error_title"), tr("models.error_prefix").format(err=err))

        t = run_in_thread(do_both, on_result, on_error)
        self._t = t

    def _reset(self):
        def do_reset():
            return self.api.delete_model_run_config(self.model_name)

        def on_result(_r):
            QMessageBox.information(self, tr("models.cfg_reset_title"), tr("models.cfg_reset_msg"))
            self.accept()

        def on_error(err):
            QMessageBox.critical(self, tr("models.error_title"), tr("models.error_prefix").format(err=err))

        t = run_in_thread(do_reset, on_result, on_error)
        self._t = t


class ModelsPage(QWidget):
    def __init__(self, api: LMLoaderAPI):
        super().__init__()
        self.api = api
        self._threads: list = []
        self._models: list[dict] = []
        self._loaded_names: set[str] = set()
        self._loaded_info: dict = {}
        self._refreshing = False
        self._refresh_thread = None
        self._relayout_timer = QTimer(self)
        self._relayout_timer.setSingleShot(True)
        self._relayout_timer.timeout.connect(self._relayout)
        self._init_ui()
        # Poll the loaded-model set so an engine that dies outside this process
        # (taskkill, crash, or the kill-on-close job object taking it down when
        # the window closes) cannot leave a stale "已加载" badge on screen. The
        # backend already reaps dead engines in /api/models/loaded; without this
        # poll the cards keep whatever snapshot they got when the page was last
        # opened. Mirrors the pattern already used by the logs page.
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_loaded)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Debounced responsive re-layout (hide secondary columns when narrow).
        self._relayout_timer.start(150)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel(tr("models.title"))
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        self.scan_btn = QPushButton(tr("models.scan_files"))
        self.scan_btn.clicked.connect(self._scan)
        header.addWidget(self.scan_btn)

        self.refresh_btn = QPushButton(tr("models.refresh"))
        self.refresh_btn.clicked.connect(self._refresh)
        header.addWidget(self.refresh_btn)

        layout.addLayout(header)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.NoSelection)
        self.list.setFocusPolicy(Qt.NoFocus)
        self.list.setSpacing(6)
        self.list.setResizeMode(QListView.Adjust)
        # The row is laid out to fit (see pick_layout_mode), so a horizontal
        # scrollbar would only ever appear for the few px QListWidget adds on
        # top of the item's own size hint — and when it did, it pushed 删除
        # out of sight. Keep it off: the responsive modes are the mechanism
        # for coping with narrow windows.
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setStyleSheet(
            "QListWidget { background: transparent; border: none; }"
            "QListWidget::item { background: transparent; }"
        )
        layout.addWidget(self.list, 1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("subtitle")
        layout.addWidget(self.status_label)

    def on_navigate(self):
        self._refresh()
        self._poll_timer.start(POLL_INTERVAL_MS)

    def on_leave(self) -> None:
        """Stop polling when navigating away so a hidden page issues no requests."""
        self._poll_timer.stop()

    # ---------------------------------------------------------------- polling

    def _poll_loaded(self) -> None:
        """Refresh only the loaded-model set, and only rebuild the cards when
        that set actually changed.

        A full ``_refresh()`` on every tick would rebuild all 8 cards several
        times a minute, which resets scroll position and makes the buttons
        flicker. Fetching just the loaded set is one cheap request and keeps the
        displayed state honest.

        A failed request returns ``None`` rather than ``[]``. That distinction is
        the whole point: ``[]`` means "the backend says nothing is loaded" and
        legitimately clears every badge, while ``None`` means "we could not ask"
        and must leave the display untouched. Using the list-returning variant
        here would conflate the two and blank the page on any timeout.
        """
        if self._refreshing:
            return
        self._refreshing = True

        def on_result(loaded):
            self._refreshing = False
            self._refresh_thread = None
            if loaded is None:
                # Query failed: keep whatever is on screen.
                return
            names = {item["model"]["name"] for item in loaded}
            info = {item["model"]["name"]: item for item in loaded}
            if names == self._loaded_names:
                # Same set, but the detail dict may have new live metrics.
                self._loaded_info = info
                return
            self._loaded_names = names
            self._loaded_info = info
            self._populate_list()
            self._update_status_counts()

        def on_error(_err):
            # A transient network blip must NOT be interpreted as "nothing is
            # loaded" — leaving the badges as they were is the safe direction.
            self._refreshing = False
            self._refresh_thread = None

        self._refresh_thread = run_in_thread(
            self.api.fetch_loaded_models_or_none, on_result, on_error
        )

    def _update_status_counts(self) -> None:
        count = len([m for m in self._models if not m.get("is_vision")])
        self.status_label.setText(
            tr("models.found").format(count=count, loaded=len(self._loaded_names))
        )

    def _refresh(self):
        if self._refreshing:
            return
        self._refreshing = True
        self.status_label.setText(tr("models.loading"))
        self.refresh_btn.setEnabled(False)

        def fetch():
            models = self.api.list_models()
            loaded = self.api.get_loaded_models()
            return models, loaded

        def on_result(result):
            models, loaded = result
            self._refreshing = False
            self._models = models
            self._loaded_names = {item["model"]["name"] for item in loaded}
            self._loaded_info = {item["model"]["name"]: item for item in loaded}
            self._populate_list()
            self._update_status_counts()
            self.refresh_btn.setEnabled(True)

        def on_error(err):
            self._refreshing = False
            self.status_label.setText(tr("models.error_prefix").format(err=err))
            self.refresh_btn.setEnabled(True)

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    # ---------------------------------------------------------------- cards

    def _relayout(self):
        if self._models:
            self._populate_list()

    def _populate_list(self):
        # Responsive layout: the densest column set whose minimum width still
        # fits the viewport. Derived from COL_WIDTHS, not a magic number.
        mode = pick_layout_mode(self.list.viewport().width())
        self.list.clear()
        for model in self._models:
            if model.get("is_vision"):
                # Vision encoders are managed through the eye button on the
                # paired text model — they are not shown as standalone rows.
                continue
            card = self._build_card(model, self._loaded_info, mode)
            item = QListWidgetItem()
            item.setSizeHint(card.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, card)

    def _build_card(self, model: dict, loaded_info: dict, mode: dict | None = None) -> QWidget:
        name = model["name"]
        is_loaded = name in self._loaded_names
        has_vision = model.get("has_vision", False)
        has_cfg = model.get("has_run_config", False)

        # A bare bool is still accepted (True == compact) so older callers and
        # probes keep working; everything below reads `hidden` / `name_min`.
        if mode is None or isinstance(mode, bool):
            mode = LAYOUT_MODES[1] if mode else LAYOUT_MODES[0]
        hidden = set(mode["hidden"])

        card = QWidget()
        card.setObjectName("modelCard")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 10, 10, 10)
        lay.setSpacing(COL_SPACING)

        fm = QFontMetrics(card.font())

        def _spacer(w: int) -> QWidget:
            """Fixed-width placeholder so columns line up across rows."""
            sp = QWidget()
            sp.setFixedWidth(w)
            sp.setFixedHeight(1)
            return sp

        # ---- name (elastic: absorbs slack, elides, never widens the row) --
        name_lbl = _ElidedLabel(name, min_width=mode["name_min"])
        name_lbl.setStyleSheet(f"font-weight: 600; font-size: 13px; color: {c('title')};")
        lay.addWidget(name_lbl, 1, Qt.AlignVCenter)

        copy_name_btn = QPushButton()
        copy_name_btn.setFixedSize(32, 30)
        copy_name_btn.setIconSize(QSize(18, 18))
        copy_name_btn.setIcon(copy_icon(c("muted")))
        copy_name_btn.setToolTip(tr("models.copy_name"))
        copy_name_btn.clicked.connect(
            lambda checked=False, n=name: self._copy_model_name(n)
        )
        lay.addWidget(copy_name_btn, 0, Qt.AlignVCenter)

        # ---- saved-config badge (fixed-width slot, placeholder otherwise) -
        cfg_slot_w = column_width("cfg", mode)
        if has_cfg:
            cfg_badge = QLabel(tr("models.cfg_badge"))
            cfg_badge.setFixedWidth(cfg_slot_w)
            cfg_badge.setAlignment(Qt.AlignCenter)
            cfg_badge.setStyleSheet(
                f"color: {c('accent')}; background: {c('accent')}22; border-radius: 8px;"
                "padding: 1px 4px; font-size: 11px; font-weight: 600;"
            )
            lay.addWidget(cfg_badge, 0, Qt.AlignVCenter)
        else:
            lay.addWidget(_spacer(cfg_slot_w))

        # ---- quant badge (fixed) ----------------------------------------
        quant_w = column_width("quant", mode)
        quant = model.get("quantization", "-")
        q_badge = QLabel(fm.elidedText(str(quant), Qt.ElideRight, quant_w - 4))
        q_badge.setFixedWidth(quant_w)
        q_badge.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        q_badge.setStyleSheet(f"color: {c('accent')}; font-size: 12px; font-weight: 600;")
        lay.addWidget(q_badge, 0, Qt.AlignVCenter)

        # ---- author (optional column) -----------------------------------
        if "author" not in hidden:
            author_w = column_width("author", mode)
            author = _extract_author(model)
            if model.get("backend") == "ollama":
                author = "ollama"
            author_lbl = QLabel(fm.elidedText(author or "-", Qt.ElideRight, author_w - 2) if author else "-")
            author_lbl.setFixedWidth(author_w)
            author_lbl.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
            lay.addWidget(author_lbl, 0, Qt.AlignVCenter)

        # ---- params / arch / format / size (fixed, aligned) --------------
        # In the compact mode there is room to show these two, but not at
        # their full width — the name column gets the slack instead, so give
        # them a readable share rather than the 8-character stubs the fixed
        # widths produced ("12…", "ge…").
        if "params" not in hidden:
            params_w = column_width("params", mode)
            params = model.get("parameters", "-") or "-"
            p_lbl = QLabel(fm.elidedText(str(params), Qt.ElideRight, params_w - 8))
            p_lbl.setFixedWidth(params_w)
            p_lbl.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
            lay.addWidget(p_lbl, 0, Qt.AlignVCenter)

        if "arch" not in hidden:
            arch_w = column_width("arch", mode)
            arch = model.get("architecture", "-") or "-"
            a_lbl = QLabel(fm.elidedText(arch, Qt.ElideRight, arch_w - 12))
            a_lbl.setFixedWidth(arch_w)
            a_lbl.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
            lay.addWidget(a_lbl, 0, Qt.AlignVCenter)

        if "fmt" not in hidden:
            fmt_w = column_width("fmt", mode)
            fmt_lbl = QLabel("Ollama" if model.get("backend") == "ollama" else "GGUF")
            fmt_lbl.setFixedWidth(fmt_w)
            fmt_lbl.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
            lay.addWidget(fmt_lbl, 0, Qt.AlignVCenter)

        size_w = column_width("size", mode)
        size_bytes = model.get("size", 0)
        size_str = f"{size_bytes / (1024**3):.1f} GB" if size_bytes else "-"
        s_lbl = QLabel(size_str)
        s_lbl.setFixedWidth(size_w)
        s_lbl.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
        lay.addWidget(s_lbl, 0, Qt.AlignVCenter)

        # elastic gap: info columns hug the name, actions stay right-aligned
        lay.addStretch(1)

        # ---- status badge (fixed-width slot so action columns align) -----
        if is_loaded:
            badge_text, badge_color = tr("models.status_loaded"), c("green")
        elif model.get("status") == "loading":
            badge_text, badge_color = tr("models.status_loading"), c("warn")
        elif model.get("status") == "failed":
            badge_text, badge_color = tr("models.status_failed"), c("red")
        else:
            badge_text, badge_color = tr("models.status_unloaded"), c("muted")
        badge = self._badge(badge_text, badge_color)
        badge.setFixedWidth(column_width("badge", mode))
        badge.setAlignment(Qt.AlignCenter)
        lay.addWidget(badge, 0, Qt.AlignVCenter)

        # ---- engine info when loaded (fixed slot, placeholder otherwise) --
        eng_slot_w = column_width("engine", mode)
        if is_loaded and name in loaded_info:
            info = loaded_info[name]
            eng = QLabel(f"{info.get('engine_slot', '-')} :{info.get('port', '-')}")
            eng.setFixedWidth(eng_slot_w)
            eng.setStyleSheet(f"color: {c('accent')}; font-size: 12px;")
            tip_bits = [f"{info.get('engine_slot', '-')} :{info.get('port', '-')}",
                        tr("models.cfg_ctx_short").format(n=info.get("ctx_size", "-")),
                        tr("models.cfg_parallel_short").format(n=info.get("parallel", "-"))]
            eng.setToolTip("  ".join(tip_bits))
            lay.addWidget(eng, 0, Qt.AlignVCenter)
        else:
            lay.addWidget(_spacer(eng_slot_w))

        # ---- eye (vision toggle; fixed slot, placeholder when no vision) --
        eye_slot_w = column_width("eye", mode)
        if has_vision:
            eye = QPushButton()
            eye.setFixedSize(eye_slot_w, 32)
            eye.setIconSize(eye.size() - QSize(10, 10))
            if is_loaded:
                vision_on = bool(loaded_info.get(name, {}).get("mmproj"))
                if vision_on:
                    eye.setIcon(eye_icon(c("green"), slashed=False))
                    eye.setToolTip(tr("models.vision_on_tip"))
                else:
                    eye.setIcon(eye_icon(c("muted"), slashed=True))
                    eye.setToolTip(tr("models.vision_off_tip"))
                eye.clicked.connect(
                    lambda checked, n=name, on=vision_on: self._toggle_vision(n, not on)
                )
            else:
                # Not loaded: the eye stays visible and loads the model WITH
                # its vision encoder (the default pairing).
                eye.setIcon(eye_icon(c("muted"), slashed=False))
                eye.setToolTip(tr("models.vision_load_with"))
                eye.clicked.connect(lambda checked, n=name: self._load(n))
            lay.addWidget(eye, 0, Qt.AlignVCenter)
        else:
            lay.addWidget(_spacer(eye_slot_w))

        # ---- gear: per-model run config -----------------------------------
        gear_w = column_width("gear", mode)
        gear = QPushButton()
        gear.setFixedSize(gear_w, 30)
        gear.setIconSize(gear.size() - QSize(9, 9))
        gear.setIcon(gear_icon(c("muted")))
        gear.setToolTip(tr("models.cfg_open"))
        gear.clicked.connect(lambda checked, n=name: self._open_config(n))
        lay.addWidget(gear, 0, Qt.AlignVCenter)

        # ---- load / unload / delete (fixed-width buttons) -----------------
        action_w = column_width("load", mode)
        if is_loaded:
            unload_btn = QPushButton(tr("models.unload"))
            unload_btn.setFixedWidth(action_w)
            unload_btn.clicked.connect(lambda checked, n=name: self._unload(n))
            lay.addWidget(unload_btn, 0, Qt.AlignVCenter)
        else:
            load_btn = QPushButton(tr("models.load"))
            load_btn.setObjectName("primary")
            load_btn.setFixedWidth(action_w)
            load_btn.clicked.connect(lambda checked, n=name: self._load(n))
            lay.addWidget(load_btn, 0, Qt.AlignVCenter)

        del_btn = QPushButton(tr("models.delete"))
        del_btn.setObjectName("danger")
        del_btn.setFixedWidth(column_width("delete", mode))
        del_btn.clicked.connect(lambda checked, mid=model["id"]: self._delete(mid, name))
        lay.addWidget(del_btn, 0, Qt.AlignVCenter)

        return card

    def _badge(self, text: str, color: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {color}; background: {color}22; border-radius: 10px;"
            "padding: 2px 10px; font-size: 12px; font-weight: 600;"
        )
        return lbl

    # ---------------------------------------------------------------- actions

    def _copy_model_name(self, model_name: str):
        QGuiApplication.clipboard().setText(model_name)
        self.status_label.setText(tr("models.name_copied"))

    def _scan(self):
        self.scan_btn.setEnabled(False)
        self.status_label.setText(tr("models.scanning"))

        def do_scan():
            return self.api.scan_models()

        def on_result(result):
            count = result.get("scanned", 0)
            self.status_label.setText(tr("models.scan_complete").format(count=count))
            self.scan_btn.setEnabled(True)
            self._refresh()

        def on_error(err):
            self.status_label.setText(tr("models.error_prefix").format(err=err))
            self.scan_btn.setEnabled(True)

        t = run_in_thread(do_scan, on_result, on_error)
        self._threads.append(t)

    def _open_config(self, model_name: str):
        self.status_label.setText(tr("models.cfg_loading").format(name=model_name))

        def fetch():
            # VRAM total lets the dialog cap parallel slots per model+ctx.
            vram_mb = None
            try:
                mon = self.api.get_monitor()
                gpus = (mon.get("gpu") or {}).get("gpus") or []
                if gpus and gpus[0].get("memory_total"):
                    vram_mb = int(gpus[0]["memory_total"])
            except Exception:
                pass
            return self.api.get_model_run_config(model_name), vram_mb

        def on_result(result):
            config, vram_mb = result
            dlg = ModelConfigDialog(self.api, model_name, config, parent=self,
                                    vram_total_mb=vram_mb)
            dlg.exec()
            self._refresh()

        def on_error(err):
            self.status_label.setText(tr("models.error_prefix").format(err=err))
            QMessageBox.critical(self, tr("models.error_title"), str(err))

        t = run_in_thread(fetch, on_result, on_error)
        self._threads.append(t)

    def _load(self, model_name: str):
        self.status_label.setText(tr("models.loading_model").format(name=model_name))

        def do_load():
            return self.api.load_model(model_name)

        def on_result(result):
            self.status_label.setText(tr("models.loaded").format(name=model_name))
            self._refresh()

        def on_error(err):
            self.status_label.setText(tr("models.error_prefix").format(err=err))

        t = run_in_thread(do_load, on_result, on_error)
        self._threads.append(t)

    def _unload(self, model_name: str):
        self.status_label.setText(tr("models.unloading").format(name=model_name))

        def do_unload():
            return self.api.unload_model(model_name)

        def on_result(result):
            self.status_label.setText(tr("models.unloaded").format(name=model_name))
            self._refresh()

        def on_error(err):
            self.status_label.setText(tr("models.error_prefix").format(err=err))

        t = run_in_thread(do_unload, on_result, on_error)
        self._threads.append(t)

    def _toggle_vision(self, model_name: str, enabled: bool):
        action = tr("models.vision_enabling") if enabled else tr("models.vision_disabling")
        self.status_label.setText(action.format(name=model_name))

        def do_toggle():
            return self.api.toggle_vision(model_name, enabled)

        def on_result(result):
            if enabled:
                self.status_label.setText(tr("models.vision_enabled").format(name=model_name))
            else:
                self.status_label.setText(tr("models.vision_disabled").format(name=model_name))
            self._refresh()

        def on_error(err):
            self.status_label.setText(tr("models.error_prefix").format(err=err))
            self._refresh()

        t = run_in_thread(do_toggle, on_result, on_error)
        self._threads.append(t)

    def _delete(self, model_id: int, model_name: str):
        reply = QMessageBox.question(
            self, tr("models.confirm_delete_title"),
            tr("models.confirm_delete_msg").format(name=model_name),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        import httpx

        def do_delete():
            resp = httpx.delete(f"{self.api.base_url}/api/models/{model_id}", timeout=10.0)
            resp.raise_for_status()
            return resp.json()

        def on_result(result):
            self.status_label.setText(tr("models.deleted").format(name=model_name))
            self._refresh()

        def on_error(err):
            self.status_label.setText(tr("models.error_prefix").format(err=err))

        t = run_in_thread(do_delete, on_result, on_error)
        self._threads.append(t)
