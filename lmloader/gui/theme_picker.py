"""The UI entry point for an image-derived theme — 选图 → 取色 → 预览 → 应用.

``theme_extract`` turns a picture into 47 tokens and ``styles.apply_theme`` can
already paint the window from such a map; this module is the missing hand that
lets the user pick the picture. Nothing here repaints the app itself: applying
writes ``ui.theme``/``ui.custom_theme`` and emits, and the window re-reads the
config through ``ui_config.active_theme()`` — one source of truth for startup
and for a live switch.
"""

import hashlib
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from lmloader.gui.i18n import tr
from lmloader.gui.styles import c
from lmloader.gui.ui_config import (
    load_custom_theme, save_custom_theme, save_ui_setting, theme_dir,
)

# The 12 colours worth showing a human; the other 35 are derived from these and
# only differ in one wash step, so a 47-cell grid would be a wall of near-dupes.
SWATCHES = ["bg", "surface", "fg", "title", "muted", "accent", "accent_soft",
            "bubble_user", "green", "red", "warn", "sidebar"]

_theme_extract = None


def extractor():
    """Import the extractor on first use.

    ``theme_extract`` pulls in PIL and numpy, and this module is reachable from
    the settings page at boot — paying that import cost on every launch to serve
    a button almost nobody presses is how the startup spinner came back before.
    """
    global _theme_extract
    if _theme_extract is None:
        from lmloader.gui import theme_extract
        _theme_extract = theme_extract
    return _theme_extract


def ink_on(colour: str) -> str:
    """Black or white — whichever actually reads on this swatch.

    A token-coloured label cannot inherit the theme's ``fg``: ``bg`` and ``title``
    sit at opposite ends of the ramp, so one of them is unreadable either way.
    Picking by contrast is what the palette itself does for ``on_accent``.
    """
    te = extractor()
    rgb = te.to_rgb(colour)
    white, black = (255, 255, 255), (17, 17, 17)
    return "#FFFFFF" if te.contrast(white, rgb) >= te.contrast(black, rgb) else "#111111"


def contrast_score(ex) -> tuple[int, int]:
    targets = extractor().CONTRAST_TARGETS
    total = len(targets)
    ok = sum(1 for key, ratio in ex.contrast.items() if key in targets and ratio >= targets[key])
    return ok, total


def store_source(path: str | Path) -> Path:
    """Copy the picked picture under ``~/.lmloader/themes`` and return its path.

    The picker may be a temp folder or a camera roll the user clears later; the
    theme keeps working only if the source it was derived from survives. Named by
    content hash so re-picking the same file is a no-op instead of a new copy.
    """
    src = Path(path)
    dest_dir = theme_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    # "重新取色" feeds the stored path back in. Its stem already ends in the
    # digest, so hashing it again would append a second suffix and leave a fresh
    # multi-megabyte copy of the same picture on every click.
    if src.parent.resolve() == dest_dir.resolve():
        return src
    digest = hashlib.sha1(src.read_bytes()).hexdigest()[:8]
    dest = dest_dir / f"{src.stem}-{digest}{src.suffix.lower()}"
    if not dest.exists():
        shutil.copy2(src, dest)
    return dest


def pick_image(parent: QWidget) -> Path | None:
    start = theme_dir() if theme_dir().exists() else Path.home()
    chosen, _ = QFileDialog.getOpenFileName(
        parent, tr("theme.pick_title"), str(start), tr("theme.pick_filter"))
    return Path(chosen) if chosen else None


def _ghost(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setStyleSheet(
        f"QPushButton {{ color: {c('fg')}; background: {c('btn')};"
        f" border: 1px solid {c('control_border')}; border-radius: 6px;"
        f" padding: 4px 12px; }}")
    return button


def _primary(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setStyleSheet(
        f"QPushButton {{ color: {c('on_accent')}; background: {c('accent')};"
        f" border: 1px solid {c('accent')}; border-radius: 6px;"
        f" padding: 4px 12px; font-weight: 600; }}")
    return button


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {c('muted')}; font-size: 12px;")
    return label


class PreviewDialog(QDialog):
    """The picture, the colours taken from it, and a base to rebuild them on.

    The base radios re-run the extractor rather than just recolouring the swatch
    grid: a light base flips the whole ramp (canvas, washes, ink), so previewing
    it honestly means showing what would actually be applied.
    """

    def __init__(self, parent, source: Path, ex):
        super().__init__(parent)
        self.source = source
        self.ex = ex
        self.setWindowTitle(tr("theme.preview_title"))
        self.resize(760, 340)
        self._build()
        self._paint()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(16)

        left = QVBoxLayout()
        self._pic = QLabel()
        self._pic.setScaledContents(False)
        self._pic.setStyleSheet(f"border: 1px solid {c('border')}; border-radius: 8px;"
                                f" background: {c('bg')};")
        self._pic.setPixmap(QPixmap(str(self.source)).scaled(
            220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        left.addWidget(self._pic)
        self._pic_hint = _hint(tr("theme.original").format(size="220×220"))
        left.addWidget(self._pic_hint)
        left.addStretch(1)
        root.addLayout(left)

        mid = QVBoxLayout()
        self._swatch_hint = _hint("")
        mid.addWidget(self._swatch_hint)
        self._grid_host = QWidget()
        grid = QGridLayout(self._grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self._cells = []
        for index in range(len(SWATCHES)):
            cell = QLabel()
            cell.setAlignment(Qt.AlignCenter)
            grid.addWidget(cell, index // 3, index % 3)
            self._cells.append(cell)
        mid.addWidget(self._grid_host)

        self._dark = QRadioButton(tr("theme.base_dark"))
        self._light = QRadioButton(tr("theme.base_light"))
        self._dark.setChecked(self.ex.base == "dark")
        self._light.setChecked(self.ex.base == "light")
        for radio in (self._dark, self._light):
            radio.toggled.connect(self._on_base_switched)
        bases = QHBoxLayout()
        bases.addWidget(self._dark)
        bases.addWidget(self._light)
        bases.addStretch(1)
        mid.addLayout(bases)
        self._contrast_hint = _hint("")
        mid.addWidget(self._contrast_hint)
        mid.addStretch(1)
        root.addLayout(mid, 1)

        right = QVBoxLayout()
        right.addStretch(1)
        self._apply = _primary(tr("theme.apply"))
        self._apply.clicked.connect(self.accept)
        right.addWidget(self._apply)
        cancel = _ghost(tr("theme.cancel"))
        cancel.clicked.connect(self.reject)
        right.addWidget(cancel)
        right.addWidget(_hint(tr("theme.apply_hint")))
        root.addLayout(right)

    def _on_base_switched(self, on: bool):
        if not on:
            return  # the radio being *un*checked always arrives with its sibling checked
        base = "dark" if self._dark.isChecked() else "light"
        try:
            self.ex = extractor().extract_palette(self.source, base=base)
        except Exception as err:  # a bad picture cannot be half-re-extracted
            QMessageBox.warning(self, tr("settings.error_title"),
                                tr("settings.theme_error").format(err=err))
            return
        self._paint()

    def _paint(self):
        ok, total = contrast_score(self.ex)
        self._swatch_hint.setText(tr("theme.swatches").format(
            n=len(self.ex.tokens), shown=len(SWATCHES)))
        for cell, token in zip(self._cells, SWATCHES):
            colour = self.ex.tokens.get(token, "#000000")
            cell.setText(f"{token}\n{colour}")
            cell.setStyleSheet(
                f"background: {colour}; color: {ink_on(colour)};"
                f" border: 1px solid {c('border')}; border-radius: 6px;"
                f" padding: 6px; font-size: 11px;")
        self._contrast_hint.setText(tr("theme.contrast").format(ok=ok, total=total))


def run_picker(parent: QWidget, source: Path | None = None):
    """Ask for a picture, extract, preview. Returns the accepted palette or None."""
    source = source or pick_image(parent)
    if source is None:
        return None
    try:
        ex = extractor().extract_palette(source)
    except Exception as err:
        QMessageBox.warning(parent, tr("settings.error_title"),
                            tr("settings.theme_error").format(err=err))
        return None
    if PreviewDialog(parent, source, ex).exec() != QDialog.DialogCode.Accepted:
        return None
    return ex


def apply_palette(source: Path, ex) -> Path:
    """Persist an accepted palette *and* select it as the active theme.

    Both writes belong here: ``ui.custom_theme`` without ``ui.theme=custom`` is a
    palette nobody paints with, and the two half-steps were exactly the gap that
    let a headless caller apply colours and still get the built-in theme.

    The contrast tally is stored next to the tokens so the settings page can
    report it at startup without decoding the picture and re-running the
    extractor just to draw a label.
    """
    stored = store_source(source)
    save_custom_theme(ex.base, str(stored), ex.tokens, contrast_score(ex))
    save_ui_setting("theme", "custom")
    return stored


class CustomThemeRow(QWidget):
    """The 自定义主题 line inside the settings page's appearance group.

    Emits the theme name the window should switch to; the settings page forwards
    it to ``MainWindow._on_theme_changed``, which reloads the config. Applying
    from here is immediate on purpose — waiting for 保存设置 would leave a
    previewed palette sitting in a dialog nobody can find again.
    """

    theme_applied = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        line = QHBoxLayout()
        line.setSpacing(8)
        self._thumb = QLabel()
        self._thumb.setFixedSize(42, 42)
        self._thumb.setStyleSheet(f"border: 1px solid {c('border')}; border-radius: 6px;")
        line.addWidget(self._thumb)
        self._pick = _primary(tr("settings.theme_pick"))
        self._pick.clicked.connect(self._on_reextract)
        line.addWidget(self._pick)
        self._swap = _ghost(tr("settings.theme_swap"))
        self._swap.clicked.connect(self._on_new_picture)
        self._swap.setVisible(False)
        line.addWidget(self._swap)
        self._reset = _ghost(tr("settings.theme_reset"))
        self._reset.clicked.connect(self._on_reset)
        line.addWidget(self._reset)
        line.addStretch(1)
        layout.addLayout(line)

        self._hint = _hint("")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        self.refresh()

    def custom(self) -> dict:
        return load_custom_theme()

    def _stored_source(self) -> Path | None:
        raw = self.custom().get("source") or ""
        path = Path(raw)
        return path if raw and path.exists() else None

    def refresh(self):
        custom = self.custom()
        source = self._stored_source()
        has_palette = bool(custom.get("tokens"))
        self._pick.setText(tr("settings.theme_reextract") if source
                           else tr("settings.theme_pick"))
        self._swap.setVisible(has_palette)
        if not has_palette:
            self._thumb.setPixmap(QPixmap())
            self._hint.setText(tr("settings.theme_none"))
            return
        if source is not None:
            self._thumb.setPixmap(QPixmap(str(source)).scaled(
                40, 40, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        else:
            self._thumb.setPixmap(QPixmap())
        base = tr("settings.theme_dark") if custom["base"] == "dark" else tr("settings.theme_light")
        name = Path(custom.get("source", "")).name
        if source is None:
            # The copy in the data dir was deleted; the palette still applies,
            # only 重新取色 is impossible until another picture is picked.
            self._hint.setText(f"{name} · {base} · {tr('settings.theme_none')}")
            return
        ok, total = (custom.get("contrast") + [0, 0])[:2]
        self._hint.setText(tr("settings.theme_source").format(
            file=name, base=base, ok=ok, total=total))

    def _finish(self, source: Path, ex):
        if ex is None:
            return
        apply_palette(source, ex)   # writes ui.custom_theme and selects it
        self.refresh()
        self.theme_applied.emit("custom")

    def _on_reextract(self):
        """Re-derive the palette from the picture already in the data dir."""
        source = self._stored_source()
        if source is None:
            self._on_new_picture()
            return
        self._finish(source, run_picker(self, source))

    def _on_new_picture(self):
        source = pick_image(self)
        if source is None:
            return
        self._finish(source, run_picker(self, source))

    def _on_reset(self):
        """Fall back to the built-in palette, keeping the picture for later.

        Deliberately not ``clear_custom_theme()``: 恢复内置 is a switch, not a
        delete, so 自定义 still shows up in the combo with its colours intact.
        The window resolves the theme from disk, so the choice has to be written
        before the emit.
        """
        base = self.custom().get("base", "dark")
        base = base if base in ("dark", "light") else "dark"
        save_ui_setting("theme", base)
        self.theme_applied.emit(base)
