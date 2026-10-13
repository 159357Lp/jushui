"""Shared vector icons for GUI pages.

Emoji / unicode glyphs do not render reliably under the app font on Windows,
so every icon is rasterised from an inline SVG path instead. models.py and
logs.py each used to keep their own copy of these helpers — this module is the
single source so a tweak to an icon lands on both pages.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer


_EYE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>'
    '<circle cx="12" cy="12" r="3"/>{slash}</svg>'
)
_SLASH_SVG = '<line x1="3" y1="3" x2="21" y2="21"/>'
_COPY_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="8" y="8" width="12" height="12" rx="1.5"/>'
    '<path d="M16 8V5.5A1.5 1.5 0 0 0 14.5 4h-9A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H8"/>'
    '</svg>'
)
_GEAR_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="3"/>'
    '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>'
    '</svg>'
)
# Arrow into a tray: "save this answer to disk". Reads as download at 15px,
# where a floppy disk collapses into an unreadable blob.
_DOWNLOAD_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 3v12"/>'
    '<path d="m7.5 10.5 4.5 4.5 4.5-4.5"/>'
    '<path d="M4 17v2.5A1.5 1.5 0 0 0 5.5 21h13a1.5 1.5 0 0 0 1.5-1.5V17"/>'
    '</svg>'
)
# Paperclip for the attachment picker.
_ATTACH_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 11.5 12.6 19a4.5 4.5 0 0 1-6.4-6.4l7.8-7.7a3 3 0 0 1 4.2 4.2l-7.7 7.8a1.5 1.5 0 0 1-2.1-2.1l7.1-7.2"/>'
    '</svg>'
)


def icon_from_svg(svg: str, size: int = 18) -> QIcon:
    """Rasterise an SVG string into a QIcon at `size` px (transparent bg)."""
    renderer = QSvgRenderer(svg.encode("utf-8"))
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    renderer.render(painter)
    painter.end()
    return QIcon(pix)


def eye_icon(color: str, slashed: bool = False, size: int = 20) -> QIcon:
    """Eye glyph; `slashed` marks a disabled / unloaded vision encoder."""
    svg = _EYE_SVG.format(color=color, slash=_SLASH_SVG if slashed else "")
    return icon_from_svg(svg, size)


def copy_icon(color: str, size: int = 18) -> QIcon:
    """Overlapping-sheets glyph used by the copy buttons."""
    return icon_from_svg(_COPY_SVG.format(color=color), size)


def gear_icon(color: str, size: int = 18) -> QIcon:
    """Gear glyph for the per-model config button."""
    return icon_from_svg(_GEAR_SVG.format(color=color), size)


def download_icon(color: str, size: int = 18) -> QIcon:
    """Arrow-into-tray glyph for 'save this answer to a file'."""
    return icon_from_svg(_DOWNLOAD_SVG.format(color=color), size)


def attach_icon(color: str, size: int = 18) -> QIcon:
    """Paperclip glyph for the attachment picker."""
    return icon_from_svg(_ATTACH_SVG.format(color=color), size)


def app_icon_path() -> str:
    """Absolute path to the bundled multi-resolution application icon."""
    from pathlib import Path

    # lmloader/gui/icons.py -> repo root
    root = Path(__file__).resolve().parent.parent.parent
    return str(root / "lmloader_icon.ico")


def app_icon() -> QIcon:
    """Application icon for the window and the tray.

    Returns a null QIcon when the file is absent (e.g. a wheel install that
    skipped the asset) so callers can fall back rather than crash.
    """
    path = app_icon_path()
    if not Path(path).exists():
        return QIcon()
    return QIcon(path)
