"""背景图层：把主题源图画在整个窗口的最底下。

为什么不用 QSS 的 ``background-image``：它按图片原始尺寸绘制、**不随控件缩放**，
窗口一 resize 就会被裁切或平铺。这里改成在控件的 ``paintEvent`` 里
``drawPixmap``，按当前尺寸做 cover 缩放，缩放结果按 (源图, 尺寸) 缓存。

透明度也不走色值：Qt 把 8 位十六进制读成 ``#AARRGGBB``（见 styles.tint 的注释），
所以浓度用 ``QPainter.setOpacity`` 表达，面板半透明用 ``rgba()`` 函数式写法。
"""

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from lmloader.gui import styles, ui_config

_CACHE_LIMIT = 6
_cache: dict[tuple[str, int, int, float], QPixmap] = {}
_MEAN_LIMIT = 8
_mean_cache: dict[str, str | None] = {}


def average_colour(source: str) -> str | None:
    """The picture's own average pixel as #RRGGBB, or None if it can't be read.

    Decoded at 32×32 through ``QImageReader.setScaledSize`` so the whole
    multi-megabyte file never lands in memory just to answer "is this painting
    light or dark", and the answer is cached per path: every slider notch asks.
    """
    if source in _mean_cache:
        return _mean_cache[source]

    colour = None
    reader = QImageReader(source)
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid():
        reader.setScaledSize(QSize(32, 32))
        image = reader.read()
        if not image.isNull():
            image = image.convertToFormat(QImage.Format_RGB32)
            total = [0, 0, 0]
            for y in range(image.height()):
                for x in range(image.width()):
                    p = image.pixelColor(x, y)
                    total[0] += p.red()
                    total[1] += p.green()
                    total[2] += p.blue()
            count = image.width() * image.height()
            colour = "#{:02X}{:02X}{:02X}".format(*(round(v / count) for v in total))
    if len(_mean_cache) >= _MEAN_LIMIT:
        _mean_cache.pop(next(iter(_mean_cache)), None)
    _mean_cache[source] = colour
    return colour


def cover_pixmap(source: str, size: QSize, dpr: float = 1.0) -> QPixmap:
    """`source` scaled to *cover* `size`, cropped from the centre.

    Cached per size because a resize storm must not re-decode a multi-megabyte
    picture on every paint; the cache is keyed on the device-pixel target so a
    HiDPI screen gets a genuinely sharp bitmap instead of an upscaled one.
    """
    target = QSize(max(1, round(size.width() * dpr)), max(1, round(size.height() * dpr)))
    key = (source, target.width(), target.height(), dpr)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    image = QImage(source)
    if image.isNull():
        return QPixmap()
    scaled = image.scaled(target, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                          Qt.TransformationMode.SmoothTransformation)
    x = max(0, (scaled.width() - target.width()) // 2)
    y = max(0, (scaled.height() - target.height()) // 2)
    pixmap = QPixmap.fromImage(scaled.copy(x, y, target.width(), target.height()))
    pixmap.setDevicePixelRatio(dpr)
    if len(_cache) >= _CACHE_LIMIT:
        _cache.pop(next(iter(_cache)), None)
    _cache[key] = pixmap
    return pixmap


def clear_cache() -> None:
    _cache.clear()


class WallpaperHost(QWidget):
    """Central widget: paints the picture, then lets the pages sit on top of it.

    Children only show it through their own ``rgba()`` fills, which is exactly
    what :func:`styles.with_panel_alpha` produces while a wallpaper is active.
    """

    def paintEvent(self, event):
        super().paintEvent(event)
        state = styles.wallpaper_state()
        if not state:
            return
        pixmap = cover_pixmap(state["source"], self.size(), self.devicePixelRatioF())
        if pixmap.isNull():
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(styles.hexify(styles.c("bg"))))
        painter.setOpacity(state["strength"])
        painter.drawPixmap(0, 0, pixmap)
        painter.end()


def preview_wallpaper(source: str, enabled: bool, strength: int, panel_alpha: int):
    """Repaint with these numbers right now; writes nothing to config.

    The sliders must answer while they are being dragged — waiting for 保存设置
    would leave the user guessing at every notch. The durable write stays in the
    commit handler so a preview that gets abandoned never lands on disk.
    """
    app = QApplication.instance()
    if app is None:
        return None
    state = None if (not enabled or not source) else {
        "enabled": True, "source": str(source),
        "strength": strength / 100, "panel_alpha": panel_alpha / 100}
    base, overrides = ui_config.active_theme()
    styles.apply_theme(app, base, overrides, wallpaper=state)
    for widget in app.topLevelWidgets():
        # topLevelWidgets 里什么都有（对话框、行控件、隐藏的辅助窗口），
        # 只有主窗口才有 centralWidget，别对着普通 QWidget 找它。
        host = getattr(widget, "centralWidget", None)
        if callable(host) and isinstance(host(), WallpaperHost):
            host().update()
    return state
