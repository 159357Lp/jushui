"""调色板令牌与页面内联样式：色值必须是 Qt 能按 CSS 意思解析的实色。

Qt 的 ``QColor`` 把 8 位十六进制读成 **#AARRGGBB**，不是 CSS 的 #RRGGBBAA，所以
``"#58a6ff22"`` 会画出一块 12% 透明度的**青柠绿**，而且不报错、不回退、肉眼只在渲染后
才看得出来。深色主题曾经有 8 个令牌踩这个坑，页面里还有三处把 ``"22"`` 直接拼在
``c('accent')`` 后面。这个套件把三件事钉住：

* 两套内置调色板的每个令牌都是 #RRGGBB，且 Qt 解析出来的就是字面那三个通道；
* 页面源码里不再出现"颜色后面拼两位十六进制"的写法；
* ``styles.tint()`` 混色确实落在底衬与目标色之间，并跟随当前（含自定义）调色板；
* ``apply_theme()`` 除了样式表还要下发 QPalette —— 两套模板都没点名 QDialog /
  QMessageBox / 裸 QWidget，没有调色板它们就一直画系统浅灰。

反向对照有两条：往源码里种一条 ``{c('green')}66``，守卫必须抓到它；把应用调色板退回
系统默认，弹窗像素探针必须报红。
"""

import re
from pathlib import Path

import pytest

from lmloader.gui import styles
from lmloader.gui.styles import _PALETTES, c, tint

GUI_ROOT = Path(styles.__file__).parent
HEX6 = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _py_files():
    return sorted(p for p in GUI_ROOT.rglob("*.py") if p.name != "styles.py")


def _alpha_suffixed(source: str):
    """Lines that paint a colour and then glue two hex digits onto it."""
    hits = []
    for lineno, line in enumerate(source.splitlines(), 1):
        if not re.search(r"\b(color|background|border)", line):
            continue
        for m in re.finditer(r"\}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])", line):
            hits.append((lineno, line.strip()[:90]))
    return hits


@pytest.mark.parametrize("base", ["dark", "light"])
def test_palette_tokens_are_plain_six_digit(base):
    for token, value in _PALETTES[base].items():
        assert HEX6.match(value), f"{base}.{token} = {value!r} 不是 #RRGGBB"


@pytest.mark.parametrize("base", ["dark", "light"])
def test_qt_reads_each_token_as_authored(base):
    """The trap only bites inside Qt, so ask Qt: what QColor actually gets must be
    the three channels the token spells, at full opacity."""
    from PySide6.QtGui import QColor

    for token, value in _PALETTES[base].items():
        colour = QColor(value)
        assert colour.isValid(), f"{base}.{token} = {value!r} Qt 解析失败"
        assert (colour.red(), colour.green(), colour.blue()) == \
               tuple(int(value[i:i + 2], 16) for i in (1, 3, 5)), f"{base}.{token} 被读成了别的色"
        assert colour.alpha() == 255, f"{base}.{token} = {value!r} 被当成了带透明度的色"


def test_no_page_glues_an_alpha_suffix_onto_a_colour():
    offenders = {}
    for path in _py_files():
        hits = _alpha_suffixed(path.read_text(encoding="utf-8"))
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"这些行会被 Qt 读成 #AARRGGBB: {offenders}"


def test_alpha_suffix_guard_fires_on_a_planted_style():
    """Negative control: the guard above is worthless unless it can report red."""
    planted = (
        'QLabel { color: #ffffff; }',                       # 无插值，不该报
        "lbl.setStyleSheet(f\"color: {c('green')};\")",      # 纯色值，不该报
        "lbl.setStyleSheet(f\"border-color: {c('green')}66;\")",   # 种下去的坑
    )
    hits = _alpha_suffixed("\n".join(planted))
    assert hits == [(3, planted[2])], f"守卫没抓到种下去的 {planted[2]!r}: {hits}"
    assert _alpha_suffixed("\n".join(planted[:2])) == [], "守卫见谁都报"


@pytest.fixture
def palette_restored():
    """These tests poke the live theme globals *and* the QApplication; put every
    one of them back exactly as found."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    saved = (styles.current_theme, _PALETTES.get("custom"),
             app.palette() if app is not None else None,
             app.styleSheet() if app is not None else None)
    yield
    styles.current_theme, custom, palette, sheet = saved
    if custom is None:
        _PALETTES.pop("custom", None)
    else:
        _PALETTES["custom"] = custom
    if app is not None:
        app.setPalette(palette)
        app.setStyleSheet(sheet)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("base", ["dark", "light"])
def test_tint_lands_between_backdrop_and_colour(base, palette_restored):
    styles.current_theme = base
    over = c("bg")
    for token, alpha in (("accent", 0x22 / 255), ("green", 0x33 / 255), ("red", 0x55 / 255)):
        top, washed = c(token), tint(token, alpha)
        assert HEX6.match(washed)
        for i in (1, 3, 5):
            lo, hi = sorted((int(over[i:i + 2], 16), int(top[i:i + 2], 16)))
            assert lo - 1 <= int(washed[i:i + 2], 16) <= hi + 1, \
                f"{base}/{token}: {washed} 不在 {over}↔{top} 之间"
        assert washed != top, f"{base}/{token} 洗淡等于没洗"
    assert tint("accent", 0.0).lower() == over.lower()


def test_tint_follows_a_custom_palette(palette_restored):
    """A wash must move with the theme, or an image-derived palette gets GitHub blue
    highlights pasted onto it."""
    styles._PALETTES["custom"] = dict(_PALETTES["dark"], bg="#000000", accent="#FF0000")
    styles.current_theme = "custom"
    assert tint("accent", 0.5) == "#800000"
    assert tint("#00FF00", 0.25, over="accent") == "#BF4000"


# ------------------------------------------------ 样式表没点名的控件（弹窗那一类）

def _corner(widget) -> str:
    img = widget.grab().toImage()
    px = img.pixelColor(6, 6)
    return f"#{px.red():02X}{px.green():02X}{px.blue():02X}".upper()


@pytest.mark.parametrize("base", ["dark", "light"])
def test_apply_theme_publishes_a_matching_qpalette(base, qapp, palette_restored):
    from PySide6.QtGui import QPalette

    styles.apply_theme(qapp, base)
    role, group = QPalette.ColorRole, QPalette.ColorGroup
    for role_name, token in (("Window", "bg"), ("Base", "bg"), ("Text", "fg"),
                             ("WindowText", "fg"), ("Button", "btn"), ("Highlight", "accent"),
                             ("HighlightedText", "on_accent"), ("ToolTipBase", "surface"),
                             ("PlaceholderText", "dim")):
        got = getattr(role, role_name)
        assert qapp.palette().color(got).name().upper() == c(token).upper(), role_name
    assert qapp.palette().color(group.Disabled, role.Text).name().upper() == c("dim").upper()


def test_dialogs_paint_the_theme_background(qapp, palette_restored):
    """The bug no stylesheet assertion could see: QDialog / QMessageBox aren't named
    in either template, so they painted the platform's light grey in the dark theme."""
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QDialog, QMessageBox, QWidget

    styles.apply_theme(qapp, "dark")
    want = c("bg").upper()
    for widget in (QMessageBox(), QDialog(), QWidget()):
        widget.resize(320, 160)
        assert _corner(widget) == want, type(widget).__name__

    # 反向对照：退回"只下发样式表"的状态（系统默认调色板），探针必须报红
    probe = QMessageBox()
    probe.resize(320, 160)
    saved = qapp.palette()
    qapp.setPalette(QPalette())
    try:
        assert _corner(probe) != want, "系统默认调色板也画出了 bg —— 这个探针测不出问题"
    finally:
        qapp.setPalette(saved)
