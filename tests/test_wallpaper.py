"""背景图层：底图画在窗口最底下，面板靠 rgba 半透明透出来。

钉住四件会悄悄坏掉的事：

* 透明度只能走 ``rgba()`` 函数式写法和 ``QPainter.setOpacity``——八位十六进制会被
  Qt 读成 ``#AARRGGBB``（见 styles.tint 的注释），一旦有人图省事拼 ``bg + "b3"``，
  整窗颜色会串成另一种色；
* 滑块洗的是卡片（bg/surface/surface_hover/bubble_user/chip_bg），文字、边框、
  强调色必须保持实色，否则正文跟着画一起虚掉；
* 容器（正文区、历史栏、输入区外壳）走 styles.canvas()：铺了画就全透，
  这样没对话时看到的就是画本身，气泡照样把它挡住；
* QPalette 拿不到 alpha（``QColor`` 不认 ``rgba()``），弹窗那一层继续用实色；
* 滑块的百分比要能落盘、能重启恢复，且绝不把 tokens 写丢。

反向对照：每个判据都配一条"关掉背景图层就该回到实色"的断言。
"""

import re
from pathlib import Path

import pytest
import yaml

from lmloader.gui import styles, ui_config


HEX6 = re.compile(r"^#[0-9A-Fa-f]{6}$")
HEX8 = re.compile(r"#[0-9A-Fa-f]{8}\b")


@pytest.fixture
def scratch(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("ui:\n  theme: custom\n  language: zh\n", encoding="utf-8")
    monkeypatch.setattr(ui_config, "config_path", lambda: cfg)
    monkeypatch.setattr("lmloader.gui.theme_picker.theme_dir", lambda: tmp_path / "themes")
    return cfg


@pytest.fixture
def picture(tmp_path):
    from PIL import Image

    path = tmp_path / "shot.png"
    img = Image.new("RGB", (60, 40))
    for x in range(60):
        for y in range(40):
            img.putpixel((x, y), (31, 58, 147) if y < 28 else (244, 245, 250))
    img.save(path)
    return path


@pytest.fixture
def theme_with_picture(scratch, picture):
    """一份真实取过色的配置：后面所有判据都建立在"有源图"之上。"""
    from lmloader.gui.theme_extract import extract_palette
    from lmloader.gui.theme_picker import apply_palette

    ex = extract_palette(picture)
    stored = apply_palette(picture, ex)
    return {"base": ex.base, "tokens": ex.tokens, "source": stored}


@pytest.fixture(autouse=True)
def palette_restored(qapp):        # 依赖 qapp：这样每个用例都一定抓得到那个实例
    """apply_theme 改的是全局；每个用例结束后原样放回。

    autouse 是必须的：滑块的即时预览会走真的 QApplication.setStyleSheet，
    漏一个用例，后面任何量尺寸、读 c() 的用例都在测另一套主题。
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    theme, custom, state = styles.current_theme, styles._PALETTES.get("custom"), styles.wallpaper_state()
    sheet = app.styleSheet() if app is not None else ""
    yield
    styles.current_theme = theme
    if custom is None:
        styles._PALETTES.pop("custom", None)
    else:
        styles._PALETTES["custom"] = custom
    styles.set_wallpaper(state)
    if app is not None and app.styleSheet() != sheet:
        app.setStyleSheet(sheet)


class FakeApp:
    def __init__(self):
        self.sheets = []
        self.palette = None

    def setStyleSheet(self, qss):
        self.sheets.append(qss)

    def setPalette(self, palette):
        self.palette = palette


# ---------------------------------------------------------------- 令牌 → 该画成什么

def test_panel_alpha_only_softens_the_cards():
    """滑块洗的是卡片（气泡/输入/标签/悬停），文字边框强调色一律实色。"""
    tokens = styles.theme_tokens("light")
    paint = styles.with_panel_alpha(tokens, 0.72)
    for role in styles.WALLPAPER_ROLES:
        assert paint[role].startswith("rgba("), f"{role} 没变半透明，画会被它挡住"
    for role in ("fg", "title", "border", "accent", "sidebar", "muted"):
        assert HEX6.match(paint[role]), f"{role} 被洗成半透明了，正文会跟着虚"
    assert paint["fg"] == tokens["fg"]


def test_canvas_goes_see_through_only_while_a_picture_is_painted(palette_restored, qapp):
    """容器问 canvas()，卡片问 c()：没铺画时两者必须完全一样。"""
    styles.set_wallpaper(None)
    assert styles.canvas("bg") == styles.c("bg")
    assert styles.canvas("surface") == styles.c("surface")
    styles.set_wallpaper({"enabled": True, "source": "x.png",
                          "strength": 0.55, "panel_alpha": 1.0})
    assert styles.canvas("bg") == "transparent", "铺了画容器还不透，正文区就是一片纯色"
    assert styles.canvas("surface") == "transparent"
    assert styles.c("bg") != "transparent", "卡片不能跟着一起透"
    styles.set_wallpaper(None)


def test_chat_containers_ask_canvas_not_c():
    """静态守卫：聊天页那三个容器一旦改回 c()，背景层就又看不见了。"""
    src = Path(styles.__file__).with_name("pages").joinpath("chat.py").read_text(encoding="utf-8")
    for needle in ("QWidget#chatSidebar {{ background-color: {canvas('surface')}",
                   "QScrollArea#chatDisplay {{ background-color: {canvas('bg')}",
                   "chatMessages {{ background-color: {canvas('bg')}",
                   "QWidget#chatComposer {{ background-color: {canvas('surface')}"):
        assert needle in src, f"容器样式改回实色了：{needle}"


def test_alpha_is_spelled_as_rgba_never_eight_digit_hex():
    """八位十六进制是 Qt 的坑：它按 #AARRGGBB 解析，颜色会整个串掉。"""
    paint = styles.with_panel_alpha(styles.theme_tokens("dark"), 0.35)
    assert not any(HEX8.search(value) for value in paint.values())
    r, g, b, a = re.match(r"rgba\((\d+), (\d+), (\d+), ([\d.]+)\)",
                          paint["bg"]).groups()
    assert (int(r), int(g), int(b)) == (13, 17, 23), "bg 的三通道被改成了别的色"
    assert float(a) == 0.35


def test_panel_alpha_one_is_byte_identical_to_no_wallpaper():
    tokens = styles.theme_tokens("light")
    assert styles.with_panel_alpha(tokens, 1.0) == tokens


def test_tint_survives_a_translucent_backdrop():
    """洗色要能在半透明 bg 上算：以前它只会切十六进制，喂 rgba 直接崩。"""
    assert styles.hexify("rgba(255, 230, 230, 0.72)") == "#FFE6E6"
    assert styles.hexify("#ABC") == "#AABBCC"


def test_apply_theme_publishes_translucent_qss_and_solid_palette(palette_restored, qapp):
    tokens = styles.theme_tokens("light")
    app = FakeApp()
    styles.apply_theme(app, "light", None,
                       {"enabled": True, "source": "x.png", "strength": 0.55,
                        "panel_alpha": 0.72})
    assert "rgba(" in app.sheets[0], "样式表里没有半透明，画铺了也看不见"
    assert styles.c("bg").startswith("rgba(")
    from PySide6.QtGui import QPalette
    assert app.palette.color(QPalette.ColorRole.Window).alpha() == 255, \
        "弹窗那一层跟着变半透明，会透出桌面"


def test_turning_the_wallpaper_off_restores_solid_everywhere(palette_restored):
    """反向对照：同一份配置把开关关掉，界面必须一点 alpha 都不剩。"""
    app = FakeApp()
    styles.apply_theme(app, "light", None,
                       {"enabled": False, "source": "x.png", "strength": 0.55,
                        "panel_alpha": 0.72})
    assert "rgba(" not in app.sheets[0]
    assert HEX6.match(styles.c("bg"))
    assert styles.wallpaper_state() is None


# ------------------------------------------------------------------- 配置进出

def test_applying_a_picture_theme_turns_the_wallpaper_on(theme_with_picture):
    cfg = ui_config.load_wallpaper()
    assert cfg["enabled"] is True
    assert cfg["strength"] == 55 and cfg["panel_alpha"] == 100
    state = ui_config.wallpaper_state()
    assert state["source"] == str(theme_with_picture["source"])
    assert state["strength"] == pytest.approx(0.55)


def test_wallpaper_write_keeps_the_palette_tokens(theme_with_picture):
    """滑块只改背景层，绝不能顺手把 47 个令牌写丢。"""
    before = ui_config.load_custom_theme()
    ui_config.save_wallpaper(True, 20, 90)
    after = ui_config.load_custom_theme()
    assert after["tokens"] == before["tokens"]
    assert after["wallpaper"] == {"enabled": True, "strength": 20, "panel_alpha": 90}
    assert ui_config.wallpaper_state()["panel_alpha"] == pytest.approx(0.9)


@pytest.mark.parametrize("raw,expect", [
    ({"enabled": True, "strength": 500, "panel_alpha": -20}, (100, 0)),   # 越界要夹住
    ({"enabled": True, "strength": "abc", "panel_alpha": None}, (55, 100)),  # 坏值回落默认
    ({"strength": 30}, (30, 100)),                                        # 缺字段补默认
])
def test_wallpaper_settings_are_normalised(scratch, theme_with_picture, raw, expect):
    custom = ui_config.load_custom_theme()
    ui_config.save_custom_theme(custom["base"], custom["source"], custom["tokens"],
                                custom["contrast"], raw)
    cfg = ui_config.load_wallpaper()
    assert (cfg["strength"], cfg["panel_alpha"]) == expect


def test_no_wallpaper_without_a_source_picture(scratch, theme_with_picture):
    """源图被删了就不该有背景层——半透明面板配一张不存在的图 = 糊成一片。"""
    custom = ui_config.load_custom_theme()
    ui_config.save_custom_theme(custom["base"], "", custom["tokens"], custom["contrast"],
                                {"enabled": True, "strength": 55, "panel_alpha": 72})
    assert ui_config.wallpaper_state() is None


def test_theme_request_carries_the_wallpaper(scratch, theme_with_picture):
    base, overrides, wallpaper = ui_config.theme_request()
    assert overrides and wallpaper["enabled"]
    ui_config.save_wallpaper(False)
    assert ui_config.theme_request()[2] is None


# --------------------------------------------------------------------- 画得出来

def test_cover_pixmap_covers_the_target_and_is_cached(tmp_path, picture):
    from PySide6.QtCore import QSize

    from lmloader.gui import wallpaper

    wallpaper.clear_cache()
    pm = wallpaper.cover_pixmap(str(picture), QSize(200, 90))
    assert not pm.isNull()
    assert pm.width() >= 200 and pm.height() >= 90, "cover 缩放没铺满，边缘会露底色"
    again = wallpaper.cover_pixmap(str(picture), QSize(200, 90))
    assert again.cacheKey() == pm.cacheKey(), "同尺寸重复解码，resize 时会卡"
    assert wallpaper.cover_pixmap(str(tmp_path / "没了.png"), QSize(10, 10)).isNull()


def test_host_paints_the_picture_between_bg_and_art(picture, palette_restored):
    """像素判据：底图既不是纯 bg 也不是原图，强度 0 时必须逐像素等于纯 bg。"""
    from PySide6.QtWidgets import QApplication

    from lmloader.gui.wallpaper import WallpaperHost

    app = QApplication.instance() or QApplication([])
    styles.apply_theme(app, "light", {"bg": "#FFE6E6", "surface": "#F0D8D8"},
                       {"enabled": True, "source": str(picture), "strength": 0.55,
                        "panel_alpha": 0.72})
    host = WallpaperHost()
    host.resize(120, 80)
    corner = host.grab().toImage().pixelColor(6, 6)
    assert (corner.red(), corner.green(), corner.blue()) != (255, 230, 230), "画没画上去"

    styles.apply_theme(app, "light", {"bg": "#FFE6E6", "surface": "#F0D8D8"},
                       {"enabled": True, "source": str(picture), "strength": 0.0,
                        "panel_alpha": 0.72})
    host.update()
    from PySide6.QtTest import QTest
    QTest.qWait(60)
    off = host.grab().toImage().pixelColor(6, 6)
    assert (off.red(), off.green(), off.blue()) == (255, 230, 230), \
        "浓度 0 还画得出东西，说明这条判据抓不到'没关干净'"


# ------------------------------------------------------------------ 滑块接线

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_sliders_preview_then_commit(theme_with_picture, qapp, monkeypatch):
    from PySide6.QtTest import QTest

    from lmloader.gui.theme_picker import CustomThemeRow

    row = CustomThemeRow()
    assert row._wp_enable.isChecked(), "取色后背景层默认就该开着"
    assert row._strength.value() == 55 and row._panel.value() == 100
    assert row._readouts[row._strength].text() == "55%"

    row._strength.setValue(30)
    assert row._readouts[row._strength].text() == "30%"
    assert styles.wallpaper_state()["strength"] == pytest.approx(0.30), "拖动过程中界面没跟着变"
    assert ui_config.load_wallpaper()["strength"] == 55, "还在拖就先写了盘"
    QTest.qWait(600)                       # 等 400ms 防抖落盘
    assert ui_config.load_wallpaper()["strength"] == 30

    row._panel.setValue(60)                # 卡片一旦调虚，QSS 必须出现 rgba()
    assert styles.c("bg").startswith("rgba(") and styles.c("bubble_user").startswith("rgba(")

    emitted = []
    row.theme_applied.connect(emitted.append)
    row._wp_enable.setChecked(False)
    assert emitted == ["custom"]
    assert ui_config.load_wallpaper()["enabled"] is False
    assert styles.wallpaper_state() is None, "关掉开关后还留着背景层状态"


def test_row_hides_the_layer_without_a_palette(scratch, qapp):
    from lmloader.gui.theme_picker import CustomThemeRow

    row = CustomThemeRow()
    assert not row._wp_box.isVisibleTo(row), "还没取过色就摆出背景图层控件"
