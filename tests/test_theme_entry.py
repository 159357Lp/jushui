"""自定义主题入口：选图 → 取色 → 落盘 → 重绘，每一步都得是可判据的。

这个套件钉住四件容易悄悄坏掉的事：

* ``apply_palette`` 必须**同时**写 ``ui.custom_theme`` 和 ``ui.theme=custom``——
  只写调色板的话，界面下次启动还是内置主题，用户会以为"应用"没生效；
* ``active_theme()`` 是唯一的判据来源（启动、设置页、取色器都读它），所以
  "半写的配置"必须明确回落到内置主题，而不是渲染出一半新一半旧的界面；
* 源图复制进数据目录，``重新取色`` 不必再挑一次文件；目录不能落在仓库里；
* 色板每格的读写色按该格亮度单独选——统一用 ``fg`` 时，``bg`` 和 ``title``
  一个太深一个太浅，总有一格看不见（原型截图就是这么露馅的）。

反向对照：每个判据都配一条"坏状态必须报红"的断言，见 ``_falls_back`` 与
``test_per_swatch_ink_is_load_bearing``。
"""

import re
from pathlib import Path

import pytest
import yaml

from lmloader.gui import styles, ui_config
from lmloader.gui.theme_picker import SWATCHES, ink_on


HEX6 = re.compile(r"^#[0-9A-Fa-f]{6}$")


@pytest.fixture
def scratch(monkeypatch, tmp_path):
    """A private config.yaml and a private themes dir — never his real ones.

    The settings page resolves its own path at construction, so both readers have
    to be redirected: without this, 保存设置 would overwrite the repo's config.
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text("ui:\n  theme: light\n  language: zh\n", encoding="utf-8")
    monkeypatch.setattr(ui_config, "config_path", lambda: cfg)
    monkeypatch.setattr("lmloader.gui.pages.settings._get_config_path", lambda: cfg)
    themes = tmp_path / "themes"
    monkeypatch.setattr("lmloader.gui.theme_picker.theme_dir", lambda: themes)
    return cfg


@pytest.fixture
def picture(tmp_path):
    """A two-band navy/white picture: enough colour for the extractor to chew on."""
    from PIL import Image

    path = tmp_path / "shot.png"
    img = Image.new("RGB", (60, 40))
    for x in range(60):
        for y in range(40):
            img.putpixel((x, y), (31, 58, 147) if y < 28 else (244, 245, 250))
    img.save(path)
    return path


def read_ui(cfg):
    return (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("ui") or {}


def test_apply_palette_persists_and_selects(scratch, picture):
    from lmloader.gui.theme_picker import apply_palette

    stored = apply_palette(picture, _extract(picture))
    ui = read_ui(scratch)
    assert ui["theme"] == "custom", "应用了调色板却没换主题，下次启动还是内置那套"
    assert len(ui["custom_theme"]["tokens"]) == 47
    assert HEX6.match(next(iter(ui["custom_theme"]["tokens"].values())))
    assert Path(ui["custom_theme"]["source"]) == stored and stored.exists()
    assert stored.parent.name == "themes"


def test_reapplying_the_same_picture_does_not_pile_up_copies(scratch, picture):
    from lmloader.gui.theme_picker import apply_palette

    first = apply_palette(picture, _extract(picture))
    apply_palette(picture, _extract(picture))
    assert list(first.parent.glob("*")) == [first], "同一张图每次应用都复制一份"


def test_retaking_colour_from_the_stored_picture_copies_nothing(scratch, picture):
    """「重新取色」喂回来的是数据目录里那条路径，不是用户原始文件。

    存下来的文件名茎部已经带哈希，再按同一规则命名会一直往后追加哈希段，
    每次点击多留一份几 MB 的重复图——他的 themes 目录里就是这么攒出三份的。
    """
    from lmloader.gui.theme_picker import apply_palette

    first = apply_palette(picture, _extract(picture))
    again = apply_palette(first, _extract(first))
    assert again == first, f"源图已在数据目录里还重新复制: {again.name}"
    assert [p.name for p in first.parent.glob("*")] == [first.name]


def test_selection_is_what_makes_a_palette_visible(scratch, picture, monkeypatch):
    """反向对照：把"选中主题"这一步抹掉，界面必须还是内置那套。

    没有这条，``test_apply_palette_persists_and_selects`` 只要 tokens 写进去就会
    变绿，而真正的用户症状——应用了却没换肤——恰好测不出来。
    """
    from lmloader.gui import theme_picker

    monkeypatch.setattr(theme_picker, "save_ui_setting", lambda key, value: None)
    theme_picker.apply_palette(picture, _extract(picture))
    assert read_ui(scratch)["theme"] == "light"
    assert ui_config.active_theme() == ("light", None)


def test_theme_dir_stays_out_of_the_repository():
    """A theme's source image is user data; it must not show up as untracked junk."""
    repo = Path(styles.__file__).resolve().parents[2]
    resolved = ui_config.theme_dir().resolve()
    assert repo not in resolved.parents and resolved != repo, f"{resolved} 在仓库里"


def _extract(path):
    from lmloader.gui.theme_extract import extract_palette
    return extract_palette(path)


# ------------------------------------------------------------- 配置 → 该用哪套色

def test_active_theme_hands_back_the_tokens(scratch, picture):
    from lmloader.gui.theme_picker import apply_palette

    ex = _extract(picture)
    apply_palette(picture, ex)
    base, overrides = ui_config.active_theme()
    assert base == ex.base
    assert overrides["accent"] == ex.tokens["accent"]
    assert len(overrides) == 47


def test_builtin_selection_ignores_a_stored_palette(scratch, picture):
    """恢复内置 is a switch, not a delete: the picture stays, but must not paint."""
    from lmloader.gui.theme_picker import apply_palette

    apply_palette(picture, _extract(picture))
    ui_config.save_ui_setting("theme", "light")
    assert ui_config.active_theme() == ("light", None)


@pytest.mark.parametrize("broken", [
    {"theme": "custom"},                                        # 从没取过色
    {"theme": "custom", "custom_theme": {"base": "dark"}},      # 只有基底没有色
    {"theme": "custom", "custom_theme": {"tokens": {}}},        # 半写的空表
    {"theme": "custom", "custom_theme": {"tokens": {"bg": 7}}},  # 坏类型
    {"theme": "nonsense"},                                      # 不存在的主题名
])
def test_active_theme_falls_back_on_a_half_written_config(scratch, broken):
    cfg = scratch
    cfg.write_text(yaml.safe_dump({"ui": broken}, allow_unicode=True), encoding="utf-8")
    base, overrides = ui_config.active_theme()
    assert base in ("dark", "light"), base
    assert overrides is None, f"这份配置渲染不出完整界面，却还被当成自定义主题: {broken}"


def test_a_missing_source_picture_keeps_the_palette(scratch, picture, qapp):
    """删掉数据目录里的图，界面颜色不该跟着丢——只是重新取色要重新挑文件。"""
    from lmloader.gui.theme_picker import CustomThemeRow, apply_palette

    stored = apply_palette(picture, _extract(picture))
    stored.unlink()
    row = CustomThemeRow()
    assert "尚未取色" in row._hint.text() or "No picture" in row._hint.text()
    assert row._pick.text() in ("选择图片…", "Choose image…"), "图没了却还许诺重新取色"
    assert ui_config.active_theme()[1] is not None, "提示行把调色板也一起判没了"


# --------------------------------------------------------------------- 色板可读性

def test_ink_on_picks_the_readable_end():
    assert ink_on("#0D1117") == "#FFFFFF"      # 深色底 → 白字
    assert ink_on("#F5F6FA") == "#111111"      # 浅色底 → 黑字
    for colour in ("#373B81", "#B7BAED", "#D4D6FA", "#41A36A", "#EB5E68"):
        from lmloader.gui.theme_extract import contrast, to_rgb
        assert contrast(to_rgb(ink_on(colour)), to_rgb(colour)) >= 4.5, colour


def test_per_swatch_ink_is_load_bearing():
    """反向对照：整张色板统一用主题前景色，必然有格子看不见。"""
    from lmloader.gui.theme_extract import contrast, to_rgb

    for base in ("dark", "light"):
        palette = styles.theme_tokens(base)
        worst_with_fg = min(contrast(to_rgb(palette["fg"]), to_rgb(palette[t]))
                            for t in SWATCHES if t in palette)
        worst_with_ink = min(contrast(to_rgb(ink_on(palette[t])), to_rgb(palette[t]))
                             for t in SWATCHES if t in palette)
        assert worst_with_fg < 4.5, f"{base}: 统一 fg 竟然也全都读得清，这个判据没用力"
        assert worst_with_ink >= 4.5, f"{base}: 逐格选色后仍有 {worst_with_ink:.2f}:1 的格子"


# ------------------------------------------------------------------- 设置页接线

class _FakeApi:
    base_url = "http://127.0.0.1:1"
    token = ""

    def _headers(self):
        return {}

    def get_loaded_models(self):
        return []


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_settings_page_offers_a_custom_entry_and_a_row(scratch, qapp):
    from lmloader.gui.pages.settings import SettingsPage
    from lmloader.gui.theme_picker import CustomThemeRow

    page = SettingsPage(_FakeApi())
    assert page.theme_combo.findData("custom") >= 0, "下拉里没有自定义这一项"
    assert isinstance(page.theme_row, CustomThemeRow)

    emitted = []
    page.theme_changed.connect(emitted.append)
    page._on_custom_theme("custom")
    assert emitted == ["custom"]
    assert page.theme_combo.currentData() == "custom", "按钮换了主题，下拉还停在旧选项"


def test_save_settings_writes_before_it_repaints(scratch, qapp, monkeypatch):
    """顺序判据：窗口是照着磁盘上的配置重绘的，先发信号就等于用旧色画新主题。"""
    from PySide6.QtWidgets import QMessageBox

    from lmloader.gui.pages.settings import SettingsPage

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    page = SettingsPage(_FakeApi())
    index = page.theme_combo.findData("dark")
    page.theme_combo.setCurrentIndex(index)

    seen = []
    def on_theme(theme):
        seen.append((theme, read_ui(scratch).get("theme")))
    page.theme_changed.connect(on_theme)
    page._save_settings()
    assert seen == [("dark", "dark")], f"信号先于落盘，重绘读到的是旧主题: {seen}"
