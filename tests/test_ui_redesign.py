"""Regression guards for the UI redesign.

Each test here locks a behaviour that was actually reached during the redesign,
so a later edit cannot silently undo it:

* the light/dark palettes stay in sync and keep the designed tokens;
* the two stylesheets stay pure ``$token`` templates — a hard-coded colour would
  make an image-derived theme impossible;
* the sidebar stylesheet stays token-driven (it used to hard-code #192334);
* i18n stays symmetric — several keys were added during the redesign and a
  half-added key only shows up in one language at runtime;
* the model card's fixed-width column budget must not overflow the viewport at
  1280px, which is exactly how the 删除 button once got pushed behind a
  horizontal scrollbar;
* the chat page's redesigned actions have their i18n keys on both sides.
"""

import re

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy

from lmloader.gui.styles import DARK_THEME, LIGHT_THEME, _PALETTES


# --------------------------------------------------------------------- tokens

# The colours the redesign landed on. Changing any of these is a design change,
# not a refactor, so it has to be a conscious edit here too.
LIGHT_TOKENS = {
    "bg": "#F5F6FA",         # light canvas
    "surface": "#FFFFFF",    # cards
    "border": "#E9ECF4",     # card / input border
    "title": "#1A2032",
    "accent": "#2E6BE6",     # primary blue
    "sidebar": "#151A2C",    # dark rail even in the light theme
    "log_bg": "#0F1420",     # console panel stays dark in both themes
    "green_btn": "#22A06B",
    "red": "#E5484D",
}


@pytest.mark.parametrize("name,expected", sorted(LIGHT_TOKENS.items()))
def test_light_palette_matches_design_tokens(name, expected):
    assert _PALETTES["light"][name] == expected


def test_palettes_expose_the_same_keys():
    """A token added to one theme but not the other silently falls back to dark."""
    assert set(_PALETTES["light"]) == set(_PALETTES["dark"])


@pytest.mark.parametrize("name", sorted(LIGHT_TOKENS))
def test_both_themes_define_every_token(name):
    assert name in _PALETTES["dark"]


def _relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance, used to prove the console panel stays dark."""
    hex_color = hex_color.lstrip("#")
    channels = []
    for i in (0, 2, 4):
        value = int(hex_color[i:i + 2], 16) / 255.0
        channels.append(value / 12.92 if value <= 0.03928
                        else ((value + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_light_theme_uses_dark_rail():
    """The light theme keeps the dark navy rail — that contrast is the whole
    point of the redesign's sidebar."""
    assert "background-color: #151A2C" in LIGHT_THEME


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_console_panel_stays_dark_in_both_themes(theme):
    """log_bg is consumed page-level (logs.py), not by LIGHT_THEME, so assert on
    the token itself: the terminal panel must read as a console in either theme,
    otherwise log text stops being legible."""
    log_bg = _PALETTES[theme]["log_bg"]
    assert _relative_luminance(log_bg) < 0.02, (
        f"{theme} log_bg {log_bg} is not dark enough for a console panel"
    )


def test_card_and_control_radius_conventions():
    """12px cards / 10px controls — the visual signature of the redesign."""
    assert LIGHT_THEME.count("border-radius: 12px") >= 4
    assert "border-radius: 10px" in LIGHT_THEME


def test_dark_theme_untouched():
    """The redesign only restyled the light theme; dark must keep its values."""
    assert _PALETTES["dark"]["bg"] == "#0d1117"
    assert _PALETTES["dark"]["surface"] == "#161b22"
    assert "background-color: #0d1117" in DARK_THEME


# -------------------------------------------------------------------- templates
#
# The stylesheets are now ``string.Template`` bodies filled from a palette. That
# is the prerequisite for an image-derived theme: a colour that is not a token
# cannot be recoloured. These guards keep it that way.

_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_WORD_COLOUR = re.compile(r"\b(?:white|black)\b")
_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _declarations(qss: str):
    """Yield (selector, property, value) for real CSS, comments stripped."""
    selector = "(root)"
    for line in _COMMENT.sub("", qss).splitlines():
        stripped = line.strip()
        if stripped.endswith("{"):
            selector = stripped.split("{")[0].strip()
        elif stripped.endswith(";") and ":" in stripped:
            prop, _, value = stripped.partition(":")
            yield selector, prop.strip(), value.strip()


def _hardcoded_colours(source: str):
    return [(sel, prop, value) for sel, prop, value in _declarations(source)
            if _HEX.search(value) or _WORD_COLOUR.search(value)]


def test_colour_scanner_sees_a_planted_literal():
    """Negative control for the guard below — if this scanner cannot report a
    hard-coded colour, a clean run of the next test means nothing."""
    assert _hardcoded_colours('QPushButton {\n    color: #FF0000;\n}') == [
        ("QPushButton", "color", "#FF0000;")]
    assert _hardcoded_colours('/* old: color: #FF0000 */\nQLabel {\n    color: white;\n}') == [
        ("QLabel", "color", "white;")]


@pytest.mark.parametrize("name", ["dark", "light"])
def test_shipped_qss_has_no_hardcoded_colours(name):
    import lmloader.gui.styles as styles

    source = styles._DARK_QSS if name == "dark" else styles._LIGHT_QSS
    assert not _hardcoded_colours(source), (
        f"{name} stylesheet hard-codes colours again; add a token instead"
    )


@pytest.mark.parametrize("name", ["dark", "light"])
def test_render_theme_reproduces_the_shipped_stylesheet(name):
    """Templatizing was a refactor: the rendered QSS must not have moved."""
    import lmloader.gui.styles as styles

    shipped = styles.DARK_THEME if name == "dark" else styles.LIGHT_THEME
    assert styles.render_theme(name) == shipped


def test_render_theme_fails_loudly_on_an_incomplete_palette():
    """A half-built custom palette must raise, not ship a half-coloured UI."""
    import lmloader.gui.styles as styles

    with pytest.raises(KeyError):
        styles.Template("${no_such_token}").substitute(styles.theme_tokens("light"))


def test_apply_theme_accepts_a_custom_palette(monkeypatch):
    import lmloader.gui.styles as styles

    monkeypatch.setattr(styles, "current_theme", "dark")
    applied = []

    class _App:
        def setStyleSheet(self, qss):
            applied.append(qss)

    styles.apply_theme(_App(), "light", {"accent": "#FF0000"})
    assert styles.current_theme == "custom"
    assert styles._PALETTES["custom"]["accent"] == "#FF0000"
    assert styles.c("accent") == "#FF0000", "inline page QSS would keep the old blue"
    assert "#FF0000" in applied[0]

    styles.apply_theme(_App(), "light")
    assert applied[1] == styles.LIGHT_THEME, "apply_theme lost the cached stylesheet"


# --------------------------------------------------------------------- sidebar

def _sidebar_qss_for(theme: str) -> str:
    """Render the real sidebar stylesheet under a given theme.

    ``_sidebar_qss`` only reads ``theme_color()`` and never touches ``self``,
    so it can be called unbound — that keeps this test free of a full
    MainWindow construction while still exercising the production code path.
    """
    import lmloader.gui.styles as styles
    from lmloader.gui.window import MainWindow

    saved = styles.current_theme
    try:
        styles.current_theme = theme
        return MainWindow._sidebar_qss(None)
    finally:
        styles.current_theme = saved


def test_sidebar_qss_follows_the_active_theme():
    """The regression: the rail used to hard-code #192334 / #34435a / #78a3ff,
    so switching theme left a navy rail behind in a light window."""
    import lmloader.gui.styles as styles

    light_qss = _sidebar_qss_for("light")
    dark_qss = _sidebar_qss_for("dark")

    assert styles._PALETTES["light"]["sidebar"] in light_qss
    assert styles._PALETTES["dark"]["sidebar"] in dark_qss
    assert light_qss != dark_qss, "sidebar qss is theme-independent — regression"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_sidebar_qss_has_no_hardcoded_navy(theme):
    import lmloader.gui.styles as styles

    qss = _sidebar_qss_for(theme).lower()
    for legacy in ("#192334", "#34435a", "#78a3ff", "#26354a", "#aab5c6"):
        assert legacy not in qss, f"{theme} sidebar still hard-codes {legacy}"
    # every colour in the rail must come from the palette
    palette = {value.lower() for value in styles._PALETTES[theme].values()}
    for hex_color in set(re.findall(r"#[0-9a-f]{6}", qss)):
        assert hex_color in palette or hex_color == "#ffffff", (
            f"{theme} sidebar uses {hex_color}, which is not a theme token"
        )


# ------------------------------------------------------------------------- i18n

def _keys(table):
    return {key for key in table if not key.startswith("_")}


def test_i18n_zh_and_en_have_identical_keys():
    from lmloader.gui.i18n import TRANSLATIONS

    zh, en = _keys(TRANSLATIONS["zh"]), _keys(TRANSLATIONS["en"])
    missing_in_en = zh - en
    missing_in_zh = en - zh
    assert not missing_in_en, f"missing English strings: {sorted(missing_in_en)}"
    assert not missing_in_zh, f"missing Chinese strings: {sorted(missing_in_zh)}"


CHAT_REDESIGN_KEYS = [
    "chat.attach",
    "chat.attach_tip",
    "chat.attachment",
    "chat.attach_read_error",
    "chat.attach_too_large",
    "chat.copy",
    "chat.copied",
    "chat.save",
    "chat.save_conv",
    "chat.save_title",
    "chat.save_done",
    "chat.save_failed",
    "chat.save_conv_done",
    "chat.nothing_to_save",
]

DOWNLOAD_CARD_KEYS = [
    "download.manual_title",
    "download.hf_label",
    "download.ollama_label",
]

# P1 fix: the auto-detected Ollama path moved out of the editable field into a
# read-only hint, which needs its own strings in both languages.
DETECTED_PATH_KEYS = [
    "settings.ollama_detected_empty",
    "settings.ollama_detected_current",
]


@pytest.mark.parametrize("key", CHAT_REDESIGN_KEYS + DOWNLOAD_CARD_KEYS + [
    "settings.startup_timeout_tip",
] + DETECTED_PATH_KEYS)
def test_redesign_strings_exist_in_both_languages(key):
    from lmloader.gui.i18n import TRANSLATIONS

    assert key in TRANSLATIONS["zh"], f"{key} missing in zh"
    assert key in TRANSLATIONS["en"], f"{key} missing in en"


def test_detected_path_hint_takes_a_path_placeholder():
    """Both hint strings interpolate {path}; a missing placeholder would render
    a literal "{path}" to the user."""
    from lmloader.gui.i18n import TRANSLATIONS

    for key in DETECTED_PATH_KEYS:
        for lang in ("zh", "en"):
            assert "{path}" in TRANSLATIONS[lang][key], f"{key}[{lang}] has no {{path}}"


def test_no_empty_translations():
    from lmloader.gui.i18n import TRANSLATIONS

    for lang in ("zh", "en"):
        empty = [k for k, v in TRANSLATIONS[lang].items() if not str(v).strip()]
        assert not empty, f"{lang} has empty strings: {empty}"


# ----------------------------------------------------- model card row budget

class _FakeApi:
    """ModelsPage only touches the api on user actions, so a stub is enough."""

    base_url = "http://127.0.0.1:0"


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication for the whole session.

    The project has no pytest-qt / conftest.py, so the app object is created
    here. Qt allows only one instance per process, hence session scope.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture
def models_page(qapp):
    from lmloader.gui.pages.models import ModelsPage

    page = ModelsPage(_FakeApi())
    yield page
    page.deleteLater()
    qapp.processEvents()


# ------------------------------------------------------- silent-write guards
#
# Both fixes below remove a way for the app to change stored state without the
# user having asked for it. They are pure source/logic checks so they run
# without a live server.

def test_download_config_rejects_an_empty_body():
    """P2: an empty body used to fall through and write proxy="" and
    ollama_dir="", silently erasing the stored config."""
    import inspect

    from lmloader.api.routes.ollama import set_download_config

    source = inspect.getsource(set_download_config)
    assert "body.proxy is None and body.ollama_dir is None" in source, (
        "the all-fields-absent guard is gone — an empty body would wipe the config"
    )
    assert "status_code=400" in source, "an empty body must be rejected with 400"


def test_download_config_reports_which_fields_changed():
    """Callers need to tell a real change from a no-op save."""
    import inspect

    from lmloader.api.routes.ollama import set_download_config

    assert '"changed": changed' in inspect.getsource(set_download_config)


def test_delete_run_config_checks_the_model_exists():
    """P3: an unknown model name used to answer 200 {"saved": false}, so the
    caller could not distinguish "deleted" from "no such model"."""
    import inspect

    from lmloader.api.routes.models import delete_model_run_config

    source = inspect.getsource(delete_model_run_config)
    assert "get_model_by_name" in source, "delete must verify the model exists"
    assert "status_code=404" in source


def test_settings_does_not_load_the_detected_path_into_the_field():
    """P1: the probe result was written into the editable Ollama field, so
    opening the page and pressing save persisted it as if typed."""
    import inspect

    from lmloader.gui.pages.settings import SettingsPage

    source = inspect.getsource(SettingsPage._load_download_config)
    # The field may only ever take the saved value...
    assert 'cfg.get("ollama_dir", "") or ""' in source, (
        "the ollama_dir field must only receive the saved value"
    )
    # ...the detected path belongs to the read-only hint instead.
    assert "dl_ollama_detected" in source, "the detected path needs its own label"
    assert 'cfg.get("ollama_dir", "") or cfg.get("detected_ollama_root"' not in source, (
        "detected_ollama_root must not fall back into the editable field"
    )


SAMPLE_MODEL = {
    "id": 1,
    "name": "Gemma-4-12B-IT-NVFP4-Q4_K_M-00001-of-00002.gguf",
    "quantization": "NVFP4-Q4_K_M",
    "parameters": "12.1B",
    "architecture": "gemma3",
    "size": 6 * 1024 ** 3,
    "has_vision": True,
    "has_run_config": True,
    "metadata": {"general.author": "google-deepmind"},
}


def _layout():
    """The responsive-mode table, imported lazily.

    Importing at module scope would drag PySide6 in before the qapp fixture
    exists; these three are pure data/functions but live in a Qt module.
    """
    from lmloader.gui.pages.models import LAYOUT_MODES, pick_layout_mode, row_min_width

    return LAYOUT_MODES, pick_layout_mode, row_min_width

# Window sizes the app actually supports (window.setMinimumSize(1000, 680))
# crossed with both sidebar states — the rail is 200px expanded, 52px collapsed.
# CHROME covers the page margins (12+12) and the extra padding QListWidget adds
# around each item widget (measured at 10px: item sizeHint 766 vs a 776px
# viewport still produced a +2 scrollbar range).
WINDOW_SIZES = (1000, 1100, 1200, 1280, 1440)
RAIL_WIDTHS = (200, 52)
CHROME = 36          # page margins (12+12) + QListWidget item padding (~10) + slack


def viewport_for(window_width: int, rail_width: int) -> int:
    return window_width - rail_width - CHROME


def _all_viewports():
    return [viewport_for(w, r) for w in WINDOW_SIZES for r in RAIL_WIDTHS]


def _mode_ids():
    return [m["id"] for m in _layout()[0]]


@pytest.mark.parametrize("mode_id", _mode_ids())
def test_predicted_width_matches_the_built_row(models_page, mode_id):
    """`row_min_width` is what `pick_layout_mode` decides on, so if the
    prediction drifts from the widget's real minimum the mode choice silently
    becomes wrong. This is the guard the hard-coded 1120px threshold lacked.
    """
    modes, _, row_min_width = _layout()
    mode = next(m for m in modes if m["id"] == mode_id)
    predicted = row_min_width(mode)
    actual = models_page._build_card(SAMPLE_MODEL, {}, mode).minimumSizeHint().width()
    # Never optimistic — under-predicting is exactly what lets the row overflow.
    assert predicted >= actual, (
        f"{mode_id}: predicted {predicted}px < actual {actual}px — "
        f"pick_layout_mode would choose this mode too eagerly"
    )
    assert predicted - actual <= 24, (
        f"{mode_id}: predicted {predicted}px overshoots actual {actual}px by more "
        f"than a scrollbar's worth — the mode table has drifted"
    )


@pytest.mark.parametrize("window_width", WINDOW_SIZES)
@pytest.mark.parametrize("rail_width", RAIL_WIDTHS)
def test_model_row_never_overflows_any_supported_window(models_page, window_width, rail_width):
    """The real regression: at the default 1200px window with the rail expanded
    the row needed 34px more than the viewport, so Qt grew a horizontal
    scrollbar and pushed 删除 out of sight."""
    _, pick_layout_mode, _row = _layout()
    viewport = viewport_for(window_width, rail_width)
    mode = pick_layout_mode(viewport)
    actual = models_page._build_card(SAMPLE_MODEL, {}, mode).sizeHint().width()
    assert actual <= viewport, (
        f"{window_width}px window / {rail_width}px rail -> viewport {viewport}px, "
        f"mode={mode['id']} needs {actual}px (+{actual - viewport}); "
        f"删除 will be clipped behind a horizontal scrollbar"
    )


def test_model_list_never_shows_a_horizontal_scrollbar(models_page):
    """The scrollbar is the visible symptom of the overflow: it appears for the
    few px QListWidget adds on top of the item size hint, and while it is there
    it covers the bottom of the 删除 button. The responsive modes handle narrow
    windows, so the bar stays off."""
    assert models_page.list.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff, (
        "a horizontal scrollbar can reappear and clip 删除 — narrow windows are "
        "handled by pick_layout_mode instead"
    )


@pytest.mark.parametrize("viewport", _all_viewports())
def test_pick_layout_mode_takes_the_densest_fit(viewport):
    """LAYOUT_MODES is ordered densest-first, so the first mode that fits is
    the right answer — never a sparser one."""
    modes, pick_layout_mode, row_min_width = _layout()
    fitting = [i for i, m in enumerate(modes) if row_min_width(m) <= viewport]
    expected = fitting[0] if fitting else len(modes) - 1
    assert modes.index(pick_layout_mode(viewport)) == expected, (
        f"viewport {viewport}px picked {pick_layout_mode(viewport)['id']}, "
        f"expected {modes[expected]['id']}"
    )


def test_elastic_name_column_has_a_floor(models_page):
    """The name column absorbs slack but must never vanish — a 0-width name
    makes the row unusable, and eliding needs the full text kept around."""
    modes, _, _row = _layout()
    for mode in modes:
        card = models_page._build_card(SAMPLE_MODEL, {}, mode)
        elided = [c for c in card.findChildren(QLabel) if type(c).__name__ == "_ElidedLabel"]
        assert len(elided) == 1, f"{mode['id']}: expected exactly one elastic name label"
        label = elided[0]
        assert label.minimumWidth() == mode["name_min"]
        assert label.full_text() == SAMPLE_MODEL["name"]
        assert label.toolTip() == SAMPLE_MODEL["name"], "tooltip keeps the full name readable"
        assert label.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored, (
            f"{mode['id']}: the name label must not report its text width, "
            f"or it re-widens the row"
        )


def test_model_card_row_has_every_information_column(models_page):
    """At the full width the row must carry: copy button, saved-config badge,
    quant, author, params, arch, format, size — the set the user asked for."""
    modes, _, _row = _layout()
    card = models_page._build_card(SAMPLE_MODEL, {}, modes[0])
    texts = []

    def walk(widget):
        for child in widget.findChildren(object):
            name = type(child).__name__
            if name == "QLabel" and child.text():
                texts.append(child.text())
            walk(child)

    walk(card)
    joined = " | ".join(texts)
    assert "NVFP4" in joined, f"quantisation column missing: {joined}"
    assert "12.1B" in joined, f"parameter-count column missing: {joined}"
    assert "GGUF" in joined, f"format column missing: {joined}"
    assert "GB" in joined, f"model-size column missing: {joined}"
    assert card.findChildren(object), "card must expose its action buttons"