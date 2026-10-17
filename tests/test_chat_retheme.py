"""ChatPage inline styles must follow the palette, not the theme it was built in.

``apply_theme()`` only repaints the application-wide stylesheet. This page
additionally styles ~30 widgets by calling ``c()`` inside per-widget QSS, and
those strings are frozen when the widget is constructed. Without a retheme hook
an injected (image-derived) palette would leave the chat page half in the old
colours — which is exactly the bug this suite pins down:

* every widget registered through ``_bind_style`` repaints on ``retheme()``;
* callables are re-evaluated, so a style can depend on state, not only palette;
* message bubbles and conversation rows are rebuilt, and keep their state;
* any *new* direct ``setStyleSheet`` call is refused unless retheme accounts
  for it (``test_unaccounted_inline_style_is_reported`` is the guard).
"""

import ast
import inspect
import re
from pathlib import Path

import pytest
from PySide6.QtWidgets import QWidget

import lmloader.gui.pages.chat as chat_mod
from lmloader.gui import styles
from lmloader.gui.pages.chat import ChatPage, ConversationRow
from lmloader.infra.chat_db import ChatDatabase

CHAT_SOURCE = Path(inspect.getfile(chat_mod)).read_text(encoding="utf-8")

# Keep the harness pieces from the system-prompt suite rather than duplicating
# them: same fake API, same stubbed worker/thread, same throwaway config.yaml.
from tests.test_chat_system_prompt import _FakeApi, _RecordingWorker, _StubThread


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ui_config(tmp_path, monkeypatch):
    import lmloader.gui.ui_config as ui_config_mod

    cfg = tmp_path / "config.yaml"
    cfg.write_text("ui: {}\n", encoding="utf-8")
    monkeypatch.setattr(ui_config_mod, "config_path", lambda: cfg)
    return cfg


@pytest.fixture
def restore_theme(qapp):
    """apply_theme() mutates module globals shared with the rest of the suite."""
    before = styles.current_theme
    yield
    styles.apply_theme(qapp, before)


@pytest.fixture
def page(qapp, tmp_path, ui_config, monkeypatch, restore_theme):
    # A tmp database: ChatPage.init kicks off chat_db.init(), and that must
    # never open the developer's real lmloader.db.
    store = ChatDatabase()
    store.db_path = tmp_path / "retheme.db"
    monkeypatch.setattr(chat_mod, "chat_db", store)
    monkeypatch.setattr(chat_mod, "ChatStreamWorker", _RecordingWorker)
    monkeypatch.setattr(chat_mod, "QThread", _StubThread)
    _RecordingWorker.captured = None

    p = ChatPage(_FakeApi())
    yield p
    p.deleteLater()


# ------------------------------------------------------------------- palettes

def _custom_palette(base: str = "light") -> dict:
    """Replace every token with a value guaranteed to be foreign to `base`.

    Sequential #0A00xx values can't collide with a real palette (no shipped
    token has green and blue both zero), so "the old colour is gone" is a
    meaningful assertion rather than an accident of the replacement.
    """
    tokens = styles.theme_tokens(base)
    overrides = {name: f"#0A{i:04X}" for i, name in enumerate(sorted(tokens))}
    assert not set(overrides.values()) & set(tokens.values())
    return overrides


def _old_values(base: str = "light") -> set[str]:
    return {v for v in styles.theme_tokens(base).values() if v != "transparent"}


def _all_styles(root: QWidget) -> list[str]:
    out = [root.styleSheet()]
    out.extend(child.styleSheet() for child in root.findChildren(QWidget))
    return [s for s in out if s]


def _bodies_html(page) -> str:
    """The rich-text of every message body — colours live here too."""
    from PySide6.QtWidgets import QLabel

    return "\n".join(
        w.text() for row in page._message_rows for w in row.findChildren(QLabel)
        if w.objectName() == "messageBody"
    )


# ------------------------------------------------------------------ bound set

def test_page_registers_every_inline_style(page):
    """A silently-shrinking binding list would make the other tests vacuous."""
    assert len(page._style_bindings) >= 25, [id(w) for w, _ in page._style_bindings]
    widgets = {id(w) for w, _ in page._style_bindings}
    assert len(widgets) == len(page._style_bindings), "a widget is bound twice"


def test_retheme_repaints_every_bound_widget(page, qapp):
    styles.apply_theme(qapp, "light")
    page._new_conversation()
    before = [(w, w.styleSheet()) for w, _ in page._style_bindings]
    assert all(sheet for _, sheet in before), "a bound widget was never styled"

    overrides = _custom_palette()
    styles.apply_theme(qapp, "light", overrides)
    page.retheme()

    for widget, old in before:
        assert widget.styleSheet() != old, "bound widget kept the old palette"
    painted = "\n".join(widget.styleSheet() for widget, _ in page._style_bindings)
    for value in _old_values():
        assert value not in painted, f"{value} survived the theme switch"
    # And the new palette really arrived.
    assert overrides["accent"] in painted


def test_callable_bindings_are_re_evaluated(page, qapp):
    """Callables must be invoked at retheme time, not captured at bind time."""
    from PySide6.QtWidgets import QLabel

    styles.apply_theme(qapp, "light")
    probe = QLabel("probe")
    page._bind_style(probe, lambda: f"color: {styles.c('accent')};")
    assert probe.styleSheet() == f"color: {styles.theme_tokens('light')['accent']};"

    styles.apply_theme(qapp, "light", {"accent": "#FF1122"})
    page.retheme()
    assert probe.styleSheet() == "color: #FF1122;"


def test_retheme_is_idempotent(page, qapp):
    styles.apply_theme(qapp, "light", _custom_palette())
    page.retheme()
    once = [(w, w.styleSheet()) for w, _ in page._style_bindings]
    page.retheme()
    assert [(w, w.styleSheet()) for w, _ in once] == once


# ------------------------------------------------------------------- rebuilds

def _fill_transcript(page) -> None:
    page._new_conversation()
    conv = page._current_conv()
    conv["system_prompt"] = "只回答结论。"
    conv["messages"] = [
        {"role": "user", "content": "为什么 OOM"},
        {"role": "assistant", "content": "## 结论\n\n`KV cache` 太大",
         "_reasoning": "先看显存占用", "_stats": "12 tok/s"},
    ]
    page._apply_prompt_to_ui()
    page._render_messages()


def test_message_bubbles_follow_the_palette(page, qapp):
    styles.apply_theme(qapp, "light")
    _fill_transcript(page)
    before = "\n".join(s for row in page._message_rows for s in _all_styles(row))
    assert before, "no message widgets were built"
    old_accent = styles.theme_tokens("light")["accent"]
    assert old_accent in before

    overrides = _custom_palette()
    styles.apply_theme(qapp, "light", overrides)
    page.retheme()

    after = "\n".join(s for row in page._message_rows for s in _all_styles(row))
    assert after != before
    assert old_accent not in after
    assert overrides["accent"] in after


def test_markdown_body_colours_follow_the_palette(page, qapp):
    """Headings/code spans embed c() into the rich-text HTML, not only QSS."""
    styles.apply_theme(qapp, "light")
    _fill_transcript(page)
    old_accent = styles.theme_tokens("light")["accent"]
    assert old_accent in _bodies_html(page)

    overrides = _custom_palette()
    styles.apply_theme(qapp, "light", overrides)
    page.retheme()

    body_html = _bodies_html(page)
    assert old_accent not in body_html
    assert overrides["accent"] in body_html


def test_conversation_rows_follow_the_palette(page, qapp):
    styles.apply_theme(qapp, "light")
    page._new_conversation()
    page._new_conversation()
    rows = [
        page.conv_list.itemWidget(page.conv_list.item(i))
        for i in range(page.conv_list.count())
    ]
    assert all(isinstance(r, ConversationRow) for r in rows)
    before = [r.styleSheet() for r in rows]
    old = styles.theme_tokens("light")

    overrides = _custom_palette()
    styles.apply_theme(qapp, "light", overrides)
    page.retheme()

    after = [r.styleSheet() for r in rows]
    assert after != before
    for sheet in after:
        for value in _old_values():
            assert value not in sheet
    # The selection highlight survives the rebuild, in the new colours.
    assert page.current_conv_id == 1
    assert overrides["accent_surface"] in after[1]
    assert overrides["accent_surface"] not in after[0]


def test_thinking_button_follows_the_palette(page, qapp):
    styles.apply_theme(qapp, "light")
    before = (page.thinking_btn.styleSheet(), page.thinking_btn.text())
    overrides = _custom_palette()
    styles.apply_theme(qapp, "light", overrides)
    page.retheme()
    assert page.thinking_btn.styleSheet() != before[0]
    # and it still reads its own toggle state
    page._thinking_on = not page._thinking_on
    page.retheme()
    assert page.thinking_btn.text() != before[1]


# ---------------------------------------------------------------- the guard

REBUILT_BY_RETHEME = {
    "ConversationRow._apply_style": "row.retheme() re-runs it",
    "ChatPage._update_thinking_btn": "called by retheme()",
    "ChatPage._refresh_prompt_ui": "called by retheme()",
    "ChatPage._prompt_state_error": "transient state, restyled by _refresh_prompt_ui",
    "ChatPage._render_attachments": "called by retheme()",
    "ChatPage._build_prompt_panel": "styles prompt_state only, then _refresh_prompt_ui runs",
    "ChatPage._icon_action": "inside rebuilt message rows",
    "ChatPage._create_message_row": "inside rebuilt message rows",
    "ChatPage._create_message_bubble": "inside rebuilt message rows",
    "ChatPage._append_message_bubble": "inside rebuilt message rows",
    "ChatPage._append_assistant_message": "inside rebuilt message rows",
}
MACHINERY = {"ChatPage._bind_style", "ChatPage.retheme"}
COLOURLESS = {"ChatPage._build_sidebar": "conv_list QSS carries no palette colour"}


def _unaccounted(source: str) -> list[str]:
    """Direct setStyleSheet calls that retheme() cannot reach."""
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        for fn in node.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            who = f"{node.name}.{fn.name}"
            styled = any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "setStyleSheet"
                for call in ast.walk(fn)
            )
            if not styled or who in MACHINERY or who in {**REBUILT_BY_RETHEME, **COLOURLESS}:
                continue
            offenders.append(who)
    return sorted(offenders)


def test_unaccounted_inline_style_is_reported():
    """Negative control: plant a stale style in a method retheme() never runs."""
    planted = CHAT_SOURCE.replace(
        "    def _show_conv_menu(self, pos):",
        "    def _planted_probe(self):\n"
        '        self.chat_title.setStyleSheet(f"color: {c(\'accent\')};")\n\n'
        "    def _show_conv_menu(self, pos):",
        1,
    )
    assert planted != CHAT_SOURCE, "anchor drifted"
    assert _unaccounted(planted) == ["ChatPage._planted_probe"]
    assert _unaccounted(CHAT_SOURCE) == []


def test_retheme_allowlist_is_not_stale():
    """A renamed or deleted method must not linger in the allowlist."""
    for who in list(REBUILT_BY_RETHEME) + list(COLOURLESS):
        cls, _, name = who.partition(".")
        target = getattr(chat_mod, cls)
        assert hasattr(target, name), f"{who} no longer exists"


def _declared_colourless_style(source: str) -> str:
    """The QSS argument of the single unbound call site in _build_sidebar."""
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.ClassDef) and node.name == "ChatPage"):
            continue
        for fn in node.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "_build_sidebar"):
                continue
            calls = [
                call for call in ast.walk(fn)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "setStyleSheet"
            ]
            assert len(calls) == 1, f"expected one unbound style, got {len(calls)}"
            return ast.get_source_segment(source, calls[0].args[0])
    raise AssertionError("_build_sidebar not found")


def test_declared_colourless_style_has_no_colour():
    """The one unbound call site is only excused because it paints nothing."""
    qss = _declared_colourless_style(CHAT_SOURCE)
    assert not re.search(r"\bc\(|#[0-9A-Fa-f]{3,8}", qss), qss


def test_window_rethemes_chat_page_on_theme_change():
    """The page is useless if the window never calls it."""
    from lmloader.gui import window as window_mod

    src = inspect.getsource(window_mod.MainWindow._on_theme_changed)
    assert "self.chat_page.retheme()" in src
