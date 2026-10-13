"""Tests for the chat bubble actions and the attachment pipeline.

Three user-reported problems are pinned here:
  * copy/save buttons showed text labels on every bubble — wanted icons
  * the user's own message carried copy/save — not wanted
  * attachments were capped at 200 KB, which blocked every image
"""

import base64
import inspect
import struct
import zlib
from pathlib import Path

import pytest

from lmloader.gui.i18n import tr
from lmloader.gui.pages.chat import ChatPage, IMAGE_EXTS, TEXT_EXTS


# --- a real 2x2 PNG, built without external deps ---------------------------

def _png_bytes() -> bytes:
    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 2 for _ in range(2))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class _FakeApi:
    base_url = "http://127.0.0.1:1"
    token = ""

    def _headers(self):
        return {}

    def get_loaded_models(self):
        return []

    def list_conversations(self):
        return []


@pytest.fixture
def page(qapp, tmp_path):
    p = ChatPage(_FakeApi())
    p._tmp = tmp_path
    yield p
    p.deleteLater()


# --- icon-only actions -----------------------------------------------------

def test_user_bubble_has_no_action_row():
    """Regression: the user's own message used to carry copy/save too."""
    src = inspect.getsource(ChatPage._append_message_bubble)
    assert "_message_action_row" not in src, (
        "user bubble must not render a copy/save row"
    )


def test_assistant_bubble_still_has_the_action_row():
    src = inspect.getsource(ChatPage._append_assistant_message)
    assert "_message_action_row" in src


def test_action_buttons_are_icon_only(qapp):
    row = ChatPage._message_action_row(ChatPage.__new__(ChatPage), "文本", "a.md")
    from PySide6.QtWidgets import QPushButton
    btns = row.findChildren(QPushButton)
    assert len(btns) == 2
    for b in btns:
        assert b.text() == "", "buttons must show an icon, not a text label"
        assert not b.icon().isNull()
        assert b.toolTip(), "the tooltip must carry the label"
        assert b.width() <= 32 and b.height() <= 28


def test_icons_module_exposes_the_needed_glyphs():
    from lmloader.gui import icons
    for fn in ("copy_icon", "download_icon"):
        assert hasattr(icons, fn), f"icons.{fn} missing"


def test_download_icon_renders_a_non_empty_pixmap(qapp):
    from lmloader.gui.icons import download_icon
    icon = download_icon("#888888", 15)
    assert not icon.isNull()
    assert not icon.pixmap(15, 15).isNull()


# --- attachment size limits -----------------------------------------------

def test_the_200kb_ceiling_is_gone():
    src = inspect.getsource(ChatPage._add_attachment)
    assert "200 * 1024" not in src
    assert "attach_too_large" not in src


def test_text_cap_is_measured_in_characters_not_bytes():
    """UTF-8 CJK costs 3 bytes/char; a byte cap would cut Chinese to 1/3."""
    src = inspect.getsource(ChatPage)
    assert "TEXT_PREVIEW_LIMIT = 400_000" in src


def test_a_file_over_the_old_200kb_limit_is_accepted(page):
    big = page._tmp / "big.txt"
    big.write_text("line\n" * 100_000, encoding="utf-8")   # ~500 KB
    assert big.stat().st_size > 200 * 1024
    page._add_attachment(str(big))
    assert len(page._attachments) == 1
    assert page._attachments[0]["kind"] == "text"
    assert page._attachments[0]["text"]


def test_chinese_file_is_not_truncated_by_a_byte_cap(page):
    cn = page._tmp / "cn.txt"
    # 7 chars per line x 40_000 lines ~= 280k characters (1.2 MB on disk).
    # Under the 400k CHARACTER cap, well over the old 200 KB BYTE cap —
    # exactly the case a byte-based limit would wrongly truncate.
    cn.write_text("中文字符测试\n" * 40_000, encoding="utf-8")
    page._add_attachment(str(cn))
    entry = page._attachments[0]
    assert cn.stat().st_size > 200 * 1024, "must exceed the old byte ceiling"
    assert not entry.get("note"), (
        f"CJK text must not be wrongly truncated (got note={entry.get('note')!r})"
    )
    assert entry["text"], "content must be intact"


def test_an_oversized_file_is_truncated_with_a_visible_note(page):
    big = page._tmp / "huge.txt"
    big.write_text("x" * 500_000, encoding="utf-8")
    page._add_attachment(str(big))
    entry = page._attachments[0]
    assert entry.get("note"), "a truncated file must say so"
    assert len(entry["text"]) == ChatPage.TEXT_PREVIEW_LIMIT


# --- image attachments -----------------------------------------------------

def test_image_extensions_cover_the_common_formats():
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
        assert ext in IMAGE_EXTS


def test_image_becomes_a_data_uri(page):
    img = page._tmp / "x.png"
    png = _png_bytes()
    img.write_bytes(png)
    page._add_attachment(str(img))
    entry = page._attachments[0]
    assert entry["kind"] == "image"
    assert entry["data_uri"].startswith("data:image/png;base64,")
    assert base64.b64decode(entry["data_uri"].split(",", 1)[1]) == png


def test_image_jpeg_mime_is_correct(page):
    img = page._tmp / "x.jpg"
    img.write_bytes(_png_bytes())        # content is really PNG, name says jpg
    page._add_attachment(str(img))
    assert page._attachments[0]["data_uri"].startswith("data:image/jpeg;base64,")


def test_image_never_goes_into_the_text_prompt(page):
    """An image pasted as text is useless and would blow the context."""
    img = page._tmp / "y.png"
    img.write_bytes(_png_bytes())
    page._add_attachment(str(img))
    assert page._attachment_payload() == ""


def test_image_content_uses_openai_multimodal_shape(page):
    img = page._tmp / "z.png"
    img.write_bytes(_png_bytes())
    page._add_attachment(str(img))
    content = page._build_user_content("这是什么颜色")
    assert isinstance(content, list)
    assert [p["type"] for p in content] == ["text", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/")


def test_image_only_message_has_no_empty_text_part(page):
    img = page._tmp / "w.png"
    img.write_bytes(_png_bytes())
    page._add_attachment(str(img))
    content = page._build_user_content("")
    assert [p["type"] for p in content] == ["image_url"]


def test_text_only_content_stays_a_plain_string(page):
    """Regression guard: the engine and the DB both expect a string here."""
    txt = page._tmp / "a.txt"
    txt.write_text("hello", encoding="utf-8")
    page._add_attachment(str(txt))
    assert isinstance(page._build_user_content("问题"), str)


def test_text_and_image_together(page):
    txt = page._tmp / "b.md"
    txt.write_text("# doc", encoding="utf-8")
    img = page._tmp / "c.png"
    img.write_bytes(_png_bytes())
    page._add_attachment(str(txt))
    page._add_attachment(str(img))
    content = page._build_user_content("总结")
    assert isinstance(content, list)
    assert [p["type"] for p in content] == ["text", "image_url"]
    assert "```" in page._attachment_payload(), "text still rides in the prompt"


# --- rendering list content -----------------------------------------------

def test_content_to_text_handles_plain_string():
    assert ChatPage._content_to_text("hello") == "hello"


def test_content_to_text_flattens_multimodal_parts():
    """Rendering must not assume str: an image turn is stored as a list."""
    content = [
        {"type": "text", "text": "问题"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]
    out = ChatPage._content_to_text(content)
    assert "问题" in out
    assert "1" in out


def test_content_to_text_handles_none_and_empty_list():
    assert ChatPage._content_to_text(None) == ""
    assert ChatPage._content_to_text([]) == ""


def test_content_to_text_ignores_unknown_part_types():
    content = [{"type": "text", "text": "hi"}, {"type": "weird", "x": 1}]
    assert "hi" in ChatPage._content_to_text(content)


# --- i18n ------------------------------------------------------------------

def test_new_attachment_keys_exist_in_both_languages():
    from lmloader.gui.i18n import TRANSLATIONS
    for key in ("chat.attach_truncated", "chat.attach_binary",
                "chat.attach_image", "chat.attached_images"):
        assert key in TRANSLATIONS["zh"], f"zh missing {key}"
        assert key in TRANSLATIONS["en"], f"en missing {key}"


def test_text_extension_table_is_not_empty():
    assert ".md" in TEXT_EXTS
    assert ".json" in TEXT_EXTS


def test_translations_stay_aligned():
    from lmloader.gui.i18n import TRANSLATIONS
    zh, en = TRANSLATIONS["zh"], TRANSLATIONS["en"]
    assert set(zh) == set(en), "zh/en key sets diverged"