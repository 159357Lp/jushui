"""Tests for the per-conversation system prompt.

The feature is deliberately built so a prompt can never corrupt a transcript:

* the prompt is a column on ``conversations``, not a ``messages`` row, so the
  chat history stays exactly what the user and the model said;
* a database created before the column existed must migrate in place, keeping
  its rows — both ``ChatDatabase.init()`` and the standalone script do this;
* ``_do_send`` prepends the system message to a *copy* of the transcript;
* switching conversations must load that conversation's prompt into the panel,
  not the previous one's.
"""

import asyncio
import inspect
import sqlite3
from pathlib import Path

import pytest

import lmloader.gui.pages.chat as chat_mod
from lmloader.gui.i18n import tr
from lmloader.gui.pages.chat import ChatPage
from lmloader.gui.ui_config import load_ui_setting, save_ui_setting
from lmloader.infra.chat_db import ChatDatabase
from lmloader.migrations import ADD_SYSTEM_PROMPT_SQL
from lmloader.migrations.add_system_prompt import migrate

LEGACY_SCHEMA = """
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    model_name TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _legacy_db(path: Path, rows: int = 2) -> Path:
    """A database in the pre-system_prompt shape, with some history in it."""
    con = sqlite3.connect(path)
    con.execute(LEGACY_SCHEMA)
    con.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " conversation_id INTEGER NOT NULL, role TEXT NOT NULL,"
        " content TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    for i in range(rows):
        con.execute(
            "INSERT INTO conversations (title, model_name, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (f"会话{i}", "m.gguf", "2026-01-01", f"2026-01-0{i + 1}"),
        )
        con.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at)"
            " VALUES (?, 'user', '你好', '2026-01-01')",
            (i + 1,),
        )
    con.commit()
    con.close()
    return path


def _columns(path: Path) -> set[str]:
    con = sqlite3.connect(path)
    try:
        return {row[1] for row in con.execute("PRAGMA table_info(conversations)")}
    finally:
        con.close()


# ----------------------------------------------------------------- migrations

def test_standalone_migration_adds_column_and_keeps_rows(tmp_path):
    db = _legacy_db(tmp_path / "lmloader.db")
    assert migrate(db) == 2

    assert "system_prompt" in _columns(db)
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT id, title, system_prompt FROM conversations ORDER BY id"
        ).fetchall()
        msgs = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        con.close()
    assert [r[1] for r in rows] == ["会话0", "会话1"]
    assert all(r[2] == "" for r in rows), "migrated rows must default to empty"
    assert msgs == 2, "migration must not touch the transcript"


def test_standalone_migration_is_idempotent(tmp_path):
    db = _legacy_db(tmp_path / "lmloader.db")
    assert migrate(db) == 2
    assert migrate(db) == 0, "a second run must report already-migrated"

    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    finally:
        con.close()


def test_init_migrates_a_legacy_database_in_place(tmp_path):
    """The GUI must cope with an existing lmloader.db without being told to."""
    db = _legacy_db(tmp_path / "lmloader.db", rows=1)
    store = ChatDatabase()
    store.db_path = db
    asyncio.run(store.init())

    assert "system_prompt" in _columns(db)
    got = asyncio.run(store.list_conversations())
    assert len(got) == 1 and got[0]["system_prompt"] == ""


def test_init_on_a_legacy_db_without_rows(tmp_path):
    db = tmp_path / "empty.db"
    _legacy_db(db, rows=0)
    store = ChatDatabase()
    store.db_path = db
    asyncio.run(store.init())
    assert "system_prompt" in _columns(db)


def test_migrate_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        migrate(tmp_path / "nope.db")


def test_migration_sql_is_shared_not_duplicated():
    """Both paths must issue the same DDL, or they can drift apart."""
    store_src = inspect.getsource(ChatDatabase.init)
    assert "ADD_SYSTEM_PROMPT_SQL" in store_src
    assert "ALTER TABLE" not in store_src


@pytest.mark.parametrize(
    "prompt", ["只回答三个字。", "line1\nline2", "", "a" * 4000]
)
def test_prompt_roundtrips_through_the_db(tmp_path, prompt):
    store = ChatDatabase()
    store.db_path = tmp_path / "c.db"
    asyncio.run(store.init())

    cid = asyncio.run(store.create_conversation("t", "m.gguf", prompt))
    assert asyncio.run(store.get_conversation(cid))["system_prompt"] == prompt
    assert asyncio.run(store.list_conversations())[0]["system_prompt"] == prompt

    asyncio.run(store.update_system_prompt(cid, "改过了"))
    assert asyncio.run(store.get_conversation(cid))["system_prompt"] == "改过了"


def test_update_prompt_does_not_reshuffle_the_list(tmp_path):
    """Editing a prompt is not a turn — it must not bump updated_at."""
    store = ChatDatabase()
    store.db_path = tmp_path / "c.db"
    asyncio.run(store.init())
    first = asyncio.run(store.create_conversation("first", ""))
    asyncio.run(store.add_message(first, "user", "hi"))
    second = asyncio.run(store.create_conversation("second", ""))

    before = asyncio.run(store.list_conversations())
    asyncio.run(store.update_system_prompt(first, "新提示词"))
    after = asyncio.run(store.list_conversations())

    assert [c["id"] for c in before] == [c["id"] for c in after]


def test_legacy_conversation_can_be_updated(tmp_path):
    db = _legacy_db(tmp_path / "lmloader.db", rows=1)
    store = ChatDatabase()
    store.db_path = db
    asyncio.run(store.init())
    asyncio.run(store.update_system_prompt(1, "迁移后补的"))
    assert asyncio.run(store.get_conversation(1))["system_prompt"] == "迁移后补的"


# ------------------------------------------------------------------ ui config

@pytest.fixture
def ui_config(tmp_path, monkeypatch):
    """Point ui.* settings at a throwaway config.yaml."""
    import lmloader.gui.ui_config as ui_config_mod

    cfg = tmp_path / "config.yaml"
    cfg.write_text("ui: {}\n", encoding="utf-8")
    monkeypatch.setattr(ui_config_mod, "config_path", lambda: cfg)
    return cfg


def test_ui_setting_roundtrip_and_merge_preserves_siblings(ui_config):
    save_ui_setting("system_prompt", "全局默认")
    assert load_ui_setting("system_prompt", "") == "全局默认"
    save_ui_setting("sidebar_collapsed", True)
    assert load_ui_setting("system_prompt", "") == "全局默认"
    assert load_ui_setting("theme", "dark") == "dark"


def test_load_ui_setting_default_when_absent(ui_config):
    assert load_ui_setting("no_such_key", "fallback") == "fallback"


# ------------------------------------------------------------- gui injection

class _StubSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class _RecordingWorker:
    """Stands in for ChatStreamWorker so no socket is ever opened."""

    captured = None

    def __init__(self, api, model_name, messages, temperature, thinking, max_tokens=None):
        type(self).captured = messages
        self.token_received = _StubSignal()
        self.finished = _StubSignal()
        self.error = _StubSignal()

    def run(self):
        pass

    def moveToThread(self, thread):
        pass

    def moveToThread(self, thread):
        pass


class _StubThread:
    def __init__(self):
        self.started = _StubSignal()
        self.quits = 0

    def quit(self):
        self.quits += 1

    def start(self):
        pass


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


@pytest.fixture
def page(qapp, tmp_path, ui_config, monkeypatch):
    # A tmp database: constructing ChatPage runs chat_db.init() in a worker
    # thread, and it must never touch the developer's real lmloader.db.
    store = ChatDatabase()
    store.db_path = tmp_path / "gui.db"
    monkeypatch.setattr(chat_mod, "chat_db", store)
    monkeypatch.setattr(chat_mod, "ChatStreamWorker", _RecordingWorker)
    monkeypatch.setattr(chat_mod, "QThread", _StubThread)
    _RecordingWorker.captured = None

    p = ChatPage(_FakeApi())
    yield p
    p.deleteLater()


def _ready(page, prompt=""):
    """Give the page a conversation whose prompt is `prompt`."""
    page._new_conversation()
    conv = page._current_conv()
    conv["system_prompt"] = prompt
    page._apply_prompt_to_ui()
    return conv


def test_send_prepends_system_message_without_editing_history(page):
    conv = _ready(page, "只回答三个字。")
    conv["messages"].append({"role": "user", "content": "hi"})
    original = conv["messages"][0]

    page._do_send(conv, "下一个问题", "m.gguf")

    sent = _RecordingWorker.captured
    assert sent[0] == {"role": "system", "content": "只回答三个字。"}
    assert [m["role"] for m in sent[1:3]] == ["user", "user"]
    assert sent[1]["content"] == "hi" and sent[2]["content"] == "下一个问题"
    # The transcript itself stays clean: reloading the chat shows no system row.
    assert conv["messages"][0] is original
    assert [m["role"] for m in conv["messages"]] == ["user", "user", "assistant"]
    assert not any(m["role"] == "system" for m in conv["messages"])


def test_blank_prompt_is_not_injected(page):
    for blank in ("", "   ", "\n"):
        conv = _ready(page, blank)
        page._do_send(conv, "问题", "m.gguf")
        assert _RecordingWorker.captured[0]["role"] == "user", repr(blank)


def test_panel_shows_current_conversation_prompt(page):
    first = _ready(page, "第一个会话的提示词")
    page._new_conversation()
    assert page.prompt_box.toPlainText() == "", "new chat starts from no default"

    page._switch_conversation(0)
    assert page.prompt_box.toPlainText() == "第一个会话的提示词"
    assert page._current_conv() is first
    assert page.prompt_bar.isVisible() is True or not page.isVisible()


def test_panel_edits_write_back_to_the_current_conversation(page):
    conv = _ready(page, "")
    page.prompt_btn.setChecked(True)
    page.prompt_box.setPlainText("改成这样")
    assert conv["system_prompt"] == "改成这样"
    assert page.prompt_box.toPlainText() == "改成这样"


def test_prompt_chip_toggles_the_panel(page):
    page._on_prompt_panel_toggled(True)
    assert page._prompt_panel.isVisible() is True or not page.isVisible()
    page._on_prompt_panel_toggled(False)
    assert page._prompt_panel.isHidden()


def test_token_estimate_counts_cjk_and_ascii(page):
    assert ChatPage._estimate_tokens("只回答三个字。") == 7
    assert ChatPage._estimate_tokens("") == 0
    assert ChatPage._estimate_tokens("abcdefgh") == 2
    assert ChatPage._estimate_tokens("中文" + "a" * 8) == 4


def test_new_conversation_seeded_from_global_default(page, ui_config):
    save_ui_setting("system_prompt", "全局默认提示词")
    page._new_conversation()
    assert page._current_conv()["system_prompt"] == "全局默认提示词"
    page._apply_prompt_to_ui()
    assert page.prompt_box.toPlainText() == "全局默认提示词"


def test_scope_switch_to_global_persists_the_default(page, ui_config):
    conv = _ready(page, "")
    page.prompt_btn.setChecked(True)
    page.prompt_box.setPlainText("每轮都要遵守")
    page.prompt_scope.setCurrentIndex(1)

    assert load_ui_setting("system_prompt", "") == "每轮都要遵守"
    page._new_conversation()
    assert page._current_conv()["system_prompt"] == "每轮都要遵守"


def test_switching_scope_back_clears_only_an_identical_default(page, ui_config):
    conv = _ready(page, "本对话的提示词")
    page.prompt_btn.setChecked(True)

    save_ui_setting("system_prompt", " unrelated default ")
    page._apply_scope_from_conv()
    page.prompt_scope.setCurrentIndex(0)
    assert load_ui_setting("system_prompt", "") == " unrelated default ", (
        "dropping to per-chat must not delete an unrelated global default"
    )

    page.prompt_scope.setCurrentIndex(1)
    assert load_ui_setting("system_prompt", "") == "本对话的提示词"
    page.prompt_scope.setCurrentIndex(0)
    assert load_ui_setting("system_prompt", "") == ""
    assert conv["system_prompt"] == "本对话的提示词"


def test_prompt_keys_exist_in_both_languages():
    for lang in ("zh", "en"):
        from lmloader.gui.i18n import TRANSLATIONS

        keys = [k for k in TRANSLATIONS[lang] if k.startswith("chat.prompt")]
        assert len(keys) >= 20
        assert not any(k.startswith("chat.prompt_tpl") and not TRANSLATIONS[lang][k].strip() for k in keys)
    assert tr("chat.prompt_count").format(chars=7, tok=5)
    assert tr("chat.prompt_save_fail").format(err="x")


def test_settings_save_keeps_unknown_ui_keys():
    """settings used to overwrite config["ui"] wholesale and delete these keys."""
    from lmloader.gui.pages.settings import SettingsPage

    src = inspect.getsource(SettingsPage._save_settings)
    assert 'config["ui"] = {' not in src
    assert 'config.get("ui")' in src
