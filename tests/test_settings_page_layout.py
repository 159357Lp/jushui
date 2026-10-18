"""设置页的结构判据：三张引擎卡 + 一张版本表，自定义引擎和内置引擎同表同权。

重排前更新/回退长在两张卡片里、可插拔引擎单独一大块还没有回退；这三条一旦回退
成旧布局，测试就该报红。行为（提交给后端什么参数）在 _pydiag/shot_settings_page.py
里连着截图一起判。
"""

import pytest
from PySide6.QtWidgets import QApplication, QGroupBox, QHBoxLayout, QLabel, QPushButton


class _FakeApi:
    base_url = "http://127.0.0.1:1"
    token = ""

    def _headers(self):
        return {}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qapp):
    from lmloader.gui.pages.settings import SettingsPage
    return SettingsPage(_FakeApi())


DISCOVERED = [
    {"name": "primary", "kind": "llama", "binary_path": "D:/e/official/llama-server.exe",
     "formats": [".gguf"], "source": "config"},
    {"name": "ternary", "kind": "llama", "binary_path": "D:/e/ternary/llama-server.exe",
     "formats": [".gguf", ".gggf"], "source": "config"},
    {"name": "vllm@vllm-server", "kind": "vllm", "binary_path": "D:/vllm/vllm-server.exe",
     "formats": [".safetensors"], "source": "scan"},
    {"name": "rwkv-runner", "kind": "rwkv", "binary_path": "D:/e/rwkv/runner.exe",
     "formats": [".bin"], "source": "custom"},
]
CUSTOMS = [
    {"name": "rwkv-runner", "kind": "rwkv", "binary_path": "D:/e/rwkv/runner.exe",
     "history": [{"binary_path": "D:/e/rwkv/old.exe", "version": "b1719"}]},
    {"name": "onnx-clip", "kind": "onnx", "binary_path": "D:/e/onnx/clip.exe"},
]


def _titles(page):
    return [g.title() for g in page.findChildren(QGroupBox)]


def _buttons_in_row(page, row: int) -> list[QPushButton]:
    widget = page.version_table.cellWidget(row, 5)
    return [] if widget is None else widget.findChildren(QPushButton)


def test_three_engine_cards_share_one_row(page):
    from lmloader.gui.i18n import tr

    titles = _titles(page)
    for key in ("settings.primary_engine", "settings.ternary_engine",
                "settings.pluggable_engine"):
        assert tr(key) in titles, f"缺卡片：{key}"

    rows = [l for l in page.findChildren(QHBoxLayout)
            if l.count() == 3 and all(l.itemAt(i).widget() is not None for i in range(3))]
    cards = {tr("settings.primary_engine"), tr("settings.ternary_engine"),
             tr("settings.pluggable_engine")}
    assert any({l.itemAt(i).widget().title() for i in range(3)} == cards for l in rows), \
        "三张引擎卡不在同一行"


def test_update_controls_live_in_the_table_not_the_cards(page):
    from lmloader.gui.i18n import tr

    table = page.version_table
    assert table.rowCount() == 2, "槽位行没建出来"
    for row, slot in enumerate(("primary", "ternary")):
        texts = [b.text() for b in _buttons_in_row(page, row)]
        for key in ("settings.check_update", "settings.update_engine",
                    "settings.rollback_engine"):
            assert tr(key) in texts, f"{slot} 行少了 {key}"
        assert getattr(page, f"{slot}_check_btn").parentWidget() is not None

    cards = [g for g in page.findChildren(QGroupBox)
             if g.title() in {tr("settings.primary_engine"), tr("settings.ternary_engine")}]
    assert cards, "两张内置引擎卡不见了"
    for card in cards:
        assert not [b for b in card.findChildren(QPushButton)
                    if b.text() == tr("settings.check_update")], "更新按钮还留在卡片里"


def test_registered_and_scanned_engines_get_their_own_rows(page):
    from lmloader.gui.i18n import tr

    page._fill_engine_rows(DISCOVERED, CUSTOMS)
    table = page.version_table
    assert table.rowCount() == 2 + 2 + 1, "两条登记 + 一条扫描，行数不对"

    texts = [table.item(r, 0).text() for r in range(table.rowCount())]
    assert texts[:2] == [tr("settings.engine_row_primary"), tr("settings.engine_row_ternary")]
    assert "rwkv-runner" in texts and "onnx-clip" in texts
    assert table.item(0, 3) is not None and ".gguf" in table.item(0, 3).text()

    swap, rollback = page._custom_actions["rwkv-runner"]
    assert swap.text() == tr("settings.swap_binary")
    assert rollback.isEnabled(), "有历史却不让回退"
    swap2, rollback2 = page._custom_actions["onnx-clip"]
    assert not rollback2.isEnabled(), "没历史也能点回退"

    scan_row = texts.index("vllm@vllm-server")
    assert table.item(scan_row, 5).text() == tr("settings.action_none")
    assert _buttons_in_row(page, scan_row) == []


def test_only_one_save_button_for_the_whole_page(page):
    from lmloader.gui.i18n import tr

    saves = [b for b in page.findChildren(QPushButton) if b.text() == tr("settings.save")]
    assert len(saves) == 1
    assert tr("settings.download_save") not in [b.text() for b in page.findChildren(QPushButton)]
    download = [g for g in page.findChildren(QGroupBox) if g.title() == tr("settings.download")]
    assert download, "下载卡没了"
    assert tr("settings.download_saved_hint") in [l.text() for l in download[0].findChildren(QLabel)], \
        "下载卡没说明它跟着哪个按钮保存"
