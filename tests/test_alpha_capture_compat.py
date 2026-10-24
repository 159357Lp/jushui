"""不透明度采集脚本对 ``theme_picker`` 的依赖要跨版本成立。

``CustomThemeRow`` 的预览有两种形态：加了 40ms 防抖的那版把重绘收进 ``_preview_timer``，
没有防抖时 ``valueChanged`` 直接调 ``preview_wallpaper``，行上根本没有这个名字的属性。
``demo_capture_alpha.py`` 曾经把两个定时器都写死，结果在只有 commit 定时器的代码上第一帧
就 ``AttributeError`` —— 拍一半崩，比拍不出来更难查。

钉住五件事：

* ``_quiet()`` 只停对象上真正存在的定时器；
* 正在跑的定时器确实被停掉，不是"查不到就算通过"；
* 没有任何定时器的对象也不炸；
* 反向对照一：写死属性名的旧写法在无预览定时器的行上必须恰好报一次 ``AttributeError``；
* 反向对照二：滑杆自身的改动不落盘 —— 采集脚本敢把预览留在进程内的前提。
"""

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import demo_capture_alpha as alpha
from lmloader.gui import theme_picker


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def head_shaped_row(qapp):
    """真的滑杆行，但剥掉只有防抖版才有的预览定时器。"""
    row = theme_picker.CustomThemeRow()
    if hasattr(row, "_preview_timer"):
        del row._preview_timer
    assert not hasattr(row, "_preview_timer"), "没剥干净，反向对照会假绿"
    return row


def test_quiet_survives_row_without_preview_timer(head_shaped_row):
    alpha._quiet(head_shaped_row)


def test_quiet_really_stops_live_timers(qapp):
    row = theme_picker.CustomThemeRow()
    timers = {}
    for name in ("_preview_timer", "_commit_timer"):
        if hasattr(row, name):
            timers[name] = QTimer(row)
            timers[name].start(5000)
            setattr(row, name, timers[name])
    assert timers, "行上一个定时器都没有，这条判据会是假绿"
    for timer in timers.values():
        assert timer.isActive(), "定时器没跑起来，测不出停没停"

    alpha._quiet(row)

    still_running = [n for n, t in timers.items() if t.isActive()]
    assert not still_running, f"这些定时器还在跑：{still_running}"


def test_quiet_tolerates_object_without_any_timer():
    alpha._quiet(type("Bare", (), {})())


def test_hardcoded_timer_names_are_the_actual_bug(head_shaped_row):
    """变异锚点：把修复前那两行原样跑一遍，必须在 HEAD 形状上报红。"""

    def hardcoded(row):
        row._commit_timer.stop()
        row._preview_timer.stop()

    with pytest.raises(AttributeError):
        hardcoded(head_shaped_row)


def test_slider_move_does_not_write_config(qapp, monkeypatch):
    writes = []
    monkeypatch.setattr(theme_picker, "save_wallpaper",
                        lambda *a, **k: writes.append(a))
    monkeypatch.setattr(theme_picker, "save_custom_theme",
                        lambda *a, **k: writes.append(a))
    monkeypatch.setattr(theme_picker, "save_ui_setting",
                        lambda *a, **k: writes.append(a))

    row = theme_picker.CustomThemeRow()
    if not row._wp_enable.isChecked():
        row._wp_enable.setChecked(True)
    for value in (100, 63, 0):
        alpha._quiet(row)
        row._panel.setValue(value)
    alpha._quiet(row)

    assert not writes, f"滑杆改动直接落盘了 {len(writes)} 次：{writes}"
