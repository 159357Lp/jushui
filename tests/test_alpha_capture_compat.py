"""不透明度采集脚本对 ``theme_picker`` 的依赖要跨版本成立。

``CustomThemeRow`` 的预览有两种形态：加了 40ms 防抖的那版把重绘收进 ``_preview_timer``，
没有防抖时 ``valueChanged`` 直接调 ``preview_wallpaper``，行上根本没有这个名字的属性。
``demo_capture_alpha.py`` 曾经把两个定时器都写死，结果在只有 commit 定时器的代码上第一帧
就 ``AttributeError`` —— 拍一半崩，比拍不出来更难查。

钉住六件事：

* ``_quiet()`` 只停对象上真正存在的定时器；
* 正在跑的定时器确实被停掉，不是"查不到就算通过"；
* 没有任何定时器的对象也不炸；
* 反向对照一：写死属性名的旧写法在无预览定时器的行上必须恰好报一次 ``AttributeError``；
* 反向对照二：滑杆自身的改动不落盘 —— 采集脚本敢把预览留在进程内的前提；
* 预览改进程级样式全局之后必须还原回去，否则后面页面的尺寸判据会被带红（实测顺序相关）。

素材自备（一张 tmp 里的 PNG + 一份演示主题）而不是读 ``config.yaml``：在只含已提交代码
的干净检出里根本没有配置文件，靠本机壁纸会让判据变成假红 —— 这条踩过一次。
"""

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import demo_capture_alpha as alpha
from lmloader.gui import styles, theme_picker, ui_config


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def style_restored(qapp):
    """把主题与壁纸全局原样放回。

    这几条判据要真的构造滑杆行，而勾选壁纸开关会走真实预览链路，把
    ``styles._wallpaper`` 改成非空 —— 那是进程级全局，后面的页面样式表会跟着换内衬，
    于是别处的按钮尺寸判据会被带红（实测 ``test_chat_actions`` 的图标按钮 30 > 28）。
    """
    saved = (styles._wallpaper, styles.current_theme, qapp.palette(), qapp.styleSheet())
    yield
    styles.set_wallpaper(saved[0])
    styles.current_theme = saved[1]
    qapp.setPalette(saved[2])
    qapp.setStyleSheet(saved[3])


@pytest.fixture()
def head_shaped_row(qapp, style_restored):
    """真的滑杆行，但剥掉只有防抖版才有的预览定时器。"""
    row = theme_picker.CustomThemeRow()
    if hasattr(row, "_preview_timer"):
        del row._preview_timer
    assert not hasattr(row, "_preview_timer"), "没剥干净，反向对照会假绿"
    return row


def test_quiet_survives_row_without_preview_timer(head_shaped_row):
    alpha._quiet(head_shaped_row)


def test_quiet_really_stops_live_timers(qapp, style_restored):
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


def test_slider_move_does_not_write_config(qapp, tmp_path, style_restored, monkeypatch):
    """拖不透明度滑杆只重绘、不落盘 —— 整段采集脚本敢这么干的前提。

    素材自备：壁纸源必须是真实存在的文件，否则 ``_stored_source()`` 判定不可用，
    预览直接空转。实测在只含已提交代码、没有 config.yaml 的检出里，靠本机配置
    会直接得到 ``styles._wallpaper is None`` —— 判据会跟着假红。
    """
    from PySide6.QtGui import QPixmap

    pic = tmp_path / "wp.png"
    canvas = QPixmap(64, 64)
    canvas.fill("#10151C")
    assert canvas.save(str(pic), "PNG"), "演示壁纸没写出来"

    theme = {"base": "dark", "source": str(pic), "tokens": {"bg": "#10151C"},
             "contrast": [11, 11]}
    monkeypatch.setattr(theme_picker, "load_custom_theme", lambda: dict(theme))
    monkeypatch.setattr(theme_picker, "load_wallpaper",
                        lambda: {"enabled": True, "strength": 100, "panel_alpha": 50})
    monkeypatch.setattr(ui_config, "active_theme", lambda: ("dark", {}))

    writes = []
    for name in ("save_wallpaper", "save_custom_theme", "save_ui_setting"):
        monkeypatch.setattr(theme_picker, name, lambda *a, **k: writes.append(a))

    row = theme_picker.CustomThemeRow()
    before = (styles._wallpaper, styles.current_theme, qapp.styleSheet())
    row._wp_enable.setChecked(True)
    # 勾选走的是 commit 路径，两个版本都会在这里写盘；本判据只管滑杆，故清零后再称。
    writes.clear()

    for value in (100, 63, 0):
        alpha._quiet(row)
        row._panel.setValue(value)
    alpha._quiet(row)

    assert (styles._wallpaper, styles.current_theme, qapp.styleSheet()) != before, \
        "预览链路没真的改过全局样式，这条判据和 style_restored 都是空转"
    assert styles._wallpaper is not None, "预览没把壁纸状态装上，重绘路径根本没被走到"
    assert not writes, f"滑杆改动直接落盘了 {len(writes)} 次：{writes}"
