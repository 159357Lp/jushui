"""采集「面板不透明度」滑杆的真实连续帧，供短视频使用。

滑杆走的是应用自己的 valueChanged → preview_wallpaper 链路（只重绘、不落盘）；
save_wallpaper 在采集期被打成空函数，绝不写 config.yaml。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtWidgets import QApplication, QScrollArea

import demo_capture as dc
from lmloader.gui import styles, theme_picker, ui_config

OUT = dc.OUT / "alpha_sweep"


def main() -> None:
    # 落盘口子先堵死：预览链路本身不写，只有 commit 定时器会写
    theme_picker.save_wallpaper = lambda *a, **k: None
    ui_config.save_wallpaper = lambda *a, **k: None

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("掬水")
    dc.seed_demo_db()
    demo = dc.install_demo_theme()
    styles.set_wallpaper(None)
    styles.apply_theme(app, "dark")

    win = dc.build_window(app) if hasattr(dc, "build_window") else None
    if win is None:
        from lmloader.gui.window import MainWindow
        win = MainWindow()
        win.setAttribute(Qt.WA_DontShowOnScreen, True)
        win._guardian_timer.stop()
        win._conn_timer.stop()
        if win._collapsed:
            win._toggle_sidebar()
        win.resize(dc.W, dc.H)
        win.show()
        dc.pump(app, 2.5)

    dc.apply_demo_theme(app, win, demo)

    settings = win.settings_page
    row = settings.theme_row
    scroll = settings.findChild(QScrollArea)
    win._navigate(5)
    dc.pump(app, 1.2)
    if scroll is not None:
        scroll.ensureWidgetVisible(row, 0, 60)
    dc.pump(app, 0.6)

    if not row._wp_enable.isChecked():
        row._wp_enable.setChecked(True)
    row._strength.setValue(100)
    dc.pump(app, 0.8)

    def step(value: int, name: str) -> None:
        row._commit_timer.stop()
        row._preview_timer.stop()
        row._panel.setValue(value)          # 触发真实的 valueChanged → preview
        dc.pump(app, 0.12)
        row._commit_timer.stop()
        row._preview_timer.stop()
        win.grab().save(str(OUT / name))
        print(f"  [frame] {name}  panel={row._panel.value()}% "
              f"readout={row._readouts[row._panel].text()}")

    OUT.mkdir(parents=True, exist_ok=True)
    print("[sweep] 设置页：面板不透明度 100% → 0% → 50%")
    idx = 0
    for v in range(100, -1, -5):
        step(v, f"settings_{idx:02d}.png")
        idx += 1
    for v in range(5, 51, 5):
        step(v, f"settings_{idx:02d}.png")
        idx += 1
    print(f"[sweep] 设置页共 {idx} 帧")

    print("[sweep] 聊天页：三个不透明度对照")
    win._navigate(0)
    dc.pump(app, 1.0)
    chat = win.chat_page
    if chat.conv_list.count():
        chat.conv_list.setCurrentRow(3)
        dc.pump(app, 1.2)
    for v in (100, 50, 0):
        step(v, f"chat_{v:03d}.png")

    # 收尾：把预览状态清掉，不留在半透明态（进程马上退出，配置从未被写）
    styles.set_wallpaper(None)
    print("[done] 输出目录:", OUT)


if __name__ == "__main__":
    main()
