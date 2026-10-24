"""拍摄掬水各页静帧，供功能演示视频使用。

- 原生平台渲染 + WA_DontShowOnScreen：中文不会出现豆腐块，也不会弹窗打断你
- 聊天页接一份临时演示库（demo_screenshots/_demochat.db），不读取你的真实会话
- 壁纸镜头在进程内应用，不写 config.yaml
- 所有数据来自正在运行的后端（只读 GET），不加载/卸载模型、不发送生成请求
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_SCALE_FACTOR", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "demo_screenshots" / "shots_2026-10-24"
DEMO_DB = ROOT / "demo_screenshots" / "_demochat.db"
DEMO_ART = Path("D:/掬水/vibe_images/jushui_demo_wallpaper_1792830981975_1e3aa8ac.png")
W, H = 1920, 1080

DEMO_CONVS = [
    (
        "掬水 和 LM Studio 的差别",
        [
            ("user", "用三点说清掬水和 LM Studio 的差别。"),
            (
                "assistant",
                "**一、加载不是暴力上卡。** 掬水先读 GGUF 头里的张量字节、KV 几何和 MoE 专家维度，"
                "再按当前整卡余量解出 `-ngl` 与 `--n-cpu-moe`，解不出来就不硬上。\n\n"
                "**二、多模型共存是一本联合账。** 超预算时按 0.80 分页上限做联合解，"
                "刚性模型不让步，MoE 对等模型选择\"减专家重载\"而不是被驱逐。\n\n"
                "**三、每次生成都留档。** 速度、命中率、缓存占用、权重分账、加载前后机器空闲"
                "全部进运行表，长对话变慢时先看账再看模型。",
            ),
        ],
    ),
    (
        "MTP 投机解码怎么配",
        [
            ("user", "MTP 的 draft 深度该怎么选？"),
            (
                "assistant",
                "先确认 GGUF 里真有 MTP 张量，再动 draft 深度。默认 2 是通用最优：\n\n"
                "```text\n# 2026-10-22 实测（Qwen3.5-9B NVFP4 / RTX 5080）\n"
                "d1  148 t/s\nd2  160 t/s\nd3  162 t/s\nd4  141 t/s\n```\n\n"
                "深训 MTP 的模型在 d2 已接近峰值，d3 只多 1%；浅训模型在 d2 也不掉速。"
                "三层以上收益转负，别只看单次峰值，还要把草稿层的 KV 计价算进去。",
            ),
        ],
    ),
    (
        "给大模型省显存",
        [
            ("user", "27B 的 NVFP4 想跑长上下文，显存不够怎么办？"),
            (
                "assistant",
                "按代价从低到高：\n\n"
                "- **KV 量化**：`q8_0 → q4_0`，长上下文下省得最多，质量损失可接受。\n"
                "- **提示缓存设上限**：`--cache-ram 1024`，避免第 6 条对话分支把卡挤穿。\n"
                "- **MoE 专家分层**：`--n-cpu-moe N`，只把 N 层专家留在内存，比整层下沉便宜。\n"
                "- **视觉编码器按需挂**：不用的对话不必挂着 mmproj。\n\n"
                "最后才考虑降上下文：超过可用余量的上下文不是配置问题，是预算问题。",
            ),
        ],
    ),
    (
        "周报润色",
        [
            ("user", "把这段周报改得紧凑一点。"),
            (
                "assistant",
                "本周完成三件事：一是把模型扫描的重复解析去掉，启动从 58 秒降到 0.4 秒；"
                "二是空闲轮询改用连接池，待机占用从 35% 降到 0.4%；"
                "三是把启动链路改成非阻塞，窗口出现时间从 15 秒降到 2.2 秒。"
                "下周计划补齐长上下文下的显存拐点实测。",
            ),
        ],
    ),
]


def seed_demo_db() -> None:
    """建一份独立演示库；绝不碰真实 lmloader.db。"""
    from lmloader.infra import chat_db as chat_db_mod

    if DEMO_DB.exists():
        DEMO_DB.unlink()
    chat_db_mod.chat_db.db_path = DEMO_DB
    asyncio.run(chat_db_mod.chat_db.init())

    now = datetime.now()
    with sqlite3.connect(DEMO_DB) as db:
        for i, (title, messages) in enumerate(DEMO_CONVS):
            ts = (now - timedelta(hours=5 - i)).isoformat(timespec="seconds")
            cur = db.execute(
                "INSERT INTO conversations (title, model_name, system_prompt, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (title, "qwen35-9b-mtp-nvfp4.gguf", "", ts, ts),
            )
            conv_id = cur.lastrowid
            for j, (role, content) in enumerate(messages):
                db.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (conv_id, role, content, (now - timedelta(hours=5 - i, minutes=-j)).isoformat(timespec="seconds")),
                )
        db.commit()
    print(f"[demo-db] {DEMO_DB.name}: {len(DEMO_CONVS)} 个会话")


def pump(app, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def install_demo_theme() -> dict:
    """拍摄进程内把自定义主题换成演示图，绝不落盘到 config.yaml。

    你的真实配置里那张壁纸是私人图，设置页的外观卡会显示缩略图和文件名；
    换成合成演示图后，画面里出现的所有素材都可以公开。
    """
    from lmloader.gui import theme_picker, ui_config
    from lmloader.gui.theme_extract import extract_palette

    ext = extract_palette(str(DEMO_ART), "dark")
    ratios = list(ext.contrast.values())
    demo = {
        "base": "dark",
        "source": str(DEMO_ART),
        "tokens": ext.tokens,
        "contrast": [sum(1 for v in ratios if v >= 4.5), len(ratios)],
        "wallpaper": {"enabled": True, "strength": 100, "panel_alpha": 50},
    }
    ui_config.load_custom_theme = lambda: dict(demo)
    theme_picker.load_custom_theme = lambda: dict(demo)
    theme_picker.load_wallpaper = lambda: dict(demo["wallpaper"])
    return demo


def apply_demo_theme(app, win, demo: dict) -> None:
    from lmloader.gui import styles

    styles.apply_theme(
        app, demo["base"], demo["tokens"],
        wallpaper={"enabled": True, "source": demo["source"],
                   "strength": 1.0, "panel_alpha": 0.5},
    )
    for page in (win.chat_page, win.logs_page):
        if hasattr(page, "retheme"):
            page.retheme()
    win._apply_sidebar_style()
    win.centralWidget().update()
    for page in (win.models_page, win.download_page):
        if hasattr(page, "_refresh"):
            page._refresh()


def apply_solid_theme(app, win) -> None:
    from lmloader.gui import styles

    styles.set_wallpaper(None)
    styles.apply_theme(app, "dark")
    for page in (win.chat_page, win.logs_page):
        if hasattr(page, "retheme"):
            page.retheme()
    win._apply_sidebar_style()
    win.centralWidget().update()


def shots(app, win, navigate, names_to_pages):
    """navigate: 导航函数 index->None；names_to_pages: [(文件名, 页索引, 等待秒)]"""
    for filename, index, wait in names_to_pages:
        navigate(index)
        pump(app, wait)
        path = OUT / filename
        win.grab().save(str(path))
        print(f"[shot] {filename} {path.stat().st_size // 1024} KB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_demo_db()

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QScrollArea

    from lmloader.gui import styles
    from lmloader.gui.window import MainWindow

    app = QApplication([])
    app.setApplicationName("掬水")

    # 干净的内置深色主题（不读你的图片主题，避免把个人壁纸带进视频）
    demo_theme = install_demo_theme()
    styles.set_wallpaper(None)
    styles.apply_theme(app, "dark")

    win = MainWindow()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win._guardian_timer.stop()          # 审批弹窗不能盖在拍摄画面上
    win._conn_timer.stop()
    if win._collapsed:
        win._toggle_sidebar()
    win.resize(W, H)
    win.show()
    pump(app, 3.0)

    def navigate(index: int) -> None:
        win._navigate(index)

    # ── 常规页面 ────────────────────────────────────────────────
    shots(app, win, navigate, [
        ("01_chat.png", 0, 2.5),
        ("02_models.png", 1, 3.5),
        ("05_monitor.png", 3, 3.0),
        ("06_logs.png", 4, 3.0),
        ("07_settings_top.png", 5, 2.0),
        ("09_run_stats.png", 6, 4.0),
    ])

    # ── 聊天：选内容最丰富的那条会话 ───────────────────────────
    navigate(0)
    pump(app, 1.5)
    chat = win.chat_page
    target_row = 0
    if chat.conv_list.count():
        target_row = next(
            (i for i, conv in enumerate(chat.conversations)
             if "LM Studio" in (conv.get("title") or "")),
            0,
        )
        chat.conv_list.setCurrentRow(target_row)
        pump(app, 1.8)
        win.grab().save(str(OUT / "01b_chat_conversation.png"))
        print(f"[shot] 01b_chat_conversation.png (row={target_row})")
    chat.prompt_btn.setChecked(True)
    pump(app, 1.2)
    win.grab().save(str(OUT / "01c_chat_prompt.png"))
    print("[shot] 01c_chat_prompt.png")
    chat.prompt_btn.setChecked(False)
    pump(app, 0.3)

    # ── 模型运行配置对话框 ─────────────────────────────────────
    from lmloader.gui.pages.models import ModelConfigDialog
    from lmloader.gui.api_client import LMLoaderAPI

    api = LMLoaderAPI()
    target = "Qwen3.8-27B-NVFP4-MTP-LOW.gguf"
    try:
        cfg = api.get_model_run_config(target)
        vram_mb = None
        try:
            mon = api.get_monitor()
            gpus = (mon.get("gpu") or {}).get("gpus") or []
            if gpus and gpus[0].get("memory_total"):
                vram_mb = int(gpus[0]["memory_total"])
        except Exception:
            pass
        dlg = ModelConfigDialog(api, target, cfg, parent=win, vram_total_mb=vram_mb)
        dlg.setAttribute(Qt.WA_DontShowOnScreen, True)
        dlg.show()
        pump(app, 1.0)
        dlg.grab().save(str(OUT / "04_models_config.png"))
        print("[shot] 04_models_config.png")
        dlg.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 配置对话框未拍到: {exc}")

    # ── 下载中心：HuggingFace 与 Ollama 各一张 ─────────────────
    navigate(2)
    pump(app, 0.5)
    dl = win.download_page
    try:
        dl.search_input.setText("qwen3")
        dl._search()
        deadline = time.time() + 30
        while time.time() < deadline:
            pump(app, 0.3)
            if dl.results_table.rowCount() >= 2:
                break
        pump(app, 0.8)
        win.grab().save(str(OUT / "03_download_hf.png"))
        print(f"[shot] 03_download_hf.png (rows={dl.results_table.rowCount()})")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] HF 搜索结果未拍到: {exc}")

    try:
        dl.tabs.setCurrentIndex(1)
        dl.ollama_input.setText("deepseek")
        dl._ollama_search()
        deadline = time.time() + 30
        while time.time() < deadline:
            pump(app, 0.3)
            if dl.ollama_table.rowCount() >= 2:
                break
        pump(app, 0.8)
        win.grab().save(str(OUT / "03b_download_ollama.png"))
        print(f"[shot] 03b_download_ollama.png (rows={dl.ollama_table.rowCount()})")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Ollama 结果未拍到: {exc}")

    # ── 设置：外观卡用演示主题，滚到外观 / 下载 / 路径 ─────────
    apply_demo_theme(app, win, demo_theme)
    navigate(5)
    pump(app, 1.5)
    scroll = win.settings_page.findChild(QScrollArea)
    if scroll is not None:
        bar = scroll.verticalScrollBar()
        for name, ratio in (("08_settings_mid.png", 0.55), ("08b_settings_bottom.png", 1.0)):
            bar.setValue(int(bar.maximum() * ratio))
            pump(app, 1.0)
            win.grab().save(str(OUT / name))
            print(f"[shot] {name} (scroll {bar.value()}/{bar.maximum()})")
    apply_solid_theme(app, win)
    pump(app, 0.6)

    # ── 速度档案：展开 NVFP4-LOW 那一行 ────────────────────────
    navigate(6)
    pump(app, 2.5)
    rs = win.run_stats_page
    try:
        stats = api.fetch_run_stats_or_none()
        models = (stats or {}).get("models") or []
        loaded = api.fetch_loaded_models_or_none()
        rs._render_live(loaded)
        rs._render_archive(stats or {"models": [], "total": 0})
        pump(app, 0.8)
        from PySide6.QtWidgets import QPushButton

        folds = [b for b in rs.findChildren(QPushButton) if b.text() in ("\u25b8", "\u25be")]
        target_index = next(
            (i for i, m in enumerate(models) if "NVFP4-MTP-LOW" in m.get("model_name", "")),
            0,
        )
        if target_index < len(folds):
            folds[target_index].click()
        pump(app, 1.2)
        win.grab().save(str(OUT / "09b_run_stats_expanded.png"))
        print(f"[shot] 09b_run_stats_expanded.png (models={len(models)}, folds={len(folds)}, open={target_index})")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 档案展开未拍到: {exc}")

    # ── 壁纸镜头：演示图取色 + 壁纸，仅进程内，不写 config ─────
    try:
        apply_demo_theme(app, win, demo_theme)
        navigate(0)
        if chat.conv_list.count():
            chat.conv_list.setCurrentRow(target_row)
        pump(app, 2.5)
        win.grab().save(str(OUT / "10_theme_wallpaper.png"))
        print("[shot] 10_theme_wallpaper.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 壁纸镜头未拍到: {exc}")
    finally:
        apply_solid_theme(app, win)

    print(f"\n[done] {OUT}")
    print(f"[privacy] 演示库 {DEMO_DB.name}；真实会话未读取")


if __name__ == "__main__":
    main()
