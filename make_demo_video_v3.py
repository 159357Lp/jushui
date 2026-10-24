"""掬水功能全览视频 v3 — 146 条功能 / 13 个能力域。

截图实物 + 自动生成的清单卡（条目解析自 功能清单.md）+ 旁白 + 字幕 + BGM。
结束前做机械覆盖校验：所有场景声明的 ID 并集必须等于清单全集。
"""

import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
SHOTS = ROOT / "demo_screenshots" / "shots_2026-10-24"
INVENTORY = ROOT / "功能清单.md"
OUT_DIR = ROOT / "demo_screenshots"
OUTPUT = OUT_DIR / "掬水功能全览_2026-10-24.mp4"
COVERAGE_MD = OUT_DIR / "视频覆盖对照.md"
COVERAGE_JSON = OUT_DIR / "video_coverage_2026-10-24.json"
TMP = Path(os.environ.get("TEMP", "C:/Users/75181/AppData/Local/Temp")) / "jushui_video_v3"
WALL = Path("D:/掬水/vibe_images/jushui_demo_wallpaper_1792830981975_1e3aa8ac.png")

WIDTH, HEIGHT, FPS = 1920, 1080, 30
FADE = int(0.6 * FPS)
WATERMARK_TEXT = "不雨花犹落"

ACCENT = (94, 234, 212)
TEXT = (230, 240, 255)
MUTED = (127, 147, 173)
ID_COLOR = (111, 216, 200)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("msyhbd.ttc", "msyh.ttc") if bold else ("msyh.ttc", "simhei.ttf")):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


# ── 清单解析 ─────────────────────────────────────────────────────
CHAPTER_ORDER = ["SHELL", "CHAT", "MOD", "DL", "MON", "SRV",
                 "SET", "ARC", "ENG", "API", "CLI", "THEME", "DB"]
CHAPTER_META = {
    "SHELL": ("桌面壳层", "DESKTOP SHELL"),
    "CHAT": ("聊天工作台", "CHAT WORKBENCH"),
    "MOD": ("模型管理", "MODEL MANAGER"),
    "DL": ("下载中心", "DOWNLOAD CENTER"),
    "MON": ("资源监控", "RESOURCE MONITOR"),
    "SRV": ("本地服务与日志", "LOCAL SERVICE & LOGS"),
    "SET": ("设置", "SETTINGS"),
    "ARC": ("速度档案", "SPEED ARCHIVE"),
    "ENG": ("引擎与加载", "ENGINE & LOADING"),
    "API": ("兼容层", "COMPATIBILITY API"),
    "CLI": ("命令行与部署", "CLI & DEPLOYMENT"),
    "THEME": ("主题与壁纸", "THEME & WALLPAPER"),
    "DB": ("存储与安全", "STORAGE & SECURITY"),
}


def parse_inventory() -> "OrderedDict[str, list[tuple[str, str]]]":
    text = INVENTORY.read_text(encoding="utf-8")
    rows = re.findall(r"^\| ([A-Z]+-\d+) \| ([^|]+?) \|", text, re.M)
    items: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict((k, []) for k in CHAPTER_ORDER)
    for fid, name in rows:
        key = fid.split("-")[0]
        if key == "SEC":
            key = "DB"
        items[key].append((fid, name.strip()))
    return items


INV = parse_inventory()
ALL_IDS = [fid for group in INV.values() for fid, _ in group]


# ── 场景定义 ─────────────────────────────────────────────────────
def img(src, chapter, ids, narration, subtitle):
    return {"kind": "img", "src": src, "chapter": chapter, "ids": ids,
            "narration": narration, "subtitle": subtitle}


def card(chapter, narration, subtitle):
    return {"kind": "card", "chapter": chapter,
            "ids": [fid for fid, _ in INV[chapter]],
            "narration": narration, "subtitle": subtitle}


def opening():
    return {"kind": "opening", "chapter": "OPEN", "ids": [], "subtitle":
            "掬水 v0.2.0 · 13 个能力域 · 146 条功能",
            "narration": "掬水，版本零点二点零。这一支视频按十三个能力域、一百四十六条功能逐条过一遍，先看界面实物，再看引擎、接口和底座。"}


def limits_card():
    return {"kind": "limits", "chapter": "OPEN", "ids": [], "subtitle": "诚实边界 · 不把这些算作卖点",
            "narration": "为了不夸大：设置页那三项页面级参数是存根；README 说的完整 SPA 只是一张静态页；Ollama 方言是子集；设了 token 后删除和下载会鉴权失败；还有一段死代码。这些在清单里都标了限制。"}


def closing():
    return {"kind": "closing", "chapter": "OPEN", "ids": [], "subtitle":
            "MIT 开源 · gitee.com/buyuhuayouluo/lmloader",
            "narration": "以上是掬水的完整功能面。每条功能都能在功能清单里找到代码出处。MIT 开源，仓库地址 gitee 点 com 斜杠 buyuhuayouluo 斜杠 lmloader。"}


SCENES = [
    opening(),
    card("SHELL", "第一域是桌面壳层，共十一条。七页侧栏、侧栏收起、手绘矢量图标、两秒一次的连接看门狗、单实例互斥、后端自动托管、托盘、关闭询问、配置变更审批、中英双语与退出兜底。",
         "SHELL · 桌面壳层 11 条"),
    img("01_chat.png", "SHELL", ["SHELL-01", "SHELL-04", "SHELL-06", "SHELL-09"],
        "启动后就是聊天工作台，左侧七页导航，右上角常驻连接状态。关闭窗口默认最小化到托盘；高危配置变更会在这里弹出审批，默认拒绝。",
        "七页导航 · 连接看门狗 · 审批锁"),
    img("01b_chat_conversation.png", "CHAT", ["CHAT-01", "CHAT-03", "CHAT-04", "CHAT-05", "CHAT-07", "CHAT-16"],
        "选一段真实会话。回复流式生成，标题、列表、行内代码按 Markdown 子集渲染；思考内容单独折叠；每轮结尾附耗时、token 数和生成速度。",
        "流式输出 · Markdown · 思考折叠 · 实测脚注"),
    img("01c_chat_prompt.png", "CHAT", ["CHAT-06", "CHAT-10", "CHAT-11", "CHAT-12", "CHAT-13", "CHAT-14", "CHAT-15"],
        "提示词面板分会话级和全局两种作用域，三套模板，六百毫秒防抖落库，实时统计 token。图片和文本都能作为附件，按路径去重，也能整段导出。",
        "双作用域系统提示词 · 图片/文本附件 · 导出"),
    card("CHAT", "聊天工作台共十八条。会话历史、删除二次确认、流式输出与停止、Markdown 渲染、思考折叠卡与开关、实测脚注、模型选择与未加载即载、决策模型保护、图片和文本附件、会话级与全局系统提示词、导出、消息复制另存、复读守卫、请求侧反读参数、SSE 错误穿透。",
         "CHAT · 聊天工作台 18 条"),
    img("02_models.png", "MOD", ["MOD-01", "MOD-02", "MOD-03", "MOD-04", "MOD-09", "MOD-11", "MOD-13", "MOD-14"],
        "模型管理列出扫描到的每一个 GGUF，显示量化、参数量、架构、大小与加载状态。视觉模型旁边有眼睛图标，可以热切换视觉编码器。",
        "六列模型卡 · 一键加载卸载 · 视觉热切换"),
    img("04_models_config.png", "MOD", ["MOD-05", "MOD-06", "MOD-07", "MOD-08"],
        "每模型十二项运行配置：上下文、GPU 层数、线程、batch、ubatch、并行槽、Flash Attention、KV 类型、MTP、draft 深度和 CPU-MoE。没保存的项按引擎生效值回显。",
        "每模型 12 项运行配置 · 按生效值回显"),
    card("MOD", "模型管理共十六条。六列模型卡、响应式三档列、一键加载卸载、视觉开关、十二项运行配置、并行槽显存估算、按生效值回显、保存与保存并加载、扫描清理、外部路径登记、改名复制、删除记录、已存配置徽标、三秒轮询、GGUF 元数据解析、扫描指纹缓存。",
         "MOD · 模型管理 16 条"),
    img("03_download_hf.png", "DL", ["DL-01", "DL-02", "DL-03", "DL-04", "DL-07"],
        "下载中心接 HuggingFace 镜像。输入关键词，列出作者、大小、下载量，展开每个模型下的全部 GGUF 文件逐一下载，支持断点续传。",
        "HF 搜索 · GGUF 展开 · 断点续传"),
    img("03b_download_ollama.png", "DL", ["DL-05", "DL-06"],
        "另一个标签页是 Ollama 库，显示 tag、能力和上下文长度，拉取后走同一套下载器落地并登记。",
        "Ollama 模型库 · 拉取即登记"),
    card("DL", "下载中心共十条。HuggingFace 搜索、GGUF 逐文件展开、仓库直连下载、断点续传分块、Ollama 库浏览与拉取、进度与停止、批量清除已完成、代理与镜像配置；还有一条限制：设了 token 后下载 worker 不带鉴权头。",
         "DL · 下载中心 10 条"),
    img("05_monitor.png", "MON", ["MON-01", "MON-02", "MON-03", "MON-04", "MON-05", "MON-06", "MON-07", "MON-08", "MON-09"],
        "资源监控每两秒刷新。显存和内存不是笼统一个数，而是权重、KV、循环状态、计算缓冲、其他进程、引擎开销、系统其他七类分账，多只模型不重复计数。",
        "六卡总览 · 显存/内存七类分账"),
    card("MON", "资源监控共九条。六卡总览、显存与内存七类分账、多模型不重复计数、磁盘忙闲、GPU 温度与型号、引擎槽状态、已加载模型清单、离线降级、两秒刷新。",
         "MON · 资源监控 9 条"),
    img("06_logs.png", "SRV", ["SRV-01", "SRV-02", "SRV-03", "SRV-04", "SRV-05", "SRV-06", "SRV-07"],
        "本地服务页把服务地址和每模型调用地址分成两行，避免抄进 Base URL 后拼出四零四。下面是终端日志流，按级别着色，可暂停、复制、清空。",
        "服务/调用地址分行 · 实时日志流"),
    card("SRV", "本地服务与日志共八条。服务状态三态、已加载模型实时行、服务地址与调用地址分行、端点清单、终端日志流、级别筛选与暂停、复制全部与清空、开发者日志中间件。",
         "SRV · 本地服务与日志 8 条"),
    img("07_settings_top.png", "SET", ["SET-01", "SET-02", "SET-03", "SET-04", "SET-05", "SET-06"],
        "设置页先看到双引擎槽和引擎版本表，可以检查更新、一键回退、重新扫描，也能登记自定义引擎。",
        "双引擎槽 · 检查更新 · 一键回退"),
    img("08_settings_mid.png", "SET", ["SET-09"],
        "外观区是图片取色：从任意图片提取十二个 token 的完整调色板，附 WCAG 对比度表，确认后才写配置。",
        "图片取色 · 对比度清单 · 确认后落盘"),
    img("08b_settings_bottom.png", "SET", ["SET-07", "SET-08", "SET-10", "SET-11"],
        "再往下是下载代理、Ollama 库地址、端口范围、切换时自动卸载、模型目录与 CUDA 路径，以及保存和恢复默认。",
        "端口范围 · 自动卸载 · 路径 · 保存/恢复"),
    card("SET", "设置共十二条。双引擎槽、引擎版本表、检查更新、一键回退、重新扫描引擎、自定义引擎、端口范围、切换时自动卸载、语言与主题、路径设置、保存与恢复默认；页面级的 GPU 层、上下文、线程是存根，不算能力。",
         "SET · 设置 12 条"),
    img("09_run_stats.png", "ARC", ["ARC-01", "ARC-02", "ARC-09"],
        "速度档案的归档视图，每模型按解码速度取高、中、低三次，附运行次数；正在跑的这一次单独一行，三秒刷新。",
        "高/中/低归档 · 正在跑的这一次"),
    img("09b_run_stats_expanded.png", "ARC", ["ARC-03", "ARC-04", "ARC-05", "ARC-06", "ARC-07", "ARC-08", "ARC-10"],
        "展开配置卡，是那一次运行的真实配置、环境与只读 argv；下面是最近十次明细，权重显存、内存、机器空闲前后、MTP 命中都分开记。",
        "三张配置卡 · 最近十次明细 · 加载前空闲快照"),
    card("ARC", "速度档案共十条，本版新增。正在跑的这一次、归档高中低、三张配置卡、最近十次明细、加载前空闲快照、三态诚实、塌陷标记、采集口径、五千行保留、损坏容错。",
         "ARC · 速度档案 10 条"),
    card("ENG", "引擎与加载共十九条，是整套系统的底盘。七种引擎种类、双槽位、格式自动路由、JIT 加载、端口自动分配、多模型共存的联合账本、自动分层解算、硬件代价表、keep-alive、决策模型钉住、孤儿进程回收、视觉编码器、提示缓存上限、MTP 投机解码、推理预算、嵌入模型识别、vLLM 专用环境、VRAM 仲裁器、进程树兜底。",
         "ENG · 引擎与加载 19 条"),
    card("API", "兼容层共九条。OpenAI 五个端点、Ollama 七个端点、keep-alive 语义对齐、决策契约、五级别名解析、统一错误面、六十三个控制面端点；Ollama 方言只是子集，Web 控制台只有一张静态页，这两点不夸大。",
         "API · 兼容层 9 条"),
    card("CLI", "命令行与部署共五条。十二条命令、无头 serve、读写命令分层、三个打包入口、Windows 下的启动批处理与桌面快捷方式。",
         "CLI · 命令行与部署 5 条"),
    img("10_theme_wallpaper.png", "THEME", ["THEME-02", "THEME-03", "THEME-04", "THEME-05", "THEME-07", "THEME-08"],
        "这是图片取色加壁纸的实际效果：底图铺满，面板半透明只软化卡片，前景墨色按对比度自动修正到可读。",
        "壁纸铺底 · 面板不透明度 · 对比度修复"),
    card("THEME", "主题共八条。深浅两套内置主题、图片取色、取色预览对话框、壁纸铺底、面板不透明度、实时预览与防抖提交、主题热切换、对比度修复。",
         "THEME · 主题与壁纸 8 条"),
    card("DB", "存储与安全共十一条。模型注册表、每模型运行配置、外部模型根目录、速度档案、会话与消息，合计六张表；安全面有 Bearer 鉴权、安全响应头、引擎路径夹紧、非回环地址告警、配置变更审批、日志隔离。",
         "DB/SEC · 存储与安全 11 条"),
    limits_card(),
    closing(),
]


# ── 画面合成 ─────────────────────────────────────────────────────
def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    sw, sh = img.size
    scale = max(w / sw, h / sh)
    img = img.resize((max(1, int(sw * scale)), max(1, int(sh * scale))), Image.LANCZOS)
    left = (img.width - w) // 2
    top = (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def _card_bg() -> Image.Image:
    if WALL.exists():
        base = _cover(Image.open(WALL).convert("RGB"), WIDTH, HEIGHT)
        base = base.filter(ImageFilter.GaussianBlur(24))
        base = Image.eval(base, lambda v: int(v * 0.46))
    else:
        base = Image.new("RGB", (WIDTH, HEIGHT), (10, 16, 28))
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([0, 0, WIDTH, HEIGHT], fill=(6, 12, 22, 118))
    grad = Image.new("L", (1, HEIGHT))
    for y in range(HEIGHT):
        grad.putpixel((0, y), int(40 + 90 * y / HEIGHT))
    grad = grad.resize((WIDTH, HEIGHT))
    d.bitmap((0, 0), grad, fill=(3, 8, 18, 255))
    return Image.alpha_composite(base.convert("RGBA"), overlay)


def render_card(scene: dict) -> Image.Image:
    bg = _card_bg()
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    key = scene["chapter"]
    idx = CHAPTER_ORDER.index(key) + 1 if key in CHAPTER_ORDER else 0
    title, en = CHAPTER_META.get(key, ("", ""))
    items = scene["ids_items"]

    d.text((140, 92), "掬水 · 功能全览", font=_font(20), fill=(*MUTED, 255))
    d.text((360, 92), f"·  第 {idx:02d} 域 / 13", font=_font(20), fill=(*MUTED, 255))
    d.text((140, 140), title, font=_font(60, True), fill=(*TEXT, 255))
    d.text((142, 224), en, font=_font(20), fill=(*MUTED, 255))
    d.rectangle([140, 268, 260, 272], fill=(*ACCENT, 255))

    d.text((1530, 40), f"{idx:02d}", font=_font(240, True), fill=(*ACCENT, 22))

    n = len(items)
    per_col = math.ceil(n / 2) if n > 4 else n
    cols = 2 if n > 4 else 1
    row_h = 52 if per_col <= 10 else 46
    start_y = 330 if per_col > 6 else 380
    for i, (fid, name) in enumerate(items):
        col = i // per_col
        row = i % per_col
        x = 142 + col * 850
        y = start_y + row * row_h
        d.text((x, y + 6), fid, font=_font(17), fill=(*ID_COLOR, 235))
        d.text((x + 116, y), name, font=_font(23), fill=(*TEXT, 255))

    d.text((142, 900), f"本域 {n} 条 · 逐条对照《功能清单.md》", font=_font(18), fill=(*MUTED, 220))
    d.text((WIDTH - 420, 900), f"总计 146 条 / 13 域", font=_font(18), fill=(*MUTED, 220))
    return Image.alpha_composite(bg, layer).convert("RGB")


def render_opening() -> Image.Image:
    bg = _card_bg()
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((150, 300), "掬水", font=_font(150, True), fill=(*TEXT, 255))
    d.text((160, 500), "桌面端本地大模型加载器  ·  v0.2.0", font=_font(34), fill=(*ACCENT, 255))
    d.rectangle([160, 570, 320, 575], fill=(*ACCENT, 255))
    d.text((160, 620), "13 个能力域   ·   146 条功能   ·   逐条实拍与清单", font=_font(26), fill=(*MUTED, 255))
    return Image.alpha_composite(bg, layer).convert("RGB")


def render_limits() -> Image.Image:
    bg = _card_bg()
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((140, 120), "诚实边界", font=_font(58, True), fill=(*TEXT, 255))
    d.text((142, 200), "NOT SOLD AS FEATURES", font=_font(20), fill=(*MUTED, 255))
    d.rectangle([140, 244, 260, 248], fill=(*ACCENT, 255))
    lines = [
        "设置页的页面级 GPU 层 / 上下文 / 线程是存根，实际参数在每模型配置里",
        "README 的「完整 SPA」言过其实，Web 控制台只有一张 10 KB 静态页",
        "Ollama 方言只覆盖常用子集，tools 参数被忽略，format 接受但不生效",
        "设了 api_token 后，模型删除与下载 worker 不带鉴权头，会 401",
        "engine/engine.py 的 EngineDetector 单例未被引用，是死代码",
        "没有全局快捷键、菜单栏、状态栏、拖拽导入、表格分页与排序",
    ]
    y = 330
    for line in lines:
        d.text((150, y), "×", font=_font(24, True), fill=(232, 140, 140, 255))
        d.text((190, y - 2), line, font=_font(24), fill=(*TEXT, 255))
        y += 78
    d.text((140, 900), "这些在《功能清单.md》§15 均有代码出处", font=_font(18), fill=(*MUTED, 220))
    return Image.alpha_composite(bg, layer).convert("RGB")


def render_closing() -> Image.Image:
    bg = _card_bg()
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((WIDTH // 2 - 120, 330), "掬水", font=_font(110, True), fill=(*TEXT, 255))
    d.text((WIDTH // 2 - 260, 500), "gitee.com/buyuhuayouluo/lmloader", font=_font(34), fill=(*ACCENT, 255))
    d.text((WIDTH // 2 - 190, 570), "MIT 开源  ·  v0.2.0  ·  2026-10-24", font=_font(24), fill=(*MUTED, 255))
    return Image.alpha_composite(bg, layer).convert("RGB")


def load_shot(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.width < WIDTH * 0.9 or img.height < HEIGHT * 0.9:
        # 小尺寸对话框：叠在对应的页面截图上，带阴影
        if "04_models_config" in path.name and (SHOTS / "02_models.png").exists():
            base = Image.open(SHOTS / "02_models.png").convert("RGB")
            base = _cover(base, WIDTH, HEIGHT)
            base = base.filter(ImageFilter.GaussianBlur(18))
            base = Image.eval(base, lambda v: int(v * 0.42))
        else:
            base = Image.new("RGB", (WIDTH, HEIGHT), (8, 12, 20))
        scale = (HEIGHT * 0.92) / img.height
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        canvas = base.convert("RGBA")
        x = (WIDTH - img.width) // 2
        y = (HEIGHT - img.height) // 2
        shadow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.rectangle([x + 14, y + 18, x + img.width + 14, y + img.height + 18], fill=(0, 0, 0, 150))
        shadow = shadow.filter(ImageFilter.GaussianBlur(18))
        canvas = Image.alpha_composite(canvas, shadow)
        canvas.paste(img, (x, y))
        return canvas.convert("RGB")
    scale = min(WIDTH / img.width, HEIGHT / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    canvas.paste(resized, ((WIDTH - resized.width) // 2, (HEIGHT - resized.height) // 2))
    return canvas


def add_subtitle(img: Image.Image, text: str) -> Image.Image:
    frame = np.array(img)
    overlay = frame.copy()
    bar_h = 92
    y_start = HEIGHT - bar_h
    cv2.rectangle(overlay, (0, y_start), (WIDTH, HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil)
    font = _font(29)
    max_chars = 48
    lines = []
    rest = text
    while len(rest) > max_chars:
        cut = rest[:max_chars]
        for sep in ("，", "。", "、", "；"):
            pos = cut.rfind(sep)
            if pos > max_chars // 2:
                cut = rest[:pos + 1]
                break
        else:
            for sep in (" ", "·", "+", "-", "/"):
                pos = cut.rfind(sep)
                if pos > max_chars // 2:
                    cut = rest[:pos + 1]
                    break
        lines.append(cut)
        rest = rest[len(cut):]
    lines.append(rest)

    line_h = 36
    total = len(lines) * line_h
    y = y_start + (bar_h - total) // 2 + 2
    for line in lines:
        bbox = d.textbbox((0, 0), line, font=font)
        d.text(((WIDTH - (bbox[2] - bbox[0])) // 2, y), line, font=font, fill=(255, 255, 255))
        y += line_h
    return pil


def add_watermark(img: Image.Image) -> Image.Image:
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = _font(21)
    bbox = d.textbbox((0, 0), WATERMARK_TEXT, font=font)
    d.text((WIDTH - (bbox[2] - bbox[0]) - 26, HEIGHT - (bbox[3] - bbox[1]) - 14),
           WATERMARK_TEXT, font=font, fill=(255, 255, 255, 120))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def compose(scene: dict) -> Image.Image:
    kind = scene["kind"]
    if kind == "img":
        base = load_shot(SHOTS / scene["src"])
    elif kind == "opening":
        base = render_opening()
    elif kind == "limits":
        base = render_limits()
    elif kind == "closing":
        base = render_closing()
    else:
        base = render_card(scene)
    return add_watermark(add_subtitle(base, scene["subtitle"]))


# ── 音频 ─────────────────────────────────────────────────────────
def estimate_duration(text: str) -> float:
    return max(6.0, min(22.0, len(text) / 5.0 + 2.0))


def make_audio(scenes: list[dict]) -> list[float]:
    import edge_tts

    TMP.mkdir(parents=True, exist_ok=True)
    durations = []
    for i, scene in enumerate(scenes):
        mp3 = TMP / f"scene_{i:02d}.mp3"
        wav = TMP / f"scene_{i:02d}.wav"
        if wav.exists():
            out = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(wav)], capture_output=True, text=True)
            cached = float(out.stdout.strip())
            # 缓存的时长必须已经满足本场景下限，否则重走一遍生成+补静音
            if cached >= max(scene.get("min_dur", 0.0), 0.0):
                durations.append(cached)
                print(f"  音频 {i:02d} {scene['subtitle'][:18]}  {cached:.1f}s (复用)")
                continue
        dur = 0.0
        try:
            asyncio.run(edge_tts.Communicate(
                scene["narration"], "zh-CN-XiaoxiaoNeural", rate="-5%").save(str(mp3)))
            out = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(mp3)], capture_output=True, text=True)
            dur = float(out.stdout.strip())
        except Exception as exc:  # 网络不可用时退化为无旁白
            print(f"  [warn] 场景 {i} TTS 失败: {exc}")
        display = max(estimate_duration(scene["narration"]), dur + 0.55,
                      scene.get("min_dur", 0.0))
        if mp3.exists() and dur > 0:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                 "-af", "apad", "-t", f"{display:.2f}", "-ar", "48000", "-ac", "1",
                 "-c:a", "pcm_s16le", str(wav)], check=True)
        else:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                 "-i", "anullsrc=r=48000:cl=mono", "-t", f"{display:.2f}",
                 "-c:a", "pcm_s16le", str(wav)], check=True)
        durations.append(display)
        print(f"  音频 {i:02d} {scene['subtitle'][:18]}  {display:.1f}s")
    return durations


def concat_audio(count: int, total: float) -> Path:
    lst = TMP / "audio.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for i in range(count):
            f.write(f"file '{(TMP / f'scene_{i:02d}.wav').as_posix()}'\n")
    out = TMP / "narration.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-af", "apad", "-t", f"{total + 0.8:.2f}",
                    "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def generate_bgm(total: float) -> Path:
    out = TMP / "bgm.wav"
    def tone(freq: float, vol: float) -> str:
        return (f"sine=frequency={freq}:duration={total:.2f},"
                f"afade=t=in:st=0:d=3,afade=t=out:st={max(total - 3, 1):.2f}:d=3,volume={vol}")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", tone(220, 0.10),
         "-f", "lavfi", "-i", tone(330, 0.06),
         "-f", "lavfi", "-i", tone(165, 0.05),
         "-filter_complex",
         "[0][1][2]amix=inputs=3:duration=longest:dropout_transition=3,"
         f"atrim=duration={total:.2f},afade=t=in:st=0:d=4,"
         f"afade=t=out:st={max(total - 4, 1):.2f}:d=4",
         "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(out)], check=True)
    return out


# ── 覆盖校验 ─────────────────────────────────────────────────────
def verify_coverage() -> None:
    covered = [fid for s in SCENES for fid in s["ids"]]
    missing = [fid for fid in ALL_IDS if fid not in set(covered)]
    stray = sorted(set(covered) - set(ALL_IDS))
    if missing or stray:
        print(f"[FAIL] 覆盖不完整：缺 {len(missing)} 条 {missing}；多出 {stray}")
        raise SystemExit(1)
    print(f"[OK] 覆盖 {len(ALL_IDS)}/{len(ALL_IDS)} 条功能，无遗漏、无杜撰")


def write_coverage_report() -> None:
    rows = []
    per_id = {}
    for i, scene in enumerate(SCENES, 1):
        name = scene["src"] if scene["kind"] == "img" else {
            "card": "清单卡", "opening": "片头", "limits": "诚实边界", "closing": "片尾",
        }[scene["kind"]]
        label = f"{scene['chapter']} · {name}" if name == "清单卡" else f"{scene['chapter']} · {name}"
        for fid in scene["ids"]:
            per_id.setdefault(fid, []).append(i)
        rows.append(f"| {i:02d} | {label} | {scene['subtitle']} | {len(scene['ids'])} |")
    lines = [
        "# 掬水功能全览视频 · 覆盖对照",
        "",
        f"- 视频：`{OUTPUT.name}`",
        f"- 场景数：{len(SCENES)}",
        f"- 清单条目：{len(ALL_IDS)}（SHELL 11 / CHAT 18 / MOD 16 / DL 10 / MON 9 / SRV 8 / SET 12 / ARC 10 / ENG 19 / API 9 / CLI 5 / THEME 8 / DB+SEC 11）",
        "- 校验方式：每个场景显式声明覆盖的功能 ID，并集与《功能清单.md》解析结果逐条比对，缺一即中止渲染。",
        "",
        "| 场景 | 画面 | 字幕 | 覆盖条数 |",
        "|---|---|---|---|",
        *rows,
        "",
        "## 每条功能出现在哪些场景",
        "",
        "| 功能 ID | 场景 |",
        "|---|---|",
    ]
    for fid in ALL_IDS:
        idxs = "、".join(f"{n:02d}" for n in per_id[fid])
        lines.append(f"| {fid} | {idxs} |")
    COVERAGE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    COVERAGE_JSON.write_text(json.dumps(
        {"total": len(ALL_IDS), "scenes": [
            {"index": i, "chapter": s["chapter"], "kind": s["kind"],
             "src": s.get("src"), "subtitle": s["subtitle"], "ids": s["ids"]}
            for i, s in enumerate(SCENES, 1)], "per_id": per_id},
        ensure_ascii=False, indent=2), encoding="utf-8")


# ── 渲染 ─────────────────────────────────────────────────────────
def main() -> None:
    if not SHOTS.exists():
        raise SystemExit(f"缺少截图目录: {SHOTS}")
    verify_coverage()

    # 把清单条目注入卡片场景
    for scene in SCENES:
        if scene["kind"] == "card":
            scene["ids_items"] = list(INV[scene["chapter"]])
            scene["ids"] = [fid for fid, _ in scene["ids_items"]]

    TMP.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print(f"  掬水功能全览 · {len(SCENES)} 场景 · {len(ALL_IDS)} 条功能")
    print("=" * 64)

    print("\n[1/5] 旁白音频...")
    durations = make_audio(SCENES)
    total = sum(durations)
    print(f"  总时长 {total / 60:.1f} 分钟")

    narration = concat_audio(len(SCENES), total)

    print("\n[2/5] 构建帧...")
    tmp_video = TMP / "video_only.mp4"
    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit("VideoWriter 打开失败")
    prev = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i, scene in enumerate(SCENES):
        # cv2.VideoWriter 按 BGR 解释数组，PIL 出来的是 RGB，必须换序
        frame = cv2.cvtColor(np.array(compose(scene)), cv2.COLOR_RGB2BGR)
        n = max(int(durations[i] * FPS), FADE + 2)
        for j in range(FADE):
            a = j / max(FADE - 1, 1)
            writer.write(cv2.addWeighted(prev, 1 - a, frame, a, 0))
        for _ in range(n - FADE):
            writer.write(frame)
        prev = frame
        print(f"  场景 {i:02d} {scene['subtitle'][:22]}  {durations[i]:.1f}s")

    # 末尾 0.8s 淡出
    for j in range(int(0.8 * FPS)):
        a = j / max(int(0.8 * FPS) - 1, 1)
        writer.write(cv2.addWeighted(prev, 1 - a, np.zeros_like(prev), a, 0))
    writer.release()

    print("\n[3/5] 背景音乐...")
    bgm = generate_bgm(total + 0.8)

    print("\n[4/5] 混音与编码...")
    final_tmp = TMP / "final.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(tmp_video), "-i", str(narration), "-i", str(bgm),
         "-filter_complex",
         "[1:a]volume=1.0[narration];[2:a]volume=0.14[bgm];"
         "[narration][bgm]amix=inputs=2:duration=first:dropout_transition=3[mixed]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "22",
         "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "192k",
         "-map", "0:v", "-map", "[mixed]", "-shortest", str(final_tmp)], check=True)

    print("\n[5/5] 落盘与校验...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(final_tmp, OUTPUT)
    write_coverage_report()
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries",
         "format=duration,size:stream=codec_name,width,height,r_frame_rate",
         "-of", "json", str(OUTPUT)], capture_output=True, text=True)
    print(probe.stdout)
    print(f"  输出: {OUTPUT}")
    print(f"  覆盖对照: {COVERAGE_MD}")


if __name__ == "__main__":
    main()
