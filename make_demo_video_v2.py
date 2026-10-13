"""LM Loader 演示视频生成器 — 截图 + 旁白 + 字幕 + 开场结尾 + 背景音乐 + 水印。"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 路径 ──
SRC_DIR = Path("D:/模型加载器/lmloader/demo_screenshots/新建文件夹")
OUT_DIR = Path("D:/模型加载器/lmloader/demo_screenshots")
AUDIO_DIR = OUT_DIR / "audio"
OUTPUT = OUT_DIR / "lmloader_final_h264.mp4"

WIDTH, HEIGHT = 1920, 1080
FPS = 30

WATERMARK_TEXT = "不雨花犹落"

# ── 分镜: (截图文件名, 旁白台词, 字幕行) ──
SCENES = [
    (
        "lmloader_opening.png",
        "LM Loader，桌面端 AI 模型加载器，平替 Ollama 加 LM Studio，集成三进制 llama-server 引擎。",
        "LM Loader — 桌面端 AI 模型加载器，平替 Ollama + LM Studio，集成三进制 llama-server 引擎",
    ),
    (
        "1.png",
        "打开 LM Loader，首先是聊天工作台。左侧管理会话历史，右侧是对话区，支持 Markdown 渲染和流式输出。",
        "聊天工作台 — 会话管理 + 流式对话",
    ),
    (
        "2.png",
        "切换到模型管理页。这里列出了所有已扫描的 GGUF 模型，显示参数量、量化类型和文件大小，一键加载，一键卸载。",
        "模型管理 — 6 个模型，一键加载卸载",
    ),
    (
        "3.png",
        "下载中心支持双源获取。在 HuggingFace 标签页输入关键词，就能搜索并浏览所有相关 GGUF 文件。",
        "下载中心 — HuggingFace 模型搜索",
    ),
    (
        "4.png",
        "搜索 qwen 的结果，从零点几 B 到几十 B 的模型都有，共找到 20 个模型、39 个 GGUF 文件，一键下载。",
        "HuggingFace 搜索 qwen — 39 个 GGUF 文件",
    ),
    (
        "5.png",
        "同时也支持 Ollama 模型库。搜索 deepseek，展开版本列表，选择需要的量化版本直接拉取。",
        "Ollama 模型库 — deepseek 系列",
    ),
    (
        "6.png",
        "资源监控页面实时显示 CPU、内存、GPU 和显存使用情况，下方展示引擎状态和已加载模型信息。",
        "资源监控 — CPU / 内存 / GPU / 显存实时曲线",
    ),
    (
        "7.png",
        "本地服务页面展示服务器运行状态。模型加载后显示推理速度 128 tokens 每秒，MTP 命中率 51%，终端日志实时滚动。",
        "本地服务 — 128.2 t/s · MTP 命中率 51%",
    ),
    (
        "8.png",
        "设置页面可以配置双引擎路径、端口范围、Flash Attention 等参数，支持一键更新引擎和一键回退。",
        "设置 — 双引擎配置 + 一键更新回退",
    ),
    (
        "9.png",
        "回到聊天页，发送一张图片让模型分析。qwen35-9B 以 128 tokens 每秒的速度输出详细的图像分析结果。",
        "视觉对话 — 128.2 t/s 图像分析",
    ),
    (
        "lmloader_closing.png",
        "LM Loader，MIT 开源协议。欢迎访问 Gitee 仓库，gitee.com/buyuhuayouluo/lmloader。",
        "MIT 开源 · gitee.com/buyuhuayouluo/lmloader",
    ),
]


def load_resize(path: Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    scale = min(WIDTH / w, HEIGHT / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    y0 = (HEIGHT - new_h) // 2
    x0 = (WIDTH - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _find_chinese_font() -> str | None:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


FONT_PATH = _find_chinese_font()


def add_subtitle(frame: np.ndarray, text: str) -> np.ndarray:
    """底部半透明黑条 + 白色中文字幕。"""
    overlay = frame.copy()
    bar_h = 80
    y_start = HEIGHT - bar_h - 20
    cv2.rectangle(overlay, (0, y_start), (WIDTH, HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    font_size = 28
    font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()

    max_chars = 50
    lines = []
    while len(text) > max_chars:
        split = text[:max_chars]
        for sep in ['，', '。', '、', '；']:
            idx = split.rfind(sep)
            if idx > max_chars // 2:
                split = text[:idx + len(sep)]
                break
        else:
            for sep in [' ', '+', '-', '/']:
                idx = split.rfind(sep)
                if idx > max_chars // 2:
                    split = text[:idx + len(sep)]
                    break
        lines.append(split)
        text = text[len(split):]
    lines.append(text)

    line_h = font_size + 8
    total_h = len(lines) * line_h
    y = y_start + (bar_h - total_h) // 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (WIDTH - tw) // 2
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += line_h

    return np.array(pil_img)


def add_watermark(frame: np.ndarray) -> np.ndarray:
    """右下角半透明水印「不雨花犹落」。"""
    pil_img = Image.fromarray(frame).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    font_size = 22
    font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()

    text = WATERMARK_TEXT
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = WIDTH - tw - 30
    y = HEIGHT - th - 30

    # 半透明白色文字
    draw.text((x, y), text, fill=(255, 255, 255, 160), font=font)

    return np.array(pil_img.convert("RGB"))


async def generate_audio(scenes: list[tuple[str, str, str]]) -> list[Path]:
    """为每段台词生成 TTS 音频。"""
    import edge_tts

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_files = []
    voice = "zh-CN-XiaoxiaoNeural"

    for i, (_, narration, _) in enumerate(scenes):
        out_path = AUDIO_DIR / f"scene_{i:02d}.mp3"
        communicate = edge_tts.Communicate(narration, voice, rate="-5%")
        await communicate.save(str(out_path))
        audio_files.append(out_path)
        print(f"  音频 {i}: {out_path.name}")

    return audio_files


def get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def generate_bgm(duration: float) -> Path:
    """用 ffmpeg 合成一段柔和的环境背景音乐。"""
    bgm_path = AUDIO_DIR / "bgm.mp3"
    # 多层 sine 波 + 慢速音量调制 = 柔和 pad 音色
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        f"sine=frequency=220:duration={duration:.1f},"
        f"afade=t=in:st=0:d=3,"
        f"afade=t=out:st={duration - 3:.1f}:d=3,"
        f"volume=0.12,"
        f"apad",
        "-f", "lavfi", "-i",
        f"sine=frequency=330:duration={duration:.1f},"
        f"afade=t=in:st=0:d=3,"
        f"afade=t=out:st={duration - 3:.1f}:d=3,"
        f"volume=0.08,"
        f"apad",
        "-f", "lavfi", "-i",
        f"sine=frequency=165:duration={duration:.1f},"
        f"afade=t=in:st=0:d=3,"
        f"afade=t=out:st={duration - 3:.1f}:d=3,"
        f"volume=0.06,"
        f"apad",
        "-filter_complex",
        "[0][1][2]amix=inputs=3:duration=longest:dropout_transition=3,"
        f"atrim=duration={duration:.1f},afade=t=in:st=0:d=4,afade=t=out:st={duration - 4:.1f}:d=4",
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(bgm_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"  背景音乐生成: {bgm_path.name} ({duration:.1f}s)")
    return bgm_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LM Loader 演示视频生成器")
    print("=" * 60)

    # 1. 生成旁白音频
    print("\n[1/5] 生成旁白音频...")
    audio_files = asyncio.run(generate_audio(SCENES))

    # 2. 获取每段音频时长
    print("\n[2/5] 计算时长...")
    durations = []
    for i, af in enumerate(audio_files):
        dur = get_audio_duration(af)
        display_dur = dur + 0.5
        durations.append(display_dur)
        print(f"  场景 {i}: 音频 {dur:.1f}s, 显示 {display_dur:.1f}s")

    total_dur = sum(durations)
    print(f"  总时长: {total_dur:.1f}s")

    # 3. 构建视频帧（含字幕 + 水印）
    print("\n[3/5] 构建视频帧...")
    all_frames = []
    FADE_FRAMES = int(0.6 * FPS)

    for i, (img_name, narration, subtitle) in enumerate(SCENES):
        img_path = SRC_DIR / img_name
        if not img_path.exists():
            print(f"  [跳过] {img_name} 不存在")
            continue

        print(f"  场景 {i}: {img_name} ({durations[i]:.1f}s)")
        img = load_resize(img_path)
        img = add_subtitle(img, subtitle)
        img = add_watermark(img)

        n_frames = int(durations[i] * FPS)

        if all_frames:
            prev = all_frames[-1]
            for j in range(FADE_FRAMES):
                alpha = j / max(FADE_FRAMES - 1, 1)
                blended = cv2.addWeighted(prev, 1 - alpha, img, alpha, 0)
                all_frames.append(blended)
            hold = n_frames - FADE_FRAMES
        else:
            black = np.zeros_like(img)
            for j in range(FADE_FRAMES):
                alpha = j / max(FADE_FRAMES - 1, 1)
                blended = cv2.addWeighted(black, 1 - alpha, img, alpha, 0)
                all_frames.append(blended)
            hold = n_frames - FADE_FRAMES

        all_frames.extend([img] * max(hold, 1))

    print(f"  总帧数: {len(all_frames)}")

    # 4. 写入视频
    print("\n[4/5] 写入视频...")
    tmp_video = OUT_DIR / "tmp_video_only.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, FPS, (WIDTH, HEIGHT))
    for frame in all_frames:
        writer.write(frame)
    writer.release()
    print(f"  视频帧写入完成: {tmp_video.name}")

    # 拼接旁白音频
    concat_list = OUT_DIR / "audio_concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for af in audio_files:
            safe_path = str(af).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    tmp_audio = OUT_DIR / "tmp_audio.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c", "copy", str(tmp_audio)],
            capture_output=True, check=True,
        )
        print(f"  旁白拼接完成: {tmp_audio.name}")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="replace")[:500] if e.stderr else "unknown"
        print(f"  旁白拼接失败: {err_msg}")
        tmp_audio = None

    # 5. 生成背景音乐 + 混音
    print("\n[5/5] 背景音乐 + 最终合成...")
    bgm_path = generate_bgm(total_dur)

    final_h264 = OUTPUT.with_name("lmloader_final_h264.mp4")

    if tmp_audio and tmp_audio.exists():
        # 旁白 + 背景音乐混合
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(tmp_video),
             "-i", str(tmp_audio),
             "-i", str(bgm_path),
             "-filter_complex",
             "[1:a]volume=1.0[narration];"
             "[2:a]volume=0.15[bgm];"
             "[narration][bgm]amix=inputs=2:duration=first:dropout_transition=3[mixed]",
             "-c:v", "libx264", "-preset", "medium", "-crf", "23",
             "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k",
             "-map", "0:v", "-map", "[mixed]",
             "-shortest", str(final_h264)],
            capture_output=True, check=True,
        )
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_video),
             "-i", str(bgm_path),
             "-filter_complex", "[1:a]volume=0.15[bgm]",
             "-c:v", "libx264", "-preset", "medium", "-crf", "23",
             "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k",
             "-map", "0:v", "-map", "[bgm]",
             "-shortest", str(final_h264)],
            capture_output=True, check=True,
        )

    # 清理临时文件
    tmp_video.unlink(missing_ok=True)
    if tmp_audio:
        tmp_audio.unlink(missing_ok=True)
    concat_list.unlink(missing_ok=True)

    size_mb = final_h264.stat().st_size / 1048576
    print(f"\n{'=' * 60}")
    print(f"  完成! 视频大小: {size_mb:.1f}MB")
    print(f"  输出: {final_h264}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
