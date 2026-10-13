"""将演示截图 + 开场/结尾图合成为 MP4 视频。

用法:
    python make_demo_video.py

输出: demo_screenshots/lmloader_demo.mp4
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEMO_DIR = Path(__file__).parent / "demo_screenshots"
VIBE_DIR = Path(__file__).parent.parent / "vibe_images"
OUTPUT = DEMO_DIR / "lmloader_demo.mp4"

# 每帧显示秒数
DURATION_OPEN = 3.0
DURATION_SHOT = 4.0
DURATION_CLOSE = 4.0
FPS = 30
WIDTH, HEIGHT = 1920, 1080


def find_latest_screenshots() -> list[Path]:
    """找到最新一轮截图（按序号 00-09 收集，取最近出现的完整序列）。"""
    shots = sorted(DEMO_DIR.glob("[0-9]*_[0-9][0-9]_*.png"))
    if not shots:
        return []
    # 按序号分组，取每个序号最近的文件
    by_index: dict[str, Path] = {}
    for s in shots:
        parts = s.stem.split("_", 2)
        if len(parts) >= 2:
            idx = parts[1]  # "00", "01", ...
            by_index[idx] = s  # 后面的覆盖前面的，保留最新的
    # 按序号排序
    result = [by_index[k] for k in sorted(by_index.keys())]
    return result


def find_opening_closing() -> tuple[Path | None, Path | None]:
    opening = None
    closing = None
    for f in VIBE_DIR.glob("lmloader_opening_*.png"):
        opening = f
    for f in VIBE_DIR.glob("lmloader_closing_*.png"):
        closing = f
    return opening, closing


def load_resize(path: Path) -> np.ndarray:
    """加载图片并 resize 到目标尺寸，保持比例加黑边。"""
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    scale = min(WIDTH / w, HEIGHT / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    # 居中放置到黑色画布
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    y0 = (HEIGHT - new_h) // 2
    x0 = (WIDTH - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def make_fade_frames(img: np.ndarray, n_frames: int, fade_in: bool = True) -> list[np.ndarray]:
    """生成淡入/淡出帧序列。"""
    frames = []
    for i in range(n_frames):
        alpha = i / max(n_frames - 1, 1) if fade_in else 1 - i / max(n_frames - 1, 1)
        alpha = max(0.0, min(1.0, alpha))
        black = np.zeros_like(img)
        blended = cv2.addWeighted(img, alpha, black, 1 - alpha, 0)
        frames.append(blended)
    return frames


def make_crossfade(img_a: np.ndarray, img_b: np.ndarray, n_frames: int) -> list[np.ndarray]:
    """生成交叉淡入淡出过渡帧。"""
    frames = []
    for i in range(n_frames):
        alpha = i / max(n_frames - 1, 1)
        blended = cv2.addWeighted(img_a, 1 - alpha, img_b, alpha, 0)
        frames.append(blended)
    return frames


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    """在画面底部添加文字标签。"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, HEIGHT - 60), (WIDTH, HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame, text, (40, HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


# 截图对应的中文标签
SHOT_LABELS = {
    "00_health": "Step 1: 服务健康检查",
    "01_engine": "Step 2: 双引擎状态",
    "02_model": "Step 3: 模型列表",
    "03_monitor": "Step 4: 加载前资源监控",
    "04_model_loaded": "Step 5: 加载模型 (Gemma-4-12B)",
    "05_monitor_after": "Step 6: 加载后资源监控",
    "06_chat": "Step 7: 流式对话",
    "07_event": "Step 8: 事件日志",
    "08_model_unloaded": "Step 9: 卸载模型",
    "09_monitor_final": "Step 10: 卸载后资源恢复",
}


def match_label(filename: str) -> str:
    for key, label in SHOT_LABELS.items():
        if key in filename:
            return label
    return ""


def main():
    opening, closing = find_opening_closing()
    shots = find_latest_screenshots()

    if not shots:
        print("未找到截图，请先运行 demo_auto.py")
        sys.exit(1)

    print(f"开场图: {opening}")
    print(f"结尾图: {closing}")
    print(f"截图 ({len(shots)} 张):")
    for s in shots:
        print(f"  {s.name}")

    # 构建帧序列
    all_frames: list[np.ndarray] = []
    FADE_FRAMES = int(0.5 * FPS)  # 0.5 秒过渡

    # 开场
    if opening:
        print("处理开场图...")
        img = load_resize(opening)
        all_frames.extend(make_fade_frames(img, FADE_FRAMES, fade_in=True))
        hold = int(DURATION_OPEN * FPS) - FADE_FRAMES
        all_frames.extend([img] * max(hold, 1))

    # 截图
    for shot in shots:
        label = match_label(shot.name)
        print(f"处理: {shot.name} -> {label}")
        img = load_resize(shot)
        if label:
            img = add_label(img, label)

        if all_frames:
            # 交叉淡入
            all_frames.extend(make_crossfade(all_frames[-1], img, FADE_FRAMES))
        else:
            all_frames.extend(make_fade_frames(img, FADE_FRAMES, fade_in=True))

        hold = int(DURATION_SHOT * FPS) - FADE_FRAMES
        all_frames.extend([img] * max(hold, 1))

    # 结尾
    if closing:
        print("处理结尾图...")
        img = load_resize(closing)
        if all_frames:
            all_frames.extend(make_crossfade(all_frames[-1], img, FADE_FRAMES))
        else:
            all_frames.extend(make_fade_frames(img, FADE_FRAMES, fade_in=True))
        hold = int(DURATION_CLOSE * FPS) - FADE_FRAMES
        all_frames.extend([img] * max(hold, 1))

    # 写入视频
    print(f"\n写入视频: {OUTPUT}")
    print(f"总帧数: {len(all_frames)}, 时长: {len(all_frames)/FPS:.1f}秒")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (WIDTH, HEIGHT))
    for frame in all_frames:
        writer.write(frame)
    writer.release()

    # 用 ffmpeg 重新编码为 H.264（兼容性更好）
    import subprocess
    h264_out = OUTPUT.with_name("lmloader_demo_h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(OUTPUT), "-c:v", "libx264", "-preset", "medium",
             "-crf", "23", "-pix_fmt", "yuv420p", str(h264_out)],
            capture_output=True, check=True,
        )
        print(f"H.264 版本: {h264_out}")
        OUTPUT.unlink()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg 不可用，保留 mp4v 格式")

    size_mb = OUTPUT.stat().st_size / 1048576 if OUTPUT.exists() else h264_out.stat().st_size / 1048576
    print(f"\n完成! 视频大小: {size_mb:.1f}MB")
    print(f"输出: {OUTPUT if OUTPUT.exists() else h264_out}")


if __name__ == "__main__":
    main()
