"""掬水 60-90 秒推广短片：实拍高光 + 面板不透明度滑杆的真实连续帧。

复用长版（make_demo_video_v3）的字体/截图/水印/旁白管线，只换节奏与字幕样式。
"""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import make_demo_video_v3 as base
from make_demo_video_v3 import HEIGHT, WIDTH, FPS, _card_bg, _font

TMP = Path(os.environ.get("TEMP", "C:/Users/75181/AppData/Local/Temp")) / "jushui_promo_short"
base.TMP = TMP

SHOTS = base.SHOTS
SWEEP = SHOTS / "alpha_sweep"
OUTPUT = base.OUT_DIR / "掬水推广短视频_2026-10-24.mp4"

ACCENT = base.ACCENT
TEXT = base.TEXT
MUTED = base.MUTED

SCENES = [
    {"kind": "opening", "min_dur": 5.0, "headline": "掬水", "tag": "",
     "narration": "掬水，桌面端本地大模型加载器。",
     "subtitle": "本地大模型 · 一键掬取"},
    {"kind": "img", "src": "01b_chat_conversation.png", "tag": "聊天工作台",
     "headline": "本地模型，直接聊", "min_dur": 7.0,
     "narration": "本地模型直接聊，流式输出，每轮附实测速度。",
     "subtitle": "流式 · Markdown · 每轮实测 t/s"},
    {"kind": "img", "src": "02_models.png", "tag": "模型管理",
     "headline": "扫到就能一键加载", "min_dur": 8.2,
     "narration": "扫描到的 GGUF 一键加载、一键卸载，视觉模型热切换。",
     "subtitle": "六列模型卡 · 视觉编码器热切换"},
    {"kind": "img", "src": "04_models_config.png", "tag": "运行配置",
     "headline": "每只模型 12 项参数", "min_dur": 7.6,
     "narration": "每只模型十二项运行配置，GPU 层数自动解算。",
     "subtitle": "上下文 · 层数 · MTP · KV 类型"},
    {"kind": "img", "src": "09b_run_stats_expanded.png", "tag": "速度档案",
     "headline": "每一次运行都留档", "min_dur": 8.2,
     "narration": "速度档案记下每一次运行的真实配置，高、中、低三档对照。",
     "subtitle": "真实配置卡 · 加载前空闲快照"},
    {"kind": "img", "src": "05_monitor.png", "tag": "资源监控",
     "headline": "显存花在哪，看得见", "min_dur": 7.2,
     "narration": "显存和内存分七类对账，谁占的一目了然。",
     "subtitle": "权重 / KV / 缓冲 / 引擎开销 分账"},
    {"kind": "seq", "tag": "主题 · 图片取色 + 壁纸",
     "headline": "面板不透明度，拖一下就懂",
     "narration": "面板不透明度可以拖：卡片变透，壁纸透上来，字还是清楚的。",
     "subtitle": "拖动即时预览，停手才落盘", "min_dur": 14.0},
    {"kind": "closing", "min_dur": 7.0, "headline": "146 条功能 · MIT 开源", "tag": "",
     "narration": "掬水，一百四十六条功能，MIT 开源。",
     "subtitle": "gitee.com/buyuhuayouluo/lmloader"},
]


def sweep_frames() -> list[Path]:
    """设置页 100%→0%→50% 的真实帧，再接聊天页三档对照。"""
    seq = [SWEEP / f"settings_{i:02d}.png" for i in range(31)]
    seq += [SWEEP / f"chat_{v:03d}.png" for v in (100, 50, 0)]
    return [p for p in seq if p.exists()]


def add_promo_caption(img: Image.Image, headline: str, tag: str, sub: str) -> Image.Image:
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # 底部渐变遮罩，保证任何画面上大标题都可读
    scrim_h = 360
    grad = Image.new("L", (1, scrim_h))
    for y in range(scrim_h):
        grad.putpixel((0, y), int(225 * (y / scrim_h) ** 1.35))
    d.bitmap((0, HEIGHT - scrim_h), grad.resize((WIDTH, scrim_h)), fill=(2, 6, 14, 255))
    d.rectangle([90, HEIGHT - 232, 96, HEIGHT - 152], fill=(*ACCENT, 255))
    d.text((120, HEIGHT - 236), headline, font=_font(52, True), fill=(255, 255, 255, 255))
    d.text((122, HEIGHT - 158), sub, font=_font(26), fill=(*ACCENT, 255))
    if tag:
        # 域标签放到底部遮罩内的同一行右侧，绝不压在界面内容上
        bbox = d.textbbox((0, 0), tag, font=_font(22))
        d.text((WIDTH - 120 - (bbox[2] - bbox[0]), HEIGHT - 154), tag,
               font=_font(22), fill=(*MUTED, 255))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def render_promo_opening(scene: dict) -> Image.Image:
    bg = _card_bg()
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((WIDTH // 2 - 210, 300), "掬水", font=_font(190, True), fill=(*TEXT, 255))
    d.rectangle([WIDTH // 2 - 90, 545, WIDTH // 2 + 90, 551], fill=(*ACCENT, 255))
    d.text((WIDTH // 2 - 330, 590), "桌面端本地大模型加载器 · v0.2.0",
           font=_font(32), fill=(*MUTED, 255))
    return Image.alpha_composite(bg, layer).convert("RGB")


def render_promo_closing(scene: dict) -> Image.Image:
    bg = _card_bg()
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((WIDTH // 2 - 330, 360), "146 条功能", font=_font(84, True), fill=(*TEXT, 255))
    d.text((WIDTH // 2 - 330, 470), "逐条有代码出处 · MIT 开源", font=_font(38), fill=(*ACCENT, 255))
    d.text((WIDTH // 2 - 330, 560), "gitee.com/buyuhuayouluo/lmloader", font=_font(30), fill=(*MUTED, 255))
    return Image.alpha_composite(bg, layer).convert("RGB")


def compose_one(scene: dict, src: Path | None) -> np.ndarray:
    if scene["kind"] == "img":
        img = base.load_shot(SHOTS / scene["src"])
    elif scene["kind"] == "opening":
        img = render_promo_opening(scene)
    elif scene["kind"] == "closing":
        img = render_promo_closing(scene)
    else:
        img = base.load_shot(src)
        # 聊天页三档对照帧放大一点，避免与设置页画面同尺寸时显得空
    img = add_promo_caption(img, scene["headline"], scene["tag"], scene["subtitle"])
    img = base.add_watermark(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def scene_frames(scene: dict) -> list[np.ndarray]:
    if scene["kind"] == "seq":
        return [compose_one(scene, p) for p in sweep_frames()]
    return [compose_one(scene, None)]


def write_bgm(total: float) -> Path:
    out = TMP / "bgm.wav"
    def tone(freq: float, vol: float, trem: str) -> str:
        return (f"sine=frequency={freq}:duration={total:.2f},{trem}"
                f"afade=t=in:st=0:d=1.5,afade=t=out:st={max(total - 2.5, 1):.2f}:d=2.5,"
                f"volume={vol}")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", tone(196, 0.10, "tremolo=f=1.8:d=0.55,"),
         "-f", "lavfi", "-i", tone(294, 0.055, "tremolo=f=3.6:d=0.4,"),
         "-f", "lavfi", "-i", tone(98, 0.07, ""),
         "-filter_complex",
         "[0][1][2]amix=inputs=3:duration=longest:dropout_transition=2,"
         f"atrim=duration={total:.2f},afade=t=in:st=0:d=1.5,"
         f"afade=t=out:st={max(total - 2.5, 1):.2f}:d=2.5",
         "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"  掬水推广短片 · {len(SCENES)} 场景")
    print("=" * 60)

    print("\n[1/4] 旁白...")
    durations = base.make_audio(SCENES)
    total = sum(durations)
    print(f"  总时长 {total:.1f}s ({total / 60:.2f} 分钟)")
    if not (58 <= total <= 92):
        print(f"[WARN] 时长 {total:.1f}s 不在 60-90 秒区间内")
    narration = base.concat_audio(len(SCENES), total)

    print("\n[2/4] 构建帧...")
    tmp_video = TMP / "video_only.mp4"
    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (WIDTH, HEIGHT))
    prev = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    fade = int(0.35 * FPS)
    for i, scene in enumerate(SCENES):
        frames = scene_frames(scene)
        n = max(int(durations[i] * FPS), fade + 2)
        for j in range(fade):
            a = j / max(fade - 1, 1)
            writer.write(cv2.addWeighted(prev, 1 - a, frames[0], a, 0))
        if len(frames) == 1:
            for _ in range(n - fade):
                writer.write(frames[0])
        else:
            hold = max((n - fade) // len(frames), 1)
            written = 0
            for f in frames:
                for _ in range(hold):
                    writer.write(f)
                    written += 1
            for _ in range(max(n - fade - written, 0)):
                writer.write(frames[-1])
        prev = frames[-1]
        print(f"  场景 {i} {scene['headline']}  {durations[i]:.1f}s  "
              f"({len(frames)} 个独特画面)")

    for j in range(int(0.5 * FPS)):
        a = j / max(int(0.5 * FPS) - 1, 1)
        writer.write(cv2.addWeighted(prev, 1 - a, np.zeros_like(prev), a, 0))
    writer.release()

    print("\n[3/4] 背景音乐 + 编码...")
    bgm = write_bgm(total + 0.5)
    final_tmp = TMP / "final.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(tmp_video), "-i", str(narration), "-i", str(bgm),
         "-filter_complex",
         "[1:a]volume=1.0[narration];[2:a]volume=0.20[bgm];"
         "[narration][bgm]amix=inputs=2:duration=first:dropout_transition=2[mixed]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "192k",
         "-map", "0:v", "-map", "[mixed]", "-shortest", str(final_tmp)], check=True)

    print("\n[4/4] 落盘...")
    import shutil
    shutil.copyfile(final_tmp, OUTPUT)
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries",
         "format=duration,size:stream=codec_name,width,height,r_frame_rate",
         "-of", "json", str(OUTPUT)], capture_output=True, text=True)
    print(probe.stdout)
    print(f"  输出: {OUTPUT}")


if __name__ == "__main__":
    main()
