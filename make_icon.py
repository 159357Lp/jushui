"""生成 LM Loader 应用图标（.ico 多尺寸 + .png 预览）。

设计约束
--------
1. 沿用 GUI 既有视觉语言：24x24 逻辑网格、圆头圆角、主色 #2E6BE6
   （见 window.py 的 _make_nav_icon 与 styles.py）。
2. 小尺寸可读性优先：16x16 是资源管理器与任务栏的实际显示尺寸，
   任何细节在此尺寸下必须仍能辨识。因此：
   - 超采样绘制（8x）后 LANCZOS 缩放，避免直接画小图产生锯齿
   - 元素数量压到最低，细节在小尺寸下会互相糊成一团
   - 主体轮廓占画布 70% 以上，四周留边保证不被裁切
3. 语义：'loader' = 把模型装进机器。图形 = 堆叠的模型层 + 下方托盘缺口，
   一眼能读成'加载'而非泛泛的芯片。

输出
----
lmloader_icon.ico   16/24/32/48/64/128/256 多尺寸（Windows 快捷方式用）
icon_256.png         预览 / 文档 / 高分屏
icon_32.png / icon_16.png  单独导出便于调试
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw

OUT_DIR = r"D:\模型加载器\lmloader"

# ---- 设计 token（与 GUI 保持一致）----
ACCENT = (46, 107, 230)        # #2E6BE6 主色
ACCENT_DEEP = (30, 78, 174)    # 底部渐深
ACCENT_LIGHT = (108, 160, 245) # 顶部提亮
DARK = (21, 26, 44)            # #151A2C 侧边栏深色，用作底板
WHITE = (255, 255, 255)
GHOST = (196, 214, 248)        # 次级层的淡蓝灰

SS = 8          # 超采样倍数
CANVAS = 24     # 逻辑网格
BASE = CANVAS * SS


def draw_icon() -> Image.Image:
    """Draw at BASE x BASE then the caller downsamples."""
    S = SS
    img = Image.new("RGBA", (BASE, BASE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def px(v):
        return v * S

    # ---- 底板：圆角方块，带轻微垂直渐变 ----
    # 直接画渐变需要逐行填充，先画圆角遮罩再合成渐变。
    plate = Image.new("RGBA", (BASE, BASE), (0, 0, 0, 0))
    pd = ImageDraw.Draw(plate)
    radius = px(5.5)
    pd.rounded_rectangle([0, 0, BASE - 1, BASE - 1], radius=radius, fill=DARK + (255,))

    grad = Image.new("RGBA", (BASE, BASE), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    for y in range(BASE):
        t = y / max(1, BASE - 1)
        # 深色在上、亮蓝在下：视觉重心落在图形主体（下半），
        # 不会像上亮下暗那样把视线带出图标。底部亮蓝也呼应 GUI 主色。
        if t < 0.45:
            k = t / 0.45
            col = tuple(int(DARK[i] + (ACCENT_DEEP[i] - DARK[i]) * k * 0.9) for i in range(3))
        else:
            k = (t - 0.45) / 0.55
            col = tuple(int(ACCENT_DEEP[i] + (ACCENT[i] - ACCENT_DEEP[i]) * k) for i in range(3))
        gd.line([(0, y), (BASE, y)], fill=col + (255,))

    plate.paste(grad, (0, 0), grad)
    img.alpha_composite(plate)

    # ---- 主体：两块已装载的模型 + 一个底座托盘 ----
    # v1 用了三层堆叠，32x32 下三层糊成一条带（描边层与实心层分不开）。
    # 改为两层实心 + 底座：元素少一半，但每笔都足够粗，小尺寸仍可读。
    #
    # 层 1（已装载，最粗）
    y1 = px(12.4)
    d.rounded_rectangle([px(4.6), y1, px(19.4), y1 + px(3.6)],
                        radius=px(1.3), fill=ACCENT_LIGHT + (255,))
    # 层 2（已装载，次粗）—— 深一档拉开层次
    y2 = px(16.8)
    d.rounded_rectangle([px(4.6), y2, px(19.4), y2 + px(3.6)],
                        radius=px(1.3), fill=ACCENT + (255,))
    # 底座托盘：比两层略宽，明确'承托'关系，同时暗示机器/机箱
    yb = px(21.0)
    d.rounded_rectangle([px(3.4), yb, px(20.6), yb + px(1.9)],
                        radius=px(0.85), fill=(255, 255, 255, 205))

    # ---- 装载箭头：向下进入最上层 ----
    # 纯白箭头在 16x16 下三角轮廓会被抗锯齿啃掉（v2 实测：糊成一团亮点）。
    # 修法：先画一层加粗的深色描边再叠白色填充，相当于给箭头描一圈
    # '负空间'，让三角的斜边在深底上重新出现。
    cx = px(12.0)
    arrow_w = px(3.1)
    outline_col = DARK + (215,)

    # 短杆（含描边）
    d.rounded_rectangle(
        [cx - px(1.35), px(1.0), cx + px(1.35), px(5.3)],
        radius=px(0.95), fill=outline_col,
    )
    d.rounded_rectangle(
        [cx - px(0.78), px(1.6), cx + px(0.78), px(4.7)],
        radius=px(0.55), fill=WHITE + (245,),
    )

    # 三角头（含描边）：描边版比白色版整体外扩
    tri_outline = [
        (cx - px(4.15), px(4.05)),
        (cx + px(4.15), px(4.05)),
        (cx, px(8.15)),
    ]
    d.polygon(tri_outline, fill=outline_col)
    d.polygon([
        (cx - arrow_w, px(4.0)),
        (cx + arrow_w, px(4.0)),
        (cx, px(7.4)),
    ], fill=WHITE + (245,))

    return img


def main() -> None:
    master = draw_icon()
    master.save(os.path.join(OUT_DIR, "_icon_master.png"))

    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = []
    for s in sizes:
        # LANCZOS 对缩放边缘更干净；16/24 这类小尺寸再过一遍轻量对比度提升
        img = master.resize((s, s), Image.LANCZOS)
        frames.append(img)
        img.save(os.path.join(OUT_DIR, f"icon_{s}.png"))

    ico_path = os.path.join(OUT_DIR, "lmloader_icon.ico")
    # Pillow 的 ico 保存：传入最大尺寸的帧列表即可写入多分辨率
    frames[-1].save(ico_path, format="ICO",
                    sizes=[(s, s) for s in sizes])

    # 校验
    with Image.open(ico_path) as ico:
        got = sorted(ico.info.get("sizes", []))
    print(f"ico written: {ico_path}")
    print(f"ico frames : {got}")
    print(f"png master : {os.path.join(OUT_DIR, '_icon_master.png')} "
          f"({master.size[0]}x{master.size[1]})")


if __name__ == "__main__":
    main()
