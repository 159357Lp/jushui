"""速度档案页离屏渲染：确认展开一行真的画出三张卡、"该引擎不传"和"其他参数"看得见。

不碰真库也不起服务：把 db 指到临时文件，塞几条合成记录，页面读的是真 get_run_stats。
"""
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFontMetrics  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QFrame, QLabel, QPlainTextEdit, QPushButton,
)

from lmloader.engine import run_stats  # noqa: E402
from lmloader.infra.database import db  # noqa: E402

HERE = Path(__file__).resolve().parent
TMP_DB = HERE / "_run_stats_render.db"

ARGS = [
    "-m", "D:/模型/Bonsai-27B-NVFP4-MTP.gguf", "--port", "8100", "--host", "127.0.0.1",
    "--n-gpu-layers", "999", "-c", "131072", "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0", "--flash-attn", "on", "--parallel", "1",
    "--threads", "8", "--reasoning", "off", "-b", "2048", "-ub", "512",
    "--cache-ram", "8192", "--spec-type", "draft-mtp", "--spec-draft-n-max", "1",
    "--some-brand-new-knob", "7",
]

app = QApplication([])

if TMP_DB.exists():
    TMP_DB.unlink()
db.db_path = TMP_DB
asyncio.run(db.init())


def rec(name, tps, ctx, args=None, env=None, engine="ternary", build="prism-stable"):
    a = list(args if args is not None else ARGS)
    if "-c" in a:
        a[a.index("-c") + 1] = str(ctx)
    snap = {"prompt_tokens": 3072, "cached_tokens": 1024}
    if tps is not None:
        snap.update({"tg_final": tps, "prompt_tps": tps * 9.0,
                     "gen_tokens": int(tps * 4)})
    return run_stats.build_record(
        model_name=name, args=a, engine_kind=engine, build=build,
        snap=snap, env=env,
    )


# 2026-10-21 实测那两个档：权重全量上卡 + 16k 是快档，同样配到 64k 就塌到 5 t/s，
# 塌陷那一轮的 prompt 缓存占用 6 条 / 4754 MiB（上限 8192）。渲染出来是为了看
# "判别量能不能直接从表上读"，不是看数字好不好看。
def occ(mib, prompts, limit=8192.0):
    return {"prompts": prompts, "mib": mib, "limit_mib": limit,
            "limit_tokens": 16128, "est": 136629}


def split(vram_w, ram_w):
    return {"vram": {"weights": vram_w, "kv": 2304.0},
            "ram": {"weights": ram_w}, "devices": {}, "total_mib": {}}


for r in [
    rec("Bonsai-27B-NVFP4", 104.0, 8192, env={
        "knobs": {"GGML_CUDA_PDL": "1", "OMP_NUM_THREADS": "8"},
        "co_resident": [], "cache": occ(536.5, 1), "mem": split(14549.4, 0.0)}),
    rec("Bonsai-27B-NVFP4", 5.4, 65536, env={
        "knobs": {}, "co_resident": [],
        "cache": occ(4754.3, 6), "mem": split(14549.4, 0.0)}),
    rec("Bonsai-27B-NVFP4", 52.0, 131072, env={
        "knobs": {"GGML_CUDA_DISABLE_GRAPHS": "1"},
        "co_resident": ["Qwen3-30B-A3B", "d3-mini"],
        "cache": occ(1290.0, 2), "mem": split(11613.0, 2936.0)}),
    rec("Bonsai-27B-NVFP4", 88.0, 32768, env={"knobs": {}, "co_resident": []}),
    rec("Bonsai-27B-NVFP4", None, 16384),
    rec("DeepSeek-R1-vLLM", 12.0, 32768, engine="vllm", build="vllm", args=[
        "serve", "/models/r1", "--host", "127.0.0.1", "--port", "8110",
        "--served-model-name", "r1", "--gpu-memory-utilization", "0.85",
        "--max-model-len", "32768",
    ]),
]:
    asyncio.run(db.add_run_stat(r))

from lmloader.gui.api_client import LMLoaderAPI  # noqa: E402
from lmloader.gui.pages.run_stats import RunStatsPage, _ConfigCard  # noqa: E402

stats = asyncio.run(db.get_run_stats(recent=10))
page = RunStatsPage(LMLoaderAPI())
page._render_archive({"models": stats, "total": sum(r["count"] for r in stats)})
page._render_live([{"model": {"name": "Bonsai-27B-NVFP4"},
                    "metrics": {"tg_final": 5.4, "n_gen": 927},
                    "cache_state": occ(4754.3, 6),
                    "memory": split(14549.4, 0.0)}])

# 展开 Bonsai 那一行：只走 rows_layout 里的行控件，别把滚动区外壳当行
clicked = False
for i in range(page.rows_layout.count()):
    row = page.rows_layout.itemAt(i).widget()
    if row is None:
        continue
    if not any(label.text() == "Bonsai-27B-NVFP4"
               for label in row.findChildren(QLabel)):
        continue
    btn = row.findChild(QPushButton)
    assert btn is not None, "行里没找到展开按钮"
    btn.click()
    clicked = True
    break
assert clicked, "没找到 Bonsai 那一行，展开按钮没点上"
page.resize(1600, 1020)
page.show()
for _ in range(4):
    app.processEvents()

out = HERE / "run_stats_page.png"
page.grab().save(str(out))

cards = page.findChildren(_ConfigCard)
texts = " ".join(label.text() for label in page.findChildren(QLabel))
boxes = page.findChildren(QPlainTextEdit)
print("saved:", out)
print("状态栏:", page.status_label.text())
print("模型行数(展开按钮数):", len(page.findChildren(QPushButton)))
print("展开后卡片:", len(cards), [c.title() for c in cards])
print("卡片宽度:", [c.width() for c in cards], "页面", page.width(), "x", page.height())
print("argv 框:", [(b.width(), b.height(),
                    round(b.document().size().height()
                          / max(b.fontMetrics().lineSpacing(), 1), 1),
                    b.verticalScrollBar().maximum()) for b in boxes])
print("argv 全可见(滚动条无需拖动):",
      all(b.verticalScrollBar().maximum() == 0 for b in boxes))
print("含「该引擎不传」:", run_stats.NOT_PASSED in texts)
print("含「其他参数」值:", "--some-brand-new-knob 7" in texts)
print("含 GGML 旋钮:", "GGML_CUDA_PDL=1" in texts)
print("含共存名单:", "Qwen3-30B-A3B" in texts, "| 只有它自己:", "只有它自己" in texts)
print("含没测到:", "没测到" in texts)
print("vLLM 行可见:", "DeepSeek-R1-vLLM" in texts)
elided = [label.text() for label in page.findChildren(QLabel)
          if not label.wordWrap() and label.text()
          and QFontMetrics(label.font()).elidedText(
              label.text(), Qt.ElideRight, max(label.width() - 2, 1)) != label.text()]
print("被省略号吃掉的标签:", elided[:6])

# 收起态单独拍一张，验证列表页不展开也读得出高/中/低
page2 = RunStatsPage(LMLoaderAPI())
page2._render_archive({"models": stats, "total": sum(r["count"] for r in stats)})
page2._render_live([{"model": {"name": "Bonsai-27B-NVFP4"},
                     "metrics": {"tg_final": 96.4, "n_gen": 512,
                                 "prompt_tokens": 3072, "acceptance": 0.71},
                     "cache_state": occ(4754.3, 6),
                     "memory": split(14549.4, 0.0)}])
page2.resize(1600, 460)
page2.show()
app.processEvents()
out2 = HERE / "run_stats_collapsed.png"
page2.grab().save(str(out2))
print("saved:", out2)

# 表格线的判据：新列真出现在行里，塌陷那轮带上崩标记，没记到的写成"没报"。
rows_text = [run_stats.table_row(v) for v in (stats[0].get("recent") or [])]
print("列数:", len(run_stats.TABLE_COLUMNS), run_stats.TABLE_COLUMNS)
for cells in rows_text:
    print("  " + " | ".join(f"{name}={cell}" for name, cell
                            in zip(run_stats.TABLE_COLUMNS, cells)
                            if name in ("生成 t/s", "命中 tok", "上下文", "缓存上限",
                                        "缓存占用", "权重 显存/内存", "同卡常驻")))

# 第三张：把窗口拉高到能装下"最近运行"那张 14 列表，验证新列在真实布局里读得出来。
page3 = RunStatsPage(LMLoaderAPI())
page3._render_archive({"models": stats, "total": sum(r["count"] for r in stats)})
page3._render_live([])
for i in range(page3.rows_layout.count()):
    row = page3.rows_layout.itemAt(i).widget()
    if row is None or not any(lb.text() == "Bonsai-27B-NVFP4"
                              for lb in row.findChildren(QLabel)):
        continue
    row.findChild(QPushButton).click()
    break
page3.resize(1600, 1500)
page3.show()
for _ in range(4):
    app.processEvents()
# 滚到底：三张卡 + argv 框吃满了高度，最近运行那张表在滚动区下方。
bar = page3.scroll.verticalScrollBar()
bar.setValue(bar.maximum())
for _ in range(4):
    app.processEvents()
out3 = HERE / "run_stats_table.png"
page3.grab().save(str(out3))
print("saved:", out3)
grid_labels = [lb.text() for lb in page3.findChildren(QLabel) if lb.text().startswith("最近")]
print("表头行含新列:", all(k in " ".join(t for t in [
      lb.text() for lb in page3.findChildren(QLabel)])
      for k in ("缓存占用", "权重 显存/内存", "命中 tok")))
print("表格线含崩标记:", any(lb.text().endswith(" 崩") for lb in page3.findChildren(QLabel)))
