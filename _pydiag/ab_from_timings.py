"""从 server.log 里把 A/B 两档的引擎实测 timings 取出来。

不靠 print_timing（这个 build 不打），靠 掬水 自己记录的响应 JSON 里的
timings 块。每个块归属到它前面最近的那次 load —— 跨 run 借证据是禁的。
"""
import json
import re
import sys

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
LOG = _args[0] if _args else "logs/server.log"

LOAD = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) [^\n]{0,120}?(显存 权重 \d+ \+ KV \d+.*?MiB[^\n]*)", re.M)
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ")

text = open(LOG, encoding="utf-8", errors="replace").read()

# 1) 找出所有 load 事件的字节位置 -> 决定每个 timings 属于哪一档
arms = []
for m in LOAD.finditer(text):
    arms.append({"ts": m.group(1), "pos": m.start(), "mem": m.group(2)})
print("load 事件:")
for a in arms:
    print(f"  {a['ts']}  {a['mem']}")

# 先扫一遍日志行首时间戳，建 pos->ts 索引，避免每个块重复全量扫描
stamps = [(c.start(), c.group(1))
          for c in re.finditer(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ", text, re.M)]


def ts_before(pos):
    lo, hi, hit = 0, len(stamps) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if stamps[mid][0] <= pos:
            hit = stamps[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return hit


# 2) 找出所有 timings 块
blocks = []
for m in re.finditer(r'"timings": \{', text):
    try:
        start = text.index("{", m.start())
        depth, i = 0, start
        while True:
            if i >= len(text) or i - start > 6000:      # 被截断的日志块，跳过
                raise ValueError("unterminated")
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        t = json.loads(text[start:i + 1])
    except Exception:
        continue
    ts = ts_before(m.start())
    blocks.append({"pos": m.start(), "ts": ts, "t": t})

# 只保留最后两次 load 之后的块（A、B 两档）
if len(arms) >= 2:
    bounds = [arms[-2]["pos"], arms[-1]["pos"]]
    sel = [("A 手动999", [b for b in blocks if bounds[0] <= b["pos"] < bounds[1]]),
           ("B 自动分层", [b for b in blocks if b["pos"] >= bounds[1]])]
else:
    sel = [("全部", blocks)]

print("\n每档引擎自报（predicted=解码, prompt=预填充）:")
for label, rows in sel:
    print(f"\n[{label}] 共 {len(rows)} 次生成")
    for b in rows:
        t = b["t"]
        print(f"  {b['ts']}  decode {t.get('predicted_per_second', 0):6.2f} t/s "
              f"({t.get('predicted_n')} tok)   prefill {t.get('prompt_per_second', 0):7.2f} t/s "
              f"({t.get('prompt_n')} tok)   接受 {t.get('draft_n_accepted')}/{t.get('draft_n')}")
    ds = [b["t"].get("predicted_per_second") for b in rows if b["t"].get("predicted_n", 0) > 50]
    if ds:
        print(f"  --> 解码均值 {sum(ds)/len(ds):.2f} t/s   （{len(ds)} 个满 200 token 的 run）")

# 3) 归属自检：每档应当恰好 1 次预热 + 3 次正式 = 4 个块。
#    --break 故意把两档都算到最后一个 load 之后，判据不变，必须报红。
BREAK = "--break" in sys.argv
EXPECT = 4
fails = []
for label, rows in (sel if not BREAK else [(sel[0][0], sel[0][1] + sel[1][1]),
                                           (sel[1][0], [])]):
    if len(rows) != EXPECT:
        fails.append(f"{label}: 期望 {EXPECT} 个 timings 块，实际 {len(rows)}")
if fails:
    print("\nRED:", "; ".join(fails))
    sys.exit(1)
print(f"\nGREEN: 两档各 {EXPECT} 个 timings 块，均归属到各自的 load 区间")
