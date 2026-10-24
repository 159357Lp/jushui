"""闭环标定：拿卡上实测的常驻账，替掉自动分层里那个 ×0.9104 的手感折扣。

今天产品里的解算是 `预算 = 空闲显存×0.9104 − 固定开销596 − 状态超出449`。
0.9104 是拍下来的：在 15492MB 空闲上一口吃掉 1388MB ≈ 6.7 层，实测把旗舰从 66 层
（短题 63.8 t/s）压到 61 层（27 t/s 量级）。本脚本不跟它讲道理，直接量四件事：

  * 每层真实值多少（引擎自报『显存 权重』逐层差分 vs parser 张量表实测）；
  * 每个 token 的 ctx 真实值多少（拟合 ctx 单价 vs 规划器自己的 KV 单价）；
  * 状态＋计算缓冲随 ctx 的增长（规划器把它当常数扣，实测 16k→64k 涨 331MB）；
  * 失败边界在哪一 MB（自报合计超出加载前空闲多少就跑，超出多少就死）。

标定模型是引擎自报合计的线性拟合 `self_total = A + P·ngl + Q·ctx`，规则
`A + P·ngl + Q·ctx ≤ 空闲 + MARGIN`。MARGIN 不是新的手感数：【2】把它夹在
"实测跑好的档里最大的越界"与"实测死过的档里最小的越界"之间，两头都有行号。

为什么目标用引擎自报而不用 nvidia-smi 的空出差值：四档 16k 阶梯的请求中最低空闲
全是 101~171MB——池子被抽到底之后 nvidia-smi 再也量不出 14.2GB 和 15.5GB 的差别，
那个口径是被裁切的，拿它拟合会把"卡满了"误读成"这档只比上档多花 47MB"。

    python _pydiag/calib_loop.py --plan              # 看阶梯与判据，不加载
    python _pydiag/calib_loop.py --rung 62@16000     # 测一档（会真的卸载/加载！）
    python _pydiag/calib_loop.py --drop 62@16000     # 删掉一行重测
    python _pydiag/calib_loop.py --show              # 打印已测的行
    python _pydiag/calib_loop.py --fit               # 拟合＋判据＋逐档 ctx 回放
    python _pydiag/calib_loop.py --break             # 在真实行上做变异反向对照
    python _pydiag/calib_loop.py --restore           # 基线配置写回并重载

判据只认 _pydiag/calib_rows.jsonl 里的实测行；反向对照也在同一批真数据上做——
换成我自己编的样例，就只是在验证我写进去的假设。
"""
import copy
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8900"
MODEL = "Qwen3.8-27B-NVFP4-MTP-LOW.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "logs", "server.log")
ROWS = os.path.join(HERE, "calib_rows.jsonl")
BASE_CFG = os.path.join(HERE, "calib_base.json")
THREADS = 8
MIB = 1024 * 1024
SAFETY_TODAY = 0.9104      # 代价表里那只 5080 行的折扣，只用来算"今天的税"
# 标定规则的绝对余量。不是手感：【2】把它钉在实测带里——必须 ≥ 实测跑好的档里最大的
# 自报越界（本轮 +55.7MB），又必须 < 实测失败档里最小的越界（本轮 +1325.6MB）。
MARGIN = 64.0

# 阶梯：16k 把层轴铺开（求每层价、看会不会崩），64k 上一档成、一档败
# —— 64k/66 在 R4 里真 502 过，它就是"标定不许越界冒进"的那枚锚。
RUNGS = [(60, 16000), (62, 16000), (64, 16000), (66, 16000), (60, 65536), (66, 65536)]

# 与 bench.py 逐字同串的固定负载：两个台架必须能互相核对，否则拟合出来的
# 头寸跟他实际会撞上的头寸不是一回事。
PROMPTS = [
    ("code", "用Python写一个函数：输入二叉树根节点，返回每层节点值的列表（层序遍历）。先给代码，再简述复杂度。"),
    ("prose", "写一段300字左右的说明文字，解释为什么笔记本电脑上大模型的解码速度主要受显存带宽而不是算力限制。"),
    ("math", "一个圆柱体积是972π立方厘米，高是12厘米，求底面半径。给出推导过程和最终数值。"),
    ("long", "\n".join(f"第{i}项：记录编号{i}，测量值{i * 7 % 97}，备注条目{i}。"
                       for i in range(1, 301))
             + "\n请把上面 300 项里编号为完全平方数的项挑出来，简述挑选依据。"),
]

RE_GEN = re.compile(r"\[timing\] 生成 (\d+) tokens，平均速度 ([\d.]+) t/s")
RE_PROMPT = re.compile(r"\[timing\] 提示词处理 (\d+) tokens @ ([\d.]+) t/s")
RE_LOADED = "模型已加载"
RE_FAIL = re.compile(r"启动失败|OutOfMemory|out of memory|cudaErrorMemoryAllocation|"
                     r"available memory|does not fit")


# ---------------------------------------------------------------- 客户端小工具
def api(path, payload=None, method=None, query=None, timeout=600):
    url = BASE + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode("utf-8", "replace")[:300]}
    except urllib.error.URLError as e:
        return {"_http_error": 0, "_body": f"连接失败：{e.reason}"}


def gpu_free_mb():
    """空闲显存：连采三次取中位数。桌面/DWM 的瞬时波动不该被算成模型的账。"""
    vals = []
    for _ in range(3):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        vals += [float(x) for x in out.replace('"', "").splitlines() if x.strip()]
        time.sleep(0.3)
    return round(statistics.median(vals), 1) if vals else None


def resident():
    for item in api("/api/models/loaded").get("loaded") or []:
        if (item.get("model") or {}).get("name") == MODEL:
            return item
    return None


def model_dict():
    for m in api("/api/models").get("models") or []:
        if m.get("name") == MODEL:
            return m
    raise SystemExit(f"接口里没有 {MODEL}，无法核对 parser 的张量表实测")


def tail_from(off):
    with open(LOG, "rb") as fh:
        fh.seek(off)
        return fh.read().decode("utf-8", "replace")


def set_cfg(ngl, ctx):
    base = json.load(open(BASE_CFG, encoding="utf-8"))
    cfg = {**base, "n_gpu_layers": ngl, "ctx_size": ctx, "threads": THREADS,
           "batch_size": min(int(base["batch_size"]), ctx)}
    r = api(f"/api/models/{urllib.parse.quote(MODEL)}/config", cfg)
    if "_http_error" in r:
        raise RuntimeError(f"配置保存失败：{r}")
    return r["config"]


def send_prompts():
    """四条固定负载，请求进行中采样空闲显存最低点。速度从引擎的 [timing] 行取。"""
    low = None
    for tag, p in PROMPTS:
        body = {"model": MODEL, "messages": [{"role": "user", "content": p}],
                "max_tokens": 300, "temperature": 0.7, "stream": False}
        r = api("/v1/chat/completions", body)
        if "_http_error" in r or "choices" not in r:
            raise RuntimeError(f"{tag} 请求失败：{r}")
        f = gpu_free_mb()
        low = f if low is None else min(low, f)
        time.sleep(0.4)
    return low


# ---------------------------------------------------------------- 测一档
def run_rung(ngl, ctx):
    if any(r["ngl"] == ngl and r["ctx"] == ctx for r in read_rows()):
        print(f"   已有 {ngl}@{ctx} 的实测行，要重测先 --drop {ngl}@{ctx}")
        return 1
    eff = set_cfg(ngl, ctx)
    unload = api("/api/models/unload", method="POST", query={"model_name": MODEL})
    # 这一档本来就没上线时接口回 400「not loaded」，那是期望状态，不是失败
    if "_http_error" in unload and "not loaded" not in (unload.get("_body") or ""):
        raise RuntimeError(f"卸载失败：{unload}")
    time.sleep(2.0)
    free_idle = gpu_free_mb()          # 规划器真会读到的那个数
    off = os.path.getsize(LOG)
    load = api("/api/models/load", {"model_name": MODEL})
    err = (f"加载接口 {load['_http_error']}：{load['_body'][:160]}"
           if "_http_error" in load else "")
    row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "ngl": ngl, "ctx": ctx,
           "threads": eff["threads"], "spec_draft_max": eff.get("spec_draft_max"),
           "free_idle": free_idle, "ok": False, "note": ""}
    text, loaded = tail_from(off), resident()
    for _ in range(60):
        if loaded or RE_FAIL.search(text):
            break
        time.sleep(2.0)
        text, loaded = tail_from(off), resident()
    if loaded is None:
        bad = RE_FAIL.search(text)
        row["note"] = err or (f"引擎自报失败：{text[bad.start():bad.start() + 160]}"
                              if bad else "轮询 120s 没有出现这个模型，日志里也没有失败行")
        append_row(row)
        print(f"× {ngl}@{ctx} 未能加载：{row['note'][:220]}", flush=True)
        return 0
    mem = (loaded.get("memory") or {}).get("vram") or {}
    tot = (loaded.get("memory") or {}).get("total_mib") or {}
    row.update({"self_w": round(float(mem.get("weights", 0)), 1),
                "self_kv": round(float(mem.get("kv", 0)), 1),
                "self_state": round(float(mem.get("recurrent", 0)), 1),
                "self_calc": round(float(mem.get("compute", 0)), 1),
                "self_total": round(float(tot.get("vram", 0)), 1),
                "ram_w": round(float(((loaded.get("memory") or {}).get("ram") or {})
                                     .get("weights", 0)), 1),
                "vision": bool(loaded.get("mmproj")),
                "ngl_api": loaded.get("gpu_layers"), "ctx_api": loaded.get("ctx_size")})
    if str(ngl) != str(row["ngl_api"]) or str(ctx) != str(row["ctx_api"]):
        row["note"] = f"引擎上的实际档不是这一档：-ngl {row['ngl_api']} ctx {row['ctx_api']}"
        append_row(row)
        print(f"× {ngl}@{ctx} {row['note']}", flush=True)
        return 1
    row["free_after_load"] = gpu_free_mb()
    try:
        row["free_low"] = send_prompts()
    except RuntimeError as e:
        row["note"] = str(e)[:220]
        append_row(row)
        print(f"× {ngl}@{ctx} 请求阶段失败：{row['note']}", flush=True)
        return 0
    # [timing] 行是请求跑完才落盘的，必须重新取一次窗口
    text = tail_from(off)
    body = text[text.rindex(RE_LOADED):] if RE_LOADED in text else ""
    gen = RE_GEN.findall(body)
    if len(gen) != len(PROMPTS):
        row["note"] = f"窗口里只有 {len(gen)} 条 [timing] 生成行，应为 {len(PROMPTS)} 条"
        append_row(row)
        print(f"× {ngl}@{ctx} {row['note']}", flush=True)
        return 1
    row["decode"] = [float(s) for _, s in gen]
    row["prompt_tokens"] = [int(n) for n, _ in RE_PROMPT.findall(body)]
    parts = row["self_w"] + row["self_kv"] + row["self_state"] + row["self_calc"]
    if abs(parts - row["self_total"]) > 1.0:
        row["note"] = f"接口合计 {row['self_total']} 与四项分解 {parts:.1f} 对不上"
        append_row(row)
        print(f"× {ngl}@{ctx} {row['note']}", flush=True)
        return 1
    row["peak"] = round(row["free_idle"] - row["free_low"], 1)
    row["short_mean"] = round(statistics.mean(row["decode"][:3]), 1)
    row["decode_mean"] = round(statistics.mean(row["decode"]), 1)
    row["ok"] = True
    append_row(row)
    print(f"✓ {ngl}@{ctx} 解码 {row['decode']} 短题均值 {row['short_mean']} | "
          f"自报 {row['self_total']}（权重 {row['self_w']} KV {row['self_kv']} "
          f"状态 {row['self_state']} 计算 {row['self_calc']}）内存权重 {row['ram_w']} | "
          f"空闲 {row['free_idle']}→加载后 {row['free_after_load']}→请求中最低 "
          f"{row['free_low']} → 真实花掉 {row['peak']}", flush=True)
    return 0


def restore():
    base = json.load(open(BASE_CFG, encoding="utf-8"))
    if "_http_error" in api(f"/api/models/{urllib.parse.quote(MODEL)}/config", base):
        print("   写回失败", flush=True)
        return 1
    api("/api/models/unload", method="POST", query={"model_name": MODEL})
    time.sleep(1.0)
    api("/api/models/load", {"model_name": MODEL})
    for _ in range(60):
        time.sleep(2.0)
        it = resident()
        if it:
            m = (it.get("memory") or {}).get("total_mib") or {}
            print(f"   基线已恢复：-ngl {it['gpu_layers']} ctx {it['ctx_size']} "
                  f"线程 {it['threads']}，显存 {m.get('vram')} 内存 {m.get('ram')}", flush=True)
            return 0
    print("   恢复后 120s 内模型没起来，需要人工看 GUI/日志", flush=True)
    return 1


# ---------------------------------------------------------------- 数据行
def read_rows():
    if not os.path.exists(ROWS):
        return []
    out = []
    with open(ROWS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def append_row(row):
    with open(ROWS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_rows(rows):
    with open(ROWS, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def drop_rung(spec):
    ngl, ctx = (int(x) for x in spec.replace("@", " ").split())
    rows = read_rows()
    keep = [r for r in rows if not (r["ngl"] == ngl and r["ctx"] == ctx)]
    if len(keep) == len(rows):
        print(f"   没有 {spec} 这一行")
        return 1
    write_rows(keep)
    print(f"   已删除 {spec}")
    return 0

# ---------------------------------------------------------------- 拟合
def slope(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def collapsed(row):
    """长题掉到短题均值的 60% 以下 = 这一档在挤兑，不能算"跑得好的头寸"。"""
    if not row.get("ok") or not row.get("short_mean") or not row.get("decode"):
        return True
    return row["decode"][3] < 0.6 * row["short_mean"]


def overs(row):
    """引擎自报合计 − 加载前空闲。正值 = 卡上真实需求量已经超过 nvidia-smi 给的空闲。"""
    return round(row["self_total"] - row["free_idle"], 1)


def dead(rows):
    """加载成功、自报到手，但请求阶段死的档——它是失败边界，不是缺数据。"""
    return [r for r in rows if not r.get("ok") and r.get("self_total")]


def fit_self(rows):
    """自报合计 = A + P·ngl + Q·ctx —— 三个未知数，正规方程手解（不引 numpy）。

    拟合目标用的是**引擎自报**而不是 nvidia-smi 的空出差值：实测四档 16k 阶梯的
    空闲最低点全卡在 101~171MB（15367→104、15411→101、15492→153、15492→171），
    也就是说池子被抽到底之后 nvidia-smi 再也量不出差别——14.2GB 和 15.5GB 的常驻
    在这只 16GB 卡上都只剩 ~120MB 空闲。那个口径是**被裁切的**，拿它当标定目标会
    把"卡满了"误读成"这档比上档只多花 47MB"。自报合计没有这个问题（权重/KV/状态/
    计算四项逐条来自引擎自己），失败那一档的自报也照拿，它是边界上唯一的实测点。
    """
    pts = [r for r in rows if r.get("self_total") and r.get("free_idle")]
    if len(pts) < 3 or len({r["ctx"] for r in pts}) < 2:
        return None
    basis = [[1.0, float(r["ngl"]), float(r["ctx"])] for r in pts]
    y = [float(r["self_total"]) for r in pts]
    mat = [[sum(b[i] * b[j] for b in basis) for j in range(3)] for i in range(3)]
    vec = [sum(b[i] * v for b, v in zip(basis, y)) for i in range(3)]
    aug = [list(mat[i]) + [vec[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r_: abs(aug[r_][col]))
        if abs(aug[piv][col]) < 1e-9:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r_ in range(3):
            if r_ != col:
                f = aug[r_][col] / aug[col][col]
                for c in range(col, 4):
                    aug[r_][c] -= f * aug[col][c]
    if any(abs(aug[i][i]) < 1e-9 for i in range(3)):
        return None
    A, P, Q = (aug[i][3] / aug[i][i] for i in range(3))
    if P <= 0 or Q <= 0:
        # 层价或 ctx 价算出非正，说明这批点根本不像"层数×每层价＋ctx×每 token 价"。
        # 拿它反解 -ngl 会越解越激进，宁认可疑也不硬编一个答案。
        return None
    return {"A": A, "P": P, "Q": Q, "n": len(pts),
            "ctxs": sorted({r["ctx"] for r in pts}),
            "pred": lambda ngl, ctx: A + P * ngl + Q * ctx,
            "err": {f"{r['ngl']}@{r['ctx']}":
                    round(100 * (A + P * r["ngl"] + Q * r["ctx"] - r["self_total"])
                         / r["self_total"], 2)
                    for r in pts}}


def plan_at(free_idle, ctx):
    """调真实产品代码问一句：这个空闲读数＋这个 ctx，今天会解出几层。

    绝不在探针里重写一遍 KV/重尾/状态的账——两处各写一遍正是 KV 估算曾经偏离
    llama.cpp 打印 16.8 倍的成因。
    """
    sys.path.insert(0, ROOT)
    from lmloader.engine import hardware_costs, offload_planner
    cfg = json.load(open(BASE_CFG, encoding="utf-8"))
    cfg.update({"ctx_size": ctx, "threads": THREADS})
    return offload_planner.plan_offload(model_dict(), cfg, free_idle, 32000.0,
                                        coefficients=hardware_costs.resolve_coefficients())


def plan_ngl(plan):
    """把规划器的解折到 -ngl 参数轴上，与标定解同口径可比。

    999 在 llama.cpp 里就是"全量"，引擎实际落到顶格档；折成 ngl_ceiling 而不是
    留 999 或组数 65，否则两栏并排打印时读不出谁更激进。
    """
    if plan is None:
        return None
    return ngl_ceiling() if plan.n_gpu_layers == 999 else plan.n_gpu_layers


_LT_CACHE = None


def layers_total():
    """规划器口径的层数分母（blk 组数；NVFP4-LOW 把草稿嵌在最后一组）。"""
    global _LT_CACHE
    if _LT_CACHE is None:
        meta = model_dict()["metadata"]
        groups = int(meta.get("blk_groups") or 0)
        _LT_CACHE = groups or int(meta["block_count"]) + (1 if meta.get("supports_mtp") else 0)
    return _LT_CACHE


def ngl_ceiling():
    """引擎认得的最高 -ngl 档 = block_count +（MTP 草稿头一层）。

    与上面那个分母不是一回事：实测 -ngl 65 把草稿头留在 CPU（bench R2 的 auto 档
    就是这么解的），66 才是全量。拟合的自报合计跟着 -ngl 参数走，所以标定解必须
    封顶在 66——封顶在 65 会把它截到"少一层草稿头"那一档，而 66 恰好是实测短题
    最快（63.8 t/s）的位置。到顶即等价于规划器词表里的 999。
    """
    meta = model_dict()["metadata"]
    return int(meta["block_count"]) + (1 if meta.get("supports_mtp") else 0)


def meas_eff(ngl):
    """实测档的 -ngl 口径：999 这类超过顶格的档折回顶格，拟合轴上才是同一档。"""
    return min(int(ngl), ngl_ceiling()) if ngl is not None else None


def calibrated_answer(cal, free_idle, ctx):
    """标定规则的反解：自报合计（拟合值）不得超过 空闲 + MARGIN，封顶在引擎最高 -ngl。"""
    raw = int((free_idle + MARGIN - cal["A"] - cal["Q"] * ctx) / cal["P"])
    ceiling = ngl_ceiling()
    eff = min(max(raw, 0), ceiling)
    return {"raw": raw, "eff": eff, "ngl": 999 if eff >= ceiling else eff,
            "label": f"{eff}(全量)" if eff >= ceiling else str(eff)}


# ---------------------------------------------------------------- 判据
def judge(rows, cal):
    bad, note = [], []
    good = [r for r in rows if r.get("ok")]
    lost = dead(rows)
    if not good:
        return [f"{len(rows)} 行里没有一档跑完固定负载——判据无法成立，不是通过"], note
    lad = sorted([r for r in good if r["ctx"] == 16000], key=lambda r: r["ngl"])
    big = sorted([r for r in good if r["ctx"] == 65536], key=lambda r: r["ngl"])

    # 【1a】每层值多少：自报权重的逐层斜率 vs parser 张量表实测
    p_layer = None
    if len(lad) >= 3:
        meta = model_dict()["metadata"]
        price = float(meta["layer_bytes"]) / int(meta["blk_groups"]) / MIB
        p_layer = slope([r["ngl"] for r in lad], [r["self_w"] for r in lad])
        if p_layer is None:
            bad.append("【1a】16k 阶梯的 -ngl 全在同一档，层价斜率算不出来")
        elif abs(p_layer - price) / price > 0.05:
            bad.append(f"【1a】自报权重的逐层斜率 {p_layer:.1f}MB/层 与 parser 张量表实测 "
                       f"{price:.1f}MB/层 差 {100 * (p_layer - price) / price:+.1f}%（>5%）："
                       f"规划器按层摊的价跟引擎真实搬的量不是一回事")
        else:
            note.append(f"【1a】逐层权重 实测斜率 {p_layer:.1f} vs parser {price:.1f}MB/层"
                        f"（{100 * (p_layer - price) / price:+.1f}%，16k 阶梯 {len(lad)} 档）")
    else:
        bad.append(f"【1a】16k 层轴只有 {len(lad)} 档（<3），层价无法拟合")

    # 【1b】上一层 GPU 会把它自己的 KV/状态/计算一起带上来：总斜率必须严格大于权重斜率
    if p_layer is not None and cal:
        p_total = cal["P"]
        d_kv, d_st = (lad[-1][k] - lad[0][k] for k in ("self_kv", "self_state"))
        if p_total < p_layer * 1.005:
            bad.append(f"【1b】拟合层价 {p_total:.1f} 不大于权重斜率 {p_layer:.1f}"
                       f"（阶梯两端 KV {d_kv:+.1f}、状态 {d_st:+.1f}）：这一阶梯的账不闭合，"
                       f"要么引擎不按层分 KV/状态，要么抓到了另一趟加载")
        else:
            note.append(f"【1b】拟合层价 {p_total:.1f}MB/层 = 权重 {p_layer:.1f}＋该层自己的"
                        f" KV/状态（阶梯两端 KV {d_kv:+.1f}、状态 {d_st:+.1f}MB）")

    # 【1c】每个 token 的 ctx 价：拟合 Q vs 规划器自己的 kv_full 单价
    if cal and lad and big:
        pl = plan_at(lad[0]["free_idle"], 16000)
        want = pl.kv_full_mb / 16000 * 1000 if pl else None
        q = cal["Q"] * 1000
        if want is None:
            bad.append("【1c】规划器返回 None，KV 单价无从核对")
        elif q < want * 0.95:
            bad.append(f"【1c】拟合出的 ctx 单价 {q:.2f}MB/1k token 低于规划器 KV 单价 "
                       f"{want:.2f}（{100 * (q - want) / want:+.1f}%）：引擎真实占用比规划器"
                       f"为 KV 预留的还少，标定口径必有一处错")
        else:
            note.append(f"【1c】ctx 单价 拟合 {q:.2f} vs 规划器 KV {want:.2f}MB/1k token"
                        f"（{100 * (q - want) / want:+.1f}%）——多出来的 "
                        f"{q - want:.2f}MB/1k 是规划器没计价的计算缓冲随 ctx 增长")
    else:
        bad.append(f"【1c】缺 16k 或 64k 的成功档（{len(lad)}/{len(big)}）或没有拟合，"
                   f"ctx 单价无从核对")

    # 【1d】状态＋计算 必须随 ctx 涨：规划器那笔固定开销盖不住两档 ctx
    ngl0 = max(min(r["ngl"] for r in lad), min([r["ngl"] for r in big] or [0]))
    p16 = next((r for r in lad if r["ngl"] == ngl0), None)
    p64 = next((r for r in big if r["ngl"] == ngl0), None)
    if p16 and p64:
        d16 = p16["self_state"] + p16["self_calc"]
        d64 = p64["self_state"] + p64["self_calc"]
        per_k = (d64 - d16) / (65536 - 16000) * 1000
        if d64 <= d16:
            bad.append(f"【1d】同 -ngl {ngl0} 下 状态＋计算 从 16k 的 {d16:.0f}MB 到 64k 的 "
                       f"{d64:.0f}MB 没有涨：要么引擎的缓冲不随 ctx 走（那【1c】的溢价来源"
                       f"就是错的），要么这两行不是同一趟配置")
        else:
            note.append(f"【1d】状态＋计算 同 -ngl {ngl0}：16k {d16:.0f}MB → 64k {d64:.0f}MB"
                        f"（+{per_k:.2f}MB/1k token）；规划器按固定 base_overhead_mb="
                        f"596＋状态超出计价，这一档增长一分都没扣")
    else:
        bad.append(f"【1d】凑不出同 -ngl 的两档 ctx（16k {[r['ngl'] for r in lad]}，"
                   f"64k {[r['ngl'] for r in big]}），缓冲增长无从核对")

    # 【2】MARGIN 的出处：必须盖住实测成功的最大越界，又必须低于实测失败的最小越界
    ok_over = [overs(r) for r in good]
    dead_over = [overs(r) for r in lost]
    if not dead_over:
        note.append("【2】本轮没有『加载成功但请求阶段死掉』的档：失败边界一侧没有实测点，"
                    "MARGIN 只被成功侧的下限约束着——不是通过")
    if max(ok_over) > MARGIN:
        bad.append(f"【2】实测跑好的档里最大越界 {max(ok_over):+.1f}MB 已超过 MARGIN={MARGIN:.0f}MB："
                   f"这个余量太紧，按它解会把实测跑得好的档也拒掉")
    elif dead_over and MARGIN >= min(dead_over):
        bad.append(f"【2】MARGIN={MARGIN:.0f}MB 不小于实测失败档的最小越界 "
                   f"{min(dead_over):+.1f}MB：按它解会把实测死过的档放行")
    else:
        note.append(f"【2】MARGIN={MARGIN:.0f}MB 卡在实测带里：成功档最大越界 "
                    f"{max(ok_over):+.1f}MB（{len(good)} 档），失败档最小越界 "
                    + (f"{min(dead_over):+.1f}MB（{lost[0]['ngl']}@{lost[0]['ctx']}，502）"
                       if dead_over else "本轮无失败档"))

    # 【3】速度必须真的随层数走，且长题不许塌、空闲不许为负
    for r in good:
        if collapsed(r):
            bad.append(f"【3】{r['ngl']}@{r['ctx']} 长题 {r['decode'][3]} t/s < 短题均值 "
                       f"{r['short_mean']} 的 60%：记成成功但实际在挤兑")
        if r["free_low"] is not None and r["free_low"] < 0:
            bad.append(f"【3】{r['ngl']}@{r['ctx']} 请求进行中空闲 {r['free_low']}MB<0，读数不可信")
    if len(lad) >= 3:
        sp = [r["short_mean"] for r in lad]
        if any(y <= x for x, y in zip(sp, sp[1:])):
            bad.append(f"【3】16k 短题速度不随 -ngl 单调上升：{list(zip([r['ngl'] for r in lad], sp))}"
                       f"——『多上一层就快一截』这个前提本轮没成立，多给层数买不到速度")
        else:
            note.append(f"【3】16k 短题随层数单调：" +
                        " → ".join(f"{r['ngl']}层 {r['short_mean']}t/s" for r in lad))
    else:
        bad.append(f"【3】16k 阶梯不足 3 档，层数↔速度的单调性没有被检查")

    # 【4】16k：标定解必须比今天的 ×0.91 解更激进，但不得越过实测成功的最高档
    if cal and lad:
        ref = lad[-1]
        today = plan_ngl(plan_at(ref["free_idle"], 16000))
        prop = calibrated_answer(cal, ref["free_idle"], 16000)
        top16 = max(meas_eff(r["ngl"]) for r in lad)
        if today is None:
            bad.append("【4】规划器在这个空闲读数上返回 None，无从对照")
        elif prop["raw"] > top16:
            bad.append(f"【4】标定解在 16k 要 -ngl {prop['raw']}（顶格才 {ngl_ceiling()}），"
                       f"高于实测成功的最高档 {top16}——没有实测支撑的外推")
        elif prop["eff"] <= today:
            bad.append(f"【4】标定解 {prop['eff']} 层并不比今天的 ×0.91 解 {today} 层更激进："
                       f"这一趟标定没买到层数，结论不成立")
        else:
            gain = lad[-1]["short_mean"] / next(r["short_mean"] for r in lad
                                                if r["ngl"] == today or abs(r["ngl"] - today) <= 1) \
                if any(abs(r["ngl"] - today) <= 1 for r in lad) else None
            note.append(f"【4】同一空闲读数 {ref['free_idle']:.0f}MB@16k：今天解 -ngl {today}，"
                        f"标定解 -ngl {prop['label']}"
                        + (f"，对应的实测短题速度差 {gain:.2f}×" if gain else "")
                        + f"（实测最高 {top16}，不外推）")
    else:
        bad.append("【4】缺 16k 阶梯或缺拟合结果，标定解与今天的解无法对照")

    # 【5】64k：标定解必须落在实测成功档与实测失败档之间，不许两头越界
    if cal and big:
        ref64 = big[0]
        prop64 = calibrated_answer(cal, ref64["free_idle"], 65536)
        tops = max(meas_eff(r["ngl"]) for r in big)
        failed = sorted(meas_eff(r["ngl"]) for r in lost if r["ctx"] == 65536)
        if prop64["eff"] > tops:
            bad.append(f"【5】标定解在 64k 给出 -ngl {prop64['label']}（有效 {prop64['eff']} 层），"
                       f"高于 64k 实测成功的最高档 {tops}：外推越界")
        elif failed and prop64["eff"] >= failed[0]:
            bad.append(f"【5】标定解在 64k 给出 -ngl {prop64['label']}，而 {failed[0]} 层那次是"
                       f"实测失败（请求阶段 502）：越过了失败边界")
        else:
            note.append(f"【5】64k（空闲 {ref64['free_idle']:.0f}MB）标定解 -ngl "
                        f"{prop64['label']}，实测成功最高 {tops}"
                        + (f"、实测失败最低 {failed[0]}" if failed else "、本轮无失败档")
                        + f"；今天同一读数解 -ngl {plan_ngl(plan_at(ref64['free_idle'], 65536))}")
    else:
        bad.append("【5】64k 上没有成功档或缺拟合，大 ctx 的边界判据无法成立")

    # 【6】拟合本身站不站得住
    if cal:
        worst = max(abs(v) for v in cal["err"].values())
        if worst > 3.0:
            bad.append(f"【6】自报合计的线性拟合逐档误差 {cal['err']}，最大 {worst}%>3%："
                       f"self_total = A＋P·ngl＋Q·ctx 盖不住这批读数，标定不可用")
        else:
            note.append(f"【6】拟合 自报合计 = {cal['A']:.0f}＋{cal['P']:.1f}·ngl＋"
                        f"{cal['Q'] * 1000:.2f}MB/1k·ctx，{cal['n']} 档最大误差 {worst:.2f}%"
                        f"（含失败档 {len(lost)} 行）")
    else:
        bad.append("【6】拟合不成立（自报点<3 或只有一个 ctx 或斜率非正）")
    return bad, note


# ---------------------------------------------------------------- 回放
def replay(cal, rows):
    """把两个 owner 摆在同一批空闲读数上，逐档 ctx 对照。"""
    if not cal:
        return
    free = round(statistics.median([r["free_idle"] for r in rows if r.get("free_idle")]), 0)
    measured = {(r["ngl"], r["ctx"]): r for r in rows if r.get("self_total")}
    print(f"\n  回放（空闲读数 {free:.0f}MB，MARGIN {MARGIN:.0f}MB）：")
    print(f"  {'ctx':>6} {'今天×0.91':>9} {'标定解':>7} {'标定预测自报':>11} "
          f"{'实测证据':>9}  实测短题/长题")
    for ctx in (8000, 16000, 32000, 65536):
        today = plan_ngl(plan_at(free, ctx))
        ans = calibrated_answer(cal, free, ctx)
        prop = ans["eff"]
        near = sorted([(abs(meas_eff(n) - prop), n, r)
                       for (n, c), r in measured.items() if c == ctx])
        ev = "没测过"
        sp = ""
        if near:
            d, n, r = near[0]
            ok = "成" if r.get("ok") else "败"
            ev = f"{ok} {n}±{d}"
            if r.get("ok"):
                sp = f"{r['short_mean']}/{r['decode'][3]} t/s"
        print(f"  {ctx:>6} {str(today):>9} {ans['label']:>7} "
              f"{cal['pred'](prop, ctx):>10.0f}MB {ev:>11}  {sp}")


# ---------------------------------------------------------------- 反向对照
def break_control():
    """八把锚点，各自必须把对应那条判据打红；变异全部打在真实实测行上。

    编样例只会验证我自己写进去的假设，所以没有真数据时直接拒绝报绿。
    """
    rows = read_rows()
    if not [r for r in rows if r.get("ok")]:
        print("  没有真实测得的行（先跑 --rung），拒绝在假数据上报绿")
        return 1
    base_bad, _ = judge(copy.deepcopy(rows), fit_self(rows))
    if base_bad:
        print("  先把话说清：基线判据本身就是红的，变异对照失去意义（先修基线）")
        for b in base_bad:
            print("   -", b)
        return 1
    ok16 = [r for r in rows if r.get("ok") and r["ctx"] == 16000]
    top16 = max(r["ngl"] for r in ok16)
    ngl0 = min(r["ngl"] for r in ok16)

    def mut_price(rs):        # 【1a】层价摊错：自报权重整体低 10%
        for r in rs:
            if r.get("ok") and r["ctx"] == 16000:
                r["self_w"] = r["self_w"] * 0.9

    def mut_flat(rs):         # 【1b】每层的 KV/状态不算钱：总占用只随权重动，层价=权重斜率
        for ctx in (16000, 65536):
            sub = [r for r in rs if r.get("self_total") and r["ctx"] == ctx]
            if not sub:
                continue
            base = min(r["self_kv"] + r["self_state"] + r["self_calc"] for r in sub)
            for r in sub:
                r["self_total"] = r["self_w"] + base

    def mut_kv(rs):           # 【1c】把 64k 的 KV 砍一半 → 拟合 ctx 单价掉到规划器 KV 单价以下
        for r in rs:
            if r.get("self_total") and r["ctx"] == 65536:
                r["self_kv"] = r["self_kv"] * 0.5
                r["self_total"] = (r["self_w"] + r["self_kv"] + r["self_state"]
                                   + r["self_calc"])

    def mut_buf(rs):          # 【1d】把 64k 的状态＋计算压回 16k 的水平（缓冲不随 ctx 涨）
        for r in rs:
            if r.get("ok") and r["ctx"] == 65536:
                ref = next(x for x in rs if x.get("ok") and x["ctx"] == 16000
                           and x["ngl"] == r["ngl"])
                r["self_calc"] = ref["self_calc"]
                r["self_total"] = (r["self_w"] + r["self_kv"] + r["self_state"]
                                   + r["self_calc"])

    def mut_margin(rs):       # 【2】把失败档改成"只越界 20MB"：MARGIN 就放进来了
        for r in rs:
            if not r.get("ok") and r.get("self_total"):
                r["self_total"] = r["free_idle"] + 20.0

    def mut_free(rs):         # 【4】【5】空闲读数凭空多 2500MB → 两头都外推越界
        for r in rs:
            if r.get("free_idle"):
                r["free_idle"] = r["free_idle"] + 2500

    def mut_collapse(rs):     # 【3】最高档长题塌到三成，却仍记着成功
        for r in rs:
            if r.get("ok") and r["ngl"] == top16 and r["ctx"] == 16000:
                r["decode"] = r["decode"][:3] + [r["short_mean"] * 0.3]

    def mut_flat_speed(rs):   # 【3】层数买不到速度：16k 各档短题全压到最低档
        low = min(r["short_mean"] for r in rs
                  if r.get("ok") and r["ctx"] == 16000 and r.get("short_mean"))
        for r in rs:
            if r.get("ok") and r["ctx"] == 16000:
                r["short_mean"] = low
                r["decode"] = [low] * 4
                r["decode_mean"] = low

    anchors = [
        ("A1", "【1a】自报权重整体低 10%（层价摊错）", "【1a", mut_price),
        ("A2", "【1b】每层 KV/状态不算钱，总占用只随权重动", "【1b", mut_flat),
        ("A3", "【1c】64k 的 KV 砍一半（拟合 ctx 单价掉下界）", "【1c", mut_kv),
        ("A4", "【1d】64k 的计算缓冲压回 16k 水平", "【1d", mut_buf),
        ("A5", "【2】失败档的越界改成 +20MB", "【2", mut_margin),
        ("A6", "【4】空闲读数多 2500MB（诱导外推）", "【4", mut_free),
        ("A7", "【5】同 A6，看 64k 那半边是否也被抓住", "【5", mut_free),
        ("A8", "【3】最高档长题塌到三成", "【3", mut_collapse),
        ("A9", "【3】16k 短题压平（层数不再买速度）", "【3", mut_flat_speed),
    ]
    print("  锚点：变异 → 期望点名（每把都必须让对应判据变红）")
    fails = []
    for name, desc, want, fn in anchors:
        r2 = copy.deepcopy(rows)
        fn(r2)
        got, _ = judge(r2, fit_self(r2))
        new = [b for b in got if b not in base_bad]
        named = [b for b in new if b.startswith(want)]
        print(f"   {name} {desc}：新增红 {len(new)} 条，"
              + (f"点名{want}】→ {named[0][:96]}" if named else f"没有点名{want}】"))
        if not named:
            fails.append(f"{name} 是 no-op：变异后{want}】仍然不红")
    if fails:
        print("  反向对照失败：")
        for f in fails:
            print("   -", f)
        return 1
    print(f"  如预期报红：{len(anchors)} 把锚点各自把对应判据打红，判据确实在约束实测数字")
    return 0


# ---------------------------------------------------------------- 入口
def plan_cmd():
    print(f"  模型 {MODEL}｜线程 {THREADS}｜基线配置 {BASE_CFG}｜实测行 {ROWS}")
    print(f"  阶梯 " + ", ".join(f"{n}@{c}" for n, c in RUNGS))
    print("  每档：卸载→采空闲→加载→四条固定负载（请求中采空闲最低点）→接口自报四项分解")
    print(f"  标定模型 自报合计 = A＋P·ngl＋Q·ctx；规则 P·ngl＋Q·ctx＋A ≤ 空闲＋MARGIN"
          f"（MARGIN={MARGIN:.0f}MB，出处见【2】）")
    print("  判据：【1a】自报权重逐层斜率 vs parser 张量表实测（±5%）")
    print("        【1b】拟合层价 > 权重斜率（该层 KV/状态要跟着上卡）")
    print("        【1c】拟合 ctx 单价 ≥ 规划器 KV 单价（差额=未计价的计算缓冲）")
    print("        【1d】同 -ngl 的状态＋计算必须随 ctx 涨")
    print("        【2】MARGIN 卡在实测成功越界与实测失败越界之间")
    print("        【3】成功档长题不许塌、空闲不许为负、16k 短题随层数单调")
    print("        【4】16k 标定解 > 今天 ×0.91 的解，且不越过实测成功最高档")
    print("        【5】64k 标定解落在实测成功档与实测失败档之间")
    print("        【6】拟合逐档误差 ≤3%")
    return 0


def show_cmd():
    rows = read_rows()
    if not rows:
        print("  还没有实测行")
        return 1
    print(f"  {'档':11} {'状态':4} {'空闲':>7} {'加载后':>7} {'最低':>7} {'自报合计':>9} "
          f"{'越界':>8} {'权重':>8} {'KV':>7} {'状态':>7} {'计算':>7} {'内存权重':>8} "
          f"{'短题':>6} {'长题':>6} 备注")
    for r in rows:
        dec = r.get("decode") or [None] * 4
        state = "✓" if r.get("ok") else ("死" if r.get("self_total") else "×")
        over = overs(r) if r.get("self_total") else None
        print(f"  {str(r['ngl']) + '@' + str(r['ctx']):11} {state:4} "
              + " ".join(f"{r.get(k, '-'):>7}" for k in
                         ("free_idle", "free_after_load", "free_low"))
              + f" {r.get('self_total', '-'):>9} {('-' if over is None else f'{over:+.0f}'):>8} "
              + " ".join(f"{r.get(k, '-'):>7}" for k in
                         ("self_w", "self_kv", "self_state", "self_calc"))
              + f" {r.get('ram_w', '-'):>8} {r.get('short_mean', '-'):>6} "
              + f"{dec[3] if dec[3] is not None else '-':>6} {(r.get('note') or '')[:40]}")
    return 0


def fit_cmd():
    rows = read_rows()
    if not rows:
        print("  没有实测行：先跑 --rung，判据不能靠猜")
        return 1
    cal = fit_self(rows)
    if cal:
        print(f"  拟合 自报合计 = {cal['A']:.1f} + {cal['P']:.2f}·ngl + "
              f"{cal['Q'] * 1000:.2f}MB/1k·ctx（{cal['n']} 档，ctx {cal['ctxs']}）")
        print(f"  逐档误差 {cal['err']}")
        for r in rows:
            if r.get("self_total"):
                p = cal["pred"](r["ngl"], r["ctx"])
                print(f"   {r['ngl']}@{r['ctx']}: 自报 {r['self_total']:.0f} 拟合 {p:.0f}MB"
                      f"（{100 * (p - r['self_total']) / r['self_total']:+.2f}%）"
                      f"｜空闲 {r['free_idle']:.0f} ｜这档标定解 "
                      f"{calibrated_answer(cal, r['free_idle'], r['ctx'])['label']}")
    else:
        print("  拟合不成立（自报点<3／只有一个 ctx／斜率非正）："
              "标定解给不出来，【1b】【1c】【4】【5】【6】按缺数据报红")
    bad, note = judge(rows, cal)
    for n in note:
        print("  ℹ", n)
    replay(cal, rows)
    if bad:
        print("\n  判据变红：")
        for b in bad:
            print("   -", b)
        return 1
    print("\n  判据通过：层价、ctx 单价、缓冲增长、MARGIN 出处、两档 ctx 的解都与实测一致")
    return 0


def main():
    args = sys.argv[1:]
    if "--plan" in args:
        return plan_cmd()
    if "--show" in args:
        return show_cmd()
    if "--fit" in args:
        return fit_cmd()
    if "--restore" in args:
        return restore()
    if "--break" in args:
        return break_control()
    if "--drop" in args:
        return drop_rung(args[args.index("--drop") + 1])
    if "--rung" in args:
        ngl, ctx = (int(x) for x in args[args.index("--rung") + 1].replace("@", " ").split())
        return run_rung(ngl, ctx)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
