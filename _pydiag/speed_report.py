"""把三层数据拼成一张提速结论表：引擎裸速（llama-bench）、线上服务（HTTP 台）、
引擎自报的逐条 buffer（kv_probe），并检查三者互相自洽。

为什么要跨源对：单看一张表会被自己的口径骗。R4 那几次 HTTP 台的『显存 KV 63』
跟 planner 预计的 299 差 4.7×，当时判 planner 错——其实是 lmloader 的汇总把多条
buffer 并成了一条（任务 #21），而逐条相加后 planner 只差 1%。这张表把三个来源
摆在一起，再用六条跟口径无关的判据钉住：

  1) 同一批部分卸载档位，-ngl 越大解码越快，两侧（裸引擎、线上）都得单调；
  2) 同一个 ngl 档，线上必须比裸引擎快（MTP 草稿解码），倍速要落在 1.5~2.3；
  3) 部分卸载档位满足带宽定律 tg × 内存侧权重 ≈ 常数（由搬运带宽决定）；
  4) 【7】的引擎直连 A/B：每档的循环状态要等于由 GGUF 元数据推出来的那份，
     且随草稿深度单调——元数据与分配器是两个独立来源，谁读错都会露。
  5) 【8】的引擎解算器对表：规划器选的层数，引擎自己的离线估算（−fitp）必须认为
     放得下；它给出的"上下文"必须等于实测主干 KV ＋ 主干状态——这是第三方算术，
     与我们的公式没有共用一行代码。
  6) 【9】的 flag 轴只与该档自己的基准行比，且解码与预填两头一起看（只看解码会把
     −nopo 1 那种"读文档变慢十倍"当成提速项）；【10】的 build A/B 要求两侧同档、
     每个 build 自己随层数单调——换 build 的行绝不混进单 build 的那几张表。

任何一条不成立就说明某一侧的数据被读错了，不输出结论。

    python -X utf8 -u _pydiag/speed_report.py           # 出表
    python -X utf8 -u _pydiag/speed_report.py --break   # 反向对照
"""
import csv
import os
import re
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lmloader.engine.offload_planner import recurrent_state_mb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_results_all.csv")
LLB = os.path.join(HERE, "llbench_results.csv")
SPEC = os.path.join(HERE, "spec_flags_results.csv")
# 换引擎 build 的 A/B 写在这张单独的表里（`llbench.py --out`），主表始终只有一个
# build——两边混进同一张 config 索引的表，【1】【6】【9】就把两个被测物当同一次测量了。
BUILD_CSV = os.path.join(HERE, "llbench_builds.csv")

# 引擎直连 A/B（_pydiag/spec_flags.py，ctx 16000／-ngl 66）里那只旗舰的真实循环状态
# 几何：2026-10-21 由 lmloader.infra.gguf_parser.parse_gguf 从
# models/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-MTP-LOW.gguf 读出。
# 【7】用它反推每一档该有多少状态，再和引擎那一趟自己打印的 RS 行对——两个来源
# 一个是文件里的元数据，一个是跑起来后的分配器，谁读错了都会露出来。
SSM = {"architecture": "qwen35", "block_count": 65, "supports_mtp": True,
       "full_attention_interval": 4, "ssm_conv_kernel": 4, "ssm_state_size": 128,
       "ssm_group_count": 16, "ssm_inner_size": 6144}
DRAFT_RE = re.compile(r"n_max=(\d+)")
RS_RE = re.compile(r"RS (\d+)")

# kv_probe 在线上口径（--mmproj + draft-mtp）下逐条收到的 KV buffer（MiB）
KV_LINES = {
    16000: [("CUDA0", "KV", 283.50, "主干 q4_0"), ("CUDA0", "KV", 63.00, "草稿 f16")],
    65536: [("CUDA0", "KV", 1152.00, "主干 q4_0"), ("CUDA0", "KV", 256.00, "草稿 f16")],
}
LASTWIN_REPORTED = {16000: 63, 65536: 256}   # 修正前（末条覆盖）线上实际显示的那条

# 跨源判据只压 64k 的这四个部分卸载档：这四处两侧都是『跑得起来、速度可测』的正常
# 工况。ngl 62/66 在 64k 已经越界（要么 502 要么塌到个位数），是另一回事——那是要
# 报告的现象，不是数据错位的证据，拿来当判据就永远红。
PARTIAL = [46, 52, 56, 60]
# 内存侧还剩 ≥1.5GB 权重才叫部分卸载：全驻留那档只剩 644 MiB 的 embd 表，不受每
# token 搬运支配，不该进带宽定律。
PARTIAL_RAM_MB = 1500.0
MTP_BAND = (1.5, 2.3)


def rows(path):
    if not os.path.exists(path):
        raise SystemExit(f"缺少 {path}——先跑对应台架")
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _num(v):
    """失败行里的 '-' / 空串一律当没有。"""
    x = f(v)
    return x if x is not None and x > 0 else None


def _s(v, fmt="{:.1f}", dash="    —"):
    """没有记过就是没有——不能印成 0.0，读表的人会当成量到了 0。"""
    return dash if v is None else fmt.format(v)


def f0(v, w=7):
    """整数格；没有记过就印 —，印成 0 会被读表的人当成量到了 0 层。"""
    return ("—" if v is None else f"{v:.0f}").rjust(w)


def http_rows():
    """线上服务的所有实验行（保留变体，不做 best-of）。"""
    out = []
    for r in rows(BENCH):
        tg, ctx, ngl = _num(r["解码均值"]), _num(r["ctx"]), _num(r["ngl"])
        if tg is None or ctx is None or ngl is None:
            continue
        out.append({"name": r["实验"], "round": r["轮次"], "ctx": int(ctx),
                    "ngl": int(ngl), "tg": tg, "long": _num(r["long"]),
                    "vision": r["视觉"], "draft": r["draft"], "mtp": r["mtp"],
                    "threads": r["线程"], "ram_w": _num(r["内存权重"]),
                    "vram_w": _num(r["显存权重"]), "vram_total": _num(r["显存自报合计"]),
                    "free_at_load": _num(r["空闲显存"]), "free_min": _num(r["空闲最低"])})
    return out


def http_tier_table(all_rows=None):
    """按 ctx/ngl 取最快一次，附内存侧权重与运行中最低空闲（带宽定律要用）。"""
    out = {}
    for r in all_rows if all_rows is not None else http_rows():
        key = (r["ctx"], r["ngl"])
        if key not in out or r["tg"] > out[key]["tg"]:
            out[key] = r
    return out


def engine_table():
    out = {}
    for r in rows(LLB):
        tg, ngl = _num(r["tps"]), _num(r["n_gpu_layers"])
        if r["test"] != "tg128" or tg is None or ngl is None:
            continue
        out[r["config"]] = {"tg": tg, "pp": None, "ngl": int(ngl),
                            "sd": f(r["stddev"]) or 0.0}
    for r in rows(LLB):
        if r["test"] == "pp512":
            e = out.get(r["config"])
            if e:
                e["pp"] = _num(r["tps"])
    return out


def law_of(v):
    if not v["ram_w"] or v["ram_w"] < PARTIAL_RAM_MB:
        return None
    return v["tg"] * v["ram_w"] / 1024.0


def judge(http, bare, verbose=True):
    """三张表能不能互相解释。只回红名单；绿线在 verbose 时才打印。"""
    say = print if verbose else (lambda *a, **k: None)
    bad = []
    h_seq = [(n, http[(65536, n)]["tg"]) for n in PARTIAL if (65536, n) in http]
    e_seq = [(n, bare[f"ngl={n}"]["tg"]) for n in PARTIAL if f"ngl={n}" in bare]
    if len(h_seq) < len(PARTIAL) or len(e_seq) < len(PARTIAL):
        bad.append(f"部分卸载档位凑不齐：HTTP {h_seq} 裸引擎 {e_seq}")
        return bad
    for name, seq in (("HTTP 台", h_seq), ("裸引擎", e_seq)):
        if any(b <= a for (_, a), (_, b) in zip(seq, seq[1:])):
            bad.append(f"{name} 在 -ngl {PARTIAL} 上不单调：{seq}")
    for (n, htg), (_, etg) in zip(h_seq, e_seq):
        ratio = htg / etg
        if not MTP_BAND[0] <= ratio <= MTP_BAND[1]:
            bad.append(f"ngl {n}：MTP 倍速 {ratio:.2f} 不在 "
                       f"{MTP_BAND[0]}~{MTP_BAND[1]}，两侧数据口径可能不一致")
        else:
            say(f"   ngl {n} 倍速 {ratio:.2f}×（线上 {htg:.1f} vs 裸 {etg:.1f}）")
    laws = [law_of(http[(65536, n)]) for n in PARTIAL]
    laws = [x for x in laws if x]
    if len(laws) >= 3:
        lo, hi = min(laws), max(laws)
        if hi / lo > 1.6:
            bad.append(f"带宽定律离散太大：{lo:.0f}~{hi:.0f} GB/s，"
                       f"tg 不是由内存权重决定")
        else:
            say(f"   带宽定律：tg × 内存权重 ≈ {sum(laws)/len(laws):.0f} GB/s"
                f"（{len(laws)} 档，离散 {lo:.0f}~{hi:.0f}）")
    return bad


def spec_rows():
    """引擎直连 A/B 的原始行；文件还没生成时返回空，【7】整段跳过（不是通过）。"""
    if not os.path.exists(SPEC):
        return []
    out = []
    for r in rows(SPEC):
        tg = _num(r["解码"] if "解码" in r else r.get("tg"))
        eff = r.get("eff") or ""
        depth = DRAFT_RE.search(eff)
        rs_seen = RS_RE.search(r.get("kv_lines") or "")
        out.append({"variant": r["variant"], "tg": tg, "eff": eff,
                    "depth": int(depth.group(1)) if depth else None,
                    "no_draft": eff.strip() == "no-draft",
                    "rs_seen": float(rs_seen.group(1)) if rs_seen else None,
                    "acc": _num(r.get("acc_rate")), "vram": _num(r.get("vram_used"))})
    return out


def judge_spec(spec):
    """A/B 表能不能自洽：参数真生效、状态数对得上元数据、深度 ladder 单调。"""
    bad = []
    if not spec:
        return ["【7】没有 spec_flags_results.csv，MTP/草稿这笔账没有数据"]
    # 1) 生效配置：除了故意关草稿的那档，其余必须和 base 同深度，否则这一行
    #    根本不是在比同一个东西；深度为 None 说明连引擎的自报配置都没抓到。
    base = next((s for s in spec if s["variant"] == "base"), None)
    if not base or base["depth"] is None:
        bad.append("base 档缺失或没抓到生效配置，【7】整张表无从对照")
        return bad
    for s in spec:
        if s["depth"] is None and not s["no_draft"]:
            bad.append(f"{s['variant']} 没有生效配置记录，无法证明参数进了引擎")
    # 2) 状态数：元数据推的 vs 引擎那一趟打印的，两处必须一致（±1 MiB，打印取整）。
    for s in spec:
        if s["rs_seen"] is None:
            bad.append(f"{s['variant']} 没有 RS 行——引擎没跑起来或日志被截断")
            continue
        depth = 0 if s["no_draft"] else s["depth"]
        if depth is None:
            continue      # 上面已经报过"没有生效配置"，这里不重复判，也不拿 None 去算
        want = recurrent_state_mb(SSM, depth, 1)
        if abs(want - s["rs_seen"]) > 1.0:
            bad.append(f"{s['variant']} 深度 {depth}：元数据推 {want:.0f} MiB，"
                       f"引擎自报 {s['rs_seen']:.0f} MiB，差 {abs(want - s['rs_seen']):.0f}")
    # 3) 草稿越深，状态越大：只比深度这一条轴，速度轴交给台架自己的判据。
    ladder = [(s["depth"] if not s["no_draft"] else 0, s["rs_seen"]) for s in spec
              if s["rs_seen"] is not None and (s["depth"] is not None or s["no_draft"])]
    ladder = sorted(set(ladder))
    if any(b <= a for (_, a), (_, b) in zip(ladder, ladder[1:])):
        bad.append(f"状态随草稿深度不单调：{ladder}")
    return bad


FITAB = os.path.join(HERE, "fit_ab_results.csv")
# 引擎离线估算器与"实测主干 KV ＋ 主干状态"允许的相对差：fitp 用整除的 MiB 打印，
# 而实测那条 buffer 是引擎自己分配完报的，两者差 1~2 MiB 是取整，不是口径分歧。
FIT_CTX_TOL = 0.02


def _int(v):
    x = f(v)
    return int(x) if x is not None else None


def fit_rows():
    """引擎自带解算器（_pydiag/fit_ab.py 叫 llama-fit-params）与规划器的对表。

    文件还没生成时返回空，【8】整段跳过（不是通过）。−fitt 那个数在 llama.cpp 里是
    『解算完仍要保留的自由显存』，不是可用预算，所以这里两侧各自记的是自己的账，
    判据②拿引擎的算术去验规划器的答案，而不是比谁的字面量大。
    """
    if not os.path.exists(FITAB):
        return []
    out = []
    for r in rows(FITAB):
        out.append({"ctx": _int(r.get("ctx")), "free": _num(r.get("free_before")),
                    "margin": _num(r.get("margin")), "engine": _int(r.get("engine_ngl")),
                    "engine_ctx": _int(r.get("engine_ctx")), "p": _int(r.get("p_ngl")),
                    "p_kv": _num(r.get("p_kv")), "rs0": _num(r.get("p_rs_nodraft")),
                    "est_total": _num(r.get("est_p_total")),
                    "est_ctx_full": _num(r.get("est_full_ctx")),
                    "drift": f(r.get("free_drift")) or 0.0})
    return out


def judge_fit(fit):
    """四把：两侧各自单调、引擎算术为规划器背书、离线上下文==实测主干那份、层数不分歧。"""
    bad = []
    if not fit:
        return ["【8】没有 fit_ab_results.csv，引擎解算器这一侧没有对照"]
    if len(fit) < 2:
        return [f"【8】可比的解算档只有 {len(fit)} 个，单调性无从判起"]
    for r in fit:
        if r["ctx"] is None or r["engine"] is None or r["p"] is None:
            bad.append(f"ctx {r['ctx']}：引擎或规划器的层数缺失，这一行不能算对照")
        if r["drift"] > 60:
            bad.append(f"ctx {r['ctx']}：两次调用之间空闲漂了 {r['drift']:.0f} MiB，"
                       "两侧吃的不是同一条显存")
    seq_ok = [r for r in fit if r["ctx"] is not None]
    for key, name in (("engine", "引擎解算"), ("p", "规划器")):
        seq = [(r["ctx"], r[key]) for r in seq_ok if r[key] is not None]
        if any(b > a for (_, a), (_, b) in zip(sorted(seq), sorted(seq)[1:])):
            bad.append(f"{name} 的层数随 ctx 不单调：{sorted(seq)}")
    for r in seq_ok:
        if None in (r["est_total"], r["free"], r["margin"]):
            bad.append(f"ctx {r['ctx']}：没有引擎对这一档的估算，判据②没得验")
            continue
        avail = r["free"] - r["margin"]
        if r["est_total"] > avail:
            bad.append(f"ctx {r['ctx']}：规划器放 {r['p']} 层，引擎自己估这一档要 "
                       f"{r['est_total']:.0f} MiB，越过了 {avail:.0f} MiB 的可用")
        if r["engine"] is not None and r["p"] is not None \
                and abs(r["p"] - r["engine"]) > 1:
            bad.append(f"ctx {r['ctx']}：规划器 {r['p']} 层与引擎解算 {r['engine']} 层"
                       f"层数分歧（差 {abs(r['p'] - r['engine'])}，口径差最多 1）")
    for r in seq_ok:
        if r["ctx"] not in KV_LINES or None in (r["est_ctx_full"], r["rs0"]):
            continue
        want = KV_LINES[r["ctx"]][0][2] + r["rs0"]
        if abs(r["est_ctx_full"] - want) > want * FIT_CTX_TOL:
            bad.append(f"ctx {r['ctx']}：引擎离线上下文 {r['est_ctx_full']:.0f} MiB 与"
                       f"实测主干 KV＋主干状态 {want:.1f} MiB 上下文对不上")
    return bad


RE_NGL_SUFFIX = re.compile(r"^ngl=(\d+)\s*(.*)$")
# 解码涨但预填塌到这个比例以下，等于把"回答问题"换成"读文档"——线上是混合负载，
# 这种 flag 不能算提速项（−nopo 1 实测解码涨几个点、预填 −94%，就是这一类）。
PREFILL_FLOOR = 0.80
# 同一档、同一串 flag、两次独立 run 的实测散布（2026-10-21，ngl 46/61/66 共 8 组
# 重复行）最大 2.3%：llama-bench 报的 σ 是那趟内部各次重复的标准差，不含 run 间
# 的热漂移/页缓存差，直接拿它当临界会把 0.5% 也标成"真的动了"。
# 3% 就钉在这里当显著性的**相对下限**：临界取"组内 σ 合成带"与"3%×该档速度"里宽的
# 那个。它是被检查的断言，不是调来刚好不红的门槛——所以越过 3% 却没判红的组（该档
# 自身 σ 带更宽）会被逐条点名印出来，见 judge_flags/judge_builds 的 over_floor。
RUN_SCATTER = 0.03


def band(value, sd_a, sd_b):
    """两行的显著性带：组内 σ 合成的 2√(σa²＋σb²)，再垫 RUN_SCATTER 的相对下限。"""
    return max(2 * ((sd_a ** 2 + sd_b ** 2) ** 0.5), abs(value) * RUN_SCATTER)


def flag_rows():
    """每一次 flag 轴扫描的原始行：(build, 档, flag 串, 测试项) → t/s 与 σ。

    同一档的 flag 只与该档自己的基准行比（−ngl 46 的基准去比 −ngl 66 的数，
    量出来的是层数差不是 flag 差），所以这里把 ngl 与后缀拆开存。
    build 也进键：换引擎 build 的 A/B 是两个不同的被测物，把它们当成"同档重复、
    对不上就是台架不可复现"，会把真实的 build 差判成噪声。
    """
    out = []
    for r in rows(LLB):
        if r["test"] not in ("tg128", "pp512"):
            continue
        m = RE_NGL_SUFFIX.match(r["config"] or "")
        tg = _num(r["tps"])
        if not m or tg is None:
            continue
        out.append({"build": r["build"], "ngl": int(m.group(1)),
                    "suffix": m.group(2).strip(),
                    "test": r["test"], "tps": tg, "sd": f(r["stddev"]) or 0.0})
    return out


def flag_table(fl):
    """{(build, ngl): {flag串: {"tg":…, "pp":…, "dup_delta":[(测试项, 差, 临界, 相对差)]}}。

    只留扫过 flag 轴（≥2 个串）的档。tg128 与 pp512 是两次独立测量，重复档要各自
    和自己比：把解码 5.4 和预填 804 当成"同串重复"，量出来的是两项测试各自的数值
    差（~800 t/s），不是复现性。
    """
    tbl = {}
    for r in fl:
        e = tbl.setdefault((r["build"], r["ngl"]), {}).setdefault(
            r["suffix"], {"tg": None, "tg_sd": 0.0, "pp": None, "pp_sd": 0.0,
                          "n": 0, "dup_delta": []})
        k = "pp" if r["test"] == "pp512" else "tg"
        if e[k] is not None:
            d = abs(r["tps"] - e[k])
            e["dup_delta"].append((k, d, band(e[k], e[k + "_sd"], r["sd"]),
                                   d / e[k] if e[k] else 1.0))
        e[k], e[k + "_sd"] = r["tps"], r["sd"]
        e["n"] += 1
    return {n: v for n, v in tbl.items() if len(v) > 1}


def flag_findings(tbl):
    """"解码涨、预填塌"的名单：这是要报告的现象，不是数据错位，所以不判红。

    −nopo 1 实测解码涨 5.6~7.8%（随基准那一次的抖动）而预填从 822 掉到 45 t/s：
    单看解码轴它会被当成提速项采纳，那才是错。线上是"长 prompt 进来先预填"的
    混合负载，所以这条必须摆在表里并点名，让采纳的人看见另一半。
    """
    out = []
    for (build, ngl), flags in sorted(tbl.items()):
        base = flags.get("")
        if not base or not base["pp"] or base["tg"] is None:
            continue
        for suffix, v in sorted(flags.items()):
            if not suffix or not v["pp"] or v["tg"] is None:
                continue
            sig = band(base["tg"], base["tg_sd"], v["tg_sd"])
            gain = (v["tg"] - base["tg"]) / base["tg"] * 100
            if abs(v["tg"] - base["tg"]) > sig and gain > 0 \
                    and v["pp"] < base["pp"] * PREFILL_FLOOR:
                out.append(f"{build} ngl {ngl} `{suffix}`：解码 {gain:+.1f}% 但预填 "
                           f"{(v['pp']/base['pp']-1)*100:+.1f}%，"
                           f"跌破 {round((1-PREFILL_FLOOR)*100)}% 门槛 —— 不能当提速项采纳")
    return out


def judge_flags(tbl):
    """两把红的：每个被测档必须有它自己的基准行；重复档必须自己自洽。

    重复档的临界走 band()：组内 σ 合成带与 RUN_SCATTER 相对下限取宽的那个。散布越过
    下限却因该档自身 σ 更宽而放行的组，要单独点名印出来——不能让它躲在"全绿"里。
    """
    bad = []
    dup_groups, worst = 0, 0.0
    over_floor = []
    for (build, ngl), flags in sorted(tbl.items()):
        base = flags.get("")
        if not base or base["tg"] is None:
            bad.append(f"{build} ngl {ngl} 档没有基准行"
                       f"（config 里没有单独的 ngl={ngl}），这一档的 Δ 全是无根之比")
            continue
        for suffix, v in sorted(flags.items()):
            for kind, d, sig, rel in v["dup_delta"]:
                dup_groups += 1
                worst = max(worst, rel)
                if d > sig:
                    bad.append(f"{build} ngl {ngl} `{suffix}` 重复档自己都对不上"
                               f"（{kind}）：差 {d:.2f} t/s（临界 ±{sig:.2f}），"
                               f"台架不可复现")
                elif rel > RUN_SCATTER:
                    over_floor.append(f"{build} ngl {ngl} `{suffix}` {kind} "
                                      f"散布 {rel * 100:.1f}%（{d:.2f} t/s），"
                                      f"靠该档 σ 带 ±{sig:.2f} 放行")
    if not dup_groups:
        print("   （重复档 0 组：'同档同串必须复现'这条判据本轮没有行使，不是通过）")
    else:
        print(f"   （重复档 {dup_groups} 组，最大散布 {worst * 100:.1f}%；"
              f"显著性相对下限 {RUN_SCATTER * 100:.0f}%）")
        for msg in over_floor:
            print(f"   （越过下限但未判红：{msg}）")
    return bad


def build_table():
    """{ngl: {build: {"tg"/"pp" 与各自 σ、dup}}——只收裸档（config 里没附加 flag 的行）。

    主表和一个 build 一张 A/B 表都读进来按 build 分桶：同一档两个 build 都有才叫对照，
    只有一个 build 跑到的档拿去算 Δ，就是把缺席当成 0。
    """
    src = []
    for path in (LLB, BUILD_CSV):
        if os.path.exists(path):
            src += rows(path)
    out = {}
    for r in src:
        if r["test"] not in ("tg128", "pp512"):
            continue
        m = RE_NGL_SUFFIX.match(r["config"] or "")
        tg = _num(r["tps"])
        if not m or tg is None or m.group(2).strip():
            continue
        e = out.setdefault(int(m.group(1)), {}).setdefault(
            r["build"], {"tg": None, "tg_sd": 0.0, "pp": None, "pp_sd": 0.0,
                         "dup": []})
        k = "pp" if r["test"] == "pp512" else "tg"
        sd = f(r["stddev"]) or 0.0
        if e[k] is not None:
            d = abs(tg - e[k])
            e["dup"].append((k, d, band(e[k], e[k + "_sd"], sd),
                             d / e[k] if e[k] else 1.0))
        e[k], e[k + "_sd"] = tg, sd
    return out


def judge_builds(bt):
    """三把红的：对照至少两个 build；每档两侧都有；每个 build 自己随 -ngl 单调。

    重复行的临界同样走 band()（组内 σ 与 RUN_SCATTER 下限取大），跟【9】一个口径。
    """
    bad = []
    dup_groups, worst = 0, 0.0
    over_floor = []
    builds = sorted({b for v in bt.values() for b in v})
    if len(builds) < 2:
        return [f"【10】只有 {'/'.join(builds) or '没有'} 一个 build 的数据，"
                f"跨 build 这笔账是空的"]
    for ngl, v in sorted(bt.items()):
        if len(v) < 2:
            bad.append(f"ngl {ngl} 只有 {'/'.join(v)} 跑到了，另一侧缺席——"
                       f"这一档的跨 build Δ 没有对照")
    mono = 0
    for b in builds:
        seq = [(n, bt[n][b]["tg"]) for n in sorted(bt)
               if b in bt[n] and bt[n][b]["tg"] is not None]
        if len(seq) < 3:
            continue
        mono += 1
        if any(y <= x for (_, x), (_, y) in zip(seq, seq[1:])):
            bad.append(f"build {b} 解码随 -ngl 不单调：{seq}")
    if not mono:
        print("   （每个 build 的层数轴都不足 3 档：单调性这条判据本轮没有行使，不是通过）")
    for ngl, v in sorted(bt.items()):
        for b, e in sorted(v.items()):
            for kind, d, sig, rel in e["dup"]:
                dup_groups += 1
                worst = max(worst, rel)
                if d > sig:
                    bad.append(f"build {b} ngl {ngl} 重复档自己都对不上（{kind}）："
                               f"差 {d:.2f} t/s（临界 ±{sig:.2f}），台架不可复现")
                elif rel > RUN_SCATTER:
                    over_floor.append(f"build {b} ngl {ngl} {kind} "
                                      f"散布 {rel * 100:.1f}%（{d:.2f} t/s），"
                                      f"靠该档 σ 带 ±{sig:.2f} 放行")
    if not dup_groups:
        print("   （build 表 0 组重复行：'同档同 build 必须复现'这条判据本轮没有行使，不是通过）")
    else:
        print(f"   （build 表重复行 {dup_groups} 组，最大散布 {worst * 100:.1f}%；"
              f"显著性相对下限 {RUN_SCATTER * 100:.0f}%）")
        for msg in over_floor:
            print(f"   （越过下限但未判红：{msg}）")
    return bad


def build_findings(bt):
    """跨 build 的 Δ：超出该档噪声才点名，点名里必须带上【6】是跨 build 拼接这件事。"""
    out = []
    builds = sorted({b for v in bt.values() for b in v})
    if len(builds) != 2:
        return out
    a, z = builds
    for ngl, v in sorted(bt.items()):
        if len(v) != 2 or v[a]["tg"] is None or v[z]["tg"] is None:
            continue
        ea, ez = v[a], v[z]
        sig = band(ea["tg"], ea["tg_sd"], ez["tg_sd"])
        d = ez["tg"] - ea["tg"]
        if abs(d) > sig:
            out.append(f"ngl {ngl}：{z} 比 {a} {'快' if d > 0 else '慢'} "
                       f"{abs(d):.2f} t/s（临界 ±{sig:.2f}）——【6】的线上侧与这里的"
                       f"裸速侧不同 build，倍速里含这部分差")
    return out


def report():
    all_rows = http_rows()
    eng, http = engine_table(), http_tier_table(all_rows)
    bare = {k: v for k, v in eng.items() if "=" in k and " " not in k}

    print("【1】引擎裸速（llama-bench，ctx 640，无视觉无 MTP，逐 -ngl）")
    for k in sorted(bare, key=lambda x: bare[x]["ngl"]):
        v = bare[k]
        print(f"   -ngl {v['ngl']:>3}  解码 {v['tg']:5.1f} t/s   预填 {v['pp']:6.0f} t/s")

    print("\n【2】线上服务（HTTP 台，同一 ctx＋ngl 取最快一次）")
    print("   ctx      -ngl   解码 t/s   长文 t/s   内存权重   运行中最低空闲   裸引擎   MTP 倍速")
    for (ctx, ngl), v in sorted(http.items()):
        e = bare.get(f"ngl={ngl}")
        ratio = v["tg"] / e["tg"] if e else None
        bare_s = f"{e['tg']:5.1f}" if e else "    —"
        ratio_s = f"{ratio:4.2f}×" if ratio else "  —"
        print(f"   {ctx:>6}  {ngl:>5}   {v['tg']:7.1f}   {_s(v['long']):>7}  "
              f"{_s(v['ram_w'], '{:.0f}', '      —'):>8}   "
              f"{_s(v['free_min'], '{:.0f}', '       —')} MiB     "
              f"{bare_s:>7}   {ratio_s}")

    print("\n【3】同档位变体对照（每一行都是一次真实加载，不做合并）")
    print("   实验                   ctx    -ngl  视觉 草稿 线程   解码 t/s   长文 t/s   空闲最低")
    for r in sorted(all_rows, key=lambda x: (x["ctx"], x["ngl"], -x["tg"])):
        print(f"   {r['name']:22} {r['ctx']:>6} {r['ngl']:>5}  {r['vision'] or '-':>3} "
              f"{r['draft'] or '-':>4} {r['threads'] or '-':>4}   {r['tg']:7.1f}   "
              f"{_s(r['long']):>7}   {_s(r['free_min'], '{:.0f}', '      —')}")

    print("\n【3b】跑不起来的档位（同样是数据，说明 frontier 到哪为止）")
    for r in rows(BENCH):
        if _num(r["解码均值"]) or not _num(r["ctx"]):
            continue
        if _num(r["空闲显存"]) is None:
            continue
        why = (r["code"] or "").strip() or "未记录失败原因"
        print(f"   {r['实验']:22} ctx {r['ctx']:>6} -ngl {r['ngl']:>4}  "
              f"加载前空闲 {f(r['空闲显存']):.0f} MiB → {why}")

    base = bare["ngl=66"]
    print(f"\n【4】全驻留（-ngl 66）时的参数扫描，基准解码 {base['tg']:.1f} t/s")
    print(f"   显著性：|Δ| > max(2·√(σ基准²＋σ该项²), {RUN_SCATTER * 100:.0f}%×基准) 才算真的动了，"
          f"否则算噪声（llama-bench 的 σ 只含组内重复，run 间漂移靠这个下限垫）")
    for k in sorted(eng):
        if k.startswith("ngl=66 "):
            v = eng[k]
            sig = band(base["tg"], base["sd"], v["sd"])
            mark = "  ← 真的动了" if abs(v["tg"] - base["tg"]) > sig else ""
            print(f"   {k[8:]:32} 解码 {v['tg']:5.1f}（{v['tg'] - base['tg']:+5.2f}"
                  f"，临界 ±{sig:.2f}）  预填 {v['pp']:6.0f}{mark}")
    print(f"   （基准 ngl=66 无附加参数：预填 {base['pp']:.0f} t/s，σ={base['sd']:.2f}）")

    print("\n【5】KV：引擎逐条打印 vs lmloader 曾经报出的数")
    print("   （主干那条是全部层都在卡上的口径；-ngl 60 时同一个 1152 会拆成卡上 1080＋CPU 72）")
    for ctx, lines in sorted(KV_LINES.items()):
        s = sum(x[2] for x in lines)
        print(f"   ctx {ctx:>6}  逐条 {'＋'.join(f'{x[2]:.0f}({x[3]})' for x in lines)}"
              f" ＝ {s:.0f} MiB ｜ 修正前只报 {LASTWIN_REPORTED[ctx]} MiB"
              f"（少 {s - LASTWIN_REPORTED[ctx]:.0f}，{s / LASTWIN_REPORTED[ctx]:.1f}×）")
    print("   修正后 lmloader 报的是逐条相加（同 (device,kind) 下按打印函数＋大小去重，"
          "重复打印不重复计），锚在 tests/test_memory_breakdown.py::MTP_LOG")

    print("\n【6】跨源判据（线上 vs 裸引擎 vs 带宽定律）")
    bad = judge(http, bare)
    if bad:
        for b in bad:
            print("   ✗ " + b)

    spec = spec_rows()
    print("\n【7】引擎直连 A/B（ctx 16000／-ngl 66，逐档都核对过引擎自报的生效配置）")
    if not spec:
        print("   没有 spec_flags_results.csv——这一段的账是空的，不是通过")
    else:
        ref = next((s["tg"] for s in spec if s["variant"] == "base"), None)
        print("   变体                    深度  解码 t/s      Δ vs base   接受率   引擎RS   元数据RS   显存")
        for s in spec:
            depth = 0 if s["no_draft"] else s["depth"]
            want = recurrent_state_mb(SSM, depth, 1) if depth is not None else None
            delta = "基准" if s["variant"] == "base" else (
                f"{(s['tg'] / ref - 1) * 100:+.1f}%" if ref and s["tg"] else "—")
            print(f"   {s['variant']:22} {depth if depth is not None else '?':>3}  "
                  f"{_s(s['tg']):>9}   {delta:>9}   {_s(s['acc'], '{:.1f}', '  —')}   "
                  f"{_s(s['rs_seen'], '{:.0f}', ' —')}    "
                  f"{f'{want:.0f}' if want is not None else '—':>6}   "
                  f"{_s(s['vram'], '{:.0f}', '  —')}")
        spec_bad = judge_spec(spec)
        print("   判据：", "全绿（参数生效、状态与元数据对得上、深度单调）"
              if not spec_bad else "")
        for b in spec_bad:
            print("   ✗ " + b)
        bad = bad + spec_bad

    fit = fit_rows()
    print("\n【8】引擎自带解算器（llama-fit-params）vs 项目规划器——同一条实时空闲显存")
    if not fit:
        print("   没有 fit_ab_results.csv——跑 _pydiag/fit_ab.py，这一段的账是空的，不是通过")
    else:
        print("   ctx      空闲   保留  引擎解算  规划器   Δ    引擎估这一档  可用   离线上下文  实测主干＋状态")
        for r in sorted(fit, key=lambda x: (x["ctx"] or 0)):
            want = (KV_LINES[r["ctx"]][0][2] + r["rs0"]) if (r["ctx"] in KV_LINES
                                                            and r["rs0"]) else None
            d = (r["p"] - r["engine"]) if None not in (r["p"], r["engine"]) else None
            avail = (r["free"] - r["margin"]) if None not in (r["free"], r["margin"]) else None
            print(f"   {f0(r['ctx'], 6)} {f0(r['free'], 7)} {f0(r['margin'], 5)} "
                  f"{f0(r['engine'], 6)} {f0(r['p'], 6)} "
                  f"{('%+d' % d if d is not None else '—'):>5}   "
                  f"{f0(r['est_total'], 9)} {f0(avail, 7)} {f0(r['est_ctx_full'], 9)}   "
                  f"{_s(want, '{:>10.1f}')}")
        fit_bad = judge_fit(fit)
        print("   判据：", "全绿（两侧各自单调、引擎算术为规划器的层数背书、离线上下文等于实测主干那份、层数不分歧）"
              if not fit_bad else "")
        for b in fit_bad:
            print("   ✗ " + b)
        bad = bad + fit_bad

    tbl = flag_table(flag_rows())
    print(f"\n【9】flag 轴 × 卸载深度（同一档只与该档自己的基准行比；"
          f"临界 = max(2·√(σ基²＋σ项²), {RUN_SCATTER * 100:.0f}%×基准)）")
    if not tbl:
        print("   没有任何档扫过 flag 轴——这一段的账是空的，不是通过")
    flag_bad = judge_flags(tbl)
    for (build, ngl), flags in sorted(tbl.items()):
        base = flags.get("")
        if not base or base["tg"] is None:
            print(f"   {build} −ngl {ngl}：缺基准行，整档不印（印出来的 Δ 会是没有根号的数）")
            continue
        print(f"   {build} −ngl {ngl}（基准：解码 {base['tg']:.2f} t/s，"
              f"预填 {f0(base['pp'], 5)} t/s）")
        for suffix, v in sorted(flags.items()):
            if not suffix or v["tg"] is None:
                continue
            sig = band(base["tg"], base["tg_sd"], v["tg_sd"])
            dp = (v["pp"] / base["pp"] - 1) * 100 if base["pp"] and v["pp"] else None
            print(f"      {suffix:34} 解码 {v['tg']:6.2f}"
                  f"（{(v['tg']/base['tg']-1)*100:+6.1f}%，临界 ±{sig/base['tg']*100:4.1f}%）"
                  f" 预填 {f0(v['pp'], 6)}（{f'{dp:+6.1f}%' if dp is not None else '    —'}）"
                  f"{'  ← 真的动了' if abs(v['tg'] - base['tg']) > sig else '  噪声'}")
    for msg in flag_findings(tbl):
        print("   ⚠ " + msg)
    if flag_bad:
        for b in flag_bad:
            print("   ✗ " + b)
        bad = bad + flag_bad

    bt = build_table()
    builds = sorted({b for v in bt.values() for b in v})
    print("\n【10】引擎 build A/B（同一个 llama-bench 台架，逐档对照裸速 解码±σ／预填）")
    build_bad = []
    if len(builds) < 2:
        print(f"   只有 {'/'.join(builds) or '没有'} 一个 build 的数据——"
              f"这一段的账是空的，不是通过")
    else:
        print(f"   -ngl  " + "  ".join(f"{b:>22}" for b in builds) + "   Δ（末侧−首侧）")
        for ngl, v in sorted(bt.items()):
            cells, vals = [], []
            for b in builds:
                e = v.get(b)
                if e and e["tg"] is not None:
                    cells.append(f"{e['tg']:8.2f}±{e['tg_sd']:.2f}／{f0(e['pp'], 5):>7}")
                    vals.append(e["tg"])
                else:
                    cells.append(f"{'— 这侧没跑到 —':>22}")
            delta = (f"{vals[-1] - vals[0]:+7.2f}"
                     if len(vals) == len(builds) == 2 else "      —")
            print(f"   {ngl:>4}  " + "  ".join(cells) + f"   {delta}")
        build_bad = judge_builds(bt)
        for msg in build_findings(bt):
            print("   ⚠ " + msg)
        if build_bad:
            for b in build_bad:
                print("   ✗ " + b)
        else:
            print("   判据： 全绿（两侧同档、每个 build 自己随层数单调、重复档自洽）")
        bad = bad + build_bad
    return bad


def do_break():
    """反向对照：十把锚点各自去动 `judge`/`judge_spec`/`judge_fit` 真正读的那份数据。

    判据不许在这里重写一遍——重写版的锚点只能证明抄的那段代码会红。
    A/B/C 压跨源三张表，D/E/F 压 MTP 直连 A/B，G/H/I/J 压引擎解算器对表：
    G 口径分歧、H 逐行的显存越界、I 离线上下文与实测 buffer 的对账、J 单调性。
    """
    eng = engine_table()
    bare = {k: v for k, v in eng.items() if "=" in k and " " not in k}
    http = http_tier_table()
    base_bad = judge(deepcopy(http), deepcopy(bare), verbose=False)
    if base_bad:
        print("  ✗ 未变异就已经报红，先修数据再谈锚点：")
        for b in base_bad:
            print("    -", b)
        return 1
    hits = 0

    def once(label, needle, tamper):
        """变异后 judge 必须恰好在含 needle 的那条上红一次。"""
        h, b = deepcopy(http), deepcopy(bare)
        tamper(h, b)
        bad = [x for x in judge(h, b, verbose=False) if needle in x]
        if len(bad) != 1:
            print(f"  {label}) ✗ 变异后『{needle}』命中 {len(bad)} 次（要恰好 1 次）：{bad}")
            return 0
        print(f"  {label}) ✓ {needle}：{bad[0]}")
        return 1

    # A) 把最小卸载档的裸速抬到比全驻留还快：单调性判据必须红
    hits += once("A", "不单调",
                 lambda h, b: b["ngl=46"].__setitem__("tg", b["ngl=66"]["tg"] + 1))
    # B) 只把 64k/46 这档的线上速度缩小十倍：倍速出界，速度序仍然单调
    hits += once("B", "倍速",
                 lambda h, b: h[(65536, 46)].__setitem__("tg", h[(65536, 46)]["tg"] / 10))
    # C) 把 64k/52 的内存侧权重报成实际的 2.5 倍：带宽定律离散必须红
    hits += once("C", "带宽定律",
                 lambda h, b: h[(65536, 52)].__setitem__("ram_w", h[(65536, 52)]["ram_w"] * 2.5))

    spec = spec_rows()
    if not spec:
        print("  · 跳过 D/E/F：没有 spec_flags_results.csv，【7】本身就没有数据")
    else:
        base_spec = judge_spec(deepcopy(spec))
        if base_spec:
            print("  ✗ 【7】未变异就报红，先修数据：", base_spec)
            return 1

        def once_spec(label, needle, tamper):
            s = deepcopy(spec)
            tamper(s)
            bad = [x for x in judge_spec(s) if needle in x]
            if len(bad) != 1:
                print(f"  {label}) ✗ 变异后『{needle}』命中 {len(bad)} 次（要恰好 1 次）：{bad}")
                return 0
            print(f"  {label}) ✓ {needle}：{bad[0]}")
            return 1

        # D) 把深度 1 那档的状态说成 449（深度 2 的数）：元数据↔引擎对账必须红
        hits += once_spec("D", "元数据推",
                          lambda s: next(x for x in s if x["variant"] == "nmax1")
                          .__setitem__("rs_seen", 449.0))
        # E) 关掉草稿那档的状态抬到比深度 4 还大：深度 ladder 单调性必须红
        hits += once_spec("E", "不单调",
                          lambda s: next(x for x in s if x["variant"] == "no_mtp")
                          .__setitem__("rs_seen", 900.0))
        # F) 抹掉一档的生效配置（depth 就是 judge_spec 从引擎自报配置里解出来的
        #    那个数）：这一行不再能证明参数进了引擎，必须红
        hits += once_spec("F", "没有生效配置",
                          lambda s: next(x for x in s if x["variant"] == "ngld0")
                          .__setitem__("depth", None))
    fit = fit_rows()
    if not fit:
        print("  · 跳过 G/H/I/J：没有 fit_ab_results.csv，【8】本身就没有数据")
    else:
        base_fit = judge_fit(deepcopy(fit))
        if base_fit:
            print("  ✗ 【8】未变异就报红，先修数据：", base_fit)
            return 1

        def once_fit(label, needle, tamper):
            r = deepcopy(fit)
            tamper(r)
            bad = [x for x in judge_fit(r) if needle in x]
            if len(bad) != 1:
                print(f"  {label}) ✗ 变异后『{needle}』命中 {len(bad)} 次（要恰好 1 次）：{bad}")
                return 0
            print(f"  {label}) ✓ {needle}：{bad[0]}")
            return 1

        # G) 把最长那档的规划器层数抬到引擎解算＋3：口径分歧判据必须红
        hits += once_fit("G", "层数分歧",
                         lambda r: max(r, key=lambda x: x["ctx"])
                         .__setitem__("p", max(r, key=lambda x: x["ctx"])["engine"] + 3))
        # H) 只把 8192 这档的引擎估算抬到越过『空闲−保留』：安全判据必须红，
        #    而且不能顺手把别的档也点着（只动一行才测得出判据是逐行的）
        hits += once_fit("H", "越过",
                         lambda r: next(x for x in r if x["ctx"] == 8192)
                         .__setitem__("est_total", r[0]["free"] + 5000))
        # I) 把离线上下文说成 9000：与实测主干 KV＋状态那份对账必须红
        hits += once_fit("I", "上下文对不上",
                         lambda r: next(x for x in r if x["ctx"] == 16000)
                         .__setitem__("est_ctx_full", 9000.0))
        # J) 把最长那档的引擎层数抬到比最短档还高：ctx 变大层数还变多，单调性判据
        #    必须红。（反过来抬最短档是"更早就能多放层"，不违反非增，锚点会假绿。）
        hits += once_fit("J", "不单调",
                         lambda r: max(r, key=lambda x: x["ctx"]).__setitem__(
                             "engine", min(x["engine"] for x in r) + 4))
    tbl = flag_table(flag_rows())
    if not tbl:
        print("  · 跳过 K/L/M：没有扫过 flag 轴的档，【9】本身就没有数据")
    else:
        base_flag = judge_flags(deepcopy(tbl))
        if base_flag:
            print("  ✗ 【9】未变异就报红，先修数据：", base_flag)
            return 1

        def once_flag(label, needle, want, tamper):
            t = deepcopy(tbl)
            tamper(t)
            got = [x for x in judge_flags(t) if needle in x]
            if len(got) != want:
                print(f"  {label}) ✗ 变异后『{needle}』命中 {len(got)} 次（要 {want} 次）：{got}")
                return 0
            print(f"  {label}) ✓ {needle}" + (f"：{got[0]}" if want else "（没被误报）"))
            return 1

        small = min(tbl)
        # K) 删掉最短那档的基准行：该档所有 Δ 失去分母，必须报红（只报这一档）
        hits += once_flag("K", "没有基准行", 1,
                          lambda t: t[small].pop("", None))

        # L) 重复档自洽是双向的：越过显著性下限要拦住，落在下限之内要放行。
        #    只测反侧的话，把这条分支写成恒报错也能骗过 K。
        key = next(k for k, v in sorted(tbl[small].items()) if k)
        # 锚定 RUN_SCATTER 这条**相对**下限，所以刻意把 σ 取 0：临界只剩 3%×该档速度，
        # 5% 必红、2% 必绿，两个方向都只有"真有下限"才同时成立。
        # （不取 0 的话，组内 σ 合成带可能比 3% 还宽，锚点量的就不是下限而是 σ。）
        lv = tbl[small][key]["tg"]
        lsig = band(lv, 0.0, 0.0)
        hits += once_flag("L1", "重复档", 1,
                          lambda t: t[small][key].update(
                              {"n": 2, "dup_delta": [("tg", lv * 1.05, lsig, 1.05)]}))
        hits += once_flag("L2", "重复档", 0,
                          lambda t: t[small][key].update(
                              {"n": 2, "dup_delta": [("tg", lv * 0.02, lsig, 0.02)]}))

        # M) "解码涨、预填塌"的点名也要有正反两侧：真实数据必须点到 −nopo 1；
        #    把预填改回基准值之后必须不再点它——否则这条名单是硬编码的。
        find = [x for x in flag_findings(deepcopy(tbl)) if "-nopo 1" in x]
        if len(find) != 1:
            print(f"  M1) ✗ 真实数据里『-nopo 1』被点名 {len(find)} 次（要恰好 1 次）")
            hits += 0
        else:
            print(f"  M1) ✓ 点名预填塌陷：{find[0]}")
            hits += 1

        def repaired(t):
            b = t[small][""]
            for k, v in t[small].items():
                if "-nopo 1" in k:
                    v["pp"] = b["pp"]
        t2 = deepcopy(tbl)
        repaired(t2)
        gone = [x for x in flag_findings(t2) if "-nopo 1" in x]
        if gone:
            print(f"  M2) ✗ 修好预填后还在点名：{gone}")
            hits += 0
        else:
            print("  M2) ✓ 把预填改回基准值后不再点名（这条名单是算出来的，不是抄的）")
            hits += 1
    bt = build_table()
    bbuilds = sorted({b for v in bt.values() for b in v})
    if len(bbuilds) < 2:
        print("  · 跳过 N：A/B 表里不足两个 build，【10】本身就没有数据")
    else:
        base_build = judge_builds(deepcopy(bt))
        if base_build:
            print("  ✗ 【10】未变异就报红，先修数据：", base_build)
            return 1

        def once_build(label, needle, want, tamper):
            t = deepcopy(bt)
            tamper(t)
            got = [x for x in judge_builds(t) if needle in x]
            if len(got) != want:
                print(f"  {label}) ✗ 变异后『{needle}』命中 {len(got)} 次（要 {want} 次）：{got}")
                return 0
            print(f"  {label}) ✓ {needle}" + (f"：{got[0]}" if want else "（没被误报）"))
            return 1

        # N0) 把对照那一侧整个抹掉：只剩一个 build 时判据必须说"这笔账是空的"，
        #     不能因为没有可比的档就一条红都不报（那正是"没话可说=通过"）。
        hits += once_build("N0", "这笔账是空的", 1,
                           lambda t: [v.pop(bbuilds[-1], None) for v in t.values()])
        # N1) 单档缺席：那一档的跨 build Δ 失去对照，只许报那一档
        med = sorted(bt)[len(bt) // 2]
        hits += once_build("N1", "没有对照", 1,
                           lambda t: t[med].pop(bbuilds[0], None))
        # N2) 把某个 build 最小层的解码抬到超过它最大层：该 build 自己的层数单调性必须红
        big = max(n for n in bt if bbuilds[0] in bt[n] and bt[n][bbuilds[0]]["tg"])
        hits += once_build("N2", "不单调", 1,
                           lambda t: t[min(bt)][bbuilds[0]].__setitem__(
                               "tg", bt[big][bbuilds[0]]["tg"] + 5))
    expect = (3 + (3 if spec else 0) + (4 if fit else 0) + (5 if tbl else 0)
              + (3 if len(bbuilds) >= 2 else 0))
    if hits != expect:
        return 1
    print(f"  ✓ {hits} 把锚点各命中一次")
    return 0


if __name__ == "__main__":
    if "--break" in sys.argv[1:]:
        sys.exit(do_break())
    bad = report()
    if bad:
        print("\n判据变红，上面的表不能用")
        sys.exit(1)
    sys.exit(0)
