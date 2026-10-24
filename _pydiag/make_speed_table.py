"""把散落的测速记录合成一张表，存成本地 CSV（Excel 可直接打开）。

三个来源的字段完整度不一样，这里不假装它们同构：
  * calib_rows.jsonl —— 闭环标定的 6 档，字段齐全（引擎自报四项分解＋空闲起/最低＋逐条速度）；
  * 本轮三次补测（恢复基线后复测、draft 3、最终配置复测）—— 只有自报分解与四条速度；
  * bench_results.csv —— 台架 R1~R5 共 24 行。**表头是 26 字段，R4 起写成 30/31 字段**，
    所以 len!=26 的行只采信速度段（列 0~15 在新旧两种顺序里对齐），显存列一律留空
    并在备注写明"列位漂移"。之前就是因为这个漂移，标定不肯用 CSV 当数据源。

空值一律留空，不填 0，也不把某一趟的显存数字挪到另一趟上。

    python _pydiag/make_speed_table.py            # 生成 speed_records_2026-10-21.csv
    python _pydiag/make_speed_table.py --check    # 只核对行数与若干锚点值
    python _pydiag/make_speed_table.py --break    # 反向对照：抄数出错必须被点名
"""
import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(HERE, "calib_rows.jsonl")
BENCH = os.path.join(HERE, "bench_results.csv")
OUT = os.path.join(HERE, "speed_records_2026-10-21.csv")
MODEL = "Qwen3.8-27B-NVFP4-MTP-LOW.gguf"

FIELDS = ["来源", "时间", "台架/实验", "模型", "ctx", "ngl", "解算ngl", "线程", "draft",
          "mtp", "视觉", "code", "prose", "math", "long", "短题均值", "解码均值",
          "MTP命中率", "命中长度", "显存权重", "显存KV", "显存状态", "显存计算",
          "自报合计", "内存权重", "空闲起", "生成中最低", "越界", "结果", "备注"]

# 本轮三次补测：现场只打印过、没落进任何文件，按日志 [timing] 行的时刻手工登记。
EXTRA = [
    {"台架/实验": "recheck-66-16k-d2", "时间": "2026-10-21 14:02:32",
     "ctx": 16000, "ngl": 66, "线程": 8, "draft": 2, "mtp": "True", "视觉": "开",
     "code": 75.5, "prose": 51.3, "math": 75.0, "long": 68.4,
     "显存权重": 14160, "显存KV": 346, "显存状态": 449, "显存计算": 541,
     "内存权重": 644, "空闲起": "", "生成中最低": 81,
     "结果": "成", "备注": "恢复基线后在活实例上复测，未重新加载（同 13:40:57 那趟 PID 38744）。"
                          "长题 prefill 命中提示词缓存（6877 token @2026.9 t/s），与标定档不等价"},
    {"台架/实验": "probe-draft3-66-16k", "时间": "2026-10-21 14:11:55",
     "ctx": 16000, "ngl": 66, "线程": 8, "draft": 3, "mtp": "True", "视觉": "开",
     "code": 72.7, "prose": 44.5, "math": 77.5, "long": 67.5,
     "显存权重": 14160, "显存KV": 346, "显存状态": 598, "显存计算": 541,
     "内存权重": 644.14, "空闲起": 15562, "生成中最低": 77,
     "结果": "成", "备注": "为最快配置补测：短题均值 64.9 落在 draft 2 的 run 间散布带内，"
                          "却多驻留 149MB（越界 +83MB vs draft2 的 +4.6MB）→ 不采纳"},
    {"台架/实验": "final-66-16k-d2", "时间": "2026-10-21 14:13:36",
     "ctx": 16000, "ngl": 66, "线程": 8, "draft": 2, "mtp": "True", "视觉": "开",
     "code": 73.0, "prose": 52.0, "math": 70.3, "long": 69.4,
     "显存权重": 14160, "显存KV": 346, "显存状态": 449, "显存计算": 541,
     "内存权重": 644, "空闲起": 15492, "生成中最低": 81,
     "结果": "成", "备注": "最终配置复测：配置接口回读 13 项一致，自报 状态 449 证明按 draft 2 起"},
]


def blank():
    return {k: "" for k in FIELDS}


def from_calib(row):
    r = blank()
    dec = row.get("decode") or []
    r.update({
        "来源": "闭环标定", "时间": row.get("ts"), "台架/实验": f"calib {row['ngl']}@{row['ctx']}",
        "模型": MODEL, "ctx": row["ctx"], "ngl": row["ngl"], "线程": row.get("threads"),
        "draft": row.get("spec_draft_max"), "mtp": "True",
        "视觉": "开" if row.get("vision") else "关",
        "code": dec[0] if len(dec) > 0 else "", "prose": dec[1] if len(dec) > 1 else "",
        "math": dec[2] if len(dec) > 2 else "", "long": dec[3] if len(dec) > 3 else "",
        "短题均值": row.get("short_mean"), "解码均值": row.get("decode_mean"),
        "显存权重": row.get("self_w"), "显存KV": row.get("self_kv"),
        "显存状态": row.get("self_state"), "显存计算": row.get("self_calc"),
        "自报合计": row.get("self_total"), "内存权重": row.get("ram_w"),
        "空闲起": row.get("free_idle"), "生成中最低": row.get("free_low"),
        "越界": (round(row["self_total"] - row["free_idle"], 1)
                 if row.get("self_total") and row.get("free_idle") else ""),
        "结果": "成" if row.get("ok") else "败",
        "备注": (row.get("note") or "")[:180],
    })
    if r["结果"] == "败" and not r["code"]:
        r["备注"] = "加载成功，请求阶段 502"
    return r


BENCH_HDR = ["实验", "视觉", "ctx", "ngl", "解算ngl", "线程", "draft", "mtp", "ubatch", "kv",
             "fa", "code", "prose", "math", "解码均值", "MTP%均值", "命中长度", "显存权重",
             "显存KV", "显存状态", "显存计算", "内存权重", "显存自报合计", "预计", "空闲显存",
             "生成长度"]


def from_bench(cells):
    """26 字段的行按表头逐列取；30/31 字段的行只取到速度段，显存列留空。"""
    r = blank()
    aligned = len(cells) == len(BENCH_HDR)
    get = lambda i: cells[i] if i < len(cells) else ""
    dec = [get(11), get(12), get(13)] + ([get(14)] if not aligned else [""])
    mean_idx = 14 if aligned else 15
    sp = [float(x) for x in dec[:3] if x not in ("", "失败") and x.replace(".", "").isdigit()]
    r.update({
        "来源": "测速台架" + ("" if aligned else "（列位漂移，显存列不采信）"),
        "台架/实验": get(0), "模型": MODEL, "视觉": get(1), "ctx": get(2), "ngl": get(3),
        "解算ngl": get(4), "线程": get(5), "draft": get(6), "mtp": get(7),
        "视觉": "开" if get(1) in ("True", "开") else "关",
        "code": dec[0], "prose": dec[1], "math": dec[2], "long": dec[3],
        "短题均值": round(statistics.mean(sp), 1) if len(sp) == 3 else "",
        "解码均值": get(mean_idx),
        "MTP命中率": get(15) if aligned else get(16),
        "命中长度": get(16) if aligned else get(17),
        "结果": "失败" if dec[0] == "失败" else ("成" if dec[0] else ""),
    })
    if aligned:
        r.update({
            "显存权重": get(17), "显存KV": get(18), "显存状态": get(19),
            "显存计算": get(20), "内存权重": get(21), "自报合计": get(22),
            "空闲起": get(24),
            "备注": f"预计 {get(23)}MB；生成长度 {get(25)}",
        })
    else:
        r["备注"] = (f"该行 {len(cells)} 字段 vs 表头 {len(BENCH_HDR)}，"
                     f"提示词长度 {get(27)}｜prefill {get(28)}｜生成长度 {get(29)}")[:180]
    return r


def build():
    out = []
    if os.path.exists(ROWS):
        with open(ROWS, encoding="utf-8") as fh:
            out += [from_calib(json.loads(line)) for line in fh if line.strip()]
    for e in EXTRA:
        r = blank()
        r.update({"来源": "补测", "模型": MODEL})
        for k, v in e.items():
            r[k] = v
        r["短题均值"] = round(statistics.mean([e["code"], e["prose"], e["math"]]), 1)
        r["解码均值"] = round(statistics.mean([e["code"], e["prose"], e["math"], e["long"]]), 1)
        r["自报合计"] = (e["显存权重"] + e["显存KV"] + e["显存状态"] + e["显存计算"])
        if e.get("空闲起"):
            r["越界"] = round(r["自报合计"] - e["空闲起"], 1)
        out.append(r)
    if os.path.exists(BENCH):
        rows = list(csv.reader(open(BENCH, encoding="utf-8-sig")))
        out += [from_bench(cells) for cells in rows[1:] if cells]
    return out


def check(out):
    bad = []
    if len(out) != 33:
        bad.append(f"【行数】总行数 {len(out)}，应为 6 标定＋3 补测＋24 台架＝33")
    cal = [r for r in out if r["来源"] == "闭环标定"]
    if sum(1 for r in cal if r["结果"] == "败") != 1:
        bad.append("【失败档】标定段应有且只有 1 行失败（66@65536 的 502）")
    if not all(r["自报合计"] for r in cal):
        bad.append("【缺数】标定段有行缺自报合计")
    drift = [r for r in out if "列位漂移" in r["来源"]]
    if len(drift) != 9:
        bad.append(f"【漂移数】漂移行 {len(drift)} 条，应为 9 条（R4 的 7 行＋收尾 auto 与 t8 复跑）")
    if any(r["显存权重"] for r in drift):
        bad.append("【不抄错位】漂移行竟然填了显存列——那是错位抄来的数")
    anchor = next((r for r in out if r["台架/实验"] == "calib 66@16000"), None)
    if not anchor or anchor["code"] != 72.3 or anchor["越界"] != 4.6:
        bad.append(f"【锚1】锚点 calib 66@16000 抄错：{anchor and (anchor['code'], anchor['越界'])}")
    d3 = next((r for r in out if r["台架/实验"] == "probe-draft3-66-16k"), None)
    if not d3 or d3["越界"] != 83.0 or d3["显存状态"] != 598:
        bad.append(f"【锚2】锚点 draft3 抄错（越界应为 +83、状态 598）："
                   f"{d3 and (d3['越界'], d3['显存状态'])}")
    nomtp = next((r for r in out if r["台架/实验"] == "nomtp"), None)
    if not nomtp or nomtp["解码均值"] != "36.0":
        bad.append("【锚3】锚点 nomtp 的解码均值不是 36.0（对照组丢了）")
    return bad


def break_control():
    """反向对照：每把变异都必须让自己那条核对变红，且只红那一条以外的都算噪声。

    变异打在"已经建好的表"上，等价于抄数时手抖——不是改判据本身。
    """
    base = check(build())
    if base:
        print("  基线核对就是红的，变异对照失去意义（先修基线）")
        for b in base:
            print("   -", b)
        return 1
    cases = []

    def drop(out):
        out.pop()

    def fake_mem(out):
        next(r for r in out if "列位漂移" in r["来源"])["显存权重"] = 14160

    def wrong_speed(out):
        next(r for r in out if r["台架/实验"] == "calib 66@16000")["code"] = 99.9

    def wrong_draft(out):
        r = next(x for x in out if x["台架/实验"] == "probe-draft3-66-16k")
        r["显存状态"], r["越界"] = 449, 4.6

    def lost_control(out):
        out.remove(next(r for r in out if r["台架/实验"] == "nomtp"))

    cases = [("M1 少一行（漏记）", "【行数】", drop),
             ("M2 把漂移行的显存列硬填上", "【不抄错位】", fake_mem),
             ("M3 顶层档的 code 抄成 99.9", "【锚1】", wrong_speed),
             ("M4 draft3 的状态/越界写成 draft2 的数", "【锚2】", wrong_draft),
             ("M5 对照组 nomtp 整行丢失", "【锚3】", lost_control)]
    fails = []
    for desc, want, fn in cases:
        mutated = build()
        fn(mutated)
        got = [b for b in check(mutated) if b not in base]
        named = [b for b in got if b.startswith(want)]
        print(f"   {desc}：新增红 {len(got)} 条，"
              + (f"点名{want}→{named[0][:70]}" if named else f"没有点名{want}"))
        if not named:
            fails.append(f"{desc} 是 no-op")
    if fails:
        print("  反向对照失败：")
        for f in fails:
            print("   -", f)
        return 1
    print(f"  如预期报红：{len(cases)} 把变异各自打红对应的核对项")
    return 0


def main():
    out = build()
    if "--break" in sys.argv:
        return break_control()
    if "--check" in sys.argv:
        bad = check(out)
        for b in bad:
            print("   -", b)
        print(f"  核对：{len(out)} 行，" + ("判据全过" if not bad else f"{len(bad)} 条变红"))
        return 1 if bad else 0
    bad = check(out)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(out)
    print(f"  已写出 {OUT}：{len(out)} 行 × {len(FIELDS)} 列")
    print("  各来源行数：" + ", ".join(
        f"{src}={sum(1 for r in out if r['来源'].startswith(src))}"
        for src in ("测速台架", "闭环标定", "补测")))
    if bad:
        print("  注意，行数/锚点核对变红（文件已写出，数字要人工再看一遍）：")
        for b in bad:
            print("   -", b)
        return 1
    print("  锚点核对通过：33 行、标定段 1 行失败、9 条漂移行留空显存列、"
          "calib 66@16000／draft3／nomtp 三个数抄对")
    return 0


if __name__ == "__main__":
    sys.exit(main())
