"""A/B 实测：同一模型同一设置，只有 GPU 层数档位不同。

A = 他现在的存档（-ngl 999，22GB 权重压进 16GB 卡，靠 WDDM 分页）
B = 自动分层（-ngl 999 + --n-cpu-moe N，注意力与 KV 全驻留，专家按层下内存）

速度只认引擎自己打的 print_timing 行（按日志字节偏移只取本次请求新增的
部分，不跨 run 借证据）。跑完把基线配置原样写回。
"""
import json
import re
import sys
import time

import httpx

BASE = "http://127.0.0.1:8900"
NAME = "Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
LOG = "logs/server.log"
PROMPT = "请写一段 about 秋天的杭州的说明文字，具体一些，不要列提纲。"
MAX_TOKENS = 200
RUNS = 3

DECODE = re.compile(r"\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens .*?([\d.]+) tokens per second")
PREFILL = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens .*?([\d.]+) tokens per second")
NCMOE = re.compile(r"--n-cpu-moe (\d+)")   # 来自规划器写进事件日志的账目行
MEMO = re.compile(r"显存 权重 \d+ \+ KV \d+ \+ 状态 \d+ \+ 计算 \d+ MiB；内存 权重 \d+")


def log_off():
    return __import__("os").path.getsize(LOG)


def log_new(off):
    with open(LOG, "rb") as f:
        f.seek(off)
        return f.read().decode("utf-8", "replace")


def chat(client, off):
    r = client.post(f"{BASE}/v1/chat/completions", json={
        "model": NAME, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS, "temperature": 0, "stream": False,
    }, timeout=600)
    r.raise_for_status()
    txt = log_new(off)
    d = DECODE.findall(txt)
    p = PREFILL.findall(txt)
    out_tok = r.json().get("usage", {}).get("completion_tokens", 0)
    return {
        "decode_tps": float(d[-1][2]) if d else None,
        "decode_tokens": int(d[-1][1]) if d else 0,
        "prefill_tps": float(p[-1][2]) if p else None,
        "usage_tokens": out_tok,
    }


def set_ngl(client, ngl):
    r = client.post(f"{BASE}/api/models/{NAME}/config", json={"n_gpu_layers": ngl}, timeout=30)
    r.raise_for_status()
    return r.json()["config"]


def load(client):
    r = client.post(f"{BASE}/api/models/load", json={"model_name": NAME, "keep_previous": False},
                    timeout=900)
    r.raise_for_status()
    return r


def unload(client):
    client.post(f"{BASE}/api/models/unload", params={"model_name": NAME}, timeout=120)


def run_arm(client, label, ngl, baseline):
    cfg = set_ngl(client, ngl)
    print(f"[{label}] n_gpu_layers={ngl} ctx={cfg['ctx_size']} 已存档", flush=True)
    off = log_off()
    t0 = time.time()
    load(client)
    boot = time.time() - t0
    boot_txt = log_new(off)
    nc = NCMOE.search(boot_txt)
    memo = MEMO.search(boot_txt)
    print(f"[{label}] 加载 {boot:.1f}s  --n-cpu-moe={nc.group(1) if nc else '未出现'}", flush=True)
    print(f"[{label}] {memo.group(0) if memo else '无显存分解行'}", flush=True)

    chat(client, log_off())          # 预热，丢弃
    rows = []
    for i in range(RUNS):
        off = log_off()
        t1 = time.time()
        row = chat(client, off)
        row["wall_tps"] = row["usage_tokens"] / (time.time() - t1) if row["usage_tokens"] else None
        rows.append(row)
        print(f"[{label}] run{i + 1}: decode {row['decode_tps']} t/s "
              f"({row['decode_tokens']} tok)  prefill {row['prefill_tps']} t/s  "
              f"含往返 {row['wall_tps'] and round(row['wall_tps'], 1)} t/s", flush=True)
    unload(client)
    time.sleep(4)
    dec = [r["decode_tps"] for r in rows if r["decode_tps"]]
    pre = [r["prefill_tps"] for r in rows if r["prefill_tps"]]
    return {
        "label": label, "ngl": ngl, "ncmoe": nc.group(1) if nc else None,
        "boot_s": round(boot, 1),
        "mem": memo.group(0) if memo else None,
        "decode_mean": round(sum(dec) / len(dec), 1) if dec else None,
        "decode_runs": dec,
        "prefill_mean": round(sum(pre) / len(pre), 1) if pre else None,
    }


def main():
    baseline = json.load(open("_pydiag/ab_baseline_config.json", encoding="utf-8"))["config"]
    client = httpx.Client()
    results = []
    try:
        for label, ngl in (("A 手动999", 999), ("B 自动分层", -1)):
            results.append(run_arm(client, label, ngl, baseline))
    finally:
        body = {k: baseline[k] for k in
                ("ctx_size", "n_gpu_layers", "threads", "flash_attn", "cache_type_k",
                 "cache_type_v", "parallel", "batch_size", "ubatch_size",
                 "mtp_enabled", "spec_draft_max")}
        client.post(f"{BASE}/api/models/{NAME}/config", json=body, timeout=30)
        print("基线配置已写回:", json.dumps(body, ensure_ascii=False), flush=True)

    print("\n=== 结论 ===")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    if len(results) == 2 and results[0]["decode_mean"] and results[1]["decode_mean"]:
        a, b = results[0]["decode_mean"], results[1]["decode_mean"]
        print(f"decode t/s: A={a} → B={b}  ({(b / a - 1) * 100:+.0f}%)")


if __name__ == "__main__":
    sys.exit(main())
