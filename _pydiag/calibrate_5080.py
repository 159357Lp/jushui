"""5080 显存分页代价扫描：同一模型同一参数，只变 --n-cpu-moe。

要量的东西：解码速度对"权重压进显存多少"的曲线，拐点就是这张卡真正能用
的权重上限 —— 它就是代价表里 vram_safety_fraction 该填的数。A/B 只有两端
（20629MB→10.5 t/s，10773MB→51.7 t/s），中间是空的，拐点位置靠猜。

参数照抄他那个实例的真实命令行（含 MTP 投机与 q4_0 KV），只改 -ncmoe 和端口，
否则测的是另一套配置。每档 1 次预热 + 3 次 200 token，取中位数。

速度只认引擎响应里的 timings 块（这个 build 不打 print_timing）。显存实际
占用另从 -lv 4 的 "CUDA0 model buffer size" 行取，不靠我自己折算。
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time

import httpx

EXE = r"D:\掬水\engines\fastest-b11552\llama-server.exe"
MODEL = "models\\Qwen3.6-35B-A3B-MXFP4_MOE\\Qwen3.6-35B-A3B-MXFP4_MOE\\Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = "请写一段 about 秋天的杭州的说明文字，具体一些，不要列提纲。"
MAX_TOKENS = 200
RUNS = 3
LEVELS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 and
                           not sys.argv[1].startswith("--") else
                           "0,4,8,12,16,20,24".split(","))]
OUT = "_pydiag/cal_results.json"

BUFFER = re.compile(r"CUDA0\s+model buffer size\s*=\s*(\d+(?:\.\d+)?) MiB")


def argv(ncmoe):
    args = [EXE, "-m", MODEL, "--port", str(PORT), "--host", "127.0.0.1",
            "--n-gpu-layers", "999"]
    if ncmoe > 0:
        args += ["--n-cpu-moe", str(ncmoe)]
    args += ["-c", "32768", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
             "--flash-attn", "on", "--parallel", "1", "--threads", "8", "-lv", "4",
             "--reasoning-budget", "0", "-b", "2048", "-ub", "512",
             "--repeat-penalty", "1.1", "--dry-multiplier", "0.8", "--dry-base", "1.75",
             "--dry-allowed-length", "2", "--dry-penalty-last-n", "64",
             "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    return args


def wait_ready(proc, logf, limit=420):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"引擎提前退出 rc={proc.returncode}")
        try:
            if httpx.get(f"{BASE}/health", timeout=5).status_code == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(1.5)
        if int(time.time() - t0) % 30 == 29:
            logf.flush()
    raise RuntimeError("等不到 /health")


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    used, free = [int(v) for v in out.split(",")]
    return used, free


def foreign_engine():
    """除本脚本要起的引擎外，卡上不该有别的推理进程。00:43 那轮整批数据就是
    被 LM Studio 半路加载的 9B 污染的（同档 2.69 与 15.29 t/s 并存）。"""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process | Where-Object {$_.ProcessName -like '*llama*' -or "
         "$_.ProcessName -like '*vllm*'} | Select-Object -ExpandProperty Id"],
        capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.strip().isdigit()]


def require_exclusive():
    pids = foreign_engine()
    if pids:
        raise SystemExit(f"RED: 卡上有其它推理进程 {pids}，测出来的不是分层代价。"
                         f"先退出它们再扫。")


def gen(client):
    r = client.post(f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS, "temperature": 0, "stream": False,
    }, timeout=900)
    r.raise_for_status()
    body = r.json()
    t = body.get("timings") or {}
    return t, body.get("usage", {})


def measure(ncmoe):
    os.makedirs("_pydiag", exist_ok=True)
    require_exclusive()
    _, free_before = gpu_state()
    log_path = f"_pydiag/cal_nc{ncmoe}.log"
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.Popen(argv(ncmoe), stdout=logf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=".", env=env)
    try:
        boot = wait_ready(proc, logf)
        used, free = gpu_state()
        client = httpx.Client()
        warm, _ = gen(client)
        rows = []
        for _ in range(RUNS):
            t, usage = gen(client)
            rows.append(t)
        client.close()
        decodes = [r["predicted_per_second"] for r in rows if r.get("predicted_n", 0) >= MAX_TOKENS * 0.9]
        logf.flush()
        with open(log_path, encoding="utf-8") as f:
            buf = BUFFER.findall(f.read())
        last_ts = rows[-1] if rows else {}
        return {
            "ncmoe": ncmoe, "boot_s": round(boot, 1),
            "vram_weights_mib": float(buf[-1]) if buf else None,
            "free_before_mib": free_before,
            "foreign_pids": [p for p in foreign_engine() if p != proc.pid],
            "gpu_used_mib": used, "gpu_free_mib": free,
            "decode_median": round(statistics.median(decodes), 2) if decodes else None,
            "decode_runs": [round(x, 2) for x in decodes],
            "prefill_tps": round(last_ts.get("prompt_per_second", 0), 1),
            "accept": f"{last_ts.get('draft_n_accepted')}/{last_ts.get('draft_n')}",
            "full_len_runs": len(decodes),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        logf.close()
        # 显存没退回基线就进下一档 = 拿上一档的残余当本档的空闲，整条曲线作废。
        deadline = time.time() + 90
        while time.time() < deadline:
            _, now_free = gpu_state()
            if now_free >= free_before - 300:
                break
            time.sleep(2)
        else:
            raise SystemExit(f"RED: ncmoe={ncmoe} 退出后显存仍只余 {now_free} MiB"
                             f"（基线 {free_before}），拒绝在脏卡上继续测")
        time.sleep(3)


def main():
    results = []
    for n in LEVELS:
        r = measure(n)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n ncmoe  显存权重MiB  解码t/s(中位)  三次")
    for r in results:
        print(f"  {r['ncmoe']:>3}   {r['vram_weights_mib'] or 0:>9.0f}   "
              f"{r['decode_median'] or 0:>7.2f}    {r['decode_runs']}")


if __name__ == "__main__":
    main()
