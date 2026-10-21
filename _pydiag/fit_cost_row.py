"""把 cal_results.json 的扫描曲线拟合成本卡实测行，并自证它选得出快档。

拟合的可测系数：
  base_overhead_mb    ← 引擎 -lv 4 自报的 CUDA0 compute + recurrent 缓冲
  vram_safety_fraction ← 拐点显存权重 + KV + 上述开销，再除以当时的空闲显存
  paging_ceiling_fraction ← 同一合计峰值 ÷ 物理显存总量（共存账本的整卡上限）
  engine_context_mb   ← 卡上真实占用 − 桌面底噪 − 引擎自报合计，逐档取最大值

重尾/预热三个系数与显卡无关（是文件结构和 RAM 策略），保留通用值，
provenance 里写清楚哪些是实测、哪些是沿用。

自校验：用拟合行重算一次规划，规划出的 -ncmoe 必须落在实测"平台区"
（≥峰值 90% 速度）内。--break 把安全系数放大 1.15 倍（等于允许超配），
规划就会掉进分页塌陷区，这条断言必须报红——否则它是个只会变绿的探针。
"""
import glob
import json
import re
import statistics
import subprocess
import sys

from lmloader.engine import offload_planner as op
from lmloader.engine.hardware_costs import (
    GENERIC_COEFFICIENTS, CostCoefficients, detect_hardware_class,
    resolve_coefficients, save_measurement, table_path,
)

_paths = [a for a in sys.argv[1:] if not a.startswith("--")]
RESULTS = _paths[0] if _paths else "_pydiag/cal_results.json"
BREAK = "--break" in sys.argv
WRITE = "--write" in sys.argv
# 分页塌陷判据：不到峰值一半。平台判据：90% 以上（用于报告，不作硬约束）。
COLLAPSE_FRACTION = 0.5
PLATEAU_FRACTION = 0.9
# 自校验底线：拟合行选出的档位必须离塌陷区有这一段安全距离。
CHECK_FRACTION = 0.85

KV_BUF = re.compile(r"CUDA0 KV buffer size\s*=\s*(\d+(?:\.\d+)?) MiB")
RS_BUF = re.compile(r"(?:llama_memory_recurrent|CUDA0).*?RS buffer size\s*=\s*(\d+(?:\.\d+)?) MiB", re.S)
COMPUTE_BUF = re.compile(r"sched_reserve:.*?CUDA0 compute buffer size =\s*(\d+(?:\.\d+)?) MiB", re.S)


def buffers(ncmoe):
    """从该档的引擎原始输出里取显存侧 KV/状态/计算缓冲；取不到返回 None。"""
    path = f"_pydiag/cal_nc{ncmoe}.log"
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    kv = KV_BUF.findall(text)
    rs = RS_BUF.findall(text)
    cp = COMPUTE_BUF.findall(text)
    if not (kv and cp):
        return None
    return {"kv": float(kv[-1]), "recurrent": float(rs[-1]) if rs else 0.0,
            "compute": float(cp[-1])}


def subprocess_total():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0])


def fit(rows):
    # 只接受带 free_before 且确认过"卡上无外来推理进程"的档 —— 00:43 那轮
    # 就是被半路加载的 LM Studio 9B 污染，同档测出 2.69 与 15.29 t/s 两个数。
    usable = [r for r in rows if r.get("free_before_mib") and not r.get("foreign_pids")]
    if len(usable) < 3:
        raise SystemExit(f"RED: 独占档只有 {len(usable)}/{len(rows)} 个，不足以定拐点")
    speeds = [r["decode_median"] for r in usable if r["decode_median"]]
    peak = max(speeds)
    # 分页塌陷：不到峰值一半。平台区：塌陷之上、显存权重最大的那批。
    healthy = [r for r in usable if r["decode_median"] and
               r["decode_median"] >= peak * COLLAPSE_FRACTION]
    star = max(healthy, key=lambda r: r["vram_weights_mib"])
    plateau = [r for r in healthy if r["decode_median"] >= peak * PLATEAU_FRACTION]
    thrash = [r for r in usable if r not in healthy]
    b = buffers(star["ncmoe"])
    if not b:
        raise SystemExit(f"RED: 拐点档 cal_nc{star['ncmoe']}.log 里没解析到显存缓冲行，"
                         f"不能凭权重单项定安全系数")
    overhead = b["compute"] + b["recurrent"]
    engine_total = star["vram_weights_mib"] + b["kv"] + b["compute"] + b["recurrent"]
    free_before_load = star["free_before_mib"]
    desktop_baseline = subprocess_total() - free_before_load
    safety = engine_total / free_before_load
    # 整卡常驻总量的上限：驱动分页是全局性质，多引擎加起来一起撞，
    # 单模型的 safety 打折拦不住"两只各留 20% 余量、合起来超了"。
    ceiling = engine_total / (desktop_baseline + free_before_load)
    # 每进程 CUDA 上下文：卡上真实占用 − 桌面底噪 − 引擎自报合计，剩下的就是
    # 自报口径量不到的那份。取各档最大值（保守方向：宁可多留）。
    contexts = []
    card_total = subprocess_total()
    for r in healthy:
        rb = buffers(r["ncmoe"])
        if not rb or not r.get("gpu_used_mib"):
            continue
        used_before = card_total - r["free_before_mib"]
        self_reported = r["vram_weights_mib"] + rb["kv"] + rb["compute"] + rb["recurrent"]
        contexts.append(round(r["gpu_used_mib"] - used_before - self_reported, 1))
    if not contexts:
        raise SystemExit("RED: 没有一档能反解进程上下文，engine_context_mb 无从实测")
    return {
        "peak_tps": peak, "star": star, "plateau": plateau or healthy,
        "healthy": healthy, "thrash": thrash,
        "overhead_mb": overhead, "buffers": b, "engine_total": engine_total,
        "used_before": desktop_baseline, "free_before_load": free_before_load,
        "safety": safety, "ceiling": ceiling,
        "context_mb": max(contexts), "context_rows": contexts,
        "dropped": len(rows) - len(usable),
    }


def plan_with(coefficients, meta_model, cfg, free_mb):
    return op.plan_offload(meta_model, cfg, free_vram_mb=free_mb,
                           free_ram_mb=65536, coefficients=coefficients)


def main():
    rows = json.load(open(RESULTS, encoding="utf-8"))
    rows = [r for r in rows if r.get("decode_median")]
    if len(rows) < 3:
        print("RED: 有效档位不足 3 个，无法定拐点")
        sys.exit(1)
    f = fit(rows)
    print(f"峰值 {f['peak_tps']} t/s；平台区 {[r['ncmoe'] for r in f['plateau']]}"
          f"；塌陷区 {[r['ncmoe'] for r in f['thrash']]}")
    print(f"拐点档 ncmoe={f['star']['ncmoe']} 显存权重 {f['star']['vram_weights_mib']:.0f} MiB "
          f"缓冲 {f['buffers']}")
    print(f"桌面底噪 {f['used_before']:.0f} MiB → 规划可见空闲 {f['free_before_load']:.0f} MiB"
          f"（引擎合计 {f['engine_total']:.0f}）→ 拟合 safety={f['safety']:.4f} "
          f"overhead={f['overhead_mb']:.0f}MB 整卡上限={f['ceiling']:.4f}")
    print(f"逐档反解的进程外显存 {f['context_rows']} MiB → engine_context_mb"
          f"={f['context_mb']:.0f}（取最大，保守）")

    cls = detect_hardware_class()
    print(f"hardware_class = {cls}（现用来源：{resolve_coefficients().source}）")

    if WRITE:
        coeffs = dict(GENERIC_COEFFICIENTS)
        coeffs["vram_safety_fraction"] = round(f["safety"], 4)
        coeffs["base_overhead_mb"] = round(f["overhead_mb"], 1)
        coeffs["paging_ceiling_fraction"] = round(f["ceiling"], 4)
        coeffs["engine_context_mb"] = round(f["context_mb"], 1)
        # 循环状态单独记一份：base_overhead_mb 是按**这只校准模型**量的，里面混着
        # 它的 RS。规划器现在会按几何推每只模型自己的状态，行里不留这个数就没法
        # 判断"超出多少"，校准模型自己会被扣两遍。
        coeffs["overhead_rs_mb"] = round(f["buffers"]["recurrent"], 1)
        save_measurement(cls, coeffs, {
            "measured_at": "2026-10-19",
            "method": "_pydiag/calibrate_5080.py 同模型同参数只变 --n-cpu-moe，"
                      "取解码平台区最大显存权重档",
            "model": "Qwen3.6-35B-A3B-MXFP4_MOE.gguf (21155MiB, 41 MoE 层)",
            "peak_tps": f["peak_tps"],
            "knee_ncmoe": f["star"]["ncmoe"],
            "knee_vram_weights_mib": f["star"]["vram_weights_mib"],
            "measured_fields": ["vram_safety_fraction", "base_overhead_mb",
                                "paging_ceiling_fraction", "engine_context_mb",
                                "overhead_rs_mb"],
            "inherited_fields": ["fallback_tail_fraction", "max_tail_fraction",
                                 "prewarm_ram_margin"],
            "paging_ceiling_basis": "拐点档引擎合计峰值 ÷ 物理显存总量（常驻总量是全局性质，多引擎一起撞）",
            "engine_context_basis": "逐档 (卡上占用 − 桌面底噪 − 引擎自报合计) 取最大值；"
                                    "共存账本要逐进程计一次，单模型时被 safety 折扣吸收",
            "engine_context_rows_mib": f["context_rows"],
            "free_vram_before_load_mib": f["free_before_load"],
            "desktop_baseline_mib": f["used_before"],
        })
        print(f"已写入实测行 -> {table_path()}")

    # 自校验：拿拟合行（或现表）让规划器在"扫描时的空闲显存"下重新决策。
    model = json.loads(open("_pydiag/model_meta.json", encoding="utf-8").read())
    cfg = json.loads(open("_pydiag/ab_baseline_config.json", encoding="utf-8").read())["config"]
    fitted = CostCoefficients(**{**GENERIC_COEFFICIENTS,
                                 "vram_safety_fraction": f["safety"],
                                 "base_overhead_mb": f["overhead_mb"],
                                 # 自校验用的是校准模型本身：行里已经混着它的状态，
                                 # 再让规划器按几何扣一遍就是两笔钱。
                                 "overhead_rs_mb": f["buffers"]["recurrent"]},
                              source="拟合自本次扫描")
    if BREAK:
        fitted = CostCoefficients(**{**fitted.as_dict(),
                                     "vram_safety_fraction": f["safety"] * 1.15})
    plan = plan_with(fitted, model, cfg, f["free_before_load"])
    if plan is None:
        print("RED: 规划器返回 None，拟合行没接进去")
        sys.exit(1)
    allowed = {r["ncmoe"] for r in f["plateau"]}
    chosen = plan.cpu_moe_layers
    speed = next((r["decode_median"] for r in rows if r["ncmoe"] == chosen), None)
    knee = f["star"]["ncmoe"]
    print(f"拟合行规划结果：--n-cpu-moe {chosen}（拐点档 {knee}，90% 平台区 {sorted(allowed)}）"
          f" 实测该档 {speed} t/s" if speed else
          f"拟合行规划结果：--n-cpu-moe {chosen}（拐点档 {knee}，未实测此整数档）")
    # 判据是物理量而不是查表命中：下 CPU 的专家层数不能少于拐点档，
    # 少于就意味着留在显存里的权重超过实测能承受的天花板。
    if chosen < knee:
        print(f"RED: 拟合行允许 {chosen} < 拐点 {knee}，会把模型推进分页塌陷区")
        sys.exit(1)
    if speed is not None and speed < f["peak_tps"] * PLATEAU_FRACTION:
        print(f"RED: 拟合行选中的 {chosen} 档实测只有 {speed} t/s（峰值 {f['peak_tps']}）")
        sys.exit(1)
    print("GREEN: 拟合行选出的档位不掉进分页塌陷区")


if __name__ == "__main__":
    main()
