"""离线校验 KV 几何估算 vs 真实 GGUF + llama.cpp 自己的实测显存账。

锚点不是我把公式再算一遍：2026-10-18 23:06 那次加载 MXFP4_MOE，llama.cpp 在
-lv 4 下对 ctx=65536 报出 CUDA0 KV buffer = 128 MiB（掬水的显存分解行照抄了
这个数）。规划器估的 KV 必须落在"≥ 实测（保守）且 ≤ 4× 实测（没估到离谱）"
之间，否则 MoE 的 --n-cpu-moe 解就是拿错的预算解出来的。

其余 MoE 文件只报数不设断言（没有对应的实测锚点）。

跑法：
    python _pydiag/kv_geometry_probe.py            # 应当 GREEN
    python _pydiag/kv_geometry_probe.py --break    # 注入 8 倍 KV 头，必须报红
反向对照是必需的：只会变绿的探针等于没测。
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from lmloader.engine.offload_planner import kv_elements_per_token, plan_offload  # noqa: E402
from lmloader.infra.gguf_parser import parse_gguf  # noqa: E402

ANCHOR = r"models/Qwen3.6-35B-A3B-MXFP4_MOE/Qwen3.6-35B-A3B-MXFP4_MOE/Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
MEASURED_KV_MIB = 128.0      # llama.cpp -lv 4 实测，见模块 docstring
CTX = 65536
CACHE_BYTES_Q4_0 = 0.5625
FREE_VRAM_MB = 15360.0       # 当晚水位；不探测显卡，保证可重复
BREAK = "--break" in sys.argv

CFG = {"ctx_size": CTX, "parallel": 1, "cache_type_k": "q4_0",
       "cache_type_v": "q4_0", "n_gpu_layers": -1}

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  RED  ") + msg)
    if not cond:
        failures.append(msg)


def moe_files():
    """锚点文件优先，其余库内 MoE 文件附带报数。"""
    found, seen = [], {os.path.normpath(ANCHOR)}
    if os.path.exists(ANCHOR):
        found.append(ANCHOR)
    for root, _dirs, names in os.walk("models"):
        for n in names:
            if not n.lower().endswith(".gguf"):
                continue
            p = os.path.join(root, n)
            if os.path.normpath(p) in seen:
                continue
            seen.add(os.path.normpath(p))
            meta = parse_gguf(p)
            if meta is not None and (meta.architecture or "").endswith("moe"):
                found.append(p)
    return found


for path in moe_files():
    is_anchor = os.path.normpath(path) == os.path.normpath(ANCHOR)
    meta_obj = parse_gguf(path)
    if meta_obj is None:
        check(False, f"{path}: parse failed")
        continue
    size = os.path.getsize(path)
    meta = meta_obj.to_dict()
    name = os.path.basename(path)
    elems = kv_elements_per_token(meta)
    print(f"\n{name}  ({size / 1024 ** 2:.0f} MB, {meta['block_count']} 层, "
          f"interval={meta.get('full_attention_interval')}, "
          f"kv_heads={meta.get('attention_head_count_kv')})"
          + ("" if is_anchor else "  [仅报数]"))
    if elems is None:
        check(False, f"{name}: KV 几何字段缺失 → 规划器只能按 MHA 上限估")
        continue
    kv_mb = CTX * elems * CACHE_BYTES_Q4_0 / 1024 ** 2
    print(f"  kv/token={elems:.0f} 元素 → 预测 KV {kv_mb:.0f} MB"
          + (f" (llama 实测 {MEASURED_KV_MIB:.0f} MB)" if is_anchor else ""))

    broken_meta = dict(meta, attention_head_count_kv=meta["attention_head_count_kv"] * 8)
    plan_meta = broken_meta if BREAK else meta
    if BREAK:
        kv2 = CTX * kv_elements_per_token(broken_meta) * CACHE_BYTES_Q4_0 / 1024 ** 2
        print(f"  [--break] KV 头 ×8 → KV {kv2:.0f} MB")
    plan = plan_offload(SimpleNamespace(name=name, path=path, size=size,
                                        metadata=plan_meta, backend=None),
                        CFG, free_vram_mb=FREE_VRAM_MB, free_ram_mb=65536.0)
    if plan is None:
        check(False, f"{name}: 规划返回 None")
        continue
    print(f"  plan: -ngl={plan.n_gpu_layers} -ncmoe={plan.cpu_moe_layers} "
          f"KV预留={plan.kv_full_mb:.0f}MB")

    if is_anchor:
        # 断言恒等：--break 只是把输入改坏，判据一个字不动，否则探针永远不会红。
        check(plan.kv_full_mb >= MEASURED_KV_MIB, f"{name}: 预测 KV 不得低于实测")
        check(plan.kv_full_mb <= MEASURED_KV_MIB * 4,
              f"{name}: 预测 KV 不得高出实测 4 倍")

print()
if failures:
    print("RED: " + "; ".join(failures))
    sys.exit(1)
print("GREEN: KV 几何与 llama.cpp 实测同量级，卸载决策用的是这份预算")
