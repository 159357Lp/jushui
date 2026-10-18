"""三级存储分层规划器 — 加载时一次性推导 -ngl，杜绝超显存。

背景：-ngl 此前全靠手填（999=全放 / 调低省显存），大模型或两三模型共存
时只能凭经验试。本模块把"空闲显存 − KV 预留 − 固定开销 − 重尾层余量"
的预算数学收进来，供 run config 的"自动分层"档（n_gpu_layers == -1）。

MoE 模型走另一条账：专家权重能占权重的 95%，而一层的注意力只有约 10MB。
按整层 -ngl 卸载会把注意力一起赶下 GPU，白白慢一大截，所以 MoE 用
-ngl 999（注意力与 KV 全驻留）＋ --n-cpu-moe N（把 N 层的专家留在内存）。
GGUF 把一层的全部专家打包进一个张量，能搬的最小单位是"一层的专家"，
NInfer 那种逐专家常驻在 GGUF 上没有对应物。

三个不可约的保守性来源（都是故意的，宁慢勿崩）：
  * 重尾余量：embedding + 输出层远超单层均值（词表 150k+ 的模型单层能差
    10 倍），必须从权重总量里先单独扣除，不能让均值摊平它们。
  * KV 随 ngl 线性增长：只有上 GPU 的层需要 VRAM KV，KV 预算和解按
    ngl*(avg + kv_full/layers) <= budget 精确线性求解，不是先扣满额 KV。
    每 token 的 KV 字节按真实几何算：GQA 的 head_count_kv 远窄于 hidden，
    hybrid 模型只有每 full_attention_interval 层才存 KV。沿用
    "2×hidden×全部层"的老估算会把 35B-A3B 实测 128MB 的缓存读成 6GB，
    预算一开局就被 KV 吃光，后面的卸载决策全部作废。
  * 只信空闲显存：nvidia-smi 的 free 已经把已加载模型、桌面占用算在
    里面，多模型共存的预算扣减因此天然成立，无需自己记账。

运行中不会自动搬层：-ngl 是启动参数，水位变化只重载才生效，规划是
一次性决策。任何缺字段/测不到显存的场景返回 None，调用方保持手动语义，
绝不猜一个更紧的界。
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from lmloader.engine import hardware_costs
from lmloader.engine.hardware_costs import (
    CostCoefficients,
    GENERIC_COEFFICIENTS,
)
from lmloader.logger import logger


def _cache_bytes(cache_type: str) -> float:
    # 与 router.CACHE_BYTES 同一张表；延迟导入避开模块环。
    from lmloader.engine.router import CACHE_BYTES
    return CACHE_BYTES.get(str(cache_type or "q4_0"), 0.6)


def kv_elements_per_token(meta: dict) -> Optional[float]:
    """KV cache elements per token per slot, or None when geometry is unknown.

    Callers multiply by the cache quant's bytes/element. The legacy estimate
    (2 × embedding_length × every layer) assumes MHA with a KV set as wide as
    the hidden vector: it over-counts GQA by head_count/head_count_kv and a
    hybrid model by full_attention_interval, which on a real 35B-A3B added up to
    16x — the planner then spent most of the card's budget on a cache llama.cpp
    measured at 128 MiB and refused to offload weights it could have kept.

    Only the trunk is counted; a missing field never invents a bound, and the
    draft layer plus the residual gap to llama.cpp's own printout stay inside
    vram_safety_fraction's cover.
    """
    heads = int(meta.get("attention_head_count_kv") or 0)
    k_len = int(meta.get("attention_key_length") or 0)
    v_len = int(meta.get("attention_value_length") or 0) or k_len
    layers = int(meta.get("block_count") or 0)
    if not (heads and k_len and v_len and layers):
        return None
    interval = int(meta.get("full_attention_interval") or 0)
    # llama.cpp marks layer i as full-attention when (i+1) % interval == 0.
    kv_layers = layers if interval <= 1 else max(layers // interval, 1)
    if meta.get("supports_mtp"):
        kv_layers += 1
    return float(kv_layers * heads * (k_len + v_len))


@dataclass
class OffloadPlan:
    n_gpu_layers: int          # 最终 -ngl；999 = 全量卸载（含 MTP/辅助层）
    layers_total: int          # 含 MTP 附加层的总层数
    free_vram_mb: float
    budget_mb: float           # 扣安全系数与固定开销后的可用预算
    tail_reserve_mb: float     # embedding+output 重尾预留
    kv_full_mb: float          # 全层满 ctx×parallel 的 KV 估算
    avg_layer_mb: float
    layers_on_disk: int        # 留在 mmap/CPU 侧的层数（预热判断用）
    prewarm_suggested: bool
    prewarm_mb: float
    reason: str                # 一句中文账目，进事件日志
    cpu_moe_layers: int = 0    # -ncmoe：把这些层数的专家权重留在 CPU

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _moe_branch(meta: dict, size_mb: float, budget_mb: float,
                kv_full_mb: float):
    """Expert-weight tiering for MoE checkpoints, or None when not applicable.

    A llama.cpp MoE file packs every expert of a layer into one `_exps`
    tensor, so the smallest unit `-ncmoe` can move is a whole layer's experts —
    NInfer's per-expert residency has no GGUF equivalent. That granularity is
    worth having anyway: experts are ~95 % of the weights while a layer's
    attention is ~10 MB, so streaming experts from RAM keeps all attention (and
    all KV) resident instead of dropping whole layers like `-ngl` does.
    """
    moe_layers = int(meta.get("moe_layer_count") or 0)
    expert_mb = float(meta.get("expert_bytes") or 0) / (1024 * 1024)
    if not moe_layers or expert_mb < 1 or expert_mb >= size_mb:
        return None
    per_layer = expert_mb / moe_layers
    dense_mb = size_mb - expert_mb
    room = budget_mb - dense_mb - kv_full_mb
    if room < 0:
        # 连注意力+KV 都放不下，-ncmoe 救不了，交回按整层卸载的通用解。
        return None
    keep = min(max(int(room // per_layer), 0), moe_layers)
    return {
        "moe_layers": moe_layers,
        "per_layer_mb": per_layer,
        "dense_mb": dense_mb,
        "expert_mb": expert_mb,
        "cpu_moe_layers": moe_layers - keep,
        "expert_count": int(meta.get("expert_count") or 0),
        "expert_used": int(meta.get("expert_used") or 0),
    }


def plan_offload(
    model,
    cfg: dict,
    free_vram_mb: Optional[float],
    free_ram_mb: Optional[float],
    coefficients: Optional[CostCoefficients] = None,
) -> Optional[OffloadPlan]:
    """Derive ngl for one load. Pure: callers inject the measurements.

    ``model`` may be a ModelRecord or the GUI/API dict (size + metadata).
    Returns None when anything essential is unmeasurable — the caller must
    then keep the manual value, never a guessed bound.

    ``coefficients`` 来自 hardware_costs 的按卡实测表；不传即用通用兜底，
    函数本身绝不探测显卡，保持可单测的纯函数性质。
    """
    cost = coefficients or CostCoefficients(**GENERIC_COEFFICIENTS)
    if free_vram_mb is None or free_vram_mb <= 0:
        return None
    meta = getattr(model, "metadata", None)
    size = getattr(model, "size", None)
    if meta is None and isinstance(model, dict):
        meta = model.get("metadata") or {}
        size = model.get("size") or 0
    meta = meta or {}
    size_mb = (size or 0) / (1024 * 1024)
    layers = int(meta.get("block_count") or 0)
    hidden = int(meta.get("embedding_length") or 0)
    if not (layers and hidden and size_mb >= 1):
        return None

    layers_total = layers + (1 if meta.get("supports_mtp") else 0)
    ctx = int(cfg.get("ctx_size") or 4096)
    parallel = max(1, int(cfg.get("parallel") or 1))

    # 重尾：embedding 与 output 两层合计 vocab×hidden×2字节×2（输入+输出表）。
    # GGUF 惯例里这两张表基本保持 f16。
    vocab = int(meta.get("vocab_size") or 0)
    if vocab:
        tail_mb = (2 * vocab * hidden * 2) / (1024 * 1024)
    else:
        tail_mb = size_mb * cost.fallback_tail_fraction
    tail_mb = min(tail_mb, size_mb * cost.max_tail_fraction)

    trunk_mb = max(size_mb - tail_mb, 1.0)
    avg_layer_mb = trunk_mb / layers_total

    # KV 只按上 GPU 的层预留，但满额先算出来作为线性方程的系数。
    kv_elems = kv_elements_per_token(meta)
    kv_geometry_exact = kv_elems is not None
    if not kv_geometry_exact:
        kv_elems = 2 * hidden * layers_total
    kv_full_mb = (ctx * kv_elems * _cache_bytes(cfg.get("cache_type_k", "q4_0"))
                  / (1024 * 1024) * parallel)
    kv_src = "几何精算" if kv_geometry_exact else "按 MHA 上限估计"

    budget_mb = free_vram_mb * cost.vram_safety_fraction - cost.base_overhead_mb

    moe = _moe_branch(meta, size_mb, budget_mb, kv_full_mb)
    if moe:
        n_keep = moe["moe_layers"] - moe["cpu_moe_layers"]
        predicted_mb = moe["dense_mb"] + kv_full_mb + n_keep * moe["per_layer_mb"]
        prewarm = bool(
            moe["cpu_moe_layers"] and free_ram_mb
            and free_ram_mb >= size_mb * cost.prewarm_ram_margin)
        reason = (
            f"MoE 专家分层：空闲显存 {free_vram_mb:.0f}MB×{cost.vram_safety_fraction:.2f}−"
            f"固定开销 {cost.base_overhead_mb:.0f}MB → 预算 {budget_mb:.0f}MB"
            f"（系数来源：{cost.source}）；"
            f"权重 {size_mb:.0f}MB 中专家 {moe['expert_mb']:.0f}MB"
            f"（{moe['expert_count']} 选 {moe['expert_used']}，每层 {moe['per_layer_mb']:.0f}MB），"
            f"非专家 {moe['dense_mb']:.0f}MB＋满额 KV {kv_full_mb:.0f}MB（{kv_src}）后余 {budget_mb - moe['dense_mb'] - kv_full_mb:.0f}MB，"
            f"-ngl 999 保注意力与 KV 全驻留，--n-cpu-moe {moe['cpu_moe_layers']}"
            f"（{n_keep}/{moe['moe_layers']} 层专家留显存），GPU 侧预计占 {predicted_mb:.0f}MB"
        )
        return OffloadPlan(
            n_gpu_layers=999, layers_total=layers_total,
            free_vram_mb=round(free_vram_mb, 1), budget_mb=round(budget_mb, 1),
            tail_reserve_mb=round(tail_mb, 1), kv_full_mb=round(kv_full_mb, 1),
            avg_layer_mb=round(moe["per_layer_mb"], 1),
            layers_on_disk=moe["cpu_moe_layers"],
            prewarm_suggested=prewarm, prewarm_mb=round(size_mb, 1),
            reason=reason, cpu_moe_layers=moe["cpu_moe_layers"],
        )

    if budget_mb <= 0:
        ngl_out, ngl_eff = 0, 0
        why = "空闲显存不足，全部层留在 CPU"
    else:
        # ngl*avg + kv_full*(ngl/layers_total) <= budget 的整数解。部分卸载
        # 时重尾两张表留在 CPU 侧，不占这份预算；但 999 全量会把 size 全
        # 部压进显存，必须单独过一道全量门，不能信线性解的"≥总层数"。
        ngl_exact = budget_mb / (avg_layer_mb + kv_full_mb / layers_total)
        if ngl_exact >= layers_total:
            if size_mb + kv_full_mb <= budget_mb:
                ngl_out, ngl_eff = 999, layers_total
            else:
                ngl_out = ngl_eff = layers_total - 1
            why = ""
        else:
            ngl_out = ngl_eff = max(0, int(ngl_exact))
            why = ""
    disk = layers_total - ngl_eff
    predicted_mb = (
        size_mb + kv_full_mb if ngl_out == 999
        else ngl_eff * (avg_layer_mb + kv_full_mb / layers_total)
    )

    # 预热只在全量卸载之外有意义：真有层留在盘上按需缺页时才值得提前
    # 读进 page cache，且空闲 RAM 必须容得下整个模型并留余量。
    prewarm = False
    if disk > 0 and free_ram_mb and free_ram_mb >= size_mb * cost.prewarm_ram_margin:
        prewarm = True

    reason = (
        f"自动分层：空闲显存 {free_vram_mb:.0f}MB×{cost.vram_safety_fraction:.2f}−"
        f"固定开销 {cost.base_overhead_mb:.0f}MB → 预算 {max(budget_mb, 0):.0f}MB"
        f"（系数来源：{cost.source}）；"
        f"重尾预留 {tail_mb:.0f}MB（{'vocab 精算' if vocab else '均值兜底'}），"
        f"单层均 {avg_layer_mb:.0f}MB，满额 KV {kv_full_mb:.0f}MB（ctx {ctx}×{parallel} 槽，{kv_src}），"
        f"解出 -ngl={ngl_out}，GPU 侧预计占 {predicted_mb:.0f}MB，留盘 {disk} 层"
        + (f"（{why}）" if why else "")
    )
    return OffloadPlan(
        n_gpu_layers=ngl_out, layers_total=layers_total,
        free_vram_mb=round(free_vram_mb, 1), budget_mb=round(budget_mb, 1),
        tail_reserve_mb=round(tail_mb, 1), kv_full_mb=round(kv_full_mb, 1),
        avg_layer_mb=round(avg_layer_mb, 1), layers_on_disk=disk,
        prewarm_suggested=prewarm, prewarm_mb=round(size_mb, 1),
        reason=reason,
    )


def free_vram_mb() -> Optional[float]:
    """Sum free VRAM across CUDA devices via nvidia-smi (no console window)."""
    from lmloader.infra.proc_util import no_window_kwargs
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            **no_window_kwargs(),
        )
        vals = [float(x) for x in r.stdout.replace('"', "").splitlines() if x.strip()]
        if not vals:
            return None
        return sum(vals)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"free VRAM probe failed: {e}")
        return None


def free_ram_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return None


def plan_for_model(model, cfg: dict) -> Optional[OffloadPlan]:
    """Convenience wrapper that measures the machine, then plans.

    显卡探测只在这个非纯入口里做一次，plan_offload 依旧不碰外部状态。
    """
    return plan_offload(model, cfg, free_vram_mb(), free_ram_mb(),
                        coefficients=hardware_costs.resolve_coefficients())


def prewarm_page_cache(path: str) -> bool:
    """Read the whole model file once so mmap faults hit page cache.

    Called off the event loop by the loader; failure is benign (first token
    is just slow, nothing breaks).
    """
    try:
        with open(path, "rb") as f:
            while f.read(16 << 20):
                pass
        return True
    except OSError:
        return False
