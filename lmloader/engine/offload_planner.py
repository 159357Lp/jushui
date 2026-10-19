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
    里面，所以单次加载不会把显存顶爆。
  * 但"不顶爆"不等于"共存最优"：free 只会拦住后加载的那个，绝不为它把
    已在卡上的模型往下压，两个引擎合计仍能落在 WDDM 分页区一起掉速。
    多模型常驻由 plan_coexistence 用整卡上限定账（见下）。

多模型共存账本：paging_ceiling_fraction 是从实测拐点算出来的整卡常驻上限
（物理显存×系数，按引擎合计峰值折算，桌面底噪已在折算里扣除），每个引擎的
常驻 = predicted_vram_mb + base_overhead_mb（后者实测正好是一引擎的计算+状态
缓冲峰值）。非 MoE 与钉选决策模型没有让步余地，先整只驻留；MoE 是唯一弹性
档，按专家体量吸收余下预算，退到 dense+满额 KV 为止。内存放不下的分片只提
醒不拒绝——分层到盘本来就是本模块的用途。

运行中不会自动搬层：-ngl 是启动参数，水位变化只重载才生效，规划是
一次性决策。任何缺字段/测不到显存的场景返回 None，调用方保持手动语义，
绝不猜一个更紧的界。
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, replace
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
    predicted_vram_mb: float = 0.0   # 本档预计常驻（权重＋满额 KV，不含引擎缓冲）

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def moe_dims(meta: dict, size_mb: float) -> Optional[dict]:
    """Expert tiering geometry, or None for a checkpoint without MoE weights.

    Public because the coexistence ledger needs the same dense/expert split to
    price a MoE model's floor; a second copy of this formula is exactly how the
    KV estimate once drifted 16.8x away from llama.cpp's own printout.
    """
    moe_layers = int(meta.get("moe_layer_count") or 0)
    expert_mb = float(meta.get("expert_bytes") or 0) / (1024 * 1024)
    if not moe_layers or expert_mb < 1 or expert_mb >= size_mb:
        return None
    return {
        "moe_layers": moe_layers,
        "per_layer_mb": expert_mb / moe_layers,
        "dense_mb": size_mb - expert_mb,
        "expert_mb": expert_mb,
        "expert_count": int(meta.get("expert_count") or 0),
        "expert_used": int(meta.get("expert_used") or 0),
    }


def _manual_cpu_moe(cfg: dict, moe: dict) -> Optional[int]:
    """run config 里手填的 --n-cpu-moe；缺省/负数表示跟自动解算走。

    共存让步需要这条路：自动解只会"按预算往下退"，人想主动让出更多显存
    （或反过来钉住某一档不让联合账本压）时，没有可写的地方。上限截到
    该模型的 MoE 层数，超过就等于全下 CPU，不制造假档。
    """
    raw = cfg.get("cpu_moe_layers", -1)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return min(value, moe["moe_layers"])


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
    moe = moe_dims(meta, size_mb)
    if not moe:
        return None
    per_layer = moe["per_layer_mb"]
    dense_mb = moe["dense_mb"]
    room = budget_mb - dense_mb - kv_full_mb
    if room < 0:
        # 连注意力+KV 都放不下，-ncmoe 救不了，交回按整层卸载的通用解。
        return None
    keep = min(max(int(room // per_layer), 0), moe["moe_layers"])
    return {
        "moe_layers": moe["moe_layers"],
        "per_layer_mb": per_layer,
        "dense_mb": dense_mb,
        "expert_mb": moe["expert_mb"],
        "cpu_moe_layers": moe["moe_layers"] - keep,
        "expert_count": moe["expert_count"],
        "expert_used": moe["expert_used"],
    }


def _meta_and_size(model) -> tuple[dict, float]:
    """Accept a ModelRecord or the GUI/API dict; return (metadata, size_mb)."""
    meta = getattr(model, "metadata", None)
    size = getattr(model, "size", None)
    if meta is None and isinstance(model, dict):
        meta = model.get("metadata") or {}
        size = model.get("size") or 0
    return (meta or {}), ((size or 0) / (1024 * 1024))


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
    meta, size_mb = _meta_and_size(model)
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
        manual = _manual_cpu_moe(cfg, moe)
        auto_layers = moe["cpu_moe_layers"]
        if manual is not None:
            moe["cpu_moe_layers"] = manual
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
            f"（{n_keep}/{moe['moe_layers']} 层专家留显存）"
            + (f"，手填档覆盖自动解算的 {auto_layers} 层" if manual is not None else "")
            + f"，GPU 侧预计占 {predicted_mb:.0f}MB"
        )
        return OffloadPlan(
            n_gpu_layers=999, layers_total=layers_total,
            free_vram_mb=round(free_vram_mb, 1), budget_mb=round(budget_mb, 1),
            tail_reserve_mb=round(tail_mb, 1), kv_full_mb=round(kv_full_mb, 1),
            avg_layer_mb=round(moe["per_layer_mb"], 1),
            layers_on_disk=moe["cpu_moe_layers"],
            prewarm_suggested=prewarm, prewarm_mb=round(size_mb, 1),
            reason=reason, cpu_moe_layers=moe["cpu_moe_layers"],
            predicted_vram_mb=round(predicted_mb, 1),
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
        predicted_vram_mb=round(predicted_mb, 1),
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


@dataclass
class CoexistencePlan:
    """Two or more engines resident at once, fitted to one card ceiling."""

    feasible: bool
    ceiling_mb: float        # 整卡可常驻上限（含每个引擎的缓冲）
    total_mb: float          # 本方案下所有引擎常驻合计
    headroom_mb: float       # ceiling − total；负数即方案本身越界
    spill_mb: float          # 不在显存里的权重（落内存/磁盘）
    spill_destination: str
    plans: dict              # 模型名 -> OffloadPlan
    footprints: dict         # 模型名 -> 该档的真实常驻 MiB（含引擎缓冲）
    rigid_names: list        # 没有让步余地的模型（非 MoE 或钉选）
    reload_names: list       # 需要重载让步的已加载模型
    reason: str

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["plans"] = {k: v.as_dict() for k, v in self.plans.items()}
        return d


def _free_for_footprint(footprint_mb: float, cost: CostCoefficients) -> float:
    """The free-VRAM reading that makes plan_offload land on footprint_mb.

    plan_offload 的账是 budget = free×safety − overhead，解出的 predicted 只会
    ≤ budget，而一个引擎的真实常驻 = predicted + overhead（overhead 实测就是一
    引擎的计算+状态缓冲峰值）。取 free = footprint/safety 让 budget 恰好等于
    footprint − overhead，向下整除只会少不会多 —— 合计常驻因此恒 ≤ 整卡上限。
    把 footprint 当成 budget 再补一次 overhead，就会多放行一层专家，实测越界
    320MB 而方案依旧自称可行。

    共存账本先按整卡上限定出每个引擎的常驻额度，再翻译成 plan_offload 认识的
    空闲显存，卸载解算依旧只有它一个 owner —— 两处各写一遍正是 KV 估算曾经
    偏离 llama.cpp 打印 16.8 倍的成因。
    """
    return footprint_mb / cost.vram_safety_fraction


# 已加载模型的实际常驻与方案 allotment 相差不到这个数就不折腾重载。
YIELD_TOLERANCE_MB = 64.0


def plan_coexistence(
    entries: list,
    *,
    vram_total_mb: Optional[float],
    free_ram_mb: Optional[float] = None,
    coefficients: Optional[CostCoefficients] = None,
) -> Optional[CoexistencePlan]:
    """Joint tiering for N models that must stay resident together.

    ``entries``: ``{"model": ModelRecord|dict, "cfg": dict, "pinned": bool,
    "resident_mb": float|None, "vision_mb": float|None, "name": str}`` —
    ``resident_mb`` is the engine-self-reported footprint of an already-loaded
    instance, used only to decide who must yield; ``vision_mb`` is the mounted
    vision encoder (mmproj) that the engine's own breakdown does not count.

    Why a separate ledger instead of ``free_vram_mb()``: nvidia-smi's *free*
    figure stops a second load from overflowing, but it never moves the
    incumbent down, so the pair can sit jointly in WDDM paging territory where
    every engine on the card decays together. The ceiling here is the measured
    non-paging resident peak (物理显存 × paging_ceiling_fraction), and each
    engine's own non-weight buffers (``base_overhead_mb``), its CUDA context
    (``engine_context_mb``) and any mounted mmproj are priced per entry, so
    Σ(predicted + 这三项) is directly comparable to the ceiling. Measured on the
    5080: two engines + one d3 mmproj really occupied 14858MB while the ledger
    without those two allowances said 13902MB — 736MB past the ceiling, and
    decode fell from the 48.29 t/s plateau to 39-45 t/s.

    Returns None when anything essential is unmeasurable — same rule as
    plan_offload: the caller keeps manual semantics, never a guessed bound.
    """
    cost = coefficients or CostCoefficients(**GENERIC_COEFFICIENTS)
    if not vram_total_mb or vram_total_mb <= 0 or not entries:
        return None
    ceiling = vram_total_mb * cost.paging_ceiling_fraction

    priced = []
    vision_total = 0.0
    for index, entry in enumerate(entries):
        model = entry.get("model")
        cfg = entry.get("cfg") or {}
        meta, size_mb = _meta_and_size(model)
        name = (entry.get("name") or getattr(model, "name", None)
                or (model.get("name") if isinstance(model, dict) else None)
                or f"模型{index + 1}")
        # 每进程三份自有开销：自报口径量不到的 CUDA 上下文，和引擎挂载但从不
        # 计入 breakdown 的视觉编码器。合进 base_overhead_mb 一起走 plan_offload
        # 的预算，budget = free×safety − 这一份，footprint 才能与整卡上限同径。
        vision_mb = max(0.0, float(entry.get("vision_mb") or 0.0))
        vision_total += vision_mb
        entry_cost = replace(cost, base_overhead_mb=(
            cost.base_overhead_mb + cost.engine_context_mb + vision_mb))
        full = plan_offload(model, cfg, _free_for_footprint(size_mb + 1e6, entry_cost),
                            free_ram_mb, entry_cost)
        if full is None:
            logger.info(f"共存规划：{name} 几何不可测，交回手动语义")
            return None
        moe = moe_dims(meta, size_mb)
        rigid = moe is None or bool(entry.get("pinned"))
        footprint = full.predicted_vram_mb + entry_cost.base_overhead_mb
        resident = entry.get("resident_mb")
        # 实测口径是引擎自报（含计算缓冲与状态层，缺上下文与 mmproj），补齐到
        # 与 footprint 同径再比，否则"越界 64MB 才让位"实际是越界 300MB 才让位。
        resident_priced = (None if resident is None
                           else float(resident) + cost.engine_context_mb + vision_mb)
        # 不肯让位的模型按"卡上真实驻留"记账，而不是按 GGUF 几何推算：vLLM/d3
        # 那类引擎的缓冲构成与 llama.cpp 不同，估算只会低估；低估一只就等于
        # 把整卡上限多放行一只，翻页的仍是全体。弹性档不套用这条——它正是
        # 要被压小的那一只，用实测值当上限就永远得不出让步结论。
        if rigid and resident_priced and resident_priced > footprint:
            footprint = resident_priced
        # 弹性模型能让到的最低处：非专家权重 + 满额 KV + 引擎缓冲。
        floor = footprint if rigid else (
            size_mb - moe["expert_mb"] + full.kv_full_mb
            + entry_cost.base_overhead_mb)
        priced.append({
            "name": name, "model": model, "cfg": cfg, "moe": moe,
            "size_mb": size_mb, "full": full, "footprint": footprint,
            "floor": floor, "rigid": rigid, "cost": entry_cost,
            "resident_priced": resident_priced,
        })

    rigid_total = sum(p["footprint"] for p in priced if p["rigid"])
    elastic_floor = sum(p["floor"] for p in priced if not p["rigid"])
    rigid_names = [p["name"] for p in priced if p["rigid"]]
    need = rigid_total + elastic_floor
    if need > ceiling:
        worst = min(priced, key=lambda p: p["footprint"] - p["floor"])
        return CoexistencePlan(
            feasible=False, ceiling_mb=round(ceiling, 1),
            total_mb=round(need, 1), headroom_mb=round(ceiling - need, 1),
            spill_mb=0.0, spill_destination="未规划", plans={}, footprints={},
            rigid_names=rigid_names, reload_names=[],
            reason=(
                f"共存不可行：整卡常驻上限 {ceiling:.0f}MB（物理显存 {vram_total_mb:.0f}MB×"
                f"{cost.paging_ceiling_fraction:.4f}，来源：{cost.source}），"
                f"但{'、'.join(p['name'] for p in priced)} 即使全部退到最低档也要 "
                f"{need:.0f}MB —— 缺口 {need - ceiling:.0f}MB。"
                f"「{'、'.join(rigid_names)}」没有让步余地（非 MoE 或钉选决策模型），"
                f"最小的一档还要 {worst['floor']:.0f}MB（权重+满额 KV+引擎缓冲）。"
                f"请调小上下文/并行槽位，或先卸载一个模型。"
            ),
        )

    leftover = ceiling - need
    expert_demand = sum(p["moe"]["expert_mb"] for p in priced if not p["rigid"])
    plans, foot_map, total, spill, reloads, upgrades = {}, {}, 0.0, 0.0, [], []
    entry_spill: dict[str, float] = {}
    for p in priced:
        if p["rigid"]:
            plan, p["allotment"] = p["full"], p["footprint"]
        else:
            # 按专家体量分摊余量：只有一个弹性模型时它独占余量，正是
            # 「35B 给钉选的 d3-mini 让步」的算术形态。
            share = (leftover * p["moe"]["expert_mb"] / expert_demand
                     if expert_demand else 0.0)
            p["allotment"] = p["floor"] + share
            plan = plan_offload(p["model"], p["cfg"],
                                _free_for_footprint(p["allotment"], p["cost"]),
                                free_ram_mb, p["cost"]) or p["full"]
        footprint = plan.predicted_vram_mb + p["cost"].base_overhead_mb
        if p["rigid"]:
            footprint = max(footprint, p["footprint"])   # 让不掉的按实测计
        plans[p["name"]] = plan
        foot_map[p["name"]] = round(footprint, 1)
        total += footprint
        entry_spill[p["name"]] = max(
            0.0, p["size_mb"] - (plan.predicted_vram_mb - plan.kv_full_mb))
        spill += entry_spill[p["name"]]
        resident = p["resident_priced"]
        if resident is None:
            continue
        if footprint < resident - YIELD_TOLERANCE_MB:
            reloads.append(p["name"])
        elif footprint > resident + YIELD_TOLERANCE_MB:
            upgrades.append(p["name"])

    # 手填档能把一只模型钉在它分到的额度之外：2026-10-19 02:55 实测——上一次让步
    # 由账本代笔写回的 --n-cpu-moe 27 这轮被当成手填档尊重，合计 14907MB 越界
    # 785MB 却仍标着"可行"。联合解算没有权限覆盖用户手感，所以这里只能判不可行，
    # 并把"改成多少才塞得下"直接算给他看。
    overs = [p for p in priced
             if foot_map[p["name"]] > p["allotment"] + YIELD_TOLERANCE_MB]
    if total > ceiling + 1.0:
        hints = []
        for p in overs:
            auto = plan_offload(p["model"],
                                {**p["cfg"], "cpu_moe_layers": -1,
                                 "n_gpu_layers": -1},
                                _free_for_footprint(p["allotment"], p["cost"]),
                                free_ram_mb, p["cost"])
            if auto and auto.cpu_moe_layers:
                key, move = "cpu_moe_layers", f"--n-cpu-moe {auto.cpu_moe_layers}"
            elif auto and auto.n_gpu_layers and auto.n_gpu_layers != 999:
                key, move = "n_gpu_layers", f"-ngl {auto.n_gpu_layers}"
            else:
                key, move = "cpu_moe_layers", "-1（交给自动分层）"
            hand = p["cfg"].get(key, -1)
            hints.append(f"{p['name']} 的运行配置把 {key}={hand} 钉死了，"
                         f"它这一档要 {foot_map[p['name']]:.0f}MB，分给它只有 "
                         f"{p['allotment']:.0f}MB —— 改成 {move} 才不再越界")
        return CoexistencePlan(
            feasible=False, ceiling_mb=round(ceiling, 1),
            total_mb=round(total, 1), headroom_mb=round(ceiling - total, 1),
            spill_mb=round(spill, 1), spill_destination="未规划", plans={},
            footprints={}, rigid_names=rigid_names, reload_names=[],
            reason=(
                f"共存不可行：{'；'.join(hints)}。整卡常驻上限 {ceiling:.0f}MB"
                f"（物理 {vram_total_mb:.0f}MB×{cost.paging_ceiling_fraction:.4f}，"
                f"来源：{cost.source}），按现状合计 {total:.0f}MB 越界 "
                f"{total - ceiling:.0f}MB —— 越界付的是全体引擎一起掉速，不是崩，"
                f"所以照旧不拦加载，只把账报在这里。"
            ),
        )
    if free_ram_mb is None:
        destination = "内存（空闲内存未测到，未校验）"
    elif spill <= free_ram_mb:
        destination = f"内存（空闲 {free_ram_mb:.0f}MB 容得下）"
    else:
        destination = f"内存＋磁盘（会缺页：缺 {spill - free_ram_mb:.0f}MB 内存）"

    # 显存与内存是方向相反的两条边界：专家层推得越深（nc 越大）显存越省、内存分片越多。
    # 只按显存解算会挑出一档"合法但内存装不下"的，2026-10-19 03:30 实测就是这么掉到
    # 33.6 t/s 的（越界那一档反而 39.6–45.2）。这里把内存侧的上限一并算出来，两头
    # 没有交集时把两个数都报给他——账本没有权限替他选要违哪一边。
    fights = []
    if free_ram_mb is not None:
        for p in priced:
            nc = plans[p["name"]].cpu_moe_layers
            if p["rigid"] or not nc:
                continue
            per_layer = p["moe"]["expert_mb"] / max(1, p["moe"]["moe_layers"])
            room = max(0.0, free_ram_mb - (spill - entry_spill[p["name"]]))
            hi = int(room // per_layer)
            if nc > hi:
                fights.append(
                    f"{p['name']} 显存侧要求 --n-cpu-moe ≥ {nc}，内存侧只容得下 "
                    f"≤ {hi}（每层 {per_layer:.0f}MB，扣掉别只的分片后这只还能占 "
                    f"{room:.0f}MB 内存）")

    picks = "；".join(
        f"{p['name']} 常驻 {plans[p['name']].predicted_vram_mb:.0f}MB"
        + (f"（--n-cpu-moe {plans[p['name']].cpu_moe_layers}）"
           if plans[p["name"]].cpu_moe_layers else "（全量驻留）")
        for p in priced
    )
    reason = (
        f"共存账：整卡常驻上限 {ceiling:.0f}MB（物理 {vram_total_mb:.0f}MB×"
        f"{cost.paging_ceiling_fraction:.4f}，来源：{cost.source}；桌面底噪已含在"
        f"上限折算里）；固定档 {'、'.join(rigid_names) or '无'} 共 {rigid_total:.0f}MB，"
        f"弹性档先各自留底 {elastic_floor:.0f}MB，余量 {leftover:.0f}MB 按专家体量分账 → "
        f"{picks}；每进程另计 CUDA 上下文 {cost.engine_context_mb:.0f}MB×{len(priced)}"
        + (f"，挂载的视觉编码器 {vision_total:.0f}MB" if vision_total else "，无视觉编码器")
        + f"；合计 {total:.0f}MB "
        + (f"≤ {ceiling:.0f}MB，余 {ceiling - total:.0f}MB" if total <= ceiling
           else f"已越过 {ceiling:.0f}MB 上限 {total - ceiling:.0f}MB")
        + f"；落显存之外的权重 {spill:.0f}MB → {destination}"
        + (f"；两头没有交集：{'；'.join(fights)} —— 这一对怎么档都要缺页或越界，"
           f"只有卸载一只或调小上下文才能让两边同时留得住" if fights else "")
    )
    if reloads:
        reason += f"；需重载让步：{'、'.join(reloads)}"
    if upgrades:
        reason += f"（{'、'.join(upgrades)} 若重载可回收更多显存、decode 更快，本次不强制）"

    return CoexistencePlan(
        feasible=True, ceiling_mb=round(ceiling, 1), total_mb=round(total, 1),
        headroom_mb=round(ceiling - total, 1), spill_mb=round(spill, 1),
        spill_destination=destination, plans=plans, footprints=foot_map, rigid_names=rigid_names,
        reload_names=reloads, reason=reason,
    )


def plan_coexistence_for_models(entries: list) -> Optional[CoexistencePlan]:
    """Convenience wrapper: measures the card, then plans jointly."""
    cost = hardware_costs.resolve_coefficients()
    return plan_coexistence(entries, vram_total_mb=vram_total_mb(),
                           free_ram_mb=free_ram_mb(), coefficients=cost)


def vram_total_mb() -> Optional[float]:
    """Sum installed VRAM across CUDA devices via nvidia-smi (no console window)."""
    from lmloader.infra.proc_util import no_window_kwargs
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            **no_window_kwargs(),
        )
        vals = [float(x) for x in r.stdout.replace('"', "").splitlines() if x.strip()]
        return sum(vals) if vals else None
    except Exception as e:  # noqa: BLE001
        logger.debug(f"total VRAM probe failed: {e}")
        return None


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
