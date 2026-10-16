"""三级存储分层规划器 — 加载时一次性推导 -ngl，杜绝超显存。

背景：-ngl 此前全靠手填（999=全放 / 调低省显存），大模型或两三模型共存
时只能凭经验试。本模块把"空闲显存 − KV 预留 − 固定开销 − 重尾层余量"
的预算数学收进来，供 run config 的"自动分层"档（n_gpu_layers == -1）。

三个不可约的保守性来源（都是故意的，宁慢勿崩）：
  * 重尾余量：embedding + 输出层远超单层均值（词表 150k+ 的模型单层能差
    10 倍），必须从权重总量里先单独扣除，不能让均值摊平它们。
  * KV 随 ngl 线性增长：只有上 GPU 的层需要 VRAM KV，KV 预算和解按
    ngl*(avg + kv_full/layers) <= budget 精确线性求解，不是先扣满额 KV。
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

from lmloader.logger import logger

# 空闲显存的安全系数：桌面/显示与瞬时波动都要留（笔记本共享显存场景
# 实测 Windows 桌面本身就会吃掉一块）。
VRAM_SAFETY_FRACTION = 0.85
# 计算缓冲 + 运行期固定开销（取自 llama.cpp 常见实测；MTP 额外层已并入
# 层均值，不在此重复计）。
BASE_OVERHEAD_MB = 768.0
# 没有 vocab 元数据时的兜底：按权重总量比例预留重尾。
FALLBACK_TAIL_FRACTION = 0.12
# 重尾余量上限：embedding+output 不可能超过总权重的一半，超了说明估算失真。
MAX_TAIL_FRACTION = 0.5
# 预热页缓存的 RAM 门槛：模型文件必须放得进空闲内存且还留 25% 余量，
# 否则宁可让首推理慢慢缺页，也不把系统推进交换区。
PREWARM_RAM_MARGIN = 1.25


def _cache_bytes(cache_type: str) -> float:
    # 与 router.CACHE_BYTES 同一张表；延迟导入避开模块环。
    from lmloader.engine.router import CACHE_BYTES
    return CACHE_BYTES.get(str(cache_type or "q4_0"), 0.6)


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

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def plan_offload(
    model,
    cfg: dict,
    free_vram_mb: Optional[float],
    free_ram_mb: Optional[float],
) -> Optional[OffloadPlan]:
    """Derive ngl for one load. Pure: callers inject the measurements.

    ``model`` may be a ModelRecord or the GUI/API dict (size + metadata).
    Returns None when anything essential is unmeasurable — the caller must
    then keep the manual value, never a guessed bound.
    """
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
        tail_mb = size_mb * FALLBACK_TAIL_FRACTION
    tail_mb = min(tail_mb, size_mb * MAX_TAIL_FRACTION)

    trunk_mb = max(size_mb - tail_mb, 1.0)
    avg_layer_mb = trunk_mb / layers_total

    # KV 只按上 GPU 的层预留，但满额先算出来作为线性方程的系数。
    kv_full_mb = 2 * ctx * hidden * layers_total * _cache_bytes(
        cfg.get("cache_type_k", "q4_0")) / (1024 * 1024) * parallel

    budget_mb = free_vram_mb * VRAM_SAFETY_FRACTION - BASE_OVERHEAD_MB
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
    if disk > 0 and free_ram_mb and free_ram_mb >= size_mb * PREWARM_RAM_MARGIN:
        prewarm = True

    reason = (
        f"自动分层：空闲显存 {free_vram_mb:.0f}MB×{VRAM_SAFETY_FRACTION:.2f}−"
        f"固定开销 {BASE_OVERHEAD_MB:.0f}MB → 预算 {max(budget_mb, 0):.0f}MB；"
        f"重尾预留 {tail_mb:.0f}MB（{'vocab 精算' if vocab else '均值兜底'}），"
        f"单层均 {avg_layer_mb:.0f}MB，满额 KV {kv_full_mb:.0f}MB（ctx {ctx}×{parallel} 槽），"
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
    """Convenience wrapper that measures the machine, then plans."""
    return plan_offload(model, cfg, free_vram_mb(), free_ram_mb())


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
