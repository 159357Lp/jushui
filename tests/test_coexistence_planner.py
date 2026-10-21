"""多模型共存联合规划的回归测试。

fixture 用的是 2026-10-19 在 5080 Laptop 上实测的几何与代价行，不是编出来的
整数：35B-A3B 权重 21155MB（专家 18550MB / 41 层，每层 452MB）、ctx 32768 的
满额 KV 198MB；d3-mini 权重 3536MB、ctx 65536 的 KV 576MB；整卡常驻上限
16303×0.8662=14122MB（拐点档实测）。换卡或换引擎后数字会变，判据不变。

守的三条不变式：
  * 合计常驻绝不越过整卡上限——越过就是 WDDM 分页，全体一起掉速；
  * 钉选决策模型与非 MoE 模型绝不被压小（它没有弹性档可退）；
  * 放不下就明说放不下，绝不交一份"可行"但会崩的账。
"""
from __future__ import annotations

import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmloader.engine.hardware_costs import CostCoefficients  # noqa: E402
from lmloader.engine.offload_planner import (  # noqa: E402
    plan_coexistence,
    plan_offload,
)

# 2026-10-19 由 _pydiag/fit_cost_row.py 从实测拐点拟合的 5080 行
COST = CostCoefficients(
    vram_safety_fraction=0.9104,
    base_overhead_mb=596.4,
    fallback_tail_fraction=0.12,
    max_tail_fraction=0.5,
    prewarm_ram_margin=1.25,
    paging_ceiling_fraction=0.8662,
    # 单进程 CUDA 上下文：由 _pydiag/fit_cost_row.py 从标定行反解
    # gpu_used −(卡总量−free_before)−自报合计，nc16–26 六档落在 149–181MB，取上沿。
    engine_context_mb=180.0,
    source="nvidia-geforce-rtx-5080-laptop-gpu-sm120",
)
VRAM_TOTAL = 16303.0
CEILING = VRAM_TOTAL * COST.paging_ceiling_fraction   # 14121.7 MB

# models/Qwen3.6-35B-A3B-MXFP4_MOE/...gguf 的实测元数据（_pydiag/model_meta.json）
MOE_META = {
    "architecture": "qwen35moe", "block_count": 41, "embedding_length": 2048,
    "vocab_size": 496645, "supports_mtp": True, "expert_count": 256,
    "expert_used": 8, "moe_layer_count": 41, "expert_bytes": 19451084800,
    "attention_head_count": 16, "attention_head_count_kv": 2,
    "attention_key_length": 256, "attention_value_length": 256,
    "full_attention_interval": 4,
}
# models/d3-mini-Q4_K_M/...gguf：dense，钉选决策槽（decision_type=vllm-d3）
D3_META = {
    "architecture": "qwen35", "block_count": 32, "embedding_length": 2560,
    "vocab_size": 496645, "supports_mtp": False, "expert_count": 0,
    "expert_used": 0, "moe_layer_count": 0, "expert_bytes": 0,
    "attention_head_count": 16, "attention_head_count_kv": 4,
    "attention_key_length": 256, "attention_value_length": 256,
    "full_attention_interval": 4, "decision_type": "vllm-d3",
}
MOE_CFG = {"ctx_size": 32768, "parallel": 1, "cache_type_k": "q4_0",
           "cache_type_v": "q4_0", "n_gpu_layers": -1}
D3_CFG = {**MOE_CFG, "ctx_size": 65536}


def make_model(name, meta, size_mb):
    return SimpleNamespace(name=name, path=f"C:/tmp/{name}",
                           size=int(size_mb * 1024 ** 2), metadata=dict(meta),
                           backend=None)


def moe():
    return make_model("35B", MOE_META, 21155.0)


def d3():
    return make_model("d3", D3_META, 3535.8)


def pair(*, incumbent=None, free_ram_mb=3700.0, vram_total_mb=VRAM_TOTAL,
         coefficients=COST):
    """今天这台机器上的真实一对：35B 已在卡上（实测常驻 12897MB），再加 d3。"""
    first = {"model": moe(), "cfg": MOE_CFG, "pinned": False,
             "resident_mb": 12897.0}
    if incumbent:
        first.update(incumbent)
    entries = [first,
               {"model": d3(), "cfg": D3_CFG, "pinned": True, "resident_mb": None}]
    return plan_coexistence(entries, vram_total_mb=vram_total_mb,
                           free_ram_mb=free_ram_mb, coefficients=coefficients)


# ---------- 实测几何下的联合解 ----------

def test_fixture_geometry_matches_the_measured_sweep():
    """先钉住 fixture 本身：几何变了，下面所有断言就成了自欺。

    198→244 的来由：planner 现在把 MTP 草稿层的 KV 按 f16 计（引擎不吃
    --cache-type 的那条 buffer）。这条改动只在 NVFP4-LOW 上量过（kv_probe：
    ctx 16000 → 63 MiB、65536 → 256 MiB，正好 1 层 × 4 头 × (256+256) × 2B）；
    35B 这只没量过，244 = 主干 198 ＋ 草稿层 f16 差价 46 是公式输出，钉在这里
    是为了让下面的联合解断言有稳定基线，不是实测。
    旧锚点那句『引擎自报 KV 128 MiB』值得留个记号：128 恰是 ctx 65536 下草稿层
    f16 buffer 的大小，而 lifecycle 的末条覆盖写法只会报出最后那一条——所以它
    从来不是整张 KV 的量（任务 #21）。
    """
    m = plan_offload(moe(), MOE_CFG, 1e9, None, coefficients=COST)
    d = plan_offload(d3(), D3_CFG, 1e9, None, coefficients=COST)
    assert m.kv_full_mb == 244.0, m.kv_full_mb      # 主干 q4_0 ＋ 草稿层 f16
    assert d.kv_full_mb == 576.0, d.kv_full_mb
    assert m.predicted_vram_mb == 21399.0            # size + kv，全量驻留
    assert m.cpu_moe_layers == 0 and m.n_gpu_layers == 999


def test_incumbent_yields_so_both_fit_under_the_ceiling():
    r = pair()
    assert r.feasible, r.reason
    assert r.total_mb <= CEILING + 1.0, r.reason
    assert r.plans["35B"].cpu_moe_layers == 29, r.plans["35B"].reason
    assert r.plans["35B"].n_gpu_layers == 999        # 只搬专家，注意力与 KV 全驻留
    assert "35B" in r.reload_names, r.reload_names   # 13077MB → 9009MB，必须重载让步
    assert r.rigid_names == ["d3"], r.rigid_names


def test_pinned_dense_model_keeps_full_residency():
    r = pair()
    plan = r.plans["d3"]
    assert plan.n_gpu_layers == 999 and plan.cpu_moe_layers == 0
    assert plan.predicted_vram_mb == 4111.8, plan.predicted_vram_mb
    assert "d3" not in r.reload_names


# ---------- 每进程自报口径之外的两项：CUDA 上下文 + 挂载的视觉编码器 ----------

def pair_with_vision(mmproj_mb=644.3, *, coefficients=COST):
    """d3-mini 挂视觉编码器（实测 mmproj-d3-mini-BF16.gguf = 644.3MB）。"""
    entries = [
        {"model": moe(), "cfg": MOE_CFG, "pinned": False, "resident_mb": 12897.0},
        {"model": d3(), "cfg": D3_CFG, "pinned": True, "resident_mb": None,
         "vision_mb": mmproj_mb},
    ]
    return plan_coexistence(entries, vram_total_mb=VRAM_TOTAL,
                            free_ram_mb=3700.0, coefficients=coefficients)


def test_context_allowance_is_priced_once_per_process():
    """反向对照：上下文系数归零 → 每只引擎少记一份，方案会多放行一层专家。

    2026-10-19 实测：两只引擎自报合计 13227MB，卡上真实占用 14858MB，差的就是
    每进程 CUDA 上下文与挂载的 mmproj。不逐进程计一次，共存账比真实占用乐观
    约 1.6GB，翻页的仍是全体。
    """
    with_ctx = pair()
    naked = pair(coefficients=CostCoefficients(
        **{**COST.as_dict(), "engine_context_mb": 0.0}))
    assert with_ctx.feasible and naked.feasible
    assert abs(with_ctx.footprints["d3"] - naked.footprints["d3"]
               - COST.engine_context_mb) < 1.0, (with_ctx.footprints, naked.footprints)
    assert naked.plans["35B"].cpu_moe_layers < with_ctx.plans["35B"].cpu_moe_layers, \
        naked.plans["35B"].reason


def test_mounted_vision_encoder_pushes_the_incumbent_further_down():
    r0, r1 = pair(), pair_with_vision()
    assert r1.feasible, r1.reason
    # 挂载视觉必须让在用那只多让出专家层；具体让几层随 KV 计价变，方向不许变
    assert r1.plans["35B"].cpu_moe_layers > r0.plans["35B"].cpu_moe_layers, \
        (r0.plans["35B"].reason, r1.plans["35B"].reason)
    assert r1.total_mb <= CEILING + 1.0, r1.reason
    assert "视觉编码器" in r1.reason, r1.reason
    assert abs(r1.footprints["d3"] - pair_with_vision(0.0).footprints["d3"]
               - 644.3) < 1.0, r1.footprints


def test_hand_pinned_tier_that_breaks_the_ceiling_is_refused_not_approved():
    """反向对照：手填档把一只顶到它分到的额度之外 → 必须判不可行并给出改成多少。

    2026-10-19 02:55 实测：上次让步代笔写回的 --n-cpu-moe 27 这轮被当成手填档
    尊重，合计 14907MB 越界 785MB 却仍标着"可行"、余 -785MB，reload_names 空。
    手填档是用户的最高权限，联合解算不许覆盖它——只能把账和改法一起报出来。
    """
    entries = [
        {"model": moe(), "cfg": {**MOE_CFG, "cpu_moe_layers": 27},
         "pinned": False, "resident_mb": 8865.6},
        {"model": d3(), "cfg": D3_CFG, "pinned": True, "resident_mb": None,
         "vision_mb": 644.3},
    ]
    r = plan_coexistence(entries, vram_total_mb=VRAM_TOTAL,
                         free_ram_mb=3700.0, coefficients=COST)
    assert r.feasible is False, r.reason
    assert not r.plans and not r.reload_names, r.reason
    assert "cpu_moe_layers=27" in r.reason, r.reason
    assert "越界" in r.reason, r.reason
    assert "改成 --n-cpu-moe" in r.reason, r.reason
    # 提示里"分给它多少"必须是分给它的那份，不能顺手抄物理显存或整卡上限：
    # 用户照这个数字去调档，报大了一档就还在越界。
    figures = re.findall(r"它这一档要 (\d+)MB，分给它只有 (\d+)MB", r.reason)
    assert figures, r.reason
    for need, allot in figures:
        assert float(allot) < float(need), (allot, need, r.reason)
        assert float(allot) <= r.ceiling_mb, (allot, r.ceiling_mb, r.reason)


# ---------- 内存侧也是硬边界（两头定档）----------

def test_memory_ceiling_names_the_two_sided_tier_conflict():
    """反向对照：只按显存解出的档，内存装不下时必须把两条边界都报出来。

    2026-10-19 03:30 实测：35B 按显存上限解到 --n-cpu-moe 29，分片 13121MB 而机器
    只剩 8.5GB 内存 → 从盘上缺页，decode 33.6 t/s；反而被拒判的 nc27（越界 785MB）
    跑出 39.6–45.2。账本不许替用户选违哪一边，只把「显存要求 ≥ A / 内存只容 ≤ B」
    两个数一起摊开。
    """
    r = pair(free_ram_mb=8545.0)
    assert r.feasible, r.reason          # 显存侧仍然合法，判据不该顺手改成拒判
    assert "两头没有交集" in r.reason, r.reason
    lo = r.plans["35B"].cpu_moe_layers
    m = re.search(r"显存侧要求 --n-cpu-moe ≥ (\d+)，内存侧只容得下 ≤ (\d+)", r.reason)
    assert m, r.reason
    assert int(m.group(1)) == lo, (m.group(1), lo, r.reason)
    assert int(m.group(2)) < lo, (m.group(2), lo, r.reason)
    # 18 = (空闲 8545 − 别只的分片) ÷ 每层 452；把这只自己的分片重复扣一遍会算成 0，
    # 那是"内存一点都不能占"的假边界，会把该重解的档说成无解。
    assert int(m.group(2)) == 18, r.reason
    assert "每层 452MB" in r.reason, r.reason      # 18550MB 专家 / 41 层


def test_ample_ram_reports_no_two_sided_conflict():
    """反向对照：内存充裕时不许喊冲突（只会报警的警报器等于没有警报器）。"""
    r = pair(free_ram_mb=30000.0)
    assert r.feasible, r.reason
    assert "两头没有交集" not in r.reason, r.reason
    assert "容得下" in r.reason, r.reason


# ---------- 不变式：绝不越界、单调让步 ----------

def test_never_overcommits_across_ceiling_fractions():
    for frac in (0.40, 0.55, 0.70, 0.8662, 0.95, 1.0):
        cost = CostCoefficients(**{**COST.as_dict(), "paging_ceiling_fraction": frac})
        r = pair(coefficients=cost)
        ceiling = VRAM_TOTAL * frac
        if r.feasible:
            assert r.total_mb <= ceiling + 1.0, (frac, r.total_mb, r.headroom_mb)
            assert r.headroom_mb >= -1.0, (frac, r.reason)
        else:
            assert not r.plans, (frac, "拒绝就不该还给出一份方案")


def test_more_vram_headroom_never_means_more_cpu_layers():
    """上限越宽，留在 CPU 的专家层只该变少（更快），不该反过来。"""
    tiers = []
    for frac in (0.60, 0.75, 0.8662, 0.95):
        cost = CostCoefficients(**{**COST.as_dict(), "paging_ceiling_fraction": frac})
        r = pair(coefficients=cost)
        tiers.append(r.plans["35B"].cpu_moe_layers)
    assert tiers == sorted(tiers, reverse=True), tiers


# ---------- 反向对照：判据真的在起作用 ----------

def test_pinned_moe_cannot_yield_so_request_is_refused():
    """反向对照①：把 35B 也钉选 → 它没有让步余地，必须报不可行。

    若钉选标记没进账本，这里会照样给出一份 nc=28 的"可行"方案，
    把一个绝不能动的决策模型压到 CPU——所以这个红是判据本身有效。
    """
    entries = [
        {"model": moe(), "cfg": MOE_CFG, "pinned": True, "resident_mb": 12897.0},
        {"model": d3(), "cfg": D3_CFG, "pinned": True, "resident_mb": None},
    ]
    r = plan_coexistence(entries, vram_total_mb=VRAM_TOTAL,
                        free_ram_mb=3700.0, coefficients=COST)
    assert r.feasible is False
    assert not r.plans and not r.reload_names
    assert "钉选" in r.reason, r.reason
    assert "35B" in r.rigid_names


def test_smaller_card_refuses_instead_of_shrinking_the_pinned_model():
    """反向对照②：同一对模型塞进 8GB 卡 → 缺口 1011MB，只能拒绝。"""
    r = pair(vram_total_mb=8192.0)
    assert r.feasible is False, r.reason
    assert not r.plans
    assert "缺口" in r.reason and "卸载" in r.reason, r.reason


def test_spill_destination_flips_when_ram_halved():
    """反向对照③：内存砍半 → 分片落点必须从"内存"改成"内存＋磁盘（会缺页）"。

    用户要的是"剩余分片留在内存和磁盘"，所以内存不够只提醒不拒绝；
    但若这条判据没接进结论，两次的落点会一模一样。
    """
    roomy = pair(free_ram_mb=30000.0)
    tight = pair(free_ram_mb=1500.0)
    assert roomy.feasible and tight.feasible          # 内存不足绝不导致拒绝
    assert "磁盘" not in roomy.spill_destination, roomy.spill_destination
    assert "磁盘" in tight.spill_destination, tight.spill_destination
    assert abs(roomy.spill_mb - tight.spill_mb) < 1.0   # 分片量与内存无关，只有落点变
    assert "缺页" in tight.reason, tight.reason


def test_unmeasurable_model_returns_none_not_a_guess():
    """几何读不出来时返回 None，调用方保持手动语义，绝不猜一个更紧的界。"""
    broken = make_model("broken", {"block_count": 0, "embedding_length": 0}, 1000.0)
    r = plan_coexistence([{"model": broken, "cfg": MOE_CFG}],
                        vram_total_mb=VRAM_TOTAL, free_ram_mb=3700.0,
                        coefficients=COST)
    assert r is None
    assert plan_coexistence([], vram_total_mb=VRAM_TOTAL, coefficients=COST) is None
    assert plan_coexistence([{"model": moe(), "cfg": MOE_CFG}],
                           vram_total_mb=None, coefficients=COST) is None


# ---------- 让步方向：只压需要让位的，白送的速度不折腾 ----------

def test_incumbent_that_is_already_thin_enough_is_not_reloaded():
    # 自指：先解一遍拿到本方案额度，再把它当作"已在卡上的实测常驻"喂回去。
    # 实测口径不含每进程上下文，所以要减掉才同径——不减就会凭空多让一档。
    allotment = pair().footprints["35B"] - COST.engine_context_mb
    r = pair(incumbent={"resident_mb": allotment})
    assert r.reload_names == [], r.reload_names
    assert "需重载" not in r.reason


def test_looser_ceiling_suggests_upgrade_without_forcing_reload():
    """反向对照④：已加载档比方案更省 → 只提示"可回收"，不强制重载。"""
    r = pair(incumbent={"resident_mb": 5000.0})
    assert r.reload_names == [], r.reload_names
    assert "可回收" in r.reason, r.reason
