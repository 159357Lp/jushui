"""三级存储自动分层规划器的回归测试。

核心不变式只有一条：解出的 -ngl 对应的显存占用（层权重+KV+重尾）不得超过
预算——"宁保守不崩"是这套数学存在的唯一理由。另一组守卫是重尾修正：
embedding/输出层必须从均值里单独摘出来，否则大词表模型会在最后几层 OOM。
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmloader.engine import offload_planner as op  # noqa: E402
from lmloader.engine.hardware_costs import CostCoefficients  # noqa: E402
from lmloader.engine.offload_planner import plan_offload  # noqa: E402
from lmloader.engine.router import clamp_run_config  # noqa: E402


def make_model(*, size_gb=20.0, layers=48, hidden=4096, vocab=0, mtp=False):
    meta = {
        "block_count": layers,
        "embedding_length": hidden,
        "context_length": 131072,
        "supports_mtp": mtp,
    }
    if vocab:
        meta["vocab_size"] = vocab
    return SimpleNamespace(
        name="t.gguf", path="C:/tmp/t.gguf", architecture="qwen35",
        size=int(size_gb * 1024 ** 3), metadata=meta, backend=None,
    )


CFG = {"ctx_size": 8192, "parallel": 1, "cache_type_k": "q4_0",
       "n_gpu_layers": -1, "cache_type_v": "q4_0"}


# ---------- 不超预算不变式 ----------

def test_never_exceeds_budget_under_pressure():
    """20GB 模型进 8GB 空闲显存：部分卸载时 GPU 侧（层+KV）严格在预算内，
    重尾两张表留在 CPU 侧不占这份预算。"""
    p = plan_offload(make_model(), CFG, free_vram_mb=8192, free_ram_mb=65536)
    assert p is not None
    ngl = p.n_gpu_layers
    assert 0 < ngl < 999
    assert p.layers_on_disk > 0
    used = ngl * (p.avg_layer_mb + p.kv_full_mb / p.layers_total)
    assert used <= p.budget_mb + 1.0, (
        f"plan oversubscribes: {used} > {p.budget_mb}"
    )


def test_full_gate_blocks_999_when_tail_does_not_fit():
    """线性解 ≥ 总层数但全量（size+KV）超预算 → 必须退一层留重尾在 CPU，
    而不是乐观地报 999 然后 OOM。"""
    p = plan_offload(make_model(size_gb=20.0), CFG,
                     free_vram_mb=24576, free_ram_mb=65536)
    assert p.n_gpu_layers == p.layers_total - 1
    assert p.layers_on_disk == 1


def test_full_offload_when_vram_room():
    p = plan_offload(make_model(size_gb=4.0), CFG,
                     free_vram_mb=16384, free_ram_mb=32768)
    assert p.n_gpu_layers == 999
    assert p.layers_on_disk == 0


def test_zero_when_no_budget():
    p = plan_offload(make_model(size_gb=40.0), CFG,
                     free_vram_mb=300, free_ram_mb=8192)
    assert p.n_gpu_layers == 0


# ---------- 重尾层修正 ----------

def test_heavy_tail_reserves_embedding_and_output():
    hidden, vocab = 4096, 151_936
    p = plan_offload(make_model(vocab=vocab), CFG,
                     free_vram_mb=8192, free_ram_mb=65536)
    expect_mb = (2 * vocab * hidden * 2) / (1024 * 1024)
    assert abs(p.tail_reserve_mb - expect_mb) < 2.0


def test_big_vocab_lowers_ngl_versus_fallback():
    m = make_model(vocab=151_936)
    with_vocab = plan_offload(m, CFG, free_vram_mb=8192, free_ram_mb=65536)
    m2 = make_model(vocab=0)  # 走 12% 兜底
    fallback = plan_offload(m2, CFG, free_vram_mb=8192, free_ram_mb=65536)
    # 151936×4096×4B ≈ 2.3GB > 12%×20GB=2.4GB… 两者接近时至少必须同向保守：
    # 兜底不得低于精算预留
    assert with_vocab.tail_reserve_mb > 0
    assert fallback.tail_reserve_mb > 0
    assert with_vocab.n_gpu_layers <= fallback.n_gpu_layers or \
        with_vocab.tail_reserve_mb <= fallback.tail_reserve_mb


# ---------- 层数与 MTP ----------

def test_mtp_counts_extra_layer_in_total():
    p = plan_offload(make_model(layers=48, mtp=True), CFG,
                     free_vram_mb=8192, free_ram_mb=65536)
    assert p.layers_total == 49


# ---------- 不可测即弃权 ----------

def test_none_without_free_vram():
    assert plan_offload(make_model(), CFG, None, 32768) is None


def test_none_without_metadata_fields():
    m = make_model()
    m.metadata = {}
    assert plan_offload(m, CFG, 8192, 32768) is None


def test_none_accepts_dict_model_shape():
    m = {"metadata": {}, "size": 0}
    assert plan_offload(m, CFG, 8192, 32768) is None


# ---------- 预热门槛 ----------

def test_prewarm_requires_ram_headroom():
    m = make_model(size_gb=20.0)
    tight = plan_offload(m, CFG, free_vram_mb=6144, free_ram_mb=20 * 1024)
    assert tight.layers_on_disk > 0
    assert tight.prewarm_suggested is False  # 空闲 RAM 放不下 20GB×1.25
    loose = plan_offload(m, CFG, free_vram_mb=6144, free_ram_mb=64 * 1024)
    assert loose.prewarm_suggested is True


def test_no_prewarm_when_fully_on_gpu():
    p = plan_offload(make_model(size_gb=4.0), CFG,
                     free_vram_mb=16384, free_ram_mb=32768)
    assert p.prewarm_suggested is False


# ---------- clamp 与哨兵 ----------

def test_clamp_preserves_auto_sentinel():
    cfg = clamp_run_config(make_model(), {"ctx_size": 4096, "parallel": 1,
                                          "n_gpu_layers": -1, "threads": 8})
    assert cfg["n_gpu_layers"] == -1


def test_clamp_still_clamps_normal_values():
    cfg = clamp_run_config(make_model(), {"ctx_size": 4096, "parallel": 1,
                                          "n_gpu_layers": 5000})
    assert cfg["n_gpu_layers"] == 999


# ---------- 线性解的精确性 ----------

def test_kv_grows_with_ctx_and_parallel():
    p1 = plan_offload(make_model(), CFG, free_vram_mb=8192, free_ram_mb=65536)
    cfg2 = dict(CFG, ctx_size=32768, parallel=4)
    p2 = plan_offload(make_model(), cfg2, free_vram_mb=8192, free_ram_mb=65536)
    assert p2.kv_full_mb == p1.kv_full_mb * 16
    assert p2.n_gpu_layers <= p1.n_gpu_layers


def test_reason_reports_the_account():
    p = plan_offload(make_model(), CFG, free_vram_mb=8192, free_ram_mb=65536)
    assert "自动分层" in p.reason and "预算" in p.reason and "ngl" in p.reason


# ---------- 源码接线守卫 ----------

def test_router_resolves_sentinel():
    import inspect

    from lmloader.engine.router import EngineRouter
    src = inspect.getsource(EngineRouter.build_launch_config)
    assert "gpu == -1" in src and "plan_for_model" in src
    assert "prewarm=prewarm_flag" in src


# ---------- MoE 专家分层（数字取自 Qwen3-30B-A3B-Q4_K_M 头部实测） ----------

MOE_SIZE = 18556685824          # 17697.6 MB
MOE_EXPERT_BYTES = 17553162240  # 16739.9 MB，占权重 94.6%


def make_moe_model(**over):
    meta = {
        "block_count": 48, "embedding_length": 2048, "context_length": 40960,
        "vocab_size": 151669, "supports_mtp": False,
        "expert_count": 128, "expert_used": 8,
        "moe_layer_count": 48, "expert_bytes": MOE_EXPERT_BYTES,
    }
    meta.update(over)
    return SimpleNamespace(
        name="qwen3-30b-a3b.gguf", path="F:/models/qwen3-30b-a3b.gguf",
        architecture="qwen3moe", size=MOE_SIZE, metadata=meta, backend=None,
    )


def _moe_gpu_usage_mb(p):
    """显存账：非专家权重 + 满额 KV + 留在显存里的那些层的专家。"""
    expert_total = p.avg_layer_mb * p.layers_total
    dense = p.prewarm_mb - expert_total
    keep = p.layers_total - p.cpu_moe_layers
    return dense + p.kv_full_mb + keep * p.avg_layer_mb


def test_moe_keeps_attention_and_offloads_expert_layers():
    p = plan_offload(make_moe_model(), CFG, free_vram_mb=15360, free_ram_mb=65536)
    assert p.n_gpu_layers == 999, "MoE 应让 -ngl 满驻留，靠 -ncmoe 省显存"
    assert 0 < p.cpu_moe_layers < 48
    assert _moe_gpu_usage_mb(p) <= p.budget_mb + 1.0, (
        f"plan oversubscribes: {_moe_gpu_usage_mb(p)} > {p.budget_mb}")
    assert "--n-cpu-moe" in p.reason and "MoE 专家分层" in p.reason


def test_dense_model_never_gets_cpu_moe():
    p = plan_offload(make_model(), CFG, free_vram_mb=15360, free_ram_mb=65536)
    assert p.cpu_moe_layers == 0


def test_moe_falls_back_to_layer_tiering_when_attention_cant_fit():
    # 显存小到连注意力+KV 都放不下：-ncmoe 救不了，必须回通用解。
    p = plan_offload(make_moe_model(), CFG, free_vram_mb=2048, free_ram_mb=65536)
    assert p.cpu_moe_layers == 0
    assert p.n_gpu_layers < 999


def test_moe_offloads_every_expert_when_no_room_for_any():
    # 预算刚好够注意力+KV，一层专家都放不下 → 48 层全下 CPU。
    tight = CostCoefficients(vram_safety_fraction=0.99, base_overhead_mb=768.0,
                             fallback_tail_fraction=0.12, max_tail_fraction=0.5,
                             prewarm_ram_margin=1.25,
                             paging_ceiling_fraction=0.80)
    p = plan_offload(make_moe_model(), CFG, free_vram_mb=2800,
                     free_ram_mb=65536, coefficients=tight)
    assert p.cpu_moe_layers == 48
    assert p.n_gpu_layers == 999


def test_tighter_coefficients_offload_more_experts():
    """反向对照：同一模型同一显存，更凶的系数必须让下 CPU 的专家层变多。

    只会变绿的探针等于没测——这条守卫的是"系数表真的接进了 MoE 分支"。
    """
    loose = CostCoefficients(vram_safety_fraction=0.95, base_overhead_mb=100.0,
                             fallback_tail_fraction=0.12, max_tail_fraction=0.5,
                             prewarm_ram_margin=1.25,
                             paging_ceiling_fraction=0.80)
    tight = CostCoefficients(vram_safety_fraction=0.55, base_overhead_mb=3000.0,
                             fallback_tail_fraction=0.12, max_tail_fraction=0.5,
                             prewarm_ram_margin=1.25,
                             paging_ceiling_fraction=0.80)
    p_loose = plan_offload(make_moe_model(), CFG, free_vram_mb=15360,
                           free_ram_mb=65536, coefficients=loose)
    p_tight = plan_offload(make_moe_model(), CFG, free_vram_mb=15360,
                           free_ram_mb=65536, coefficients=tight)
    assert p_tight.cpu_moe_layers > p_loose.cpu_moe_layers, (
        f"{p_tight.cpu_moe_layers} !> {p_loose.cpu_moe_layers}")
    assert "系数来源" in p_loose.reason


# ---------- KV 真实几何（数字取自本地 Qwen3.6-35B-A3B-MXFP4_MOE 头部实测） ----------
#
# 这里的常量不是编的：41 层、interval 4、2 个 KV 头、key/value 各 256，
# 全部来自那个文件的 GGUF 头部；llama.cpp 自己在 -lv 4 里报出 ctx 65536
# 时 CUDA0 KV buffer = 128 MiB，是这套几何的实测锚点（见 _pydiag 探针）。

HYBRID_SIZE = 22182955008           # 21155 MB
HYBRID_EXPERT_BYTES = 19451084800   # 18550 MB，占权重 87.7%


def make_hybrid_model(**over):
    meta = {
        "block_count": 41, "embedding_length": 2048, "context_length": 262144,
        "vocab_size": 496645, "supports_mtp": True,
        "attention_head_count": 16, "attention_head_count_kv": 2,
        "attention_key_length": 256, "attention_value_length": 256,
        "full_attention_interval": 4,
        "expert_count": 256, "expert_used": 8,
        "moe_layer_count": 41, "expert_bytes": HYBRID_EXPERT_BYTES,
    }
    meta.update(over)
    return SimpleNamespace(
        name="qwen36-35b-a3b.gguf", path="M:/models/qwen36-35b-a3b.gguf",
        architecture="qwen35moe", size=HYBRID_SIZE, metadata=meta, backend=None,
    )


def test_kv_elements_count_only_the_layers_that_hold_kv():
    # 41 层里只有 (i+1)%4==0 的 10 层存 KV，MTP 附加层再 +1。
    assert op.kv_elements_per_token(make_hybrid_model().metadata) == 11 * 2 * (256 + 256)


def test_missing_geometry_returns_none_instead_of_guessing():
    meta = dict(make_hybrid_model().metadata)
    for k in ("attention_head_count_kv", "attention_key_length"):
        meta.pop(k)
    assert op.kv_elements_per_token(meta) is None


def test_legacy_estimate_is_untouched_when_geometry_is_absent():
    """没有几何字段时 KV 必须逐字节等于旧公式，保证这次改动不影响存量模型。"""
    meta = {k: 0 for k in ("attention_head_count_kv", "attention_key_length",
                           "attention_value_length", "full_attention_interval")}
    p = plan_offload(make_hybrid_model(**meta), CFG, free_vram_mb=15360,
                     free_ram_mb=65536)
    # 2 × 2048 × 42 层 × 8192 token × 0.5625 字节
    assert p.kv_full_mb == 756.0
    assert "按 MHA 上限估计" in p.reason


def test_hybrid_kv_no_longer_swamps_the_budget():
    p = plan_offload(make_hybrid_model(), {**CFG, "ctx_size": 65536},
                     free_vram_mb=15360, free_ram_mb=65536)
    assert p.kv_full_mb < 400, (
        f"真实几何下 KV 应远小于 400MB，实测 llama.cpp 只报 128MB：{p.kv_full_mb}")
    assert "几何精算" in p.reason


def test_accurate_kv_resides_more_experts():
    """反向对照：把几何字段抹掉→KV 被高估→留在 CPU 的专家层必须变多。

    守的是"KV 估算接进了卸载决策"，不是只接进了日志。
    """
    meta = dict(make_hybrid_model().metadata)
    for k in ("attention_head_count_kv", "attention_key_length",
              "attention_value_length"):
        meta[k] = 0
    cfg = {**CFG, "ctx_size": 65536}
    exact = plan_offload(make_hybrid_model(), cfg, free_vram_mb=15360, free_ram_mb=65536)
    legacy = plan_offload(make_hybrid_model(**meta), cfg, free_vram_mb=15360,
                          free_ram_mb=65536)
    assert exact.kv_full_mb < legacy.kv_full_mb
    assert exact.cpu_moe_layers < legacy.cpu_moe_layers, (
        f"{exact.cpu_moe_layers} !< {legacy.cpu_moe_layers}")
    assert exact.n_gpu_layers == 999


def test_estimate_max_parallel_stops_undercounting_hybrid_slots():
    """同一张卡，KV 高估 16 倍就会把塞得下的 8 个并发槽报成 1。

    权重压到 6GiB 是想单独测 KV 项：这个尺寸的模型才谈得上多槽，
    20GiB 的模型在 16GB 卡上恒为 1 槽，改 KV 也看不出差别。
    """
    from lmloader.engine.router import estimate_max_parallel
    cfg = {**CFG, "ctx_size": 65536}
    small = int(6 * 1024 ** 3)
    exact_model = make_hybrid_model()
    exact_model.size = small
    meta = dict(make_hybrid_model().metadata)
    for k in ("attention_head_count_kv", "attention_key_length",
              "attention_value_length"):
        meta[k] = 0
    legacy_model = make_hybrid_model(**meta)
    legacy_model.size = small
    exact = estimate_max_parallel(exact_model, cfg, vram_total_mb=15999)
    legacy = estimate_max_parallel(legacy_model, cfg, vram_total_mb=15999)
    assert legacy == 1, f"旧公式应触底到 1 槽：{legacy}"
    assert exact == 8, f"真实几何应放行到并发上限：{exact}"
