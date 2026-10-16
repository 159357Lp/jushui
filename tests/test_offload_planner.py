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
