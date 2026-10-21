"""三级存储自动分层规划器的回归测试。

核心不变式只有一条：解出的 -ngl 对应的显存占用（层权重+KV+重尾）不得超过
预算——"宁保守不崩"是这套数学存在的唯一理由。另一组守卫是重尾修正：
embedding/输出层必须从均值里单独摘出来，否则大词表模型会在最后几层 OOM。
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmloader.engine import offload_planner as op  # noqa: E402
from lmloader.engine.hardware_costs import CostCoefficients  # noqa: E402
from lmloader.engine.offload_planner import plan_offload  # noqa: E402
from lmloader.engine.router import clamp_run_config  # noqa: E402


def make_model(*, size_gb=20.0, layers=48, hidden=4096, vocab=0, mtp=False,
               layer_mb=0.0, tail_mb=0.0, groups=0):
    meta = {
        "block_count": layers,
        "embedding_length": hidden,
        "context_length": 131072,
        "supports_mtp": mtp,
    }
    if vocab:
        meta["vocab_size"] = vocab
    # 张量表实测字节（gguf_parser._walk_tensor_sizes 的产物）；不给就是 0，
    # planner 必须退回 vocab×hidden 的 f16 估算老路。
    # groups 是同一趟走查发布的 blk 组数（0 = 没发布，planner 用层算术）。
    if layer_mb:
        meta["layer_bytes"] = int(layer_mb * 1024 * 1024)
    if tail_mb:
        meta["tail_bytes"] = int(tail_mb * 1024 * 1024)
    if groups:
        meta["blk_groups"] = groups
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
# 更正一句：那条 128 MiB 是草稿层的 f16 buffer，不是整张 KV——引擎按 buffer
# 逐条打印，末条覆盖的读数只留得下最后一条（任务 #21）。主干那条没量过。

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
    # 上限按逐条相加的真实口径放宽：主干 10 层 q4_0 ＋ MTP 草稿层 f16。
    # 原来钉 <400 的依据是引擎自报的 128 MiB，而那只是最后一条 buffer（草稿层），
    # 见 test_draft_layer_kv_is_priced_in_f16 与任务 #21。
    assert p.kv_full_mb < 560, (
        f"真实几何下 KV 应远小于按 MHA 全层算出来的量级，实测相加只有 ~520MB：{p.kv_full_mb}")
    assert "几何精算" in p.reason


def test_draft_layer_kv_is_priced_in_f16_not_the_cache_quant():
    """MTP 草稿层的 KV 不吃 --cache-type：引擎一律按 f16 存它。

    差价钉成 128 MiB = ctx 65536 × 2 头 × (256+256) × 2B，正是 llama.cpp 在
    -lv 4 里为这只模型打印的那条草稿 KV buffer。换成 f16 cache 后差价必须还是
    128（此时主干也按 f16，草稿不再特殊），否则就是重复计价。
    """
    cfg = {**CFG, "ctx_size": 65536}
    on = plan_offload(make_hybrid_model(), cfg, free_vram_mb=15360, free_ram_mb=65536)
    off = plan_offload(make_hybrid_model(supports_mtp=False), cfg,
                       free_vram_mb=15360, free_ram_mb=65536)
    want = 65536 * 2 * (256 + 256) * 2.0 / (1024 * 1024)
    assert on.kv_full_mb - off.kv_full_mb == want, (on.kv_full_mb, off.kv_full_mb)

    f16 = {**cfg, "cache_type_k": "f16", "cache_type_v": "f16"}
    on_f = plan_offload(make_hybrid_model(), f16, free_vram_mb=15360, free_ram_mb=65536)
    off_f = plan_offload(make_hybrid_model(supports_mtp=False), f16,
                         free_vram_mb=15360, free_ram_mb=65536)
    assert on_f.kv_full_mb - off_f.kv_full_mb == want, (on_f.kv_full_mb, off_f.kv_full_mb)


def test_draft_kv_surcharge_follows_mtp_enabled():
    """关掉 MTP 的那一档不该再扣草稿层的 f16 差价——引擎那一趟根本不分配它。

    实测锚：_pydiag/spec_flags_results.csv 的 no_mtp 行逐条只有 KV 284，没有 ＋63
    （生效配置那栏也写着 no-draft）。钉的是绝对值不是差值：只比两侧之差的话，把
    这道门删掉两侧会一起变大，差值照旧，这一条就测不出任何东西。
    """
    cfg = {**CFG, "ctx_size": 65536}
    on = plan_offload(make_hybrid_model(), cfg, free_vram_mb=15360, free_ram_mb=65536)
    off = plan_offload(make_hybrid_model(), {**cfg, "mtp_enabled": False},
                       free_vram_mb=15360, free_ram_mb=65536)
    cb = op._cache_bytes("q4_0")                 # q4_0 = 0.5625 B/元素，别写死
    elems = 11 * 2 * (256 + 256)                 # 10 个 KV 层 ＋ 1 个草稿层
    trunk = 65536 * elems * cb / (1024 * 1024)
    surcharge = 65536 * 2 * (256 + 256) * (2.0 - cb) / (1024 * 1024)
    assert off.kv_full_mb == trunk, off.kv_full_mb
    assert on.kv_full_mb == trunk + surcharge, on.kv_full_mb
    assert "草稿层按 f16" in on.reason
    assert "草稿层按 f16" not in off.reason


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


# ---------- 张量表实测字节（量化模型的重尾不能再按 f16 猜） ----------
# 复现实测对象：Qwen3.8-27B-NVFP4-MTP-LOW（文件 14815MB，blk 层 13326.7MB，
# token_embd 644.1＋output 833.6＝重尾 1477.8MB，65 层＋1 草稿层）。
# 同一份元数据按 vocab×hidden×2×2 猜出来的重尾是 4872MB——大一倍，
# 于是单层均值被摊成 150MB（真实 202MB），全量门误判，解出 -ngl 65，
# 把 output.weight 留在内存：实测 3.8~48 t/s 档与 52~66 t/s 档的分水岭。

NVFP4 = dict(size_gb=14815 / 1024, layers=65, hidden=5120, vocab=248_320,
             mtp=True, layer_mb=13326.7, tail_mb=1477.8)
COEF = CostCoefficients(vram_safety_fraction=0.9, base_overhead_mb=0.0,
                        fallback_tail_fraction=0.05, max_tail_fraction=0.5,
                        prewarm_ram_margin=1.2, paging_ceiling_fraction=0.9)
FREE = 16303.0
CFG_LOW = {"ctx_size": 2048, "parallel": 1, "cache_type_k": "q4_0",
           "cache_type_v": "q4_0", "n_gpu_layers": -1}


def _nvfp4(**over):
    kw = {**NVFP4, **over}
    return make_model(**kw)


def test_measured_bytes_replace_the_f16_tail_guess():
    p = plan_offload(_nvfp4(), CFG_LOW, free_vram_mb=FREE,
                     free_ram_mb=65536, coefficients=COEF)
    assert abs(p.tail_reserve_mb - 1477.8) < 1.0, p.tail_reserve_mb
    assert abs(p.avg_layer_mb - 13326.7 / 66) < 1.0, p.avg_layer_mb
    assert "张量表实测" in p.reason
    assert p.n_gpu_layers == 999, p.reason


def test_same_model_without_measured_bytes_stays_at_layers_minus_one():
    """差分对照：只删掉实测字节，其余一字不改，解必须退回 -ngl 65。

    若 planner 根本没读 layer_bytes/tail_bytes，这两条会给出同一个计划，
    上面那条的 999 就成了恒真断言。
    """
    m = _nvfp4(layer_mb=0.0, tail_mb=0.0)
    p = plan_offload(m, CFG_LOW, free_vram_mb=FREE, free_ram_mb=65536,
                     coefficients=COEF)
    f16_tail_mb = (2 * 248_320 * 5120 * 2) / (1024 * 1024)
    assert abs(p.tail_reserve_mb - f16_tail_mb) < 2.0, p.tail_reserve_mb
    assert p.n_gpu_layers == p.layers_total - 1 == 65, p.reason


def test_inflated_layer_bytes_denies_the_full_gate():
    """反向对照：把实测块层报大到装不下，999 必须被拦——证明门确实在吃这个数。"""
    p = plan_offload(_nvfp4(layer_mb=14400.0, tail_mb=400.0), CFG_LOW,
                     free_vram_mb=FREE, free_ram_mb=65536, coefficients=COEF)
    assert p.n_gpu_layers != 999, p.reason
    assert p.layers_on_disk > 0
    used = p.n_gpu_layers * (p.avg_layer_mb + p.kv_full_mb / p.layers_total)
    assert used <= p.budget_mb + 1.0, f"{used} > {p.budget_mb}"


def test_impossible_measured_bytes_fall_back_to_estimate():
    """块层＋重尾超过文件本体 → 实测值不可信，退回 f16 估算而不是解出 0 层。"""
    p = plan_offload(_nvfp4(layer_mb=10000.0, tail_mb=9000.0), CFG_LOW,
                     free_vram_mb=FREE, free_ram_mb=65536, coefficients=COEF)
    f16_tail_mb = (2 * 248_320 * 5120 * 2) / (1024 * 1024)
    assert abs(p.tail_reserve_mb - f16_tail_mb) < 2.0, p.reason
    assert "张量表实测" not in p.reason


def test_nested_draft_group_count_replaces_the_mtp_arithmetic():
    """草稿嵌在最后一组时（NVFP4-LOW 实测 blk.64.nextn.*，65 组/block_count=65），
    每层均值必须按文件里真实的 65 组除，而不是按 65+1 的层算术摊。

    与下一条构成差分：只有 blk_groups 这一个字段变化，avg 就必须跟着变；
    否则 planner 根本没读它，这条断言是恒真的。
    """
    p = plan_offload(_nvfp4(groups=65), CFG_LOW, free_vram_mb=FREE,
                     free_ram_mb=65536, coefficients=COEF)
    assert p.layers_total == 65, p.reason
    assert abs(p.avg_layer_mb - 13326.7 / 65) < 1.0, p.avg_layer_mb
    q = plan_offload(_nvfp4(), CFG_LOW, free_vram_mb=FREE,
                     free_ram_mb=65536, coefficients=COEF)
    assert q.layers_total == 66 and abs(q.avg_layer_mb - 13326.7 / 66) < 1.0


def test_absurd_blk_groups_are_discarded():
    """反向对照：组数超出 block_count+1（DB 里的坏/旧缓存）只能被丢掉退回层算术，
    不能拿去当分母——否则 13326.7/999 会把每层估成 13MB，全量门形同虚设。"""
    p = plan_offload(_nvfp4(groups=999), CFG_LOW, free_vram_mb=FREE,
                     free_ram_mb=65536, coefficients=COEF)
    assert p.layers_total == 66, p.reason
    assert abs(p.avg_layer_mb - 13326.7 / 66) < 1.0, p.avg_layer_mb


# ---------- 循环状态（RS）：与 ctx 无关，却按草稿深度整份翻倍 ----------
# 复现对象同上，元数据 2026-10-21 由 lmloader.infra.gguf_parser.parse_gguf 从
# models/esatapedico/.../Qwen3.8-27B-NVFP4-MTP-LOW.gguf 原样读出（不是手填）。
# ENGINE_RS 是同一只文件在 ctx 16000／-ngl 66／parallel 1 下，llama.cpp `-lv 4`
# 逐条自报的 llama_memory_recurrent buffer（_pydiag/spec_flags_no_mtp.log、
# _nmax1/_base/_nmax4.log）：不开草稿 149.62，深度 1/2/4 分别 299.25/448.88/748.12。
# KV 项看不见这笔钱——它一字节都不随 ctx 变，却是这只旗舰卡上 449MB 的常驻，
# 规划器过去把它当 0，32k/40k 全驻留塌到 4~5 t/s 的账就缺在这里。
NVFP4_HYBRID = {
    "architecture": "qwen35", "block_count": 65, "embedding_length": 5120,
    "context_length": 262144, "vocab_size": 496645, "supports_mtp": True,
    "blk_groups": 65, "layer_bytes": int(13326.67 * 1024 * 1024),
    "tail_bytes": int(1477.75 * 1024 * 1024), "attention_head_count": 24,
    "attention_head_count_kv": 4, "attention_key_length": 256,
    "attention_value_length": 256, "full_attention_interval": 4,
    "ssm_conv_kernel": 4, "ssm_state_size": 128, "ssm_group_count": 16,
    "ssm_inner_size": 6144,
}
ENGINE_RS = {0: 149.62, 1: 299.25, 2: 448.88, 4: 748.12}
CFG_16K = {"ctx_size": 16000, "parallel": 1, "cache_type_k": "q4_0",
           "cache_type_v": "q4_0", "n_gpu_layers": -1,
           "mtp_enabled": True, "spec_draft_max": 2}


def _hybrid(meta=None, size_mb=14814.93):
    return SimpleNamespace(name="nvfp4.gguf", path="C:/tmp/nvfp4.gguf",
                           architecture="qwen35", size=int(size_mb * 1024 ** 2),
                           metadata=dict(NVFP4_HYBRID if meta is None else meta),
                           backend=None)


def test_recurrent_state_reproduces_the_engine_printout():
    """四个草稿深度的实测 buffer 一个都不能差：这条是整套 RS 算术的唯一锚点。"""
    for depth, measured in ENGINE_RS.items():
        got = op.recurrent_state_mb(NVFP4_HYBRID, depth, 1)
        assert abs(got - measured) < 0.01, f"深度 {depth}: 推导 {got} vs 实测 {measured}"


def test_recurrent_state_holds_on_a_second_model():
    """另一只混合模型（35B-A3B：41 层、inner_size 4096，几何与旗舰处处不同）
    也必须命中引擎自报的 188.44 MiB。只有旗舰一个样本的话，公式里任何一个
    系数写错都可能被它的数字正好掩盖——_pydiag/cal_nc*.log 里那批扫描量的是
    草稿深度 2，这里推导同档；深度 0 顺手给出 62.81，是同一份账的另一半。
    """
    assert abs(op.recurrent_state_mb(MOE_35B, 2, 1) - 188.44) < 0.01
    assert abs(op.recurrent_state_mb(MOE_35B, 0, 1) - 62.81) < 0.01


def test_recurrent_state_is_the_conv_and_ssm_split_the_engine_named():
    """引擎把状态拆成 R（卷积）＋S（SSM）两条报数：149.62 = 5.62＋144.00。
    只对齐总数而内部比例错了，换一个草稿深度就会散架，所以两项分别钉。"""
    m = dict(NVFP4_HYBRID)
    m["ssm_conv_kernel"] = 1          # 没有卷积状态，只剩 SSM
    only_ssm = op.recurrent_state_mb(m, 0, 1)
    assert abs(only_ssm - 144.00) < 0.01, only_ssm
    m = dict(NVFP4_HYBRID)
    m["ssm_inner_size"] = 0           # 缺字段：宁可不计价，也不猜
    assert op.recurrent_state_mb(m, 0, 1) is None


def test_plan_charges_only_the_state_the_cost_row_cannot_cover():
    """差分：系数行里已经混着这笔状态（overhead_rs_mb）时不重复扣，
    没混着时必须整份额外扣掉——两头都断言，才不是恒真。"""
    tight = replace(COEF, overhead_rs_mb=0.0)
    covered = replace(COEF, overhead_rs_mb=ENGINE_RS[2])
    p = plan_offload(_hybrid(), CFG_16K, 14000.0, 65536.0, coefficients=tight)
    q = plan_offload(_hybrid(), CFG_16K, 14000.0, 65536.0, coefficients=covered)
    assert abs(p.rs_full_mb - ENGINE_RS[2]) < 0.05, p.rs_full_mb
    assert abs(p.rs_charged_mb - ENGINE_RS[2]) < 0.05, p.rs_charged_mb
    assert q.rs_full_mb == pytest.approx(ENGINE_RS[2], abs=.05)
    assert q.rs_charged_mb == 0.0, q.reason
    assert abs(p.budget_mb - (q.budget_mb - ENGINE_RS[2])) < 0.05, (p.budget_mb, q.budget_mb)
    assert p.n_gpu_layers < q.n_gpu_layers, (p.reason, q.reason)


def test_draft_depth_moves_the_plan():
    """草稿深度 4 的状态比深度 0 多 598MB：同一只模型、同一档显存，
    解出的 -ngl 必须更低，否则 planner 压根没读 spec_draft_max。"""
    deep = plan_offload(_hybrid(), {**CFG_16K, "spec_draft_max": 4},
                        14000.0, 65536.0, coefficients=COEF)
    # 深度 0 在 run config 里代表"没填"（clamp_run_config 会抬成 1），所以差分
    # 用 1 而不是 0；"真的关掉 MTP"另有 test_mtp_off_does_not_charge_draft_positions。
    flat = plan_offload(_hybrid(), {**CFG_16K, "spec_draft_max": 1},
                        14000.0, 65536.0, coefficients=COEF)
    assert abs(deep.rs_full_mb - ENGINE_RS[4]) < 0.05, deep.rs_full_mb
    assert abs(flat.rs_full_mb - ENGINE_RS[1]) < 0.06, flat.rs_full_mb
    assert deep.n_gpu_layers < flat.n_gpu_layers, (deep.reason, flat.reason)


def test_mtp_off_does_not_charge_draft_positions():
    """关掉 MTP 后引擎只分配 1 份状态（实测 no_mtp 行 149.62）：规划必须跟着
    回落到 1 份，不能继续按配置里残留的 spec_draft_max 扣 3 份。"""
    p = plan_offload(_hybrid(), {**CFG_16K, "mtp_enabled": False},
                     14000.0, 65536.0, coefficients=COEF)
    assert abs(p.rs_full_mb - ENGINE_RS[0]) < 0.05, p.rs_full_mb


def test_pure_attention_model_pays_no_state():
    """interval=1 的普通注意力模型没有循环状态：返回 0.0（不是 None），
    预算一分不少。"""
    m = dict(NVFP4_HYBRID)
    m["full_attention_interval"] = 1
    assert op.recurrent_state_mb(m, 2, 1) == 0.0
    p = plan_offload(_hybrid(m), CFG_16K, 14000.0, 65536.0, coefficients=COEF)
    assert p.rs_charged_mb == 0.0, p.reason


def test_unreadable_state_geometry_is_reported_not_invented():
    """混合模型但读不到状态几何：不许拿 embedding_length 凑一个数，
    也不许沉默——日志里必须写"未计价"，事后才查得出这一档少算了什么。"""
    m = {k: v for k, v in NVFP4_HYBRID.items() if not k.startswith("ssm_")}
    assert op.recurrent_state_mb(m, 2, 1) is None
    p = plan_offload(_hybrid(m), CFG_16K, 14000.0, 65536.0, coefficients=COEF)
    assert p.rs_full_mb == 0.0 and p.rs_charged_mb == 0.0
    assert "未计价" in p.reason, p.reason


# 35B-A3B MoE 真 metadata（parse_gguf 直读文件所得，与旗舰处处不同：hidden 2048、
# 只有 2 个 KV 头、41 组、每层 256 选 8 专家）。
# 实测源：_pydiag/cal_nc*.log 九次加载逐条自报的 ctx 32768 账——
#   llama_kv_cache:  CUDA0 KV buffer size = 180.00 MiB   ← 主干（10 个全注意力层）
#   llama_kv_cache:  CUDA0 KV buffer size =  64.00 MiB   ← MTP 草稿上下文，cache_k/v=f16
# 两个数分别对上一项，才算同时钉住"混合模型只按全注意力层计 KV"和"草稿层不受
# --cache-type 影响、一律 f16"；只对总数则两个错能互相抵消。
MOE_35B = {
    "architecture": "qwen35moe", "block_count": 41, "embedding_length": 2048,
    "context_length": 262144, "vocab_size": 496645, "supports_mtp": True,
    "blk_groups": 41, "layer_bytes": 21090886144, "tail_bytes": 1080696832,
    "attention_head_count": 16, "attention_head_count_kv": 2,
    "attention_key_length": 256, "attention_value_length": 256,
    "full_attention_interval": 4, "expert_count": 256, "expert_used": 8,
    "moe_layer_count": 41,
    "ssm_conv_kernel": 4, "ssm_state_size": 128, "ssm_group_count": 16,
    "ssm_inner_size": 4096,
}


def _moe_35b():
    size_mb = 21155.53
    return SimpleNamespace(name="moe.gguf", path="C:/tmp/moe.gguf",
                           architecture="qwen35moe", size=int(size_mb * 1024 ** 2),
                           metadata=dict(MOE_35B), backend=None)


def test_full_kv_matches_both_engine_buffers_on_the_second_model():
    """满额 KV 必须拆得开：主干那 180 只按全注意力层算，草稿那 64 按 f16 算。"""
    cfg = {"ctx_size": 32768, "parallel": 1, "cache_type_k": "q4_0",
           "cache_type_v": "q4_0", "n_gpu_layers": -1,
           "mtp_enabled": True, "spec_draft_max": 2}
    p = plan_offload(_moe_35b(), cfg, 20000.0, 65536.0, coefficients=COEF)
    draft_f16 = 32768 * 2 * (256 + 256) * 2.0 / (1024 ** 2)
    assert draft_f16 == pytest.approx(64.0, abs=0.01), draft_f16
    assert p.kv_full_mb == pytest.approx(180.0 + 64.0, abs=0.6), p.reason
    assert p.kv_full_mb - draft_f16 == pytest.approx(180.0, abs=0.6), p.reason
    assert "几何精算＋草稿层按 f16" in p.reason, p.reason

