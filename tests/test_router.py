"""Tests for engine router — quantization-based routing."""

import struct

import pytest
from unittest.mock import MagicMock

from lmloader.engine.router import EngineRouter, TERNARY_QUANT_TYPES


@pytest.fixture
def router():
    return EngineRouter()


def _make_model(quantization: str, name: str = "test-model.gguf"):
    m = MagicMock()
    m.name = name
    m.quantization = quantization
    return m


@pytest.mark.parametrize("quant", ["PQ2_0_TERNARY", "TL1_TERNARY", "TL2_TERNARY", "TQ1_0", "TQ2_0"])
def test_ternary_models_route_to_ternary(router, quant):
    model = _make_model(quant)
    assert router.select_engine_slot(model) == "ternary"


@pytest.mark.parametrize("quant", ["Q4_K_M", "Q8_0", "F16", "Q5_K_S"])
def test_standard_models_route_to_primary(router, quant):
    model = _make_model(quant)
    assert router.select_engine_slot(model) == "primary"


def test_ternary_quant_types_set_is_complete():
    expected = {"PQ2_0_TERNARY", "TL1_TERNARY", "TL2_TERNARY", "TQ1_0", "TQ2_0", "I2_S", "TL1", "TL2"}
    assert TERNARY_QUANT_TYPES == expected


def _make_record(name, architecture="llama", metadata=None, path=None):
    m = MagicMock()
    m.name = name
    m.architecture = architecture
    m.metadata = metadata or {}
    m.path = path or f"D:/models/{name}"
    return m


@pytest.mark.parametrize("name,arch,expect", [
    # architecture is an embedding-only family
    ("nomic-embed-text-v1.5.Q4_K_M.gguf", "nomic-bert", True),
    ("bge-m3-Q4_K_M.gguf", "bge-m3", True),
    ("jina-embeddings-v2-base.gguf", "jina", True),
    ("e5-mistral-7b.gguf", "e5", True),
    # chat architecture but name advertises embeddings
    ("gte-Qwen2-1.5b-embed.gguf", "qwen2", True),
    ("text-embedding-ada.gguf", "", True),
    # plain chat models must NOT get the flag (this was the regression)
    ("qwen35-9b-mtp-nvfp4.gguf", "qwen2", False),
    ("Ternary-Bonsai-2-27B.gguf", "llama", False),
    ("Gemma-4-12B.gguf", "gemma", False),
    # a vision encoder is not an embedding model
    ("mmproj-Qwen2-VL.gguf", "clip", False),
])
def test_is_embedding_model_detection(router, name, arch, expect):
    model = _make_record(name, arch)
    assert router.is_embedding_model(model) is expect


def test_embeddings_flag_only_on_embedding_models(router, monkeypatch):
    # Skip engine discovery; route everything to the primary llama slot.
    monkeypatch.setattr(router, "resolve_engine", lambda m: ("primary", "llama"))

    chat = _make_record("qwen35-9b-mtp-nvfp4.gguf", "qwen2")
    assert "--embeddings" not in router.build_launch_config(chat, 8100).args

    embed = _make_record("nomic-embed-text.gguf", "nomic-bert")
    assert "--embeddings" in router.build_launch_config(embed, 8100).args


@pytest.mark.parametrize("name,expect", [
    ("Ternary-Bonsai-2-27B.gguf", True),
    ("Binary-Bonsai-2-27B-Q4_0.gguf", True),
    ("qwen35-9b-mtp-nvfp4.gguf", False),
    ("Gemma-4-12B.gguf", False),
])
def test_dry_sensitive_model_detection(router, name, expect):
    model = _make_record(name, "llama")
    assert router._is_dry_sensitive_model(model) is expect


@pytest.mark.parametrize("name", [
    "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf",
    "Binary-Bonsai-2-27B-Q4_0.gguf",
])
def test_bonsai_skips_repetition_suppression(router, monkeypatch, name):
    monkeypatch.setattr(router, "resolve_engine", lambda m: ("primary", "llama"))
    bonsai = _make_record(name, "llama")
    args = router.build_launch_config(bonsai, 8100).args
    assert "--repeat-penalty" not in args
    assert "--dry-multiplier" not in args
    assert "--dry-base" not in args


def test_non_bonsai_keeps_repetition_suppression(router, monkeypatch):
    monkeypatch.setattr(router, "resolve_engine", lambda m: ("primary", "llama"))
    chat = _make_record("qwen35-9b-mtp-nvfp4.gguf", "qwen2")
    args = router.build_launch_config(chat, 8100).args
    assert "--repeat-penalty" in args
    assert "--dry-multiplier" in args
    rp_idx = args.index("--repeat-penalty")
    assert args[rp_idx + 1] == "1.1"
    dry_idx = args.index("--dry-multiplier")
    assert args[dry_idx + 1] == "0.8"


# ------------------------------------------------------------- vLLM routing

import sys as _sys

_ROUTER_MOD = _sys.modules["lmloader.engine.router"]
_VLLM_ENGINE = {"name": "vllm@vllm", "kind": "vllm", "formats": [".safetensors"]}


def _make_hf_model(name="hf-model", path="D:/models/hf-model"):
    m = MagicMock()
    m.name = name
    m.path = path
    m.metadata = {"hf_model": True}
    m.backend = None
    return m


def test_hf_dir_resolves_to_vllm(router, monkeypatch):
    monkeypatch.setattr(_ROUTER_MOD, "find_engine_by_kind",
                        lambda k, discovered=None: _VLLM_ENGINE)
    slot, kind = router.resolve_engine(_make_hf_model())
    assert slot == "vllm@vllm" and kind == "vllm"


def test_hf_dir_without_vllm_engine_returns_none(router, monkeypatch):
    monkeypatch.setattr(_ROUTER_MOD, "find_engine_by_kind",
                        lambda k, discovered=None: None)
    slot, kind = router.resolve_engine(_make_hf_model())
    assert slot is None and kind == ""


def test_safetensors_resolves_to_vllm(router, monkeypatch):
    monkeypatch.setattr(_ROUTER_MOD, "find_engine_for_format",
                        lambda ext, discovered=None: _VLLM_ENGINE)
    m = MagicMock()
    m.name = "model.safetensors"
    m.path = "D:/models/model.safetensors"
    m.metadata = {}
    m.backend = None
    slot, kind = router.resolve_engine(m)
    assert slot == "vllm@vllm" and kind == "vllm"


def test_vllm_launch_args_handle_spaces(router, monkeypatch):
    monkeypatch.setattr(_ROUTER_MOD, "find_engine_by_kind",
                        lambda k, discovered=None: _VLLM_ENGINE)
    lc = router.build_launch_config(
        _make_hf_model(path="D:/my models/hf"), 8123)
    assert lc.kind == "vllm"
    # Paths with spaces are passed as a single token; the served name is the
    # DB name (so the GUI's model field matches) with the path as an alias;
    # the default VRAM cap is appended so vLLM does not reserve ~92% of the GPU.
    assert lc.args == [
        "serve", "D:/my models/hf",
        "--host", "127.0.0.1", "--port", "8123",
        "--served-model-name", "hf-model",
        "--gpu-memory-utilization", "0.85",
    ]


def test_vllm_served_model_name_matches_db_name(router, monkeypatch):
    # The GUI/proxy send the model's registered NAME; vLLM must serve under
    # that name or it 404s and the chat shows an empty reply.
    monkeypatch.setattr(_ROUTER_MOD, "find_engine_by_kind",
                        lambda k, discovered=None: _VLLM_ENGINE)
    lc = router.build_launch_config(
        _make_hf_model(name="Qwen2.5-0.5B-Instruct",
                      path="D:/models/Qwen2.5-0.5B-Instruct"), 8100)
    i = lc.args.index("--served-model-name")
    assert lc.args[i + 1] == "Qwen2.5-0.5B-Instruct"
    # The positional model is still the directory path.
    assert lc.args[1] == "D:/models/Qwen2.5-0.5B-Instruct"


def test_vllm_launch_args_respect_custom_memory_util(router, monkeypatch):
    monkeypatch.setattr(_ROUTER_MOD, "find_engine_by_kind",
                        lambda k, discovered=None: _VLLM_ENGINE)
    _ROUTER_MOD.settings.vllm.gpu_memory_utilization = 0.6
    _ROUTER_MOD.settings.vllm.max_model_len = 4096
    try:
        lc = router.build_launch_config(_make_hf_model(), 8123)
    finally:
        _ROUTER_MOD.settings.vllm.gpu_memory_utilization = 0.85
        _ROUTER_MOD.settings.vllm.max_model_len = None
    assert lc.args == [
        "serve", "D:/models/hf-model",
        "--host", "127.0.0.1", "--port", "8123",
        "--served-model-name", "hf-model",
        "--gpu-memory-utilization", "0.6",
        "--max-model-len", "4096",
    ]


def test_gguf_never_routes_to_vllm(router):
    # A GGUF record with the hf_model marker still routes by quant to the
    # built-in engines, so vLLM can never hijack GGUF models.
    m = _make_model("Q4_K_M", "m.gguf")
    m.path = "D:/models/m.gguf"
    m.metadata = {"hf_model": True}
    m.backend = None
    slot, kind = router.resolve_engine(m)
    assert slot == "primary" and kind == "llama"


def _d3_pkg(tmp_path, *, vision=True, video=True, index=True, tag="d3-pkg"):
    """A minimal directory that passes is_d3_package, with switchable modalities."""
    import json
    from pathlib import Path

    pkg = Path(tmp_path) / tag
    pkg.mkdir(exist_ok=True)
    (pkg / "d3_server.py").write_text("#", encoding="utf-8")
    (pkg / "MODEL_MANIFEST.json").write_text(
        json.dumps({"schema": "d3-package-manifest/1"}), encoding="utf-8"
    )
    (pkg / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5" if video else "qwen3_vl"}), encoding="utf-8"
    )
    if video:
        (pkg / "video_preprocessor_config.json").write_text("{}", encoding="utf-8")
    names = ["readout.weight"] + (["visual.blocks.0.attn.proj.weight"] if vision else [])
    if index:
        (pkg / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {n: "model.safetensors" for n in names}}), encoding="utf-8"
        )
    else:
        # No index: the router must read the tensor names out of the
        # safetensors header itself (8-byte length, then JSON).
        header = json.dumps({n: {} for n in names}).encode("utf-8")
        (pkg / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    m = _make_model("", "d3-pkg")
    m.path = str(pkg)
    m.metadata = {}
    return m


def test_d3_modalities_from_manifest_package(router, tmp_path):
    assert router.is_d3_package(_d3_pkg(tmp_path, tag="full"))
    assert router.d3_modalities(_d3_pkg(tmp_path, tag="full")) == ["text", "image", "video"]
    assert router.d3_modalities(_d3_pkg(tmp_path, tag="img", video=False)) == ["text", "image"]
    assert router.d3_modalities(_d3_pkg(tmp_path, tag="txt", vision=False)) == ["text"]


def test_d3_modalities_without_shard_index(router, tmp_path):
    model = _d3_pkg(tmp_path, index=False)
    assert router.d3_modalities(model) == ["text", "image", "video"]


def _auto_tier_plan(cpu_moe_layers):
    from lmloader.engine.offload_planner import OffloadPlan
    return OffloadPlan(
        n_gpu_layers=999, layers_total=48, free_vram_mb=15360, budget_mb=12288,
        tail_reserve_mb=0, kv_full_mb=900, avg_layer_mb=349,
        layers_on_disk=cpu_moe_layers, prewarm_suggested=False,
        prewarm_mb=17697, reason="测试用规划", cpu_moe_layers=cpu_moe_layers)


@pytest.mark.parametrize("cpu_moe,expect", [(20, True), (0, False)])
def test_auto_tier_wires_the_cpu_moe_flag(router, monkeypatch, cpu_moe, expect):
    """规划结果必须真的变成启动参数：-ncmoe 只在 MoE 分支给层数时才出现。"""
    from lmloader.engine import offload_planner
    monkeypatch.setattr(router, "resolve_engine", lambda m: ("primary", "llama"))
    monkeypatch.setattr(offload_planner, "plan_for_model",
                        lambda m, c: _auto_tier_plan(cpu_moe))
    model = _make_record("Qwen3-30B-A3B-Q4_K_M.gguf", "qwen3moe")
    args = router.build_launch_config(
        model, 8177, run_config={"n_gpu_layers": -1}).args
    assert ("--n-cpu-moe" in args) is expect
    if expect:
        assert args[args.index("--n-cpu-moe") + 1] == "20"


@pytest.mark.parametrize("cache_ram,expect", [
    (None, "1024"),      # 槽默认必须真的落到启动参数里，不能只在配置里躺着
    (512, "512"),
    (0, "0"),            # 0 = 关掉 prompt cache，仍要传下去
    (-1, None),          # 负数 = 不传，保持引擎自己的 8192
])
def test_cache_ram_ceiling_reaches_the_engine(router, monkeypatch, cache_ram, expect):
    """--cache-ram 决定整卡常驻时还能不能塞进第 6 条对话分支。

    2026-10-21 实测：引擎默认 8192 的缓存堆到 6 条分支 4754 MiB，解码从
    51-70 t/s 塌到 2.4 t/s，所以这个上限必须由 掬水 显式下发。
    """
    from lmloader.config import EngineDefaultsConfig
    monkeypatch.setattr(router, "resolve_engine", lambda m: ("primary", "llama"))
    model = _make_record("qwen35-27b-mtp-nvfp4.gguf", "qwen35")
    rc = {} if cache_ram is None else {"cache_ram_mb": cache_ram}
    args = router.build_launch_config(model, 8178, run_config=rc).args
    if expect is None:
        assert "--cache-ram" not in args
        return
    assert "--cache-ram" in args
    assert args[args.index("--cache-ram") + 1] == expect
    if cache_ram is None:
        # 默认值本身就是判据的一部分：改小改大都要在这里被看见
        assert EngineDefaultsConfig().cache_ram_mb == 1024
