"""Tests for engine router — quantization-based routing."""

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
