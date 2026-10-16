"""Per-engine subprocess environment (lifecycle._build_env).

The built-in binary/ternary llama engines must receive an untouched
environment, while a vLLM launch gets the Windows workarounds it needs
(no FlashInfer MSVC JIT at startup, HF mirror).
"""

import os

from lmloader.engine.lifecycle import LifecycleManager
from lmloader.engine.router import LaunchConfig


def _vllm_cfg():
    return LaunchConfig(
        model_path="D:/models/hf", port=8150,
        args=["serve", "D:/models/hf", "--host", "127.0.0.1", "--port", "8150"],
        engine_slot="vllm@vllm", kind="vllm",
    )


def _llama_cfg():
    return LaunchConfig(
        model_path="D:/models/m.gguf", port=8100,
        args=["-m", "D:/models/m.gguf", "--port", "8100"],
        engine_slot="primary", kind="llama",
    )


def test_vllm_env_disables_flashinfer_sampler(monkeypatch):
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    env = LifecycleManager()._build_env(_vllm_cfg())
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    # The configured mirror is passed through for any HF Hub lookups.
    assert env.get("HF_ENDPOINT")


def test_llama_env_is_untouched_by_vllm_workarounds(monkeypatch):
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    env = LifecycleManager()._build_env(_llama_cfg())
    assert "VLLM_USE_FLASHINFER_SAMPLER" not in env
    # The shared llama engine env still applies, unchanged.
    assert env["BONSAI_THINKING"] == "0"


def test_env_without_launch_config_keeps_legacy_behaviour(monkeypatch):
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    env = LifecycleManager()._build_env()
    assert "VLLM_USE_FLASHINFER_SAMPLER" not in env
    assert env["BONSAI_THINKING"] == "0"


def test_user_override_of_flashinfer_sampler_is_respected(monkeypatch):
    # An operator who actually wants the FlashInfer sampler (and has an MSVC
    # toolchain) must not be forced off it: setdefault never overrides.
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    env = LifecycleManager()._build_env(_vllm_cfg())
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "1"
