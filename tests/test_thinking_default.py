"""Tests for forcing thinking OFF on models with unreliable reasoning blocks."""

from lmloader.api.routes.chat import apply_thinking_default


BONSAI = "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"
OTHER = "qwen35-9b-mtp-nvfp4.gguf"


def test_bonsai_gets_thinking_disabled():
    body = {"model": BONSAI, "messages": [{"role": "user", "content": "hi"}]}
    out = apply_thinking_default(BONSAI, body)
    assert out["chat_template_kwargs"]["enable_thinking"] is False


def test_non_bonsai_untouched():
    body = {"model": OTHER, "messages": [{"role": "user", "content": "hi"}]}
    out = apply_thinking_default(OTHER, body)
    assert "chat_template_kwargs" not in out


def test_explicit_caller_choice_wins():
    body = {
        "model": BONSAI,
        "messages": [],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    out = apply_thinking_default(BONSAI, body)
    assert out["chat_template_kwargs"]["enable_thinking"] is True


def test_other_template_kwargs_preserved():
    body = {
        "model": BONSAI,
        "messages": [],
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    out = apply_thinking_default(BONSAI, body)
    kwargs = out["chat_template_kwargs"]
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["enable_thinking"] is False
