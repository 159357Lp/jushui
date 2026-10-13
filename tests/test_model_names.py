"""Tests for unified model name resolution."""

import pytest
from lmloader.infra.model_names import normalize_spec, spec_to_repo_tag, spec_to_registration_name


def test_normalize_spec_with_tag():
    assert normalize_spec("Qwen/Qwen3-4B-GGUF:Q4_K_M") == "Qwen/Qwen3-4B-GGUF:Q4_K_M"


def test_normalize_spec_without_tag():
    assert normalize_spec("Qwen/Qwen3-4B-GGUF") == "Qwen/Qwen3-4B-GGUF:latest"


def test_normalize_spec_filename():
    assert normalize_spec("model.gguf") == "model.gguf"


def test_spec_to_repo_tag_with_tag():
    repo, tag = spec_to_repo_tag("Qwen/Qwen3-4B-GGUF:Q4_K_M")
    assert repo == "Qwen/Qwen3-4B-GGUF"
    assert tag == "Q4_K_M"


def test_spec_to_repo_tag_without_tag():
    repo, tag = spec_to_repo_tag("Qwen/Qwen3-4B-GGUF")
    assert repo == "Qwen/Qwen3-4B-GGUF"
    assert tag == "latest"


def test_spec_to_registration_name_with_spec():
    name = spec_to_registration_name("Qwen/Qwen3-4B-GGUF:Q4_K_M", "some-file.gguf")
    assert name == "Qwen/Qwen3-4B-GGUF:Q4_K_M"


def test_spec_to_registration_name_with_filename():
    name = spec_to_registration_name("model.gguf", "model.gguf")
    assert name == "model.gguf"
