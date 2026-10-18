"""GGUF KV 缓存几何解析的回归测试。

守卫的是显存预算的输入：KV 按每 token 多少个元素算，取决于 GQA 的
head_count_kv、key/value 长度，以及 hybrid 模型到底有几层真的存 KV。
这些键的名字和 arch 前缀都随转换器变（实测过专家字段如此），所以
读不到精确键时必须走别名扫描，而不是当模型没有这个属性。

真实锚点（不是编的）：Qwen3.6-35B-A3B-MXFP4_MOE 头部
qwen35moe.attention.head_count_kv=2、key_length=256、value_length=256、
full_attention_interval=4、block_count=41；llama.cpp 在 -lv 4 里对
ctx=65536 报 CUDA0 KV buffer = 128 MiB。
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmloader.infra import gguf_parser as gp  # noqa: E402


def run_kv(fields, architecture="qwen35moe", hidden=2048, heads=16, monkeypatch=None):
    """只把 _get_int 换成字典查表，别名扫描与回退逻辑走真实代码。"""
    assert monkeypatch is not None, "run_kv 需要 monkeypatch 才能替换 _get_int"
    monkeypatch.setattr(gp, "_get_int", lambda reader, key: fields.get(key))
    reader = SimpleNamespace(fields=fields)
    return gp._read_kv_shape(reader, architecture, hidden, heads)


def test_hybrid_gqa_geometry_read_from_real_header_keys(monkeypatch):
    fields = {
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.value_length": 256,
        "qwen35moe.full_attention_interval": 4,
    }
    shape = run_kv(fields, monkeypatch=monkeypatch)
    assert shape == {
        "attention_head_count_kv": 2, "attention_key_length": 256,
        "attention_value_length": 256, "full_attention_interval": 4,
    }


def test_missing_interval_is_reported_as_unknown_not_one(monkeypatch):
    # 0 交给规划器解释为"每层都有 KV"；解析层不猜。
    shape = run_kv({"qwen35moe.attention.head_count_kv": 8,
                    "qwen35moe.attention.key_length": 128}, monkeypatch=monkeypatch)
    assert shape["full_attention_interval"] == 0


def test_mha_model_without_kv_head_key_falls_back_to_head_count(monkeypatch):
    shape = run_kv({"qwen35moe.attention.key_length": 128,
                    "qwen35moe.attention.value_length": 128},
                   heads=32, monkeypatch=monkeypatch)
    assert shape["attention_head_count_kv"] == 32


def test_head_dim_derived_when_key_length_absent(monkeypatch):
    shape = run_kv({"qwen35moe.attention.head_count_kv": 4}, hidden=4096,
                   heads=32, monkeypatch=monkeypatch)
    assert shape["attention_key_length"] == 128
    assert shape["attention_value_length"] == 128


def test_kv_keys_found_under_other_arch_prefix(monkeypatch):
    # general.architecture 与实际键前缀不一致时仍要认出来（专家字段实测踩过）。
    fields = {"llama.attention.head_count_kv": 2,
              "llama.attention.key_length": 256,
              "gemma2.full_attention_interval": 6}
    shape = run_kv(fields, architecture="qwen35moe", monkeypatch=monkeypatch)
    assert shape["attention_head_count_kv"] == 2
    assert shape["attention_key_length"] == 256
    assert shape["full_attention_interval"] == 6


def test_value_length_defaults_to_key_length(monkeypatch):
    shape = run_kv({"qwen35moe.attention.head_count_kv": 2,
                    "qwen35moe.attention.key_length": 256}, monkeypatch=monkeypatch)
    assert shape["attention_value_length"] == 256
