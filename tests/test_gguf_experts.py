"""GGUF 专家维度解析的回归测试。

这里不依赖任何真实 MoE 文件（用桩张量），因为要守的是命名与归类规则：
只有 routed expert 计入可卸载字节，shared expert 每个 token 都要跑、必须
常驻，算进去会让规划器高估可省下的显存。dense 模型必须全零，否则
is_moe 误判会把非 MoE 模型带进 -ncmoe 分支。

字段别名不是猜的：expert_used_count 与"打包 3D 形状"来自 Qwen3-30B-A3B
GGUF 头部实测——只读头部，半下载的文件也能复核。
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmloader.infra import gguf_parser as gp  # noqa: E402


def make_tensor(name, n_bytes, shape=None):
    return SimpleNamespace(name=name, n_bytes=n_bytes, shape=shape)


def run(tensors, fields, monkeypatch, architecture="qwen3moe"):
    """把 _get_int 换成字典查表，其余逻辑（张量名归类、层计数）走真实代码。"""
    monkeypatch.setattr(gp, "_get_int", lambda reader, key: fields.get(key))
    reader = SimpleNamespace(fields=fields, tensors=tensors)
    return gp._read_expert_shape(reader, architecture)


def test_dense_model_reads_zero(monkeypatch):
    tensors = [make_tensor("blk.0.ffn_gate.weight", 1024),
               make_tensor("token_embd.weight", 4096)]
    shape = run(tensors, {"qwen3moe.block_count": 32}, monkeypatch, "qwen35")
    assert shape == {"expert_count": 0, "expert_used": 0,
                     "moe_layer_count": 0, "expert_bytes": 0}


def test_routed_experts_summed_and_layers_counted(monkeypatch):
    tensors = [
        make_tensor("blk.0.ffn_gate_exps.weight", 100),
        make_tensor("blk.0.ffn_up_exps.weight", 100),
        make_tensor("blk.0.ffn_down_exps.weight", 100),
        make_tensor("blk.1.ffn_gate_exps.weight", 100),
        make_tensor("blk.0.ffn_gate.weight", 999),   # 非专家，不计
    ]
    fields = {"qwen3moe.expert_count": 128, "qwen3moe.expert_used": 8,
              "qwen3moe.moe_layer_count": 48}
    shape = run(tensors, fields, monkeypatch)
    assert shape["expert_count"] == 128
    assert shape["expert_used"] == 8
    # 元数据声明 48 层，桩张量只覆盖 2 层：以元数据为准。
    assert shape["moe_layer_count"] == 48
    assert shape["expert_bytes"] == 400


def test_shared_experts_excluded(monkeypatch):
    tensors = [
        make_tensor("blk.0.ffn_gate_exps.weight", 100),
        make_tensor("blk.0.ffn_gate_shexp.weight", 5000),
        make_tensor("blk.0.ffn_up_shared_experts.weight", 5000),
    ]
    shape = run(tensors, {"qwen3moe.expert_count": 128}, monkeypatch)
    assert shape["expert_bytes"] == 100


def test_gpt_oss_expw_naming_recognized(monkeypatch):
    tensors = [make_tensor("blk.3.ffn_expw0.gate_up", 256),
               make_tensor("blk.3.ffn_expw0.down", 256)]
    shape = run(tensors, {"gptoss.expert_count": 32}, monkeypatch, "gptoss")
    assert shape["expert_bytes"] == 512
    assert shape["moe_layer_count"] == 1


def test_layer_count_falls_back_to_census(monkeypatch):
    tensors = [make_tensor(f"blk.{i}.ffn_gate_exps.weight", 10) for i in range(20)]
    shape = run(tensors, {"qwen3moe.expert_count": 128}, monkeypatch)
    assert shape["moe_layer_count"] == 20


def test_marker_found_under_other_arch_prefix(monkeypatch):
    # general.architecture 与实际键前缀不一致时仍要认出来。
    fields = {"deepseek2.expert_count": 256}
    shape = run([make_tensor("blk.0.ffn_gate_exps.weight", 10)], fields,
                monkeypatch, "qwen3moe")
    assert shape["expert_count"] == 256


def test_expert_used_count_naming(monkeypatch):
    # Qwen3-30B-A3B-Q4_K_M.gguf 实际写的是 expert_used_count；只认
    # expert_used 会把激活专家数读成 0，规划器就没有每 token 成本可算。
    tensors = [make_tensor("blk.0.ffn_gate_exps.weight", 100)]
    shape = run(tensors, {"qwen3moe.expert_count": 128,
                          "qwen3moe.expert_used_count": 8}, monkeypatch)
    assert shape["expert_used"] == 8


def test_deepseek_n_routed_experts_naming(monkeypatch):
    tensors = [make_tensor("blk.0.ffn_gate_exps.weight", 100)]
    shape = run(tensors, {"deepseek2.n_routed_experts": 256,
                          "deepseek2.num_experts_per_tok": 8}, monkeypatch,
                "deepseek2")
    assert shape["expert_count"] == 256
    assert shape["expert_used"] == 8


def test_expert_count_from_packed_shape_without_metadata(monkeypatch):
    # llama.cpp 把一层的全部专家打包进一个 3D 张量，最外层就是专家数。
    # 转换器漏写 expert_count 时，这个数仍然算得出来。
    tensors = [make_tensor("blk.0.ffn_down_exps.weight", 100,
                           shape=(128, 768, 2048))]
    shape = run(tensors, {}, monkeypatch)
    assert shape["expert_count"] == 128


def test_dense_tensor_shapes_never_invent_experts(monkeypatch):
    # 反向对照：普通 2D 权重绝不能被打包形状的推断路径误认成专家。
    tensors = [make_tensor("blk.0.ffn_gate.weight", 100, shape=(2048, 768)),
               make_tensor("blk.0.attn_q.weight", 100, shape=(2048, 2048))]
    shape = run(tensors, {}, monkeypatch)
    assert shape == {"expert_count": 0, "expert_used": 0,
                     "moe_layer_count": 0, "expert_bytes": 0}


def test_is_moe_requires_both_count_and_bytes(monkeypatch):
    meta_only = gp.GGUFMetadata(
        architecture="qwen3moe", quantization="Q4_K_M", parameters="30B",
        tensor_count=1, kv_count=1, context_length=1, embedding_length=1,
        feed_forward_length=1, attention_head_count=1, block_count=1,
        supports_mtp=False, raw_metadata={}, expert_count=128, expert_bytes=0)
    assert meta_only.is_moe is False
    complete = gp.GGUFMetadata(
        architecture="qwen3moe", quantization="Q4_K_M", parameters="30B",
        tensor_count=1, kv_count=1, context_length=1, embedding_length=1,
        feed_forward_length=1, attention_head_count=1, block_count=1,
        supports_mtp=False, raw_metadata={}, expert_count=128, expert_bytes=10)
    assert complete.is_moe is True
