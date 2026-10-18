"""按卡代价表的回归测试。

守卫两件事：一是查表键的拼法必须与来源实现（NInfer
context_cost_hardware_class）一致，否则实测行永远命中不了；二是表必须
真的接进制预算的数学——所以有一项反向对照：同一模型同一测量，换一组系数
要能算出不同的 -ngl。只会变绿的探针等于没测。
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmloader.engine import hardware_costs as hc  # noqa: E402
from lmloader.engine.offload_planner import plan_offload  # noqa: E402


CFG = {"ctx_size": 8192, "parallel": 1, "cache_type_k": "q4_0",
       "n_gpu_layers": -1, "cache_type_v": "q4_0"}


def make_model(*, size_gb=20.0, layers=48, hidden=4096, vocab=150000):
    return SimpleNamespace(
        name="t.gguf", path="C:/tmp/t.gguf", architecture="qwen35",
        size=int(size_gb * 1024 ** 3),
        metadata={"block_count": layers, "embedding_length": hidden,
                  "context_length": 131072, "supports_mtp": False,
                  "vocab_size": vocab},
        backend=None,
    )


# ---------- 查表键 ----------

def test_hardware_class_matches_upstream_slug_rule():
    got = hc.hardware_class("NVIDIA GeForce RTX 5080 Laptop GPU", "12.0")
    assert got == "nvidia-geforce-rtx-5080-laptop-gpu-sm120"


def test_hardware_class_keeps_existing_nvidia_prefix():
    assert hc.hardware_class("NVIDIA A100", "8.0") == "nvidia-a100-sm80"


def test_hardware_class_rejects_unusable_input():
    assert hc.hardware_class("", "12.0") is None
    assert hc.hardware_class("RTX 5080", "") is None
    assert hc.hardware_class("RTX 5080", "unknown") is None


# ---------- 兜底与命中 ----------

def test_missing_table_falls_back_to_generic(tmp_path):
    coeffs = hc.resolve_coefficients("nvidia-no-such-card-sm99",
                                     path=str(tmp_path / "absent.json"))
    assert coeffs.source == "通用兜底（未收录 nvidia-no-such-card-sm99）"
    assert coeffs.vram_safety_fraction == hc.GENERIC_COEFFICIENTS["vram_safety_fraction"]


def test_measured_row_wins_over_generic(tmp_path):
    table = tmp_path / "costs.json"
    hc.save_measurement("nvidia-geforce-rtx-5080-laptop-gpu-sm120",
                        {**hc.GENERIC_COEFFICIENTS, "vram_safety_fraction": 0.70,
                         "base_overhead_mb": 1400.0},
                        {"method": "probe", "samples": 5},
                        path=str(table))
    coeffs = hc.resolve_coefficients(
        "nvidia-geforce-rtx-5080-laptop-gpu-sm120", path=str(table))
    assert coeffs.vram_safety_fraction == 0.70
    assert coeffs.base_overhead_mb == 1400.0
    assert "实测表" in coeffs.source


def test_save_measurement_preserves_other_machines(tmp_path):
    table = tmp_path / "costs.json"
    hc.save_measurement("nvidia-a-sm120", dict(hc.GENERIC_COEFFICIENTS), {},
                        path=str(table))
    hc.save_measurement("nvidia-b-sm120",
                        {**hc.GENERIC_COEFFICIENTS, "base_overhead_mb": 1.0}, {},
                        path=str(table))
    hc.save_measurement("nvidia-a-sm120",
                        {**hc.GENERIC_COEFFICIENTS, "base_overhead_mb": 2.0}, {},
                        path=str(table))
    loaded = hc.load_table(str(table))
    assert set(loaded) == {"nvidia-a-sm120", "nvidia-b-sm120"}
    assert loaded["nvidia-a-sm120"]["base_overhead_mb"] == 2.0


# ---------- 表本身的合法性 ----------

def test_corrupt_table_raises(tmp_path):
    bad = tmp_path / "costs.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(hc.HardwareCostError):
        hc.load_table(str(bad))


def test_duplicate_hardware_class_raises(tmp_path):
    bad = tmp_path / "costs.json"
    row = {"hardware_class": "nvidia-x-sm120",
           "coefficients": dict(hc.GENERIC_COEFFICIENTS)}
    bad.write_text(json.dumps({"version": 1, "machines": [row, row]}), encoding="utf-8")
    with pytest.raises(hc.HardwareCostError, match="重复"):
        hc.load_table(str(bad))


def test_missing_or_unknown_coefficient_raises(tmp_path):
    bad = tmp_path / "costs.json"
    for name, coefficients in (
        ("缺项", {k: v for k, v in hc.GENERIC_COEFFICIENTS.items()
                  if k != "base_overhead_mb"}),
        ("多项", {**hc.GENERIC_COEFFICIENTS, "made_up": 1.0}),
        ("比例倒挂", {**hc.GENERIC_COEFFICIENTS, "fallback_tail_fraction": 0.9,
                     "max_tail_fraction": 0.5}),
    ):
        bad.write_text(json.dumps({"version": 1, "machines": [
            {"hardware_class": "nvidia-x-sm120", "coefficients": coefficients}]}),
            encoding="utf-8")
        try:
            hc.load_table(str(bad))
        except hc.HardwareCostError:
            continue
        pytest.fail(f"{name} 的表没有被拒绝")


def test_generic_key_is_reserved(tmp_path):
    with pytest.raises(hc.HardwareCostError):
        hc.save_measurement("generic", dict(hc.GENERIC_COEFFICIENTS), {},
                            path=str(tmp_path / "costs.json"))


# ---------- 反向对照：表必须真的改变决策 ----------

def test_tighter_coefficients_actually_lower_ngl():
    """同一模型、同一空闲显存：把安全系数调紧、固定开销调大，解出的 -ngl
    必须变小。若不变，说明系数根本没进预算数学，整张表是摆设。"""
    model = make_model()
    loose = hc.CostCoefficients(vram_safety_fraction=0.95, base_overhead_mb=100.0,
                                fallback_tail_fraction=0.12, max_tail_fraction=0.5,
                                prewarm_ram_margin=1.25)
    tight = hc.CostCoefficients(vram_safety_fraction=0.60, base_overhead_mb=4000.0,
                                fallback_tail_fraction=0.12, max_tail_fraction=0.5,
                                prewarm_ram_margin=1.25)
    plan_loose = plan_offload(model, CFG, 12000.0, 65536.0, coefficients=loose)
    plan_tight = plan_offload(model, CFG, 12000.0, 65536.0, coefficients=tight)
    assert plan_loose.n_gpu_layers > plan_tight.n_gpu_layers
    assert "系数来源" in plan_loose.reason


# ---------- 表的位置不能跟着工作目录跑 ----------

def test_measured_row_found_when_cwd_is_elsewhere(tmp_path, monkeypatch):
    """快捷方式"起始位置"不是仓库根时，实测行仍要命中。

    裸相对路径会静默退回通用兜底，等于把按卡表变成装饰——这条守卫的是
    "从别处启动也算得出同一笔账"。
    """
    from pathlib import Path

    fake_home = tmp_path / "home"
    (fake_home / ".lmloader").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.chdir(tmp_path)  # 仓库根候选在这里不存在

    card = "nvidia-geforce-rtx-5080-laptop-gpu-sm120"
    hc.save_measurement(card, dict(hc.GENERIC_COEFFICIENTS,
                                   vram_safety_fraction=0.72), {}, )
    assert hc.table_path() == fake_home / ".lmloader" / "hardware_costs.json"
    got = hc.resolve_coefficients(gpu_class=card)
    assert got.vram_safety_fraction == 0.72
    assert "实测表" in got.source


def test_missing_everywhere_falls_back_to_generic(tmp_path, monkeypatch):
    # 反向对照：候选位置全空时必须是通用兜底，不能读到测试进程外的真表。
    from pathlib import Path

    empty_home = tmp_path / "home"
    empty_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: empty_home)
    monkeypatch.chdir(tmp_path)
    got = hc.resolve_coefficients(gpu_class="some-card-sm100")
    assert got.vram_safety_fraction == hc.GENERIC_COEFFICIENTS["vram_safety_fraction"]
    assert "未收录 some-card-sm100" in got.source
