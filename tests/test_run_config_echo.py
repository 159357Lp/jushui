"""每模型运行配置的回显必须是"真正会生效的那一份"。

存档行里没填的键是 NULL（db 层原样给回 None）。router 的 `_pick` 遇到 None 退回
引擎槽默认（线程 8），而模型页的线程框遇到 None 退回 `os.cpu_count()`（逻辑核 32）。
两个默认不一致时，"打开对话框看一眼再点保存"就会把 8 写进存档——实测线程 24/32
比 8 慢（−ngl 61 档 −2.8%，部分卸载档 8 反而比 16 快 5~10%）。

所以守卫的判据是 GET 的回显：NULL 键必须显示成槽默认，非 NULL 键必须显示成存档值，
`saved` 仍按"有没有存档行"给。反向对照：整行都没有时不能假装 saved=True。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lmloader.api.routes import models as m

DEFAULTS = {"ctx_size": 8192, "n_gpu_layers": -1, "threads": 8, "flash_attn": True,
            "cache_type_k": "q4_0", "cache_type_v": "q4_0", "parallel": 1,
            "batch_size": 2048, "ubatch_size": 512, "mtp_enabled": True,
            "spec_draft_max": 2, "cpu_moe_layers": -1}


@pytest.fixture
def stub(monkeypatch):
    """桩掉 db 的两条查询与槽默认；返回一个可改存档行的字典盒子。"""
    store = {"saved": None}
    model = SimpleNamespace(name="t.gguf", metadata={},
                            to_dict=lambda: {"name": "t.gguf"})

    async def get_model_by_name(_name):
        return model

    async def get_run_config(_name):
        return store["saved"]

    monkeypatch.setattr(m.db, "get_model_by_name", get_model_by_name)
    monkeypatch.setattr(m.db, "get_run_config", get_run_config)
    monkeypatch.setattr(m, "_defaults_for_engine_slot", lambda _model: dict(DEFAULTS))
    return store


def test_null_keys_come_back_as_the_slot_default(monkeypatch, stub):
    # 存档只填了 ctx，其余为 NULL（真库里就有这样一行：Qwen3.6-35B-A3B-UD-Q2_K_XL）
    stub["saved"] = {"ctx_size": 32768, "n_gpu_layers": None, "threads": None,
                     "flash_attn": False, "cache_type_k": None, "cache_type_v": None,
                     "parallel": None, "batch_size": None, "ubatch_size": None,
                     "mtp_enabled": False, "spec_draft_max": 2,
                     "cpu_moe_layers": None}
    out = asyncio.run(m.get_model_run_config("t.gguf"))
    assert out["saved"] is True, "有存档行的事实不能被合并改掉"
    cfg = out["config"]
    assert cfg["threads"] == 8, f"线程不能回 None：{cfg['threads']} 会被 GUI 当成逻辑核 32 保存"
    assert cfg["ctx_size"] == 32768, "填过的键必须还是存档值"
    assert cfg["spec_draft_max"] == 2
    assert cfg["cache_type_k"] == "q4_0", "NULL 键要落到槽默认，不是空串"


def test_absent_row_still_reports_not_saved(stub):
    out = asyncio.run(m.get_model_run_config("t.gguf"))
    assert out["saved"] is False, "没有存档行却报 saved=True 就是假回显"
    assert out["config"]["threads"] == 8


def test_saved_row_never_leaks_a_none_into_the_dialog(stub):
    """整行 NULL 也要全键有值：GUI 的每个控件都不该收到 None。"""
    stub["saved"] = {k: None for k in DEFAULTS}
    out = asyncio.run(m.get_model_run_config("t.gguf"))
    assert all(v is not None for v in out["config"].values()), out["config"]
    assert out["config"] == DEFAULTS
