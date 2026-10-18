"""决策模型钉选：切换连坐与闲置看门狗不得卸载决策模型。

决策模型（d3 包 / decision-type GGUF）是常驻基础设施，走 /v1/systemone
被智能体侧调用；d3 冷启动要过 Python + triton 编译，被自动卸载后下次
用到时卡顿最明显。钉选只挡"自动"路径：显式卸载（GUI 按钮、API unload）
和后端退出清理照常卸载。
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from lmloader.engine.lifecycle import (  # noqa: E402
    LifecycleManager,
    ModelInstance,
    is_decision_model,
)

# The package attribute ``lmloader.engine.lifecycle`` is shadowed by the
# ``lifecycle = LifecycleManager()`` singleton re-exported from the engine
# package __init__, so grab the real module object from sys.modules.
lifecycle_module = sys.modules["lmloader.engine.lifecycle"]


@pytest.fixture(autouse=True)
def _neutralize_process_kill(monkeypatch):
    """kill_process_tree would taskkill real PIDs at these numbers."""
    monkeypatch.setattr(lifecycle_module, "kill_process_tree", lambda pid: None)


# ---------- 测试替身 ----------

class FakeProcess:
    def __init__(self, pid: int = 999999):
        self.pid = pid
        self.returncode = None

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


class FakeRecord:
    def __init__(self, name="m.gguf", metadata=None):
        self.name = name
        self.path = f"C:/tmp/{name}"
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self):
        return {"name": self.name, "path": self.path}


class FakeLaunch:
    def __init__(self, port=8100, kind="llama"):
        self.port = port
        self.host = "127.0.0.1"
        self.engine_slot = "primary"
        self.kind = kind
        self.args = []
        self.mmproj_path = None


def make_inst(name, *, pinned=False, kind="llama", port=8100, idle_min=0.0):
    proc = FakeProcess(pid=900000 + port)
    inst = ModelInstance(
        FakeRecord(name), proc, FakeLaunch(port=port, kind=kind), pinned=pinned,
    )
    if idle_min:
        inst.last_active_at -= idle_min * 60.0
    return inst


def make_manager(entries):
    mgr = LifecycleManager()
    # Keep the fake DB and process-killing out of the unload path: status
    # writes hit a real sqlite file, and kill_process_tree would taskkill
    # real PIDs.
    async def _noop_status(*a, **k):
        return None
    mgr._safe_status_update_by_name = _noop_status
    for inst in entries:
        mgr.instances[inst.model.name] = inst
        mgr._used_ports.add(inst.launch_config.port)
    return mgr


# ---------- is_decision_model 判定 ----------

def test_d3_package_is_decision():
    assert is_decision_model(FakeRecord(), "d3") is True

def test_decision_gguf_is_decision_even_with_llama_kind():
    # router.resolve_engine reports kind "llama" for d3-gguf; the metadata
    # decision_type is the only mark, so this must not depend on kind.
    rec = FakeRecord(metadata={"decision_type": "vllm-d3"})
    assert is_decision_model(rec, "llama") is True

def test_ordinary_gguf_is_not_decision():
    assert is_decision_model(FakeRecord(metadata={}), "llama") is False

def test_none_metadata_is_not_decision():
    class NoMeta:
        name = "x"
        metadata = None
    assert is_decision_model(NoMeta(), "llama") is False


# ---------- 连坐卸载（切换/加载普通模型的全清路径） ----------

def test_switch_bulk_unload_keeps_pinned():
    pinned = make_inst("d3-package", pinned=True, kind="d3", port=8101)
    chattier = make_inst("chat.gguf", port=8102)
    mgr = make_manager([pinned, chattier])
    asyncio.run(mgr._unload_all_locked(keep_pinned=True))
    assert "d3-package" in mgr.instances
    assert "chat.gguf" not in mgr.instances
    assert 8101 in mgr._used_ports
    assert 8102 not in mgr._used_ports

def test_teardown_unload_all_clears_pinned_too():
    """退出清理/引擎热替换必须全量卸载（默认 keep_pinned=False）。"""
    pinned = make_inst("d3-package", pinned=True, kind="d3", port=8101)
    mgr = make_manager([pinned])
    asyncio.run(mgr.unload_all())
    assert mgr.instances == {}
    assert 8101 not in mgr._used_ports

def test_explicit_unload_reaches_pinned():
    """用户点卸载时钉选不拦路。"""
    pinned = make_inst("d3-package", pinned=True, kind="d3", port=8101)
    mgr = make_manager([pinned])
    assert asyncio.run(mgr.unload_model("d3-package")) is True
    assert mgr.instances == {}


# ---------- 闲置看门狗 ----------

def test_idle_sweep_skips_pinned():
    pinned = make_inst("d3-package", pinned=True, kind="d3", port=8101, idle_min=999)
    stale = make_inst("idle.gguf", port=8102, idle_min=999)
    fresh = make_inst("busy.gguf", port=8103, idle_min=1)
    mgr = make_manager([pinned, stale, fresh])
    asyncio.run(mgr._idle_sweep_once(limit=30))
    assert set(mgr.instances) == {"d3-package", "busy.gguf"}

def test_idle_sweep_still_unloads_normal_stale():
    stale = make_inst("idle.gguf", port=8102, idle_min=45)
    mgr = make_manager([stale])
    asyncio.run(mgr._idle_sweep_once(limit=30))
    assert mgr.instances == {}


# ---------- 对外可见性 ----------

def test_to_dict_exposes_pinned():
    inst = make_inst("d3-package", pinned=True, kind="d3")
    assert inst.to_dict()["pinned"] is True
    inst2 = make_inst("chat.gguf")
    assert inst2.to_dict()["pinned"] is False


def test_to_dict_exposes_is_decision_model():
    """/api/models/loaded 必须自带这个标记——GUI 拿它决定露哪个调用地址。"""
    assert make_inst("d3-package", kind="d3").to_dict()["is_decision_model"] is True

    gguf = ModelInstance(
        FakeRecord("d3-mini.gguf", metadata={"decision_type": "vllm-d3"}),
        FakeProcess(pid=908100), FakeLaunch(kind="llama"),
    )
    assert gguf.to_dict()["is_decision_model"] is True
    # 反向对照：普通聊天模型不能也标 True，否则界面五个端点全被藏掉。
    assert make_inst("chat.gguf").to_dict()["is_decision_model"] is False


# ---------- 自动路径的接线守卫（源码级） ----------

def test_load_model_gates_auto_paths():
    import inspect

    src = inspect.getsource(LifecycleManager.load_model)
    assert "_unload_all_locked(keep_pinned=True)" in src, (
        "load_model's auto_unload_on_switch sweep must keep pinned models"
    )
    assert "pinned=is_decision_model" in src, (
        "load_model must mark decision instances pinned at load time"
    )

def test_switch_model_gates_auto_paths():
    import inspect

    src = inspect.getsource(LifecycleManager.switch_model)
    assert "unload_all(keep_pinned=True)" in src

def test_watchdog_sweep_skips_pinned_source():
    import inspect

    src = inspect.getsource(LifecycleManager._idle_sweep_once)
    assert "inst.pinned" in src


# ---------- OpenAI /v1/models 必须把决策模型标出来 ----------

def _v1_models(records):
    """Call the OpenAI list route against a fake DB."""
    from unittest.mock import AsyncMock, patch

    import lmloader.api.routes.chat as chat_routes

    class FakeDB:
        async def list_models(self, *a, **k):
            return list(records)

    class FakeLifecycle:
        async def get_loaded_models(self):
            return []

    with patch.object(chat_routes, "lifecycle", FakeLifecycle()), \
            patch("lmloader.infra.database.db", FakeDB()):
        return asyncio.run(chat_routes.list_models())


def _model(name, **kw):
    rec = FakeRecord(name, metadata=kw.pop("metadata", {}))
    for attr, default in (("architecture", "qwen35"), ("quantization", "Q4_K_M"),
                          ("parameters", "4.8B"), ("size", 1), ("status", "unloaded"),
                          ("created_at", None)):
        setattr(rec, attr, kw.get(attr, default))
    return rec


def test_openai_model_list_marks_the_decision_gguf():
    payload = _v1_models([
        _model("chat.gguf"),
        _model("d3-mini.gguf", metadata={"decision_type": "vllm-d3"}),
    ])
    by_id = {item["id"]: item["metadata"] for item in payload["data"]}
    assert by_id["d3-mini.gguf"]["is_decision_model"] is True
    # The negative control: an ordinary chat GGUF must be marked False, not
    # left unlabelled, or a client cannot tell "not a chat model" from
    # "server too old to know".
    assert by_id["chat.gguf"]["is_decision_model"] is False


def test_openai_model_list_marks_d3_package_by_files():
    payload = _v1_models([_model("pkg")])
    item = payload["data"][0]
    assert item["metadata"]["is_decision_model"] is False, (
        "a nonexistent path is not a d3 package; the flag must not guess"
    )


# ---------- /v1/chat/completions 的 400 拦截 ----------

def _refuse(record):
    from unittest.mock import patch

    import lmloader.api.routes.chat as chat_routes

    class FakeDB:
        async def get_model_by_name(self, name):
            return record

    with patch("lmloader.infra.database.db", FakeDB()):
        try:
            asyncio.run(chat_routes._refuse_decision_model("m.gguf",
                                                            "/v1/chat/completions"))
        except chat_routes.HTTPException as exc:
            return exc
    return None


def test_refusal_400_fires_for_decision_gguf():
    exc = _refuse(_model("m.gguf", metadata={"decision_type": "vllm-d3"}))
    assert exc is not None and exc.status_code == 400
    assert "/v1/systemone" in str(exc.detail), "报错要把能用的端点说清楚"


def test_refusal_skips_ordinary_chat_model():
    # 反向对照：普通聊天模型必须畅通，否则这一拦就是把所有模型拦死
    assert _refuse(_model("m.gguf")) is None


def test_refusal_skips_unresolved_alias():
    from unittest.mock import patch

    import lmloader.api.routes.chat as chat_routes

    class NoRecordDB:
        async def get_model_by_name(self, name):
            return None

    with patch("lmloader.infra.database.db", NoRecordDB()):
        assert asyncio.run(chat_routes._refuse_decision_model(
            "alias", "/v1/chat/completions")) is None


# ---------- 本地服务页：调用地址要跟着模型类型变 ----------

def test_page_endpoints_follow_model_kind():
    from lmloader.gui.pages import logs as logs_page

    chat = {"is_decision_model": False}
    decision = {"is_decision_model": True}
    assert logs_page._primary_path(decision) == "/v1/systemone"
    assert logs_page._primary_path(chat) == "/v1/chat/completions"
    # 决策模型那一行只列它真能答的端点；列出五个 chat 端点等于邀请一次 400。
    assert logs_page._endpoints_for(decision) == [("POST", "/v1/systemone")]
    paths = [p for _, p in logs_page._endpoints_for(chat)]
    assert "/v1/chat/completions" in paths and "/v1/systemone" not in paths


def test_page_call_url_strings_exist_in_both_languages():
    from lmloader.gui.i18n import TRANSLATIONS

    keys = ("logs.call_url", "logs.engine_direct", "logs.engine_direct_tip",
            "logs.base_url", "logs.base_url_tip",
            "logs.decision_badge", "logs.decision_badge_tip")
    for key in keys:
        for lang in ("zh", "en"):
            assert key in TRANSLATIONS[lang], f"{key} missing in {lang}"
    # {port} 缺了会把花括号原样印在界面上。
    for lang in ("zh", "en"):
        assert "{port}" in TRANSLATIONS[lang]["logs.engine_direct"]
