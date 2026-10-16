"""死引擎回收：引擎进程被杀后状态必须纠正，不能继续报「已加载」。

复现的真实故障
--------------
GUI 关闭时 kill-on-close 作业对象会带走 llama-server；手动 taskkill 或
引擎崩溃同理。后端此时仍持有 instance 记录，于是：

- GET /api/models/loaded 仍返回该模型（界面显示"已加载"）
- 但对应端口无人监听，推理请求全部 502

修复前 _idle_watchdog 只按 idle_minutes 卸载，且 idle_unload_minutes==0
时直接 return —— 引擎被外部杀死这条路径完全无人处理。
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmloader.engine.lifecycle import LifecycleManager, ModelInstance  # noqa: E402


# ---------- 测试替身 ----------

class FakeProcess:
    """Stands in for asyncio.subprocess.Process with a controllable exit.

    Deliberately has NO ``poll()`` method: the real asyncio Process on Windows
    does not define one either. An earlier fake that included poll() let a real
    regression through — the production code called poll(), got an
    AttributeError, concluded "cannot tell liveness", and marked every healthy
    engine as dead. The fake must mirror the real API, warts included.
    """

    def __init__(self, pid: int = 4242, running: bool = True, returncode: int | None = None):
        self.pid = pid
        self._running = running
        # Mirror asyncio: a not-yet-reaped exit still reports None, so callers
        # construct an already-dead process with an explicit returncode.
        self.returncode = returncode if returncode is not None else (None if running else 0)

    def kill(self):
        self._running = False
        self.returncode = -9

    def terminate(self):
        self._running = False
        self.returncode = 15


class FakeRecord:
    def __init__(self, name="probe.gguf"):
        self.name = name
        self.path = f"C:/tmp/{name}"
        self.size = 1024
        self.metadata = {"context_length": 32768}

    def to_dict(self):
        return {"name": self.name, "path": self.path}


class FakeLaunch:
    def __init__(self, port=8100):
        self.port = port
        self.host = "127.0.0.1"
        self.engine_slot = "primary"
        self.kind = "llama"
        self.args = ["-c", "32768"]
        self.mmproj_path = None
        self.model_path = "C:/tmp/probe.gguf"


def make_manager_with(entries):
    """entries: list of (name, FakeProcess running?)"""
    mgr = LifecycleManager()
    for name, running in entries:
        proc = FakeProcess(running=running)
        inst = ModelInstance(FakeRecord(name), proc, FakeLaunch())
        mgr.instances[name] = inst
        mgr._used_ports.add(inst.launch_config.port)
    return mgr


# ---------- is_alive ----------

def test_is_alive_true_while_running():
    proc = FakeProcess(running=True)
    inst = ModelInstance(FakeRecord(), proc, FakeLaunch())
    assert inst.is_alive() is True


def test_is_alive_false_after_kill():
    proc = FakeProcess(running=True)
    inst = ModelInstance(FakeRecord(), proc, FakeLaunch())
    proc.kill()
    assert inst.is_alive() is False


def test_is_alive_false_when_returncode_set():
    proc = FakeProcess(running=False, returncode=1)
    inst = ModelInstance(FakeRecord(), proc, FakeLaunch())
    assert inst.is_alive() is False


def test_is_alive_false_when_poll_would_be_called():
    """Regression guard: production code must not touch poll().

    asyncio.subprocess.Process on Windows has no poll(); calling it raises
    AttributeError. An implementation that wrapped poll() in try/except and
    treated the error as "cannot tell" would report every healthy engine as
    dead — the exact regression this fake (which lacks poll) reproduces.
    """
    proc = FakeProcess(running=True)
    inst = ModelInstance(FakeRecord(), proc, FakeLaunch())
    assert not hasattr(proc, "poll"), "fake must mirror the real API"
    # A live engine must be reported alive without any poll() call.
    assert inst.is_alive() is True


def test_is_alive_source_never_calls_poll():
    import inspect

    src = inspect.getsource(ModelInstance.is_alive)
    assert ".poll(" not in src, (
        "is_alive must not call poll() — asyncio.subprocess.Process has no "
        "such method on Windows and it raises AttributeError"
    )


def test_is_alive_uses_returncode_only():
    import inspect

    src = inspect.getsource(ModelInstance.is_alive)
    assert "returncode" in src


# ---------- _reap_dead_instances ----------

def test_reap_drops_dead_instance():
    mgr = make_manager_with([("dead.gguf", False)])
    dead = asyncio.run(mgr._reap_dead_instances())
    assert dead == ["dead.gguf"]
    assert mgr.instances == {}


def test_reap_keeps_live_instance():
    mgr = make_manager_with([("alive.gguf", True)])
    dead = asyncio.run(mgr._reap_dead_instances())
    assert dead == []
    assert "alive.gguf" in mgr.instances


def test_reap_releases_port():
    """A dead engine must free its port so the next load can reuse it."""
    mgr = make_manager_with([("dead.gguf", False)])
    asyncio.run(mgr._reap_dead_instances())
    assert 8100 not in mgr._used_ports


def test_reap_mixed_survivors():
    mgr = make_manager_with([("a.gguf", True), ("b.gguf", False), ("c.gguf", True)])
    dead = asyncio.run(mgr._reap_dead_instances())
    assert dead == ["b.gguf"]
    assert set(mgr.instances) == {"a.gguf", "c.gguf"}


def test_reap_is_idempotent():
    mgr = make_manager_with([("dead.gguf", False)])
    assert asyncio.run(mgr._reap_dead_instances()) == ["dead.gguf"]
    assert asyncio.run(mgr._reap_dead_instances()) == []


def test_reap_does_not_kill_dead_process():
    """Must not go through unload/kill on an already-dead process."""
    mgr = make_manager_with([("dead.gguf", False)])
    proc = mgr.instances["dead.gguf"].process
    asyncio.run(mgr._reap_dead_instances())
    # kill() must never have been called (it would set returncode to -9)
    assert proc.returncode != -9


def test_reap_cancels_stderr_task():
    """The stderr reader must be cancelled, not left dangling.

    cancel() only *requests* cancellation; task.cancelled() stays False until
    the loop actually delivers it. So await the task instead of polling the
    flag right after the call.
    """
    mgr = make_manager_with([("dead.gguf", False)])
    inst = mgr.instances["dead.gguf"]

    async def scenario():
        async def _never():
            await asyncio.sleep(3600)

        task = asyncio.ensure_future(_never())
        inst._stderr_task = task
        assert await mgr._reap_dead_instances() == ["dead.gguf"]
        # Await it: propagates CancelledError, which we swallow to inspect state.
        try:
            await task
        except asyncio.CancelledError:
            pass
        return task

    task = asyncio.run(scenario())
    assert task.done()


def test_reap_empty_manager():
    mgr = LifecycleManager()
    assert asyncio.run(mgr._reap_dead_instances()) == []


# ---------- get_loaded_models: the GUI data source ----------

def test_get_loaded_models_filters_dead_immediately():
    """The regression: this must NOT report a model whose engine has died."""
    mgr = make_manager_with([("dead.gguf", False)])
    assert asyncio.run(mgr.get_loaded_models()) == []
    assert mgr.instances == {}


def test_get_loaded_models_returns_live():
    mgr = make_manager_with([("alive.gguf", True)])
    out = asyncio.run(mgr.get_loaded_models())
    assert len(out) == 1
    assert out[0]["model"]["name"] == "alive.gguf"


def test_get_loaded_models_mixed():
    mgr = make_manager_with([("alive.gguf", True), ("dead.gguf", False)])
    out = asyncio.run(mgr.get_loaded_models())
    assert [o["model"]["name"] for o in out] == ["alive.gguf"]


def test_get_loaded_models_reaps_so_second_call_stays_clean():
    """First poll corrects state; later polls don't resurrect the entry."""
    mgr = make_manager_with([("dead.gguf", False)])
    assert asyncio.run(mgr.get_loaded_models()) == []
    assert asyncio.run(mgr.get_loaded_models()) == []
    assert "dead.gguf" not in mgr.instances


# ---------- get_model_instance: inference path ----------

def test_get_model_instance_returns_none_for_dead():
    mgr = make_manager_with([("dead.gguf", False)])
    assert asyncio.run(mgr.get_model_instance("dead.gguf")) is None


def test_get_model_instance_returns_live():
    mgr = make_manager_with([("alive.gguf", True)])
    got = asyncio.run(mgr.get_model_instance("alive.gguf"))
    assert got is not None
    assert got.model.name == "alive.gguf"


def test_get_model_instance_unknown_name():
    mgr = LifecycleManager()
    assert asyncio.run(mgr.get_model_instance("nope.gguf")) is None


def test_get_model_instance_reaps_as_side_effect():
    mgr = make_manager_with([("dead.gguf", False)])
    asyncio.run(mgr.get_model_instance("dead.gguf"))
    assert mgr.instances == {}
    assert 8100 not in mgr._used_ports


# ---------- watchdog must run the reaper even with auto-unload off ----------

def test_watchdog_does_not_return_early_when_idle_disabled():
    """Regression guard for the original bug's second half.

    The old watchdog did `if limit <= 0: return` at the top, so with
    idle_unload_minutes == 0 no liveness check ever ran. Inspect the compiled
    code object rather than the source text: a docstring mentioning "return"
    must not be mistaken for a return statement.
    """
    import dis

    fn = LifecycleManager._idle_watchdog
    code = fn.__code__
    return_offsets = [
        ins.offset for ins in dis.get_instructions(code)
        if ins.opname == "RETURN_VALUE"
    ]

    # Locate the polling loop by its backward jump.
    loop_start = None
    for ins in dis.get_instructions(code):
        if ins.opname in ("JUMP_BACKWARD", "JUMP_BACKWARD_ABSOLUTE"):
            loop_start = ins.argval
            break
    assert loop_start is not None, "watchdog no longer contains its polling loop"

    # The fix removed every early exit: the coroutine now only returns through
    # CancelledError, so no RETURN_VALUE may sit before the loop. This is the
    # strongest form of the guard — the old `if limit <= 0: return` would show
    # up here immediately.
    early = [off for off in return_offsets if off < loop_start]
    assert not early, (
        f"watchdog has an early return at offset(s) {early}, before the polling "
        f"loop at {loop_start} — liveness checking would never run when "
        f"idle_unload_minutes == 0"
    )


def test_watchdog_source_calls_reaper():
    import inspect

    src = inspect.getsource(LifecycleManager._idle_watchdog)
    assert "_reap_dead_instances" in src


# ---------- end-to-end against a REAL killed process ----------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only process test")
def test_real_process_death_detected():
    """Not a fake: spawn a real child, kill it, confirm the reaper notices."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    async def scenario():
        class Wrapped:
            """Adapter over a REAL subprocess, mirroring the asyncio API.

            Exposes pid/returncode only — no poll(), because the real
            asyncio.subprocess.Process does not have one. kill() delegates to
            the underlying Popen so the test can actually kill it.
            """

            def __init__(self, p):
                self.pid = p.pid
                self._p = p
                self.returncode = None

            def kill(self):
                self._p.kill()

        mgr = LifecycleManager()
        inst = ModelInstance(FakeRecord("real.gguf"), Wrapped(proc), FakeLaunch())
        mgr.instances["real.gguf"] = inst
        mgr._used_ports.add(8100)

        # Alive while the child sleeps: returncode stays None.
        assert await mgr.get_model_instance("real.gguf") is not None
        assert len(await mgr.get_loaded_models()) == 1

        proc.kill()
        proc.wait(timeout=10)
        # Mirror what asyncio's child watcher does once the process is reaped.
        inst.process.returncode = proc.returncode

        dead = await mgr._reap_dead_instances()
        return dead, mgr

    try:
        dead, mgr = asyncio.run(scenario())
        assert dead == ["real.gguf"], f"expected reap, got {dead}"
        assert mgr.instances == {}
        assert 8100 not in mgr._used_ports
    finally:
        if proc.poll() is None:
            proc.kill()
