"""Regression guards for the audit findings, not for a feature.

Context: a review of the five bug-fix commits in this session found two defects
that sat squarely on those fixes' own main code paths:

H1 — clicking 停止 (or tripping the repetition guard) called
``_on_stream_finished`` from the main thread while the worker thread was still
running. ``ChatStreamWorker.run`` deliberately does NOT emit ``finished()`` on
the stop path, so ``worker.finished -> thread.quit`` never fired, the QThread
event loop stayed parked in ``exec()``, and ``_cleanup_stream_thread`` called
``deleteLater()`` on it. Qt then destroys the C++ QThread while the OS thread is
still alive -> ``std::terminate`` / 0xC0000409. The failure mode is documented
in this very module (``run_in_thread``'s docstring). Trigger: the feature the
commit added.

H2 — ``get_loaded_models`` swallowed every exception and returned ``[]``, so a
failed query was indistinguishable from "nothing is loaded" and blanked every
badge. Fixed by ``fetch_loaded_models_or_none``.

M1 — ``_reap_dead_instances`` took ``_lock``, which ``load_model`` holds for up
to ``startup_timeout`` (120s) while waiting for the engine's /health. A poll
during a load blocked past the client's 10s timeout, which then fed H2.

M2 — reaping cleaned the in-memory registry but not ``db.models.status``, so
external readers (``routes/models.py`` shutdown path, the OpenAI-compatible
/v1/models) kept seeing "loaded" after a restart.

These tests are structural plus behavioural: H1 is about C++ object lifetime and
cannot be provoked safely in-process, so it is pinned by inspecting the actual
control flow, and the runnable half is covered by the live probe.
"""

import asyncio
import pathlib

import pytest

# NOTE: `import lmloader.engine.lifecycle as X` does NOT give the module here.
# lmloader/engine/__init__.py does `from lmloader.engine.lifecycle import
# lifecycle`, which rebinds the attribute `lifecycle` on the package to the
# LifecycleManager singleton — shadowing the submodule. Asking for the module by
# name then yields that instance, and `.__file__` raises AttributeError.
# importlib.import_module resolves it unambiguously.
import importlib

LIFECYCLE_MOD = importlib.import_module("lmloader.engine.lifecycle")
import lmloader.gui.api_client as api_client
import lmloader.gui.pages.chat as chat_page
import lmloader.gui.pages.models as models_page

LIFECYCLE_SRC = open(LIFECYCLE_MOD.__file__, encoding="utf-8").read()


# =========================================================== H1: thread safety

def _chat_src():
    return open(chat_page.__file__, encoding="utf-8").read()


def test_cleanup_never_deletes_a_running_thread():
    """deleteLater() on a running QThread is the 0xC0000409 crash.

    The branch structure is the point: if isRunning(), deletion must be deferred
    to the thread's own finished() signal. A flat ``deleteLater()`` on the
    running branch reintroduces the crash.
    """
    src = _chat_src()
    cleanup = src.split("def _cleanup_stream_thread")[1].split("\n    def ")[0]
    assert "isRunning()" in cleanup, (
        "the cleanup path must distinguish a running thread from a finished one"
    )

    running_branch = cleanup.split("if thread.isRunning():")[1].split("else:")[0]
    assert "quit()" in running_branch, (
        "a running thread's event loop must be asked to quit, otherwise it stays "
        "parked in exec() forever"
    )
    assert "finished.connect(self._on_stream_thread_finished)" in running_branch, (
        "a running thread must be retained until its finished() callback"
    )
    assert "self._release_stream_thread(thread)" in running_branch, (
        "a thread that raced to completion must still be retired safely"
    )
    release = src.split("def _release_stream_thread")[1].split("\n    def ")[0]
    assert "deleteLater()" in release, (
        "thread deletion must only be scheduled after its finished() callback"
    )
    # The deferral must not be a blocking wait — the worker can be parked in a
    # socket read and freezing the GUI on it is unacceptable.
    assert "wait(" not in running_branch, (
        "wait() blocks the GUI thread; the worker may sit in a long socket read"
    )


def test_finished_branch_deletes_immediately():
    """A thread that already exited needs no event loop stopped first."""
    src = _chat_src()
    cleanup = src.split("def _cleanup_stream_thread")[1].split("\n    def ")[0]
    else_branch = cleanup.split("else:")[-1]
    assert "deleteLater" in else_branch, (
        "an already-finished thread can (and should) be deleted right away"
    )


def test_worker_stays_quiet_after_stop_but_thread_is_still_stopped():
    """The asymmetry is deliberate — but the thread must not be abandoned.

    ``ChatStreamWorker.run`` returns without emitting ``finished()`` so the GUI
    does not finish the turn twice. That means ``thread.quit`` has no trigger,
    which is exactly why ``_cleanup_stream_thread`` must issue it.
    """
    src = open(api_client.__file__, encoding="utf-8").read()
    run_body = src.split("def run(self):")[-1]
    assert "if self._stop.is_set():" in run_body
    # ...and the chat-side cleanup compensates.
    cleanup = _chat_src().split("def _cleanup_stream_thread")[1].split("\n    def ")[0]
    assert "quit()" in cleanup


def test_late_tokens_are_dropped_after_stop():
    """A token arriving after the turn ended must not mutate a finished reply."""
    src = _chat_src()
    handler = src.split("def _on_stream_token")[1].split("\n    def ")[0]
    assert "self._stream_conv is None" in handler, (
        "cleanup nulls _stream_conv; that is what makes late tokens a no-op"
    )
    finish = src.split("def _on_stream_finished")[1]
    assert "_cleanup_stream_thread()" in finish
    cleanup = src.split("def _cleanup_stream_thread")[1].split("\n    def ")[0]
    assert "self._stream_conv = None" in cleanup


# ================================================== H2: failure vs empty list

def test_fetch_loaded_models_returns_none_on_failure():
    api_src = open(api_client.__file__, encoding="utf-8").read()
    fn = api_src.split("def fetch_loaded_models_or_none")[1].split("\ndef ")[0]
    assert "return None" in fn, (
        "a failed query must return None, not [] — [] means 'nothing loaded'"
    )
    assert "except Exception:" in fn


def test_legacy_get_loaded_models_still_returns_a_list():
    """The old signature stays list-shaped for its other three callers."""
    api_src = open(api_client.__file__, encoding="utf-8").read()
    fn = api_src.split("def get_loaded_models")[1].split("\n    def ")[0]
    assert "return [] if out is None else out" in fn


def test_poll_uses_the_none_returning_variant():
    src = open(models_page.__file__, encoding="utf-8").read()
    poll = src.split("def _poll_loaded")[1].split("\n    def ")[0]
    assert "fetch_loaded_models_or_none" in poll, (
        "the reconciling poll must use the variant that keeps failure distinct"
    )


def test_poll_guards_none_before_iterating():
    poll = open(models_page.__file__, encoding="utf-8").read().split(
        "def _poll_loaded")[1].split("\n    def ")[0]
    result = poll.split("def on_result")[1].split("def on_error")[0]
    assert "if loaded is None:" in result
    body = result.split("if loaded is None:")[1].split("names =")[0]
    assert "return" in body, "the guard must short-circuit before the walk"


# ======================================== M1: the poll must not block on load

def test_reap_timeout_is_defined_and_bounded():
    assert 0 < LIFECYCLE_MOD.REAP_TIMEOUT <= 8.0, (
        "must stay under the GUI client's 10s timeout, otherwise the caller "
        "gives up and reads the empty result as 'nothing loaded'"
    )


def test_get_loaded_models_bounds_the_reap():
    src = LIFECYCLE_SRC
    fn = src.split("async def get_loaded_models")[1].split("\n    async def ")[0]
    assert "asyncio.wait_for" in fn, (
        "the reap on the GUI poll path must be time-boxed"
    )
    assert "asyncio.TimeoutError" in fn, (
        "a timeout must fall through to reporting the registry as-is, not raise"
    )


def test_reap_scans_without_holding_the_lock():
    """The scan must happen before taking _lock, or a long load blocks polls.

    load_model holds _lock across _wait_for_health (up to 120s). Snapshotting
    first means a poll during a load never waits at all when nothing is dead —
    which is the overwhelmingly common case.
    """
    src = LIFECYCLE_SRC
    reap = src.split("async def _reap_dead_instances")[1].split("\n    async def ")[0]
    scan = reap.split("async with self._lock:")[0]
    assert "is_alive()" in scan, (
        "the liveness scan must run before the lock is taken"
    )
    lock_at = reap.index("async with self._lock:")
    first_scan = reap.index("is_alive()")
    assert first_scan < lock_at, (
        "liveness must be checked before acquiring _lock, otherwise the poll "
        "blocks for the duration of a model load"
    )
    assert "if not candidates:" in reap, (
        "with nothing dead the lock must not be taken at all"
    )


def test_reap_rechecks_under_the_lock():
    """A concurrent load may have replaced the instance since the snapshot."""
    reap = LIFECYCLE_SRC.split(
        "async def _reap_dead_instances")[1].split("\n    async def ")[0]
    under = reap.split("async with self._lock:")[1]
    assert "current is not inst" in under, (
        "identity must be re-verified under the lock, or a freshly loaded "
        "instance gets reaped based on the previous one's corpse"
    )
    assert "if current.is_alive()" in under


# ================================== M2: the database must stop saying"loaded"

def test_reap_resets_the_database_status():
    """db.models.status is read by external callers, not just the GUI."""
    reap = LIFECYCLE_SRC.split("async def _reap_dead_instances")[1].split("\n    async def ")[0]
    assert '_safe_status_update_by_name(name, "unloaded")' in reap, (
        "reaping the in-memory registry without resetting db status leaves the "
        "stale state to resurface on restart"
    )


def test_status_update_happens_outside_the_lock():
    """The DB write must not extend the lock's critical section.

    Checking a character offset is not enough: a mutation that MOVES the call
    into the ``async with`` body leaves the text present, so an index()-based
    assertion still passes. The real distinction is whether the call sits inside
    the indented block, so this splits the method at the block boundary.
    """
    reap = LIFECYCLE_SRC.split("async def _reap_dead_instances")[1].split("\n    async def ")[0]
    call = 'await self._safe_status_update_by_name(name, "unloaded")'
    tail = reap[reap.index("async with self._lock:"):]
    lines = tail.split("\n")[1:]

    block_end = len(lines)
    for i, line in enumerate(lines):
        # The block ends at the first line back at the method-body indent level.
        if line.strip() and line.startswith(" " * 8) and not line.startswith(" " * 12):
            block_end = i
            break

    inside = "\n".join(lines[:block_end])
    outside = "\n".join(lines[block_end:])

    assert call not in inside, (
        "the DB write must not happen while holding _lock; awaiting the "
        "database inside the critical section blocks every other caller for "
        "no reason (the registry entry is already gone by then)"
    )
    assert call in outside, (
        "the DB write must still happen, just after the lock is released"
    )


def test_status_update_failure_does_not_break_reaping():
    """A DB hiccup must not resurrect the stale instance or raise to the GUI."""
    reap = LIFECYCLE_SRC.split(
        "async def _reap_dead_instances")[1].split("\n    async def ")[0]
    assert "_safe_status_update_by_name" in reap
    # _safe_status_update_by_name is itself the guarded wrapper (it try/excepts
    # internally) — assert that so this stays true.
    fn = LIFECYCLE_SRC.split(
        "async def _safe_status_update_by_name")[1].split("\n    async def ")[0]
    assert "except Exception" in fn


# ================================================ the reaper still does its job

def _mgr():
    """Reuse the fakes from test_dead_engine_reaper rather than re-inventing them.

    That module already encodes the hard-won lesson that a fake must mirror the
    real API's defects (no poll() on Windows), so a second, looser fake here
    would be a step backwards.
    """
    from tests.test_dead_engine_reaper import FakeProcess, FakeLaunch, FakeRecord
    return FakeProcess, FakeRecord, FakeLaunch


def test_dead_engine_is_still_reaped():
    from lmloader.engine.lifecycle import LifecycleManager, ModelInstance

    FakeProcess, FakeRecord, FakeLaunch = _mgr()
    mgr = LifecycleManager()
    dead = ModelInstance(FakeRecord("a.gguf"), FakeProcess(running=False), FakeLaunch(8100))
    live = ModelInstance(FakeRecord("b.gguf"), FakeProcess(running=True), FakeLaunch(8200))
    mgr.instances["a.gguf"] = dead
    mgr.instances["b.gguf"] = live
    mgr._used_ports.update({8100, 8200})

    reaped = asyncio.run(mgr._reap_dead_instances())

    assert reaped == ["a.gguf"], "only the dead engine is reaped"
    assert "a.gguf" not in mgr.instances
    assert "b.gguf" in mgr.instances, "a live engine must survive"
    assert 8100 not in mgr._used_ports, "the port must be released for reuse"
    assert 8200 in mgr._used_ports, "a live engine keeps its port"


def test_nothing_dead_means_nothing_reaped():
    from lmloader.engine.lifecycle import LifecycleManager, ModelInstance

    FakeProcess, FakeRecord, FakeLaunch = _mgr()
    mgr = LifecycleManager()
    mgr.instances["a.gguf"] = ModelInstance(
        FakeRecord("a.gguf"), FakeProcess(running=True), FakeLaunch(8100)
    )
    assert asyncio.run(mgr._reap_dead_instances()) == []


def test_get_loaded_models_filters_the_dead():
    from lmloader.engine.lifecycle import LifecycleManager, ModelInstance

    FakeProcess, FakeRecord, FakeLaunch = _mgr()
    mgr = LifecycleManager()
    mgr.instances["a.gguf"] = ModelInstance(
        FakeRecord("a.gguf"), FakeProcess(running=False), FakeLaunch(8100)
    )

    out = asyncio.run(mgr.get_loaded_models())

    assert "a.gguf" not in mgr.instances, "the dead entry must leave the registry"
    assert isinstance(out, list)
    assert all(i.get("model", {}).get("name") != "a.gguf" for i in out), (
        "a dead engine must never appear in the list the GUI polls"
    )
