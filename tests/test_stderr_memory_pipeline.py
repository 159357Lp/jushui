"""End-to-end: a real engine's -lv 4 stderr through the real reader.

The fixture is verbatim stderr from a d3-master run of d3-mini-Q4_K_M. It
proves the three halves of the memory-breakdown feature hold together: the
allocation report is parsed into a split, the verbosity-4 chatter that came
with it stays out of the Developer Logs, and the numbers survive to to_dict().
"""

import asyncio
import re
from pathlib import Path

import pytest

from lmloader.engine.lifecycle import ModelInstance
from lmloader.engine.router import LaunchConfig
from lmloader.infra import events
from lmloader.infra.database import ModelRecord

_RE_TIMESTAMP = re.compile(r"^\d+(?:\.\d+){3}\s+")
_SERVER_MODULES = {"srv", "cmn", "slot", "spec"}
_ECHO_TAGS = {"common_param", "common_init_", "common_params_fit_impl"}

@pytest.fixture(autouse=True)
def _isolated_run_stats_db(tmp_path, monkeypatch):
    """_read_stderr 会往 model_run_stats 落行，不隔离就是往真库里写。

    2026-10-22 在真库里查到 16 条本文件造的 d3-mini 假行（decode 恒 12.45、
    prompt 3 / gen 40、ctx 为空），最晚一批的时间戳正对着一次套件运行。那张表
    现在是速度档案页和塌陷判据的唯一来源，掺进合成行就等于污染结论。
    """
    from lmloader.infra import database as db_mod

    monkeypatch.setattr(db_mod.db, "db_path", tmp_path / "pipeline.db")
    asyncio.run(db_mod.db.init())
    yield


@pytest.fixture
def engine_stderr_lines():
    path = Path(__file__).parent / "fixtures" / "llama_stderr_v4.log"
    return [ln.encode("utf-8") for ln in path.read_text(encoding="utf-8").splitlines()]


class _Stderr:
    def __init__(self, lines):
        self._lines = list(lines)

    async def readline(self):
        return self._lines.pop(0) if self._lines else b""


class _Proc:
    def __init__(self, lines):
        self.stderr = _Stderr(lines)
        self.returncode = None
        self.pid = 41700


class _Recorder:
    """Stands in for event_log so the test can count what the panel would see."""

    def __init__(self):
        self.entries = []

    def info(self, tag, msg, **kw):
        self.entries.append(("info", tag, msg))

    def warning(self, tag, msg, **kw):
        self.entries.append(("warning", tag, msg))

    def error(self, tag, msg, **kw):
        self.entries.append(("error", tag, msg))


def _instance(lines, monkeypatch, kind="llama"):
    rec = _Recorder()
    monkeypatch.setattr(events.event_log, "info", rec.info)
    monkeypatch.setattr(events.event_log, "warning", rec.warning)
    monkeypatch.setattr(events.event_log, "error", rec.error)
    model = ModelRecord(
        id=1, name="d3-mini-Q4_K_M", path="models/d3-mini.gguf", size=1,
        architecture="qwen35", quantization="Q4_K_M", parameters="4.8B",
        metadata={},
    )
    cfg = LaunchConfig(
        engine_slot="d3-master", host="127.0.0.1", port=8198,
        model_path=model.path, args=[], kind=kind,
    )
    inst = ModelInstance(model, _Proc(lines), cfg)
    return inst, rec


def test_untagged_output_from_other_engines_still_reaches_the_log(monkeypatch):
    """The continuation filter is llama-only.

    The second line is deliberately shaped like a vLLM/d3-python line arriving
    after an INFO the reader already classified as a dump header: if the filter
    stopped checking the engine kind, this line would vanish with it.
    """
    inst, rec = _instance(
        [b"0.00.106.662 I srv    load_model: loading model 'x.gguf'",
         b"INFO 10-17 03:04:05: GPU KV cache size: 4,096 tokens"],
        monkeypatch, kind="vllm",
    )
    asyncio.run(inst._read_stderr())
    assert len(rec.entries) == 2, rec.entries

    inst2, rec2 = _instance(
        [b"0.00.106.662 I srv    load_model: loading model 'x.gguf'",
         b"INFO 10-17 03:04:05: GPU KV cache size: 4,096 tokens"],
        monkeypatch,
    )
    asyncio.run(inst2._read_stderr())
    assert len(rec2.entries) == 1, "a llama continuation line leaked"


def test_allocation_report_becomes_a_split(engine_stderr_lines, monkeypatch):
    inst, _rec = _instance(engine_stderr_lines, monkeypatch)
    asyncio.run(inst._read_stderr())

    bd = inst.memory_breakdown()
    assert bd["vram"] == {
        "weights": 1515.92, "kv": 96.0, "recurrent": 67.0, "compute": 102.66,
    }
    assert bd["ram"]["weights"] == 2009.38
    assert bd["ram"]["kv"] == 160.0
    # Cross-check against the engine's own summed table in the same log
    # ("CUDA0 ... self 1781 = 1515 + 163 + 102").
    assert round(bd["total_mib"]["vram"]) == 1782


def test_verbosity4_chatter_stays_out_of_the_developer_logs(
    engine_stderr_lines, monkeypatch
):
    """`-lv 4` adds libllama's whole INFO stream to the engine's stderr.

    Measured on the same load: at the old default verbosity the engine wrote 10
    tagged lines, at -lv 4 it writes 258. The extra ~200 must stay in the debug
    file. The srv/slot/cmn INFO that does reach the panel is the pre-existing
    profile (it appears in the developer log as far back as the runs were at
    verbosity 3), so this asserts the *shape* of what leaks rather than a count
    that would move with every llama.cpp release.
    """
    inst, rec = _instance(engine_stderr_lines, monkeypatch)
    asyncio.run(inst._read_stderr())

    engine_entries = [m for _l, _t, m in rec.entries if re.match(_RE_TIMESTAMP, m)]
    ours = [m for _l, _t, m in rec.entries if not re.match(_RE_TIMESTAMP, m)]

    for msg in engine_entries:
        m = re.match(
            r"^\d+(?:\.\d+){3}\s+([IWED])\s+(?:(\w+)\s+)?(\S+?)\s*(?:\(\S*\)\s*)?:", msg
        )
        assert m, f"unparseable engine line reached the panel: {msg!r}"
        level, module, tag = m.group(1), m.group(2), m.group(3)
        if level == "I":
            # libllama internals carry no srv/cmn/slot/spec module, and
            # llama-common's own parameter echo repeats arguments we passed it.
            assert module in _SERVER_MODULES, f"libllama internals leaked: {msg!r}"
            assert tag not in _ECHO_TAGS, f"launch-argument echo leaked: {msg!r}"

    # Everything lmloader writes itself is the memory summary and the timing lines.
    assert all(("显存" in o or "生成" in o or "提示词" in o) for o in ours), ours

    # Coarse flood guard for the whole fixture run, not a precise target.
    assert len(rec.entries) < 50, f"the panel gained engine internals: {len(rec.entries)}"

    # The lines that must still get through: startup notices and the summary.
    msgs = " \n".join(m for _lvl, _tag, m in rec.entries)
    assert "model loaded" in msgs
    assert "显存 权重" in msgs, "no memory summary was emitted"


def test_dump_body_is_swallowed_but_a_crash_trace_is_not(monkeypatch):
    """Both lines are untagged, so only the wording separates a chat-template
    body line from a ggml abort that follows a tagged INFO line."""
    inst, rec = _instance(
        [b"0.02.241.373 I srv          init: init: chat template, example_format: X",
         b"You are a helpful assistant<|im_end|>",
         b"0.02.241.800 I srv          init: init: chat template, thinking = 1",
         b"ggml/src/ggml-cuda/ggml-cuda.cu:94 CUDA error: out of memory"],
        monkeypatch,
    )
    asyncio.run(inst._read_stderr())
    msgs = [m for _l, _t, m in rec.entries]
    assert not any("helpful assistant" in x for x in msgs), msgs
    assert any("CUDA error" in x for x in msgs), msgs


def test_errors_still_surface(engine_stderr_lines, monkeypatch):
    inst, rec = _instance(engine_stderr_lines, monkeypatch)
    asyncio.run(inst._read_stderr())
    assert all(lvl != "error" for lvl, _tag, _m in rec.entries), \
        "a clean run must not produce errors"

    failing = [ln for ln in engine_stderr_lines if b"model loaded" in ln]
    inst2, rec2 = _instance(
        failing + [b"0.02.737.699 E srv  llama_server: exiting due to model loading error"],
        monkeypatch,
    )
    asyncio.run(inst2._read_stderr())
    assert any(lvl == "error" for lvl, _tag, _m in rec2.entries)


def test_to_dict_carries_the_breakdown(engine_stderr_lines, monkeypatch):
    inst, _rec = _instance(engine_stderr_lines, monkeypatch)
    asyncio.run(inst._read_stderr())
    d = inst.to_dict()
    assert d["memory"]["vram"]["kv"] == 96.0
    assert d["memory"]["ram"]["weights"] == 2009.38
