"""速度档案：一次生成的"跑多快"和"当时怎么配的"必须落在同一行里。

这个功能唯一的风险是**静默缺项**——映射表漏一个 flag，页面上那条配置就凭空消失，
看起来像"这次没设这个参数"，而实际上是它被丢了。所以判据全部围绕"缺项会不会被发现"：
未识别的 argv token 必须原样冒出来、识别了的不能再冒出来、中位必须是一条真记录
（插值出来的数没有对应配置）、JSON 坏了不能退化成空 dict。

每条"会生效"的判据都配反向对照：裁剪真超上限才删、排名真把空速度的行排除在外、
写库真失败时 stderr 排空照常跑完。
"""

from __future__ import annotations

import asyncio
import importlib
import re
import time
from pathlib import Path

import pytest

from lmloader.engine import run_stats
from lmloader.infra import database as db_mod
from lmloader.infra.database import _stat_view, db

# lmloader.engine 的惰性导出把单例命名为 lifecycle，和子模块同名，
# `from lmloader.engine import lifecycle` 拿到的是实例而不是模块。
lc_mod = importlib.import_module("lmloader.engine.lifecycle")

ARGS = [
    "-m", "D:/模型/Bonsai.gguf", "--port", "8100", "--host", "127.0.0.1",
    "--n-gpu-layers", "999", "-c", "131072", "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0", "--flash-attn", "on", "--parallel", "1",
    "--threads", "8", "--reasoning", "off", "-b", "2048", "-ub", "512",
    "--cache-ram", "2048", "--spec-type", "draft-mtp", "--spec-draft-n-max", "1",
]
ARGS_NO_MTP = ARGS[:ARGS.index("--spec-type")]


def tagged(ctx):
    """换掉 -c 的值当这一条记录的身份标记。

    必须在原位置改值，不能再追加一个 -c —— 追加的话 argv 里出现两次 -c，
    解出来的 cfg 是后者，"每条记录配置不同"这个前提就假了。
    """
    out = list(ARGS)
    out[out.index("-c") + 1] = str(ctx)
    return out


def rec(name, tps=None, args=None, **snap_extra):
    snap = {}
    if tps is not None:
        snap["tg_final"] = tps
    snap.update(snap_extra)
    return run_stats.build_record(
        model_name=name, args=ARGS if args is None else args,
        engine_kind="ternary", build="prism-stable", snap=snap,
    )


def series(name, pairs):
    """[(decode_tps, 标记用的 ctx_size)] -> 记录列表。"""
    return [rec(name, tps, args=tagged(ctx)) for tps, ctx in pairs]


@pytest.fixture
def anydb(tmp_path, monkeypatch):
    file_db = tmp_path / "run_stats.db"
    monkeypatch.setattr(db_mod.db, "db_path", file_db)
    asyncio.run(db.init())
    return file_db


def insert(*records, keep=5000):
    for r in records:
        asyncio.run(db.add_run_stat(r, keep_per_model=keep))


def _raw(name):
    import aiosqlite

    async def fetch():
        async with aiosqlite.connect(db.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cur = await conn.execute(
                "SELECT * FROM model_run_stats WHERE model_name = ? ORDER BY id",
                (name,))
            return [dict(r) for r in await cur.fetchall()]

    return asyncio.run(fetch())


def one(name, recent=10):
    rows = [m for m in asyncio.run(db.get_run_stats(recent=recent))
            if m["model_name"] == name]
    assert len(rows) == 1, rows
    return rows[0]


def view_of(record):
    insert(record)
    return one(record["model_name"])["high"]


# ------------------------------------------------------------------------ 两列入库


def test_argv_stays_verbatim_and_known_flags_are_resolved(anydb):
    """args_json 是逐字原文，cfg_json 是从它解出来的生效配置。"""
    row = view_of(rec("bonsai.gguf", 88.5))

    assert row["args"] == ARGS, "argv 原文被改动 —— 它就不再是'不可能不全'的那份真相了"
    assert row["cfg"]["ctx_size"] == 131072
    assert row["cfg"]["n_gpu_layers"] == 999
    assert row["cfg"]["cache_ram_mb"] == 2048
    assert row["cfg"]["ubatch_size"] == 512
    assert row["cfg"]["build"] == "prism-stable"
    assert row["decode_tps"] == pytest.approx(88.5)


def test_unmapped_flag_survives_as_other_args_and_mapped_one_does_not(anydb):
    """映射表漏项的表现必须是"其他参数里冒一条"，不是"这条配置凭空消失"。

    反向对照在同一条断言里：认得的 flag 不许也跑进 other，否则这张映射表就是摆设，
    页面上一切都堆进"其他参数"、看着照样"全"。
    """
    extra = ["--some-brand-new-knob", "7", "--no-value-flag"]
    other = view_of(rec("bonsai.gguf", 50.0, args=ARGS + extra))["cfg"]["_other_args"]

    assert other == extra, other
    for known in ("-m", "-c", "--spec-type", "--cache-ram"):
        assert known not in other, f"{known} 已被 CFG_FLAGS 认得，不该出现在其他参数里"


def test_mtp_switch_is_derived_from_argv_not_a_separate_flag(anydb):
    """argv 里"有没有 --spec-type"就是 MTP 的开关，页面要能直接答"这次开没开"。"""
    assert view_of(rec("on.gguf", 90.0))["cfg"]["mtp_enabled"] is True
    off = view_of(rec("off.gguf", 60.0, args=ARGS_NO_MTP))["cfg"]
    assert "mtp_enabled" not in off, "没传 spec-type 时该说「这个引擎不传」，不是「关」"
    assert view_of(rec("none.gguf", 60.0,
                       args=[*ARGS[:ARGS.index("--spec-type")],
                             "--spec-type", "none"]))["cfg"]["mtp_enabled"] is False


def test_vllm_and_d3_argv_resolve_to_their_own_keys(anydb):
    """vLLM/d3 的 flag 和 llama 系完全不同：认得的解出来，不认得的必须冒出来。

    位置参数（"serve" <path>、d3 的脚本路径）没有 flag 名，天然落在其他参数里——
    这不是缺陷，是"没有静默消失"的证据。
    """
    vllm = view_of(rec("vllm.gguf", 12.0, args=[
        "serve", "/models/x", "--host", "127.0.0.1", "--port", "8100",
        "--served-model-name", "x", "--gpu-memory-utilization", "0.85",
        "--max-model-len", "32768",
    ]))["cfg"]
    assert vllm["gpu_memory_utilization"] == "0.85"
    assert vllm["max_model_len"] == 32768
    assert "ctx_size" not in vllm and "n_gpu_layers" not in vllm, (
        "llama 的键不该出现在 vLLM 记录里"
    )
    assert vllm["_other_args"] == ["serve", "/models/x", "--served-model-name", "x"]

    d3 = view_of(rec("d3.gguf", 9.0, args=[
        "D:/engines/d3/d3_server.py", "--model", "D:/engines/d3",
        "--host", "127.0.0.1", "--port", "8120", "--name", "d3-mini",
        "--verify", "none",
    ]))["cfg"]
    assert d3["port"] == 8120 and d3["host"] == "127.0.0.1"
    assert d3["_other_args"] == [
        "D:/engines/d3/d3_server.py", "--model", "D:/engines/d3",
        "--name", "d3-mini", "--verify", "none",
    ], d3.get("_other_args")


# ---------------------------------------------------------------- 最高 / 中位 / 最低


def test_high_median_low_are_real_rows_each_with_its_own_config(anydb):
    """中位取排序后中间那条**真实记录**。插值出来的数没有对应配置，放进这张表就是假行。

    ctx 标记故意和速度反着长：这样"配置跟错了行"会立刻让断言变红。
    """
    insert(*series("m.gguf", [(88.0, 1), (104.0, 2), (52.0, 3), (71.0, 4), (90.0, 5)]))
    m = one("m.gguf")

    assert m["count"] == 5 and m["ranked_count"] == 5
    assert m["high"]["decode_tps"] == 104.0 and m["high"]["cfg"]["ctx_size"] == 2
    assert m["low"]["decode_tps"] == 52.0 and m["low"]["cfg"]["ctx_size"] == 3
    assert m["median"]["decode_tps"] == 88.0, "5 条的中位是排序后第 3 条，不是插值"
    assert m["median"]["cfg"]["ctx_size"] == 1


def test_even_count_median_is_still_an_observed_row(anydb):
    """反向对照：偶数条不许取两数平均 —— 4 条时中位是第 2 条本身。"""
    insert(*series("e.gguf", [(50.0, 1), (60.0, 2), (70.0, 3), (80.0, 4)]))
    med = one("e.gguf")["median"]
    assert med["decode_tps"] == 60.0
    assert med["cfg"]["ctx_size"] == 2, "中位行的配置必须和它的速度来自同一条记录"


def test_ranking_ignores_rows_without_a_decode_speed(anydb):
    """只有 prefill、没跑完生成的行不能当"最低速"—— 那等于把缺失当 0 来排名。

    反向对照在同一条断言里：这条记录照样要出现在 count 和最近列表里，
    排除只发生在排名上，不是把行藏起来。
    """
    insert(rec("r.gguf", None, args=tagged(1)), *series("r.gguf", [
        (40.0, 2), (80.0, 3)]))
    m = one("r.gguf")

    assert m["count"] == 3 and m["ranked_count"] == 2
    assert m["low"]["cfg"]["ctx_size"] == 2, "没有 decode 速度的行被当成最低速了"
    assert m["high"]["cfg"]["ctx_size"] == 3
    assert [r["cfg"]["ctx_size"] for r in m["recent"]] == [3, 2, 1], (
        "未排名的行从最近列表里也消失了"
    )


def test_recent_is_newest_first_and_capped(anydb):
    insert(*series("q.gguf", [(10.0 + i, i) for i in range(15)]))
    recent = one("q.gguf", recent=10)["recent"]
    assert len(recent) == 10
    assert [r["cfg"]["ctx_size"] for r in recent] == [14, 13, 12, 11, 10, 9, 8, 7, 6, 5]


# ------------------------------------------------------------------------ 上限裁剪


def test_prune_keeps_the_newest_five_and_spares_another_model(anydb):
    """体积闸门是"每只模型 N 条"：超了删最旧，且不许跨模型。"""
    insert(*series("p.gguf", [(float(i), i) for i in range(8)]), keep=5)
    assert [r["cfg"]["ctx_size"] for r in one("p.gguf")["recent"]] == [7, 6, 5, 4, 3]

    insert(*series("other.gguf", [(1.0, 1), (2.0, 2)]), keep=5)
    assert one("other.gguf")["count"] == 2, (
        "另一只模型写入时被连带裁剪 —— keep 上限是按模型的，不是全表"
    )


def test_nothing_is_pruned_at_exactly_the_cap(anydb):
    """反向对照：6 条、上限 6 —— 一条都不许少。"""
    insert(*series("s.gguf", [(float(i), i) for i in range(6)]), keep=6)
    assert one("s.gguf")["count"] == 6


# ------------------------------------------------------------------ 坏记录不装好


def test_corrupt_json_shows_raw_text_not_an_empty_config():
    """JSON 解不开时返回空 dict 会伪装成"这次没配任何参数"；必须带 _parse_error。"""
    view = _stat_view({"ts": "x", "args_json": "{not json", "cfg_json": None})
    assert view["args"] == {"_parse_error": "{not json"}
    assert view["cfg"] is None, "列本身为空才是 None，两者不能混为一谈"


# ---------------------------------------------------------------------- 路由


def _routes_src():
    return (Path(lc_mod.__file__).parent.parent / "api" / "routes" / "models.py"
            ).read_text(encoding="utf-8")


def test_run_stats_route_is_declared_above_the_int_id_route():
    """/models/{model_id} 收 int；Starlette 先按路径匹配再校验类型。

    run-stats 若声明在它下面，GET /api/models/run-stats 会拿 422 而不是答案，
    GUI 只会显示"还没有运行记录"，看不出是被路由抢走了。
    """
    src = _routes_src()
    for marker in ('@router.get("/models/run-stats")', '@router.get("/models/{model_id}"'):
        assert marker in src, f"路由声明不见了：{marker}"
    assert src.index('@router.get("/models/run-stats")') < \
        src.index('@router.get("/models/{model_id}"')


def test_run_stats_route_returns_rows_and_a_total(anydb):
    """页面要的是"每只模型一行 + 全库共 N 条"，两个数都得真。"""
    from lmloader.api.routes.models import list_run_stats

    insert(*series("m.gguf", [(50.0, 1), (80.0, 2)]))
    insert(rec("n.gguf", 30.0))
    out = asyncio.run(list_run_stats(recent=10))

    assert out["total"] == 3
    assert [m["model_name"] for m in out["models"]] == ["m.gguf", "n.gguf"]
    assert out["models"][0]["high"]["decode_tps"] == 80.0
    assert out["models"][1]["median"]["cfg"]["ctx_size"] == 131072


def test_run_stats_route_clamps_the_recent_window(anydb, monkeypatch):
    """recent 是外部可传的查询参数：不许 0 条窗口，也不许把整表拖回来。

    用间谍直接看传给 db 的那个数 —— 表里只有 6 条的话，200 的上限根本区分不出来，
    断言"返回 6 条"是个假判据。
    """
    from lmloader.api.routes.models import list_run_stats

    seen = []

    async def spy(recent=10):
        seen.append(recent)
        return []

    monkeypatch.setattr(db, "get_run_stats", spy)
    for raw, want in ((0, 1), (-5, 1), (10 ** 9, 200), (10, 10)):
        asyncio.run(list_run_stats(recent=raw))
    assert seen == [1, 1, 200, 10], seen


# ------------------------------------------------------------------ 启动环境快照


def test_startup_context_keeps_gpu_knobs_and_drops_the_rest():
    """GGML_CUDA_* 这类变量不在 argv 里，却会实打实改写内核路径，必须单独留档。"""
    ctx = run_stats.startup_context(
        env={"GGML_CUDA_PDL": "1", "PATH": "C:/Windows", "OMP_NUM_THREADS": "8",
             "LMLOADER_TOKEN": "shhh"},
        co_resident=["zeta.gguf", "alpha.gguf"],
    )
    assert ctx["knobs"] == {"GGML_CUDA_PDL": "1", "OMP_NUM_THREADS": "8"}
    assert "PATH" not in ctx["knobs"] and "LMLOADER_TOKEN" not in ctx["knobs"], (
        "整份环境入库会带走密钥"
    )
    assert ctx["co_resident"] == ["alpha.gguf", "zeta.gguf"]


def test_record_carries_context_under_env(anydb):
    view = view_of(run_stats.build_record(
        model_name="c.gguf", args=ARGS, engine_kind="ternary",
        build="prism-stable", snap={"tg_final": 70.0},
        env={"co_resident": ["other.gguf"]}))
    assert view["env"]["co_resident"] == ["other.gguf"]


# ---------------------------------------------------------------- 采集接线


def _lifecycle_src():
    return Path(lc_mod.__file__).read_text(encoding="utf-8")


def test_stderr_writer_persists_the_record_and_survives_a_db_failure():
    """_read_stderr 必须真的把 _parse_timing 的返回值写进库，而不是丢掉。

    同时写库要单独包 try：整条 reader 外层本来就有 except，少了内层的 try，
    sqlite 一抖只是**提前结束排空**——面板上看不出问题，引擎却会因为管道塞满
    冻在生成中间。
    """
    body = _lifecycle_src().split('if "print_timing" in cleaned:', 1)[1] \
        .split("if self.parse_memory_line", 1)[0]

    assert "rec = self._parse_timing(cleaned)" in body, "返回值又被丢掉了"
    assert "await db.add_run_stat(rec)" in body
    after = body.split("if rec is not None", 1)[1].split("await db.add_run_stat", 1)[0]
    assert "try:" in after, "写库没有独立的 try —— DB 异常会打断 stderr 排空"


def test_run_env_is_snapshotted_before_the_instance_becomes_resident():
    """co_resident 要答"当时卡上还有谁"：读 self.instances 必须早于登记自己，
    而且不许把自己补进去。

    两条各配一条变异锚点（见 _pydiag/run_stats_break.py 的 B5/B6）：快照挪到登记
    之后 -> 位置断言红；co_resident 手写加上自己 -> 参数断言红。
    """
    load = _lifecycle_src().split("async def load_model", 1)[1]
    assert "run_stats.startup_context" in load, "采集点被整块删掉了"
    snap_at = load.index("run_stats.startup_context")
    assert snap_at < load.index("self.instances[model.name] = instance"), (
        "先登记再快照，模型会把自己也算成共存模型"
    )
    arg = re.search(r"co_resident=([^,\n]+)", load[snap_at:snap_at + 400])
    assert arg, load[snap_at:snap_at + 400]
    assert arg.group(1).strip().rstrip(",") == "list(self.instances)", arg.group(1)


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


def _instance(lines, args=None, kind="ternary"):
    from lmloader.engine.lifecycle import ModelInstance
    from lmloader.engine.router import LaunchConfig
    from lmloader.infra.database import ModelRecord

    model = ModelRecord(id=1, name="x.gguf", path="x.gguf", size=1,
                        architecture="qwen35", quantization="Q4_K_M",
                        parameters="27B", metadata={})
    cfg = LaunchConfig(engine_slot="prism-stable", host="127.0.0.1", port=8198,
                       model_path="x.gguf",
                       args=ARGS if args is None else args, kind=kind)
    return ModelInstance(model, _Proc(lines), cfg)


def _quiet_events(monkeypatch, sink):
    from lmloader.infra import events

    for level in ("info", "warning", "error"):
        monkeypatch.setattr(
            events.event_log, level,
            lambda tag, msg, _l=level, **kw: sink.append((_l, msg)))


TIMING_PAIR = [
    b"1.00 I slot print_timing: id  0 | task 0 | prompt eval time =   100.00 ms /"
    b"     2 tokens (   50.00 tokens per second)",
    b"1.00 I slot print_timing: id  0 | task 0 |        eval time =   400.00 ms /"
    b"    40 tokens (  100.00 tokens per second)",
]


def test_a_db_failure_mid_stream_does_not_stop_the_drain(monkeypatch, anydb):
    """端到端反向对照：写库全程抛错，后面的引擎错误行照常到达面板。"""
    seen = []
    _quiet_events(monkeypatch, seen)

    async def boom(rec):
        raise RuntimeError("disk is locked")

    monkeypatch.setattr(db, "add_run_stat", boom)

    inst = _instance([*TIMING_PAIR,
                      b"1.00 E srv  llama_server: exiting due to model loading error"])
    inst.run_env = run_stats.startup_context({"GGML_CUDA_PDL": "1"}, ["y.gguf"])

    asyncio.run(inst._read_stderr())

    assert any("生成 40" in m for _l, m in seen), seen
    assert any("model loading error" in m for _l, m in seen), (
        "写库失败把后面的引擎错误行一起带走了"
    )


# ------------------------------------------------------------------ 页面渲染规则


def test_config_lines_lists_every_known_label_and_marks_what_was_not_passed():
    """已知项一律整列出来，没传的标"—（该引擎不传）"。

    反向对照在同一条断言里：vLLM 那份 argv 里 llama 的旋钮必须逐项标"不传"，
    而 Bonsai 那份传了的必须带真值 —— 只有两边都对，才说明"缺"是引擎的缺，
    不是页面把值弄丢了。
    """
    llama, _ = run_stats.split_args(ARGS)
    llama["engine_kind"], llama["build"] = "ternary", "prism-stable"
    lines = dict((label, text) for _g, label, text in run_stats.config_lines(llama))
    assert len([1 for g, _l, _t in run_stats.config_lines(llama) if g == "known"]) \
        == len(run_stats.CFG_LABELS)
    assert lines["上下文"] == "131072"
    assert lines["prompt 缓存上限"] == "2048"
    assert lines["MTP 投机"] == "开"
    assert lines["DRY"] == run_stats.NOT_PASSED, "没传 dry 系列要显式标不传"

    vllm, _ = run_stats.split_args(["--port", "8100", "--max-model-len", "32768",
                                    "--gpu-memory-utilization", "0.85"])
    vlines = dict((label, text) for _g, label, text in run_stats.config_lines(vllm))
    assert vlines["上下文"] == run_stats.NOT_PASSED
    assert vlines["-ub"] == run_stats.NOT_PASSED
    assert vlines["GPU 层数"] == run_stats.NOT_PASSED


def test_config_lines_keeps_unlabelled_keys_and_other_args_apart():
    """解出来但没标签的键走 extra，argv 里没被认出的 token 走 other。

    两组分开的意义：extra 是"映射表知道这个键、只是页面还没起中文名"，
    other 是"映射表压根不认识"。后者才需要人去补 CFG_FLAGS。
    """
    cfg = {"ctx_size": 4096, "brand_new_resolved_key": "x"}
    lines = run_stats.config_lines({**cfg, "_other_args": ["--mystery", "3"]})
    groups = {label: group for group, label, _t in lines}
    assert groups["brand_new_resolved_key"] == "extra"
    assert groups["其他参数"] == "other"
    text = dict((label, t) for _g, label, t in lines)
    assert text["其他参数"] == "--mystery 3"


def test_config_lines_of_a_corrupt_record_is_one_error_line_not_a_blank_card():
    lines = run_stats.config_lines({"_parse_error": "{not json"})
    assert lines == [("error", "配置解析失败（下面是原文）", "{not json")]
    assert run_stats.config_lines(None)[0][0] == "error"


def test_env_lines_distinguish_none_resident_from_unknown():
    """"只有它自己"和"这条记录没记环境"必须写成两种话。

    共存模型是实测过的头号掉速根因，把"没记到"读成"当时独占显存"会直接误导结论。
    """
    as_map = lambda lines: {label: text for _g, label, text in lines}
    solo = as_map(run_stats.env_lines({"co_resident": []}))
    unknown = as_map(run_stats.env_lines({}))
    assert solo["同卡常驻模型"] == "只有它自己"
    assert "同卡常驻模型" not in unknown
    assert run_stats.env_lines({"knobs": {}})[0][1] == "性能环境变量"
    assert as_map(run_stats.env_lines({"knobs": {}}))["性能环境变量"] == "无"


def test_format_mem_and_table_row_never_write_zero_for_a_missing_number():
    assert run_stats.format_mem({
        "vram": {"weights": 7291.5, "kv": 2304.0}, "ram": {}}) == "显存 权重 7292 + KV 2304 MiB"
    assert run_stats.format_mem({}) == run_stats.NOT_PASSED

    row = run_stats.table_row({"ts": "t", "decode_tps": None, "prefill_tps": 900.0,
                               "cfg": {"ctx_size": 4096}, "env": {}})
    assert len(row) == len(run_stats.TABLE_COLUMNS)
    assert row[run_stats.TABLE_COLUMNS.index("生成 t/s")] == "没测到"
    assert row[run_stats.TABLE_COLUMNS.index("提示 t/s")] == "900.0"
    assert row[run_stats.TABLE_COLUMNS.index("-ub")] == run_stats.NOT_PASSED
    assert row[run_stats.TABLE_COLUMNS.index("同卡常驻")] == "没记到", (
        "这条记录压根没记环境，不能写成「这个引擎不传」"
    )
    filled = run_stats.table_row({"ts": "t", "decode_tps": 1.0, "cfg": {},
                                  "env": {"co_resident": []}})
    assert filled[run_stats.TABLE_COLUMNS.index("同卡常驻")] == "只有它自己"


# ------------------------------------------------- 实时判别量（占用 / 分账 / 塌陷）


CACHE_LINE = ("1.30.100.001 I srv        update:  - cache state: 2 prompts,"
              " 536.515 MiB (limits: 8192.000 MiB, 16128 tokens, 136629 est)")


def test_cache_state_line_gives_occupancy_not_the_configured_limit():
    """这一行是"当下占了多少"，跟 --cache-ram 设的上限必须分开存。

    反向对照：把 limit_mib 当占用用（2026-10-21 那次塌陷就是 4754 占 8192 的上限，
    只看配置会以为没事）时，mib/limit_mib 两条断言各自报红。
    """
    inst = _instance([])
    assert inst.cache_state is None, "没解析过就该是 None，不能凭空造 0"
    assert inst.parse_cache_state_line("1.0 I llama_kv_cache: nothing here") is False
    assert inst.cache_state is None

    assert inst.parse_cache_state_line(CACHE_LINE) is True
    cs = inst.cache_state
    assert cs["prompts"] == 2 and cs["mib"] == pytest.approx(536.515)
    assert cs["limit_mib"] == pytest.approx(8192.0)
    assert cs["limit_tokens"] == 16128 and cs["est"] == 136629


def test_cache_state_parse_tolerates_builds_without_the_est_field():
    """老引擎不打 ", N est"。整条判据不能因为尾部少一项就整体解析失败。"""
    inst = _instance([])
    older = CACHE_LINE.replace(", 136629 est", "")
    assert inst.parse_cache_state_line(older) is True
    assert inst.cache_state["est"] is None
    assert inst.cache_state["prompts"] == 2


def test_occupancy_of_the_last_update_before_the_record_is_the_one_stored(monkeypatch, anydb):
    """整份真 stderr 过真 reader：占用取"这条生成收尾前最后一次"，不是第一次。

    fixture 里先打 0 条 / 0 MiB、任务开始后打 2 条 / 536.515 MiB。只记第一次的实现
    （`if self.cache_state is None`）会存成 0 条，看着像缓存真的空 —— 这条就为它而红。
    """
    _quiet_events(monkeypatch, [])
    fixture = Path(__file__).parent / "fixtures" / "llama_stderr_v4.log"
    lines = [ln.encode("utf-8") for ln in fixture.read_text(encoding="utf-8").splitlines()]

    asyncio.run(_instance(lines)._read_stderr())

    rows = _raw("x.gguf")
    assert len(rows) == 1, rows
    view = _stat_view(rows[0])
    cache = view["env"]["cache"]
    assert cache["prompts"] == 2 and cache["mib"] == pytest.approx(536.515), cache
    assert cache["limit_mib"] == pytest.approx(8192.0)
    assert view["cfg"]["ctx_size"] == 131072, "同一条记录还得带着当时的配置"


def test_cache_state_reaches_the_loaded_model_payload():
    """实时区（/models/loaded）也要能读到占用，不然正在跑的这一轮没人能看。"""
    inst = _instance([])
    inst.parse_cache_state_line(CACHE_LINE)
    payload = inst.to_dict()
    assert payload["cache_state"]["mib"] == pytest.approx(536.515)
    assert "memory" in payload, "分账是同一行的另一半，不许被挪走"


def test_table_row_separates_zero_occupancy_from_never_reported():
    """0 条 / 0 MiB 是"缓存真的是空的"，NOT_PASSED 是"这一轮没证据"。"""
    idx = lambda key: run_stats.TABLE_COLUMNS.index(key)
    empty = {"ts": "t", "cfg": {}, "env": {}}
    assert run_stats.table_row(empty)[idx("缓存占用")] == run_stats.NOT_PASSED
    assert run_stats.table_row(empty)[idx("权重 显存/内存")] == run_stats.NOT_PASSED
    # 分账报到了 KV 却没报到权重 = 这一项没证据，不等于"权重 0 MiB 上卡"。
    partial = {"ts": "t", "cfg": {}, "env": {"mem": {"vram": {"kv": 96.0}, "ram": {}}}}
    assert run_stats.table_row(partial)[idx("权重 显存/内存")] == run_stats.NOT_PASSED, (
        "没报的分账不许折成 0 / 0，那会被读成全量上卡"
    )

    zero = {"ts": "t", "cfg": {}, "env": {"cache": {
        "prompts": 0, "mib": 0.0, "limit_mib": 8192.0, "limit_tokens": 16128, "est": None}}}
    assert run_stats.table_row(zero)[idx("缓存占用")] == "0 条 / 0 MiB（上限 8192 MiB, 16128 tok）"

    full = {"ts": "t", "cfg": {}, "env": {"cache": {
        "prompts": 6, "mib": 4754.323, "limit_mib": 8192.0, "limit_tokens": 16128, "est": 1}}}
    assert run_stats.table_row(full)[idx("缓存占用")] == "6 条 / 4754 MiB（上限 8192 MiB, 16128 tok）"

    # 全量上卡 = 内存权重 0；这一列把 2026-10-21 那个 2×2 的两个变量并到同一行里。
    weights = {"ts": "t", "cfg": {}, "env": {"mem": {
        "vram": {"weights": 14549.4}, "ram": {"weights": 0.0}}}}
    assert run_stats.table_row(weights)[idx("权重 显存/内存")] == "14549 / 0"
    part = {"ts": "t", "cfg": {}, "env": {"mem": {
        "vram": {"weights": 11613.0}, "ram": {"weights": 2936.0}}}}
    assert run_stats.table_row(part)[idx("权重 显存/内存")] == "11613 / 2936"


def test_collapse_marker_needs_both_a_slow_rate_and_a_long_generation():
    """崩 = decode<10 且生成≥20，与离线判据同口径；短轮次的低速度不是塌陷样本。"""
    idx = run_stats.TABLE_COLUMNS.index("生成 t/s")
    cell = lambda **kw: run_stats.table_row({"ts": "t", "cfg": {}, "env": {}, **kw})[idx]
    assert cell(decode_tps=5.4, gen_tokens=927) == "5.4 崩"
    assert cell(decode_tps=57.6, gen_tokens=927) == "57.6"
    assert cell(decode_tps=5.4, gen_tokens=3) == "5.4", "生成 3 个 token 不算塌陷"
    assert cell(decode_tps=5.4) == "5.4", "没有生成长度就没资格判崩"
    assert cell(decode_tps=None, gen_tokens=927) == "没测到"


def test_columns_have_a_getter_and_a_mono_rule_and_no_stale_names():
    """列名、table_row 的取值、等宽规则三者必须同时对得上。

    改列名时最容易漏的是等宽集合：它还留着旧名，那一列就静默换回正文字体。
    """
    cols = run_stats.TABLE_COLUMNS
    assert len(cols) == len(set(cols)), "重名列会让取值只命中其中一个"
    assert run_stats.MONO_COLS <= set(cols), (
        f"等宽集合里有已改名的列：{sorted(run_stats.MONO_COLS - set(cols))}")
    probe = {"ts": "t", "decode_tps": 50.0, "prefill_tps": 900.0, "cfg": {}, "env": {}}
    assert len(run_stats.table_row(probe)) == len(cols)
    for name in cols:
        assert run_stats.table_row(probe)[cols.index(name)]


def test_env_lines_label_the_occupancy_separately_from_the_limit():
    as_map = lambda lines: {label: text for _g, label, text in lines}
    with_cache = as_map(run_stats.env_lines(
        {"cache": {"prompts": 2, "mib": 536.515, "limit_mib": 8192.0,
                   "limit_tokens": 16128, "est": None}}))
    assert with_cache["prompt 缓存占用"] == "2 条 / 537 MiB（上限 8192 MiB, 16128 tok）"
    assert as_map(run_stats.env_lines({}))["prompt 缓存占用"] == run_stats.NOT_PASSED


# ------------------------------------------------- 加载前的机器空闲量（第三类塌陷根因）

HOST = {"free_ram_mb": 921.6, "free_vram_mb": 136.0}


def test_format_host_keeps_zero_free_apart_from_never_measured():
    """空闲 0.0G 是"这台机器真没地方放权重了"的确诊；没测到只能写没报。

    2026-10-22 那次 -ngl 999 ctx 16128 报 bad allocation 时显存还剩 136 MiB、
    系统内存只剩 0.9G —— 只看显存会把主机内存不足误判成"卡满了"。
    """
    both = {"free_ram_mb": 17715.2, "free_vram_mb": 15467.0}
    assert run_stats.format_host(both) == "内存 17.3G / 显存 15.1G"
    assert run_stats.format_host(HOST) == "内存 0.9G / 显存 0.1G"
    assert run_stats.format_host({"free_ram_mb": 0.0, "free_vram_mb": 1024.0}) == (
        "内存 0.0G / 显存 1.0G"), "0 空闲必须原样写出来，那是结论不是缺数据"
    assert run_stats.format_host({"free_ram_mb": None,
                                 "free_vram_mb": 1024.0}) == "显存 1.0G", "一侧没测到就只写另一侧"
    assert run_stats.format_host(None) == run_stats.NOT_PASSED
    assert run_stats.format_host({}) == run_stats.NOT_PASSED
    assert run_stats.format_host({"free_ram_mb": None,
                                 "free_vram_mb": None}) == run_stats.NOT_PASSED


def test_the_host_column_and_the_card_read_the_same_numbers():
    idx = run_stats.TABLE_COLUMNS.index("机器空闲 前→后")
    assert run_stats.table_row({"ts": "t", "cfg": {}, "env": {}})[idx] == (
        run_stats.NOT_PASSED)
    row = {"ts": "t", "cfg": {}, "env": {"host": HOST}}
    assert run_stats.table_row(row)[idx] == "内存 0.9G / 显存 0.1G"
    labels = {label: text for _g, label, text in run_stats.env_lines(row["env"])}
    assert labels["机器空闲 加载前→后"] == "内存 0.9G / 显存 0.1G"


def test_host_shows_what_this_load_ate_when_both_moments_are_measured():
    """前→后是同一条快照里的两个时刻：只有加载前的数字说不出这次加载吃掉了多少。

    2026-10-22 那次 -ngl 999 就是 17.3G 可用 → 加载完只剩 0.9G，第一次请求才 bad allocation；
    只有右半边（事后才补测）时也不许把左边编成"?"以外的大数。
    """
    arrow = run_stats.host_with_after(
        {"free_ram_mb": 17715.2, "free_vram_mb": 15467.0},
        {"free_ram_mb": 921.6, "free_vram_mb": 136.0})
    assert run_stats.format_host(arrow) == "内存 17.3→0.9G / 显存 15.1→0.1G"
    # 加载后再测失败：原样保留加载前那份，宁可少一个时刻也不编数。
    assert run_stats.host_with_after(HOST, None) == HOST
    assert run_stats.host_with_after(HOST, "junk") == HOST
    # 加载前没测到、加载后测到了：左边写 ? 而不是 0。
    only_after = run_stats.host_with_after(None, {"free_ram_mb": 921.6})
    assert run_stats.format_host(only_after) == "内存 ?→0.9G"
    assert run_stats.host_with_after(None, None) is None


def test_the_after_load_reading_lands_between_spawn_and_registration():
    """两次探测的先后决定这一列能不能读：早于 spawn 的是"加载前"，晚于加载完成的是"后"。

    合并点若挪到 self.instances 登记之后（变异锚点 B31），日志页上"模型已加载"那行
    就没有机器空闲了；挪到 spawn 之前则两个时刻变成同一个数，箭头永远显示 0 消耗。
    """
    load = _lifecycle_src().split("async def load_model", 1)[1]
    merge_at = load.index("run_stats.host_with_after")
    assert merge_at > load.index("create_subprocess_exec"), "后一次探测早于 spawn，两个时刻会相等"
    assert merge_at < load.index("self.instances[model.name] = instance"), (
        "合并晚于登记，加载完成的日志行读不到这一列")


def test_host_free_is_measured_before_the_engine_gets_to_allocate():
    """这一列答的是"这次加载之前还剩多少"，所以探测必须早于 spawn。

    挪到 spawn 之后（变异锚点 B25）读到的是引擎自己刚吃过的量：那行日志会稳定显示
    "内存 0.xG / 显存 0.xG"，把判别量变成噪声，而且再也不会区分"本来就满"和"被自己吃满"。
    """
    load = _lifecycle_src().split("async def load_model", 1)[1]
    probe_at = load.index("_host_free_snapshot()")
    assert probe_at < load.index("create_subprocess_exec"), (
        "探测晚于 spawn —— 引擎已经开始分配，读到的不再是加载前的空闲量")
    at = load.index("run_stats.startup_context")
    assert re.search(r"host=host_free", load[at:at + 300]), load[at:at + 300]


def test_a_broken_or_hanging_host_probe_cannot_fail_a_load(monkeypatch):
    """探测挂了/卡住 -> 这一项写成没报，加载照常进行。

    反向对照：去掉 wait_for 或去掉 try（变异锚点 B26）时，这两条分别因为"卡住 4s"
    和"异常上抛"报红 —— 探测失败绝不能变成加载失败。
    """
    from lmloader.engine import offload_planner

    def raiser():
        raise RuntimeError("psutil missing")

    monkeypatch.setattr(offload_planner, "free_ram_mb", raiser)
    monkeypatch.setattr(offload_planner, "free_vram_mb", lambda: 15109.0)
    assert asyncio.run(lc_mod._host_free_snapshot()) is None

    monkeypatch.setattr(offload_planner, "free_ram_mb", lambda: 17715.2)
    monkeypatch.setattr(lc_mod, "_HOST_PROBE_TIMEOUT", 0.2)
    monkeypatch.setattr(offload_planner, "free_vram_mb", lambda: time.sleep(4.0))

    async def timed():
        # 计时取 await 自己那一段：asyncio.run 收尾时会 join 卡住的线程池线程，
        # 那是测试框架的事（真后端的 loop 一直在跑，加载不会被它拖住）。
        started = time.time()
        out = await lc_mod._host_free_snapshot()
        return out, time.time() - started

    out, waited = asyncio.run(timed())
    assert out is None
    assert waited < 2.0, f"探测超时没生效，加载被观测数据拖住了 {waited:.1f}s"


def test_host_free_reaches_the_loaded_model_payload():
    """实时区看的是 /models/loaded：没测到要写 None，不许折成 0。"""
    inst = _instance([])
    inst.run_env = run_stats.startup_context({}, [], HOST)
    assert inst.to_dict()["host_free"]["free_ram_mb"] == 921.6
    assert _instance([]).to_dict()["host_free"] is None


def test_a_record_stores_the_host_snapshot_it_was_loaded_with(anydb):
    """端到端：加载时的空闲量进 env_json，页面从同一条记录读出来。"""
    view = view_of(run_stats.build_record(
        model_name="c.gguf", args=ARGS, engine_kind="ternary", build="prism-stable",
        snap={"tg_final": 4.0, "gen_tokens": 900},
        env=run_stats.startup_context({}, [], HOST)))
    assert view["env"]["host"] == HOST
    assert run_stats.table_row(view)[run_stats.TABLE_COLUMNS.index("机器空闲 前→后")] == (
        "内存 0.9G / 显存 0.1G")


# ------------------------------------------------------------------ GUI 接线


def _page_src():
    return (Path(lc_mod.__file__).parent.parent / "gui" / "pages" / "run_stats.py"
            ).read_text(encoding="utf-8")


def _window_src():
    return (Path(lc_mod.__file__).parent.parent / "gui" / "window.py"
            ).read_text(encoding="utf-8")


def test_nav_lists_the_archive_and_pages_are_registered_in_the_same_order():
    """侧栏第 7 项 = 栈第 7 页：两处顺序不一致就会点进错误的页面。"""
    src = _window_src()
    nav = src.split("nav_items = [", 1)[1].split("]", 1)[0]
    assert '"run_stats"' in nav, "侧栏没有这一项"
    assert nav.count("(") == 7, nav
    order = re.findall(r'"nav\.([a-z_]+)"', nav)
    assert order == ["chat", "models", "download", "monitor", "logs", "settings",
                     "run_stats"], order
    pages = re.findall(r'self\.pages\["([a-z_]+)"\]', src)
    assert pages == order, (pages, order)


def test_settings_icon_is_no_longer_the_fallback_branch():
    """设置页的齿轮原先写在 else 里；加第 7 页后 else 会被档案页共用同一个图标。

    反向对照：新增页必须有自己的分支，else 只剩它自己。
    """
    src = _window_src()
    body = src.split("def _make_nav_icon", 1)[1].split("def _init_pages", 1)[0]
    assert "elif index == 5:" in body, "齿轮还挂在 else 上 —— 两页会同图标"
    assert body.count("else:") == 1


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_every_string_the_page_uses_exists_in_both_languages(lang):
    """页面和路由用到的每个 tr() 键必须在 zh 和 en 表里都有。

    缺一个键时 tr 会把键名原样印到界面上（"runstats.unreachable"），中文界面看着
    像功能坏了 —— 所以两条语言线各自判一次，而不是只判当前语言。
    """
    from lmloader.gui import i18n

    keys = set(re.findall(r'tr\("([a-z_]+\.[a-z_]+)"', _page_src()))
    keys |= set(re.findall(r'\("(runstats\.[a-z_]+|nav\.run_stats)"', _page_src()))
    keys.add("nav.run_stats")  # 侧栏标签在 window.py 里查，这里一起判
    assert keys, "页面里一个 tr() 都没有"
    table = i18n.TRANSLATIONS[lang]
    missing = sorted(k for k in keys if k not in table)
    assert not missing, f"{lang} 缺 {missing}"


def test_page_reads_off_thread_and_stops_polling_when_left():
    """3 秒轮询要在离开页面时停，取数要在别的线程。

    另外"后端没响应"必须写成读不到，不许折成"还没有运行记录" —— 这是假绿灯。
    """
    src = _page_src()
    assert "run_in_thread(fetch, on_result, on_error)" in src
    assert "self._timer.stop()" in src.split("def on_leave", 1)[1].split("def ", 1)[0]
    assert "self._timer.start()" in src.split("def on_navigate", 1)[1].split("def ", 1)[0]
    unreachable = src.split("if stats is None:", 1)[1].split("def ", 1)[0]
    assert 'tr("runstats.unreachable")' in unreachable
    assert 'tr("runstats.empty")' not in unreachable, (
        "查不到被折成了「还没有记录」—— 后端挂了界面看不出来"
    )


def test_api_client_distinguishes_failure_from_empty():
    """客户端把 GET 失败折成空 dict，页面就没有三态可分。"""
    from lmloader.gui import api_client

    src = Path(api_client.__file__).read_text(encoding="utf-8")
    body = src.split("def fetch_run_stats_or_none", 1)[1].split("\n    def ", 1)[0]
    assert "-> dict | None" in body
    assert "except Exception:\n            return None" in body, (
        "失败必须回 None，不能回 {}"
    )
    assert "return []" not in body


def test_one_generation_writes_exactly_one_row_end_to_end(monkeypatch, anydb):
    """真日志过真 reader：一次生成 = 一行档案，prompt 行不单独成行。

    计数恰好为 1 是这条的全部意义：0 = 接线没通，2 = 把同一次运行拆成两行，
    那"最高/最低"就会被一次运行虚占两席。
    """
    _quiet_events(monkeypatch, [])

    fixture = Path(__file__).parent / "fixtures" / "llama_stderr_v4.log"
    lines = fixture.read_text(encoding="utf-8").splitlines()
    timing = [ln.encode("utf-8") for ln in lines
              if re.search(r"print_timing:.*(prompt eval time|eval time)", ln)]
    assert len(timing) == 2, timing

    asyncio.run(_instance(timing)._read_stderr())

    rows = _raw("x.gguf")
    assert len(rows) == 1, rows
    view = _stat_view(rows[0])
    assert view["decode_tps"] == pytest.approx(12.45)
    assert view["prefill_tps"] == pytest.approx(13.97)
    assert view["prompt_tokens"] == 3 and view["gen_tokens"] == 40
    assert view["cfg"]["ctx_size"] == 131072
    assert "co_resident" not in view["env"], (
        "没经过 load_model 就不该凭空冒出共存模型；只有 mem 是实例自己算的"
    )
    assert "mem" in view["env"]


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_expanding_a_row_builds_real_widgets_not_source_strings(qapp, anydb):
    """展开一行必须真的建出控件 —— 只读源码断言的话，缺 import 也照样"绿"。

    这条不是洁癖：`QHBoxLayout` 漏进 import 列表时，源码里那行字在、卡片文字也在，
    只有真点一下展开才 NameError。
    """
    from PySide6.QtWidgets import QLabel, QPlainTextEdit, QPushButton

    from lmloader.gui.pages.run_stats import RunStatsPage, _ConfigCard

    insert(
        *series("bonsai.gguf", [(88.0, 1), (104.0, 2), (52.0, 3)]),
        run_stats.build_record(
            model_name="bonsai.gguf",
            args=ARGS + ["--brand-new-knob", "7"],
            engine_kind="ternary", build="prism-stable",
            snap={"tg_final": 30.0},
            env={"knobs": {"GGML_CUDA_PDL": "1"}, "co_resident": ["other.gguf"]},
        ),
    )
    stats = asyncio.run(db.get_run_stats(recent=10))
    assert stats, "库里应该有记录"

    page = RunStatsPage(object())
    page._render_archive({"models": stats, "total": sum(r["count"] for r in stats)})
    qapp.processEvents()

    fold = page.findChild(QPushButton)
    assert fold is not None, "模型行没画出展开按钮"
    assert page.findChildren(_ConfigCard) == [], "收起时不该建卡片"

    fold.click()
    qapp.processEvents()

    assert len(page.findChildren(_ConfigCard)) == 3, "展开后应有高/中/低三张卡"

    # argv 不许被剪进滚动条后面：这一栏的全部意义就是原文一个字都不少。
    page.resize(1280, 1020)
    page.show()
    qapp.processEvents()
    box = page.findChild(QPlainTextEdit)
    assert box.verticalScrollBar().maximum() == 0, (
        f"argv 框 {box.height()}px 装不下 {len(box.toPlainText())} 字，尾部被藏起来了"
    )

    texts = " ".join(box.toPlainText() for box in page.findChildren(QPlainTextEdit))
    assert "-m" in texts and "Bonsai" in texts, texts
    joined = " ".join(label.text() for label in page.findChildren(QLabel))
    assert run_stats.NOT_PASSED in joined, (
        "该引擎不传的旋钮必须显式标出来，不能凭空消失"
    )
    assert "--brand-new-knob 7" in joined, "映射表漏的 flag 必须冒到「其他参数」里"
    assert "GGML_CUDA_PDL=1" in joined
    assert "other.gguf" in joined


def test_collapsing_and_re_expanding_does_not_leak_widgets(qapp, anydb):
    """收起再展开不许往同一个 body 上叠第二套布局。

    Qt6 对"给已有布局的 widget 再装一套"既不警告也不报错：新布局直接不生效，
    往里加的控件变成没有父项的顶层窗口留在内存里。所以数卡片查不出来（还是 3 张），
    只有顶层窗口计数会涨 —— 反复展开收起十次就是几十个隐形窗口。
    """
    import gc

    from PySide6.QtWidgets import QPushButton

    from lmloader.gui.pages.run_stats import RunStatsPage, _ConfigCard

    insert(*series("m.gguf", [(10.0, 1), (20.0, 2), (30.0, 3)]))
    rows = asyncio.run(db.get_run_stats(recent=10))

    page = RunStatsPage(object())
    page._render_archive({"models": rows, "total": sum(r["count"] for r in rows)})
    fold = page.findChild(QPushButton)

    fold.click()
    qapp.processEvents()
    baseline = len(qapp.topLevelWidgets())
    cards = len(page.findChildren(_ConfigCard))

    for _ in range(4):
        fold.click()
        fold.click()
    gc.collect()
    qapp.processEvents()

    assert len(page.findChildren(_ConfigCard)) == cards
    assert len(qapp.topLevelWidgets()) == baseline, (
        f"重复展开漏出 {len(qapp.topLevelWidgets()) - baseline} 个无父项的顶层控件"
    )


def test_live_area_shows_the_occupancy_reading_of_the_running_model(qapp):
    """实时区必须把"占了多少"和"权重分账"当场打出来 —— 塌陷是在跑的时候发生的，
    等它进档案页已经晚了。缺这段拼接时（变异锚点 B21）这一条报红。
    """
    from PySide6.QtWidgets import QLabel

    from lmloader.gui.pages.run_stats import RunStatsPage

    page = RunStatsPage(object())
    page._render_live([{
        "model": {"name": "Qwen3.8-27B-NVFP4-MTP-LOW.gguf"},
        "metrics": {"tg_final": 5.4, "gen_tokens": 927},
        "cache_state": {"prompts": 6, "mib": 4754.323, "limit_mib": 8192.0,
                        "limit_tokens": 16128, "est": 1},
        "memory": {"vram": {"weights": 14549.4}, "ram": {"weights": 0.0}},
        "host_free": HOST,
    }, {
        "model": {"name": "quiet.gguf"},
        "metrics": {},
        "cache_state": None,
        "memory": None,
        "host_free": None,
    }])
    qapp.processEvents()
    joined = " ".join(label.text() for label in page.findChildren(QLabel))
    assert "缓存 6 条 / 4754 MiB（上限 8192 MiB, 16128 tok）" in joined, joined
    assert "权重 14549 / 0" in joined, joined
    assert "空闲 内存 0.9G / 显存 0.1G" in joined, joined
    # 没报的那只不许冒出 0
    assert joined.count("缓存 0 条") == 0, "没解析到过就要写没报，不能显示成空缓存"
    assert joined.count("内存 0.0G") == 0, "没测到的空闲量不能显示成 0，那会被读成已确诊"
