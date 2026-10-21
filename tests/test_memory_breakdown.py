"""The engine-reported weights/KV memory split parsed from llama.cpp stderr.

Fixture lines are verbatim from a real d3-master run of d3-mini-Q4_K_M with
--n-gpu-layers 12, so both the CPU and the CUDA side of every buffer class are
present. The expected totals are cross-checked against the engine's own
common_memory_breakdown_print table from that same run.
"""

import types

import pytest

from lmloader.engine.lifecycle import ModelInstance as M

# Captured stderr allocation report (the "-lv 4" lines), in the order printed.
ENGINE_LOG = [
    "0.01.186.062 I load_tensors:   CPU_Mapped model buffer size =  2009.38 MiB",
    "0.01.186.063 I load_tensors:        CUDA0 model buffer size =  1515.92 MiB",
    "0.01.740.751 I llama_context:  CUDA_Host  output buffer size =     3.79 MiB",
    "0.01.740.904 I llama_kv_cache:        CPU KV buffer size =   160.00 MiB",
    "0.01.753.695 I llama_kv_cache:      CUDA0 KV buffer size =    96.00 MiB",
    "0.01.767.245 I llama_memory_recurrent:        CPU RS buffer size =   134.00 MiB",
    "0.01.769.079 I llama_memory_recurrent:      CUDA0 RS buffer size =    67.00 MiB",
    "0.01.777.157 I sched_reserve:      CUDA0 compute buffer size =   102.66 MiB",
    "0.01.777.160 I sched_reserve:  CUDA_Host compute buffer size =    34.40 MiB",
]


def _instance():
    """A ModelInstance with only the state the parser touches."""
    inst = M.__new__(M)
    inst.memory_buffers = {}
    return inst


def _parse(lines):
    inst = _instance()
    for line in lines:
        inst.parse_memory_line(line)
    return inst


def test_every_allocation_line_is_recognised():
    assert sum(1 for ln in ENGINE_LOG if _instance().parse_memory_line(ln)) == len(ENGINE_LOG)


def test_vram_and_ram_split_weights_from_kv():
    bd = _parse(ENGINE_LOG).memory_breakdown()
    assert bd["vram"] == {
        "weights": 1515.92, "kv": 96.0, "recurrent": 67.0, "compute": 102.66,
    }
    assert bd["ram"] == {
        "weights": 2009.38, "kv": 160.0, "recurrent": 134.0, "compute": 34.4,
        "output": 3.79,
    }


def test_totals_match_the_engines_own_summary_table():
    """common_memory_breakdown_print reported CUDA0 self=1781, Host self=2337."""
    bd = _parse(ENGINE_LOG).memory_breakdown()
    assert round(bd["total_mib"]["vram"]) == 1782  # engine rounds to 1781
    assert round(bd["total_mib"]["ram"]) == 2342


def test_pinned_host_buffers_are_not_counted_as_vram():
    """CUDA_Host is page-locked system RAM despite the CUDA prefix."""
    assert M._device_bucket("CUDA_Host") == "ram"
    assert M._device_bucket("CUDA0") == "vram"
    assert M._device_bucket("CPU_Mapped") == "ram"
    assert M._device_bucket("Vulkan0") == "vram"


def test_repeated_reserve_overwrites_instead_of_accumulating():
    """sched_reserve reprints the compute buffers on every graph re-reserve."""
    inst = _parse(ENGINE_LOG)
    before = inst.memory_breakdown()["vram"]["compute"]
    inst.parse_memory_line(
        "9.99.999.999 I sched_reserve:      CUDA0 compute buffer size =   102.66 MiB"
    )
    assert inst.memory_breakdown()["vram"]["compute"] == before
    assert len(inst.memory_buffers[("CUDA0", "compute")]) == 1


# Verbatim from a real production-args load (`-lv 4`, --mmproj mounted,
# --spec-type draft-mtp --spec-draft-n-max 2) of Qwen3.8-27B-NVFP4-MTP-LOW at
# ctx 65536 / -ngl 60; the engine reached "model loaded" and the free-VRAM delta
# for that process was measured at 15293 MiB. This is the run that proves one
# (device, kind) can hold two real buffers: llama_kv_cache prints CUDA0 KV once
# for the main context (q4_0) and once for the MTP draft context (f16), and the
# draft's 256 MiB alone cannot explain 15293 MiB of occupancy.
MTP_LOG = [
    "0.02.158.570 I load_tensors:   CPU_Mapped model buffer size =  1876.76 MiB",
    "0.02.158.571 I load_tensors:        CUDA0 model buffer size = 12927.69 MiB",
    "0.06.291.140 I llama_context:  CUDA_Host  output buffer size =     0.95 MiB",
    "0.06.291.661 I llama_kv_cache:        CPU KV buffer size =    72.00 MiB",
    "0.06.298.124 I llama_kv_cache:      CUDA0 KV buffer size =  1080.00 MiB",
    "0.06.317.810 I llama_memory_recurrent:        CPU RS buffer size =    46.76 MiB",
    "0.06.325.536 I llama_memory_recurrent:      CUDA0 RS buffer size =   402.12 MiB",
    "0.06.347.015 I sched_reserve:      CUDA0 compute buffer size =   501.75 MiB",
    "0.06.347.021 I sched_reserve:  CUDA_Host compute buffer size =   144.63 MiB",
    "0.06.717.389 I llama_context:  CUDA_Host  output buffer size =     0.95 MiB",
    "0.06.718.414 I llama_kv_cache:      CUDA0 KV buffer size =   256.00 MiB",
    "0.06.737.947 I sched_reserve:      CUDA0 compute buffer size =   132.02 MiB",
    "0.06.737.952 I sched_reserve:  CUDA_Host compute buffer size =    84.02 MiB",
    "0.07.442.884 I reserve_compute_meta:      CUDA0 compute buffer size =   248.10 MiB",
    "0.07.442.888 I reserve_compute_meta:        CPU compute buffer size =    24.93 MiB",
    "0.07.474.433 I sched_reserve:      CUDA0 compute buffer size =   132.02 MiB",
    "0.07.474.438 I sched_reserve:  CUDA_Host compute buffer size =    84.02 MiB",
]


def test_main_and_draft_kv_are_two_buffers_not_one():
    """The collapsed summary used to report the draft's 256 MiB as the whole KV."""
    bd = _parse(MTP_LOG).memory_breakdown()
    assert bd["vram"]["kv"] == 1080.00 + 256.00
    assert bd["vram"]["weights"] == 12927.69
    assert bd["vram"]["recurrent"] == 402.12
    # 132.02 was printed twice by the same function: reprint, so 501.75+132.02
    # from sched_reserve plus the 248.10 meta reservation, not 132.02 twice.
    assert bd["vram"]["compute"] == pytest.approx(501.75 + 132.02 + 248.10)
    assert bd["ram"]["kv"] == 72.00
    assert bd["ram"]["compute"] == pytest.approx(144.63 + 84.02 + 24.93)
    # the draft context shares the main one's output buffer size
    assert bd["ram"]["output"] == 0.95


def test_collapsing_to_the_last_line_under_reports_kv_5_times():
    """Negative control: last-write-wins must not be mistaken for the split.

    Re-implementing the old rule over the same lines has to produce the number
    the GUI used to show, so a future "simplification" back to one value per
    (device, kind) fails here instead of silently shrinking the account.
    """
    collapsed = {}
    inst = _parse(MTP_LOG)
    for line in MTP_LOG:
        m = M._RE_MEMBUF.search(line)
        collapsed[(m.group("device"), m.group("kind"))] = float(m.group("mib"))
    old_kv = collapsed[("CUDA0", "KV")]
    assert old_kv == 256.0
    assert inst.memory_breakdown()["vram"]["kv"] != old_kv
    assert inst.memory_breakdown()["vram"]["kv"] / old_kv == pytest.approx(5.22, abs=.01)


def test_second_buffer_of_a_different_size_is_not_a_reprint():
    """The pair of tests that separates "sum" from "keep the last" for good."""
    inst = _parse(ENGINE_LOG)
    inst.parse_memory_line(
        "9.99.999.999 I llama_kv_cache:      CUDA0 KV buffer size =   256.00 MiB"
    )
    assert inst.memory_breakdown()["vram"]["kv"] == 96.0 + 256.0


@pytest.mark.parametrize("line", [
    "0.00.772.094 I srv  llama_server: model loaded",
    "0.00.473.158 W load: control-looking token: 128247 '</s>' was not control-type",
    "0.02.295.422 I slot print_timing: id  0 | task 0 | prompt eval time = 32.42 ms",
    "0.00.622.117 I llama_model_loader: Dumping metadata keys/values.",
    # a size in a unit we never parse must not be silently read as MiB
    "0.01.000.000 I load_tensors:        CUDA0 model buffer size =  1515.92 GiB",
])
def test_non_allocation_lines_are_ignored(line):
    assert _instance().parse_memory_line(line) is False


def test_break_wrong_bucket_fails_loudly():
    """Negative control: a parser that lumped host memory into VRAM must be caught."""
    inst = _instance()
    inst._device_bucket = types.MethodType(lambda _s, _d: "vram", inst)
    for line in ENGINE_LOG:
        inst.parse_memory_line(line)
    bd = inst.memory_breakdown()
    assert bd["ram"] == {}
    assert bd["total_mib"]["vram"] > 4000  # everything, RAM included
    assert bd["total_mib"]["vram"] != _parse(ENGINE_LOG).memory_breakdown()["total_mib"]["vram"]


def test_no_report_yet_is_distinct_from_zero():
    assert _instance().memory_breakdown() is None


def test_parenthesised_engine_tag_is_not_a_continuation():
    """llama.cpp's `slot  operator ():` puts () before the colon. Treated as an
    untagged continuation line it would be dropped from the developer log."""
    m = M._RE_ENG_LINE.match(
        "1.29.251.423 I slot  operator (): id  0 | task 0 | new prompt, n_ctx_slot = 8192"
    )
    assert m is not None
    assert (m.group(1), m.group(2), m.group(3)) == ("I", "slot", "operator")



def test_llama_engines_launch_at_verbosity_4():
    """Without -lv 4 libllama never prints the allocation report at all."""
    from lmloader.engine.router import EngineRouter
    import inspect

    src = inspect.getsource(EngineRouter.build_launch_config)
    assert '"-lv", "4"' in src
