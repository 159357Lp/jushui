"""The /api/monitor aggregation the GUI reads, with two models loaded.

The parser turns one engine's stderr into a split; this turns N engines' splits
into the payload the monitor page paints, and it is the only place where a
second loaded model can be mishandled (double-counted, or dropped with its RSS).
"""

import types

from lmloader.api.routes import monitor as m


class _Inst:
    def __init__(self, name, port, slot, bd, rss_mib):
        self.model = types.SimpleNamespace(name=name)
        self.launch_config = types.SimpleNamespace(port=port, engine_slot=slot)
        self.process = types.SimpleNamespace(pid=port + 1000)
        self._bd = bd
        self._rss = rss_mib

    def memory_breakdown(self):
        return self._bd


def _fake_psutil(monkeypatch, rss_by_pid):
    class _P:
        def __init__(self, pid):
            self._rss = rss_by_pid.get(pid, 0.0)

        def memory_info(self):
            return types.SimpleNamespace(rss=self._rss * 1024 * 1024)

    monkeypatch.setattr(m.psutil, "Process", _P)


_A = {"vram": {"weights": 1515.92, "kv": 96.0, "compute": 102.66},
      "ram": {"weights": 2009.38, "kv": 160.0}}
_B = {"vram": {"weights": 500.0, "kv": 24.0},
      "ram": {"weights": 800.0}}


def _two_models(monkeypatch):
    a = _Inst("d3-mini", 8198, "d3-master", _A, 4017.0)
    b = _Inst("qwen", 8199, "primary", _B, 1200.0)
    monkeypatch.setattr(m, "_instances", lambda: [a, b])
    _fake_psutil(monkeypatch, {a.process.pid: 4017.0, b.process.pid: 1200.0})
    return a, b


def test_pools_are_summed_over_models(monkeypatch):
    _two_models(monkeypatch)
    out = m._get_memory_breakdown()
    assert out["vram"]["weights"] == 2015.9
    assert out["vram"]["kv"] == 120.0
    assert out["vram"]["compute"] == 102.7
    assert out["ram"]["weights"] == 2809.4
    assert out["engine_rss_mib"] == 5217.0
    assert [x["name"] for x in out["models"]] == ["d3-mini", "qwen"]


def test_classes_a_model_never_reports_stay_zero(monkeypatch):
    _two_models(monkeypatch)
    out = m._get_memory_breakdown()
    # No engine in this fixture printed an RS or output line.
    assert out["vram"]["recurrent"] == 0.0
    assert out["vram"]["output"] == 0.0


def test_model_without_a_report_contributes_rss_only(monkeypatch):
    """A model still loading has no allocation report yet; its process memory is
    real and must not be folded into the engine total as if it were buffers."""
    a = _Inst("d3-mini", 8198, "d3-master", _A, 4017.0)
    quiet = _Inst("vllm-model", 8200, "vllm", None, 900.0)
    monkeypatch.setattr(m, "_instances", lambda: [a, quiet])
    _fake_psutil(monkeypatch, {a.process.pid: 4017.0, quiet.process.pid: 900.0})
    out = m._get_memory_breakdown()
    assert out["engine_rss_mib"] == 4917.0
    assert out["models"] and out["models"][0]["name"] == "d3-mini"
    assert len(out["models"]) == 1


def test_nothing_loaded_is_not_an_error(monkeypatch):
    monkeypatch.setattr(m, "_instances", lambda: [])
    out = m._get_memory_breakdown()
    assert out["models"] == []
    assert out["vram"]["weights"] == 0.0
    assert out["engine_rss_mib"] == 0.0


def test_break_summing_once_per_pool_fails_loudly(monkeypatch):
    """Negative control: assigning instead of accumulating must be caught."""
    _two_models(monkeypatch)
    real = dict(_A["vram"])
    # qwen's smaller weights would silently overwrite d3-mini's.
    assert _B["vram"]["weights"] < real["weights"]
    out = m._get_memory_breakdown()
    assert out["vram"]["weights"] != _B["vram"]["weights"]
    assert out["vram"]["weights"] > max(x["vram"]["weights"] for x in out["models"])


class _IO:
    def __init__(self, read_bytes=0, write_bytes=0, read_time=0, write_time=0):
        self.read_bytes, self.write_bytes = read_bytes, write_bytes
        self.read_time, self.write_time = read_time, write_time


def _drive_disk(monkeypatch, times, counters, prime=True):
    """Feed the disk sampler a scripted clock and counter series.

    The first reading always primes the baseline, so a test that wants a rate
    has to take a second reading.
    """
    it, ic = iter(times), iter(counters)
    monkeypatch.setattr(m.time, "monotonic", lambda: next(it))
    monkeypatch.setattr(m.psutil, "disk_io_counters", lambda: next(ic))
    monkeypatch.setattr(
        m.psutil, "disk_usage",
        lambda _d: types.SimpleNamespace(total=150 * 2**30, used=137 * 2**30,
                                         free=13 * 2**30, percent=91.3))
    m._DISK_BASE.clear()
    if prime:
        m._get_disk_info()


def test_disk_rates_are_deltas_between_two_polls(monkeypatch):
    _drive_disk(monkeypatch, [0.0, 2.0], [_IO(), _IO(
        read_bytes=20 * 2**20, write_bytes=4 * 2**20,
        read_time=1500, write_time=300)])
    second = m._get_disk_info()
    assert (second["read_mb_s"], second["write_mb_s"]) == (10.0, 2.0)
    assert second["busy_percent"] == 75.0  # 1500 ms of the 2000 ms interval
    assert second["percent"] == 91.3


def test_first_reading_reports_idle(monkeypatch):
    """Nothing to subtract yet: 0, not a division by zero and not a bogus rate."""
    _drive_disk(monkeypatch, [100.0], [_IO(read_bytes=1 << 30, read_time=9000)],
                prime=False)
    out = m._get_disk_info()
    assert (out["read_mb_s"], out["write_mb_s"], out["busy_percent"]) == (0.0, 0.0, 0.0)


def test_busy_time_is_the_longer_side_not_the_sum(monkeypatch):
    """A disk serving reads and writes at once counts both, and the sum would
    report it busier than it was."""
    _drive_disk(monkeypatch, [0.0, 1.0], [_IO(), _IO(read_time=600, write_time=700)])
    assert m._get_disk_info()["busy_percent"] == 70.0


def test_busy_time_is_capped_at_full(monkeypatch):
    _drive_disk(monkeypatch, [0.0, 1.0], [_IO(), _IO(read_time=5000)])
    assert m._get_disk_info()["busy_percent"] == 100.0


def test_cached_read_shows_throughput_but_no_busy_time(monkeypatch):
    """Windows only counts time the physical disk actually worked, so a read
    served from the file cache moves bytes without moving busy%. Truthful, not
    a broken rate."""
    _drive_disk(monkeypatch, [0.0, 1.0], [_IO(), _IO(read_bytes=300 * 2**20)])
    out = m._get_disk_info()
    assert out["read_mb_s"] == 300.0
    assert out["busy_percent"] == 0.0
