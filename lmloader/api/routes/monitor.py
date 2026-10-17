"""Resource monitor routes."""

import psutil
import time
from pathlib import Path

from fastapi import APIRouter

from lmloader.infra.proc_util import no_window_kwargs

router = APIRouter()


@router.get("/monitor")
async def get_monitor():
    cpu_pct = psutil.cpu_percent(interval=0)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    gpu_info = _get_gpu_info()

    return {
        "cpu": {
            "percent": cpu_pct,
            "cores": psutil.cpu_count(),
            "freq_mhz": _get_cpu_freq(),
        },
        "memory": {
            "total": mem.total,
            "used": mem.used,
            "available": mem.available,
            "percent": mem.percent,
        },
        "swap": {
            "total": swap.total,
            "used": swap.used,
            "percent": swap.percent,
        },
        "gpu": gpu_info,
        "disk": _get_disk_info(),
        "memory_breakdown": _get_memory_breakdown(),
    }


# Last (timestamp, counters) sample. Rates are differences between two polls,
# so the very first one has nothing to subtract and reports idle instead of a
# division by zero.
_DISK_BASE: dict = {}


def _get_disk_info() -> dict:
    """Disk busy/time, throughput, and the free space of the installed drive.

    Throughput is system-wide: psutil names its IO devices "PhysicalDrive0" on
    Windows, which cannot be tied back to a drive letter, whereas the capacity
    reading is per-partition. Reporting the drive this package lives on is the
    one that holds the models.
    """
    out = {
        "drive": "", "busy_percent": 0.0, "read_mb_s": 0.0, "write_mb_s": 0.0,
        "total": 0, "used": 0, "free": 0, "percent": 0.0,
    }
    drive = Path(__file__).resolve().drive
    if drive:
        out["drive"] = drive.rstrip("\\")
        try:
            usage = psutil.disk_usage(drive)
            out.update(
                total=usage.total, used=usage.used, free=usage.free,
                percent=usage.percent,
            )
        except Exception:
            pass

    try:
        io = psutil.disk_io_counters()
    except Exception:
        io = None
    now = time.monotonic()
    if io is not None:
        elapsed = now - _DISK_BASE.get("t", now)
        if elapsed > 0.05 and "io" in _DISK_BASE:
            prev = _DISK_BASE["io"]
            out["read_mb_s"] = round(
                (io.read_bytes - prev.read_bytes) / elapsed / (1024 ** 2), 1)
            out["write_mb_s"] = round(
                (io.write_bytes - prev.write_bytes) / elapsed / (1024 ** 2), 1)
            # read_time and write_time overlap while a disk serves both, so the
            # sum can exceed the interval; the longer of the two is the honest
            # "the disk was working for" figure.
            busy_ms = max(io.read_time - prev.read_time,
                          io.write_time - prev.write_time)
            out["busy_percent"] = round(min(100.0, busy_ms / (elapsed * 1000) * 100), 1)
        _DISK_BASE["t"] = now
        _DISK_BASE["io"] = io
    return out



# Buffer classes the engines report, in the order they should stack up.
_MEM_CLASSES = ("weights", "kv", "recurrent", "compute", "output")


def _get_memory_breakdown() -> dict:
    """Weights / KV / state split per memory pool, summed over loaded models.

    The per-class sizes come from each engine's own allocation report — on
    Windows nvidia-smi cannot attribute GPU memory to a process (it answers
    [N/A]), so nothing measured from outside could produce this split.
    """
    empty = {k: 0.0 for k in _MEM_CLASSES}
    vram, ram = dict(empty), dict(empty)
    models = []
    engine_rss_mib = 0.0

    for inst in _instances():
        bd = inst.memory_breakdown()
        rss = _rss_mib(inst)
        engine_rss_mib += rss
        if not bd:
            continue
        for pool, target in (("vram", vram), ("ram", ram)):
            for cls, mib in bd[pool].items():
                target[cls] = target.get(cls, 0.0) + mib
        models.append({
            "name": inst.model.name,
            "port": inst.launch_config.port,
            "engine_slot": inst.launch_config.engine_slot,
            "vram": {k: round(v, 1) for k, v in bd["vram"].items()},
            "ram": {k: round(v, 1) for k, v in bd["ram"].items()},
            "rss_mib": round(rss, 1),
        })

    return {
        "vram": {k: round(v, 1) for k, v in vram.items()},
        "ram": {k: round(v, 1) for k, v in ram.items()},
        "engine_rss_mib": round(engine_rss_mib, 1),
        "models": models,
    }


def _instances() -> list:
    """Snapshot of the loaded engine instances.

    Copied out of the dict because the lifecycle manager mutates it from the
    server's own task while this route is reading it.
    """
    try:
        from lmloader.engine.lifecycle import lifecycle

        return list(lifecycle.instances.values())
    except Exception:
        return []


def _rss_mib(inst) -> float:
    """Resident set of one engine process, 0 when it already exited."""
    try:
        return psutil.Process(inst.process.pid).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _get_cpu_freq() -> float:
    try:
        freq = psutil.cpu_freq()
        return freq.current if freq else 0.0
    except Exception:
        return 0.0


def _get_gpu_info() -> dict:
    """Try to get GPU info via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            **no_window_kwargs(),
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "name": parts[0],
                        "utilization": float(parts[1]),
                        "memory_used": int(parts[2]),
                        "memory_total": int(parts[3]),
                        "temperature": int(parts[4]),
                    })
            return {"available": True, "gpus": gpus}
    except Exception:
        pass
    return {"available": False, "gpus": []}
