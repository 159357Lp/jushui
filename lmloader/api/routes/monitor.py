"""Resource monitor routes."""

import psutil
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
    }


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
