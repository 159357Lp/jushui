"""Engine discovery — find runnable model engines on disk.

Scans the user-configured search directories, the bundled engines dir and
PATH for binaries declared by ENGINE_KINDS, so that non-GGUF model formats
(rwkv .bin, onnx, ...) can be routed to a matching engine automatically.
"""

import os
import shutil
from pathlib import Path

from lmloader.engine.kinds import ENGINE_KINDS
from lmloader.infra.engine_store import engine_dir
from lmloader.logger import logger


def default_scan_dirs() -> list[str]:
    """Bundled engines dir + user-configurable extra dirs."""
    dirs = [str(engine_dir())]
    cfg = _config_scan_dirs()
    for d in cfg:
        if d and d not in dirs:
            dirs.append(d)
    return dirs


def _config_scan_dirs() -> list[str]:
    try:
        import yaml
        from pathlib import Path as P
        cfg_path = P("config.yaml")
        if not cfg_path.exists():
            return []
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return data.get("engine_scan_dirs", []) or []
    except Exception:  # noqa: BLE001
        return []


def _custom_engines() -> list[dict]:
    try:
        import yaml
        from pathlib import Path as P
        cfg_path = P("config.yaml")
        if not cfg_path.exists():
            return []
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return data.get("custom_engines", []) or []
    except Exception:  # noqa: BLE001
        return []


def scan_dirs_for_binary(binary_names: list[str], scan_dirs: list[str]) -> list[Path]:
    """Locate binaries by name inside the scan dirs (non-recursive)."""
    found: list[Path] = []
    for d in scan_dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        try:
            entries = list(p.iterdir())
        except OSError:
            continue
        for item in entries:
            if not item.is_file():
                continue
            if item.name.lower() in binary_names or item.name.lower() in [
                b.lower() for b in binary_names
            ]:
                found.append(item)
    return found


def discover_engines(scan_dirs: list[str] | None = None) -> list[dict]:
    """Return discovered engines: {name, kind, binary_path, formats, source}.

    Includes bundled/configured llama engines + any extra binaries found in
    the scan dirs + user-defined custom engines.
    """
    scan_dirs = scan_dirs or default_scan_dirs()
    found: list[dict] = []

    # 1) configured llama slots (primary/ternary)
    try:
        import yaml
        cfg_path = Path("config.yaml")
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        for slot, slot_cfg in (data.get("engines") or {}).items():
            bp = (slot_cfg or {}).get("binary_path")
            if bp and Path(bp).exists():
                found.append({
                    "name": slot,
                    "kind": "llama",
                    "binary_path": str(bp),
                    "formats": list(ENGINE_KINDS["llama"]["formats"]),
                    "source": "config",
                })
    except Exception:  # noqa: BLE001
        pass

    # 2) scan dirs for every other declared kind
    for kind, spec in ENGINE_KINDS.items():
        if kind == "llama":
            continue
        names = [b.lower() for b in spec["binary_names"]]
        for d in scan_dirs:
            p = Path(d)
            if not p.is_dir():
                continue
            try:
                entries = list(p.iterdir())
            except OSError:
                continue
            for item in entries:
                if item.is_file() and item.name.lower() in names:
                    found.append({
                        "name": f"{kind}@{item.stem}",
                        "kind": kind,
                        "binary_path": str(item),
                        "formats": list(spec["formats"]),
                        "source": "scan",
                    })
                    break  # one binary per kind per dir

    # 3) custom engines from config
    for ce in _custom_engines():
        bp = ce.get("binary_path", "")
        kind = ce.get("kind", "llama")
        if bp and Path(bp).exists():
            found.append({
                "name": ce.get("name", Path(bp).stem),
                "kind": kind,
                "binary_path": str(bp),
                "formats": list(ENGINE_KINDS.get(kind, {}).get("formats", [])),
                "source": "custom",
            })

    logger.info(f"Engine discovery: {len(found)} engine(s) found")
    return found


def find_engine_for_format(ext: str, discovered: list[dict] | None = None) -> dict | None:
    """Pick an engine whose kind can run the given model format."""
    ext = ext.lower()
    discovered = discovered if discovered is not None else discover_engines()
    for eng in discovered:
        if ext in [f.lower() for f in eng.get("formats", [])]:
            return eng
    return None


def find_engine_by_kind(kind: str, discovered: list[dict] | None = None) -> dict | None:
    """Pick an engine of the given kind (used for directory-based models, e.g.
    a HuggingFace model directory that vLLM takes with no file extension)."""
    discovered = discovered if discovered is not None else discover_engines()
    for eng in discovered:
        if eng.get("kind") == kind:
            return eng
    return None
