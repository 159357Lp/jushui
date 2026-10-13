"""Engine store — keep llama-server binaries inside the loader directory.

The PrismML ternary fork (llama.cpp-prism) is a full runtime: llama-server.exe
is a thin stub and the real code lives in llama-server-impl.dll / llama.dll /
ggml-*.dll, so the whole release directory is bundled rather than a single
file. `ensure_engines()` migrates any externally-referenced engine (e.g. a
manual build under a custom path) into <project>/engines and rewrites config.yaml
to point at the bundled copy.
"""

import shutil
from pathlib import Path

import yaml

from lmloader.logger import logger


def _config_path() -> Path:
    # Backend runs with cwd = <project>/lmloader, so config.yaml resolves there.
    return Path("config.yaml").resolve()


def _project_root() -> Path:
    cfg = _config_path()
    return cfg.parent.parent if cfg.parent.name == "lmloader" else cfg.parent


def engine_dir() -> Path:
    """<project>/engines — next to lmloader/, inside the loader directory."""
    return _project_root() / "engines"


def is_bundled(path: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(engine_dir().resolve())
    except (AttributeError, OSError):
        return str(Path(path).resolve()).startswith(str(engine_dir().resolve()))


def bundled_binary() -> Path:
    return engine_dir() / "llama-server.exe"


def ensure_engines() -> dict:
    """Migrate external engine binaries into the bundled engines/ directory.

    Rewrites config.yaml binary_path for each slot when a migration happens.
    Returns {slot: {"action": ok|bundled|fallback|missing, "path": str}}.
    """
    result: dict = {}
    cfg_path = _config_path()
    if not cfg_path.exists():
        return result

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    engines = data.get("engines", {})
    changed = False

    for slot in ("primary", "ternary"):
        slot_cfg = engines.get(slot)
        if not isinstance(slot_cfg, dict):
            continue
        src_raw = slot_cfg.get("binary_path", "")
        if not src_raw:
            continue
        src = Path(src_raw)

        # Each slot installs into its own sub-directory: primary/official
        # builds live in engines/official/, the ternary build in engines/ root.
        subdir = (slot_cfg.get("update_subdir") or "").strip("/\\")
        slot_dir = engine_dir() / subdir if subdir else engine_dir()
        slot_binary = slot_dir / "llama-server.exe"

        if is_bundled(src) and src.exists():
            result[slot] = {"action": "ok", "path": str(src)}
            continue

        if not src.exists():
            if is_bundled(src):
                # Already points at this slot's bundled location but the build
                # is not installed yet (e.g. official engine before first
                # update). Keep the path; registry falls back at runtime.
                result[slot] = {"action": "missing", "path": src_raw}
                continue
            # An external path that no longer exists: normalise to the slot's
            # bundled path (install via one-click update if still missing).
            if slot_binary.exists():
                slot_cfg["binary_path"] = str(slot_binary)
                result[slot] = {"action": "fallback", "path": str(slot_binary)}
            else:
                slot_cfg["binary_path"] = str(slot_binary)
                result[slot] = {"action": "missing", "path": str(slot_binary)}
            changed = True
            continue

        # Copy the whole external release directory into the slot's dir.
        slot_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for item in src.parent.iterdir():
            if item.is_file():
                try:
                    shutil.copy2(item, slot_dir / item.name)
                    copied += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"engine copy failed {item.name}: {e}")
        slot_cfg["binary_path"] = str(slot_binary)
        changed = True
        result[slot] = {"action": "bundled", "path": str(slot_binary), "copied": copied}

    if changed:
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return result
