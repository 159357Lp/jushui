"""Engine status, discovery & update routes."""

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from lmloader.config import settings
from lmloader.engine.registry import engine_registry
from lmloader.engine.lifecycle import lifecycle
from lmloader.engine.discovery import discover_engines
from lmloader.engine.kinds import ENGINE_KINDS
from lmloader.infra.engine_store import engine_dir, bundled_binary, is_bundled
from lmloader.infra.engine_updater import (
    check_latest, cuda_runtime_available, download_and_install, local_version,
    slot_binary, slot_install_dir,
)
from lmloader.infra.guardian import config_guardian
from lmloader.logger import logger

import shutil
import re
import datetime

router = APIRouter()

_SLOTS = ("primary", "ternary")


def _slot_cfg(slot: str):
    if slot not in _SLOTS:
        raise HTTPException(status_code=400, detail=f"未知引擎槽: {slot}")
    return getattr(settings.engines, slot)


def _rewrite_slot_path(slot: str, path: Path):
    """Persist a slot's new binary_path into config.yaml and live settings."""
    cfg = _slot_cfg(slot)
    cfg.binary_path = str(path)
    cfg_path = Path("config.yaml").resolve()
    data = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    engines = data.setdefault("engines", {})
    slot_data = engines.setdefault(slot, {})
    slot_data["binary_path"] = str(path)
    cfg_path.write_text(yaml.safe_dump(data, default_flow_style=False,
                                       allow_unicode=True), encoding="utf-8")


@router.get("/engine")
async def get_engine():
    engines = await engine_registry.detect_all()
    if not engines:
        raise HTTPException(status_code=404, detail="No engine available")
    loaded = await lifecycle.get_loaded_models()
    return {
        "engines": {name: engine.to_dict() for name, engine in engines.items()},
        "loaded_models": loaded,
    }


@router.get("/engine/kinds")
async def engine_kinds():
    """Declared engine types (labels, binaries, supported formats)."""
    return {
        "kinds": [
            {
                "kind": k,
                "label": v["label"],
                "binary_names": v["binary_names"],
                "formats": v["formats"],
            }
            for k, v in ENGINE_KINDS.items()
        ]
    }


@router.get("/engine/discover")
async def engine_discover():
    """Re-scan engine search dirs + custom engines (no binary probing)."""
    discovered = discover_engines()
    try:
        cfg = yaml.safe_load(open("config.yaml", encoding="utf-8")) or {}
        scan_dirs = cfg.get("engine_scan_dirs", []) or []
    except Exception:  # noqa: BLE001
        scan_dirs = []
    return {"discovered": discovered, "scan_dirs": scan_dirs}


class ScanDirsBody(BaseModel):
    scan_dirs: list[str]


@router.post("/engine/scan-dirs")
async def engine_set_scan_dirs(body: ScanDirsBody):
    """Persist engine search directories."""
    # ---- guardian: ask user before changing config ----
    req = await config_guardian.submit(
        endpoint="POST /api/engine/scan-dirs",
        description=f"修改引擎搜索目录为：{body.scan_dirs}",
        parameters={"scan_dirs": body.scan_dirs},
    )
    approved = await config_guardian.wait_for_approval(req)
    if not approved:
        raise HTTPException(status_code=403, detail="用户已拒绝此配置变更")
    # ---- end guardian ----

    try:
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["engine_scan_dirs"] = [d for d in body.scan_dirs if d.strip()]
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"保存搜索目录失败: {e}")
    engine_registry.invalidate()
    return {"status": "ok", "scan_dirs": cfg["engine_scan_dirs"]}


class CustomEngineBody(BaseModel):
    name: str
    kind: str
    binary_path: str


def _allowed_engine_roots() -> list[Path]:
    """Directories a custom engine binary may live in.

    Registering a custom engine means the backend will later spawn whatever
    path is stored here. Without a restriction that endpoint is "run any
    executable on this machine" — which is harmless behind a token and a
    remote-code-execution hole without one, and still lets a typo point the
    engine at cmd.exe.
    """
    roots = [engine_dir().resolve()]
    for extra in (getattr(settings, "engine_scan_dirs", None) or []):
        try:
            roots.append(Path(extra).resolve())
        except OSError:
            continue
    return roots


def _validate_custom_engine_path(raw: str) -> Path:
    """Resolve *raw* and refuse anything outside the allowed engine roots."""
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as exc:  # unreadable path, bad drive letter, ...
        raise HTTPException(status_code=400, detail=f"引擎路径无法解析: {raw} ({exc})")

    roots = _allowed_engine_roots()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        allowed = ", ".join(str(r) for r in roots)
        raise HTTPException(
            status_code=400,
            detail=(
                f"引擎路径必须位于已允许的引擎目录内：{allowed}\n"
                f"收到：{resolved}\n"
                f"如需使用其他位置的引擎，请把该目录加入 config.yaml 的 engine_scan_dirs。"
            ),
        )
    if not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"引擎文件不存在: {resolved}")
    return resolved


@router.post("/engine/custom")
async def engine_add_custom(body: CustomEngineBody):
    """Register a custom engine (name/kind/binary path)."""
    if body.kind not in ENGINE_KINDS:
        raise HTTPException(status_code=400, detail=f"未知引擎类型: {body.kind}")
    if not body.name.strip() or not body.binary_path.strip():
        raise HTTPException(status_code=400, detail="名称和引擎路径不能为空")
    binary_path = _validate_custom_engine_path(body.binary_path)

    # ---- guardian: ask user before changing config ----
    req = await config_guardian.submit(
        endpoint="POST /api/engine/custom",
        description=f"注册自定义引擎：{body.name.strip()}（{body.kind}）→ {binary_path}",
        parameters={"name": body.name.strip(), "kind": body.kind, "binary_path": str(binary_path)},
    )
    approved = await config_guardian.wait_for_approval(req)
    if not approved:
        raise HTTPException(status_code=403, detail="用户已拒绝此配置变更")
    # ---- end guardian ----

    try:
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        customs = cfg.get("custom_engines", []) or []
        # Re-registering a name is how the picker replaces an entry. Carrying
        # its history over keeps 回退 working after a rename-in-place; dropping
        # it would silently erase the rollback chain the user can still see.
        previous = next((c for c in customs if c.get("name") == body.name.strip()), None)
        customs = [c for c in customs if c.get("name") != body.name.strip()]
        entry = {
            "name": body.name.strip(),
            "kind": body.kind,
            "binary_path": str(binary_path),
        }
        history = list((previous or {}).get("history") or [])
        old_path = str((previous or {}).get("binary_path") or "")
        # Re-registering also displaces a binary, so it belongs in the chain —
        # otherwise 回退 silently skips the file that was in use a minute ago.
        if old_path and Path(old_path).resolve() != binary_path:
            ver = await local_version(Path(old_path))
            history.insert(0, {
                "binary_path": old_path,
                "version": ver.get("raw") or "unknown",
                "at": datetime.datetime.now().isoformat(timespec="seconds"),
            })
        if history:
            entry["history"] = history[:_MAX_CUSTOM_HISTORY]
        customs.append(entry)
        cfg["custom_engines"] = customs
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"保存自定义引擎失败: {e}")
    engine_registry.invalidate()
    return {"status": "ok", "custom_engines": cfg["custom_engines"]}


@router.get("/engine/custom")
async def engine_list_custom():
    """Registered custom engines, with the history each swap left behind.

    Kept separate from ``/engine/discover``: discovery scans the filesystem and
    returns candidates, this returns what the user actually committed to
    config.yaml — including the previous ``binary_path`` entries the GUI offers
    to roll back to.
    """
    try:
        cfg = yaml.safe_load(open("config.yaml", encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    return {"custom_engines": cfg.get("custom_engines", []) or []}


class CustomEngineUpdateBody(BaseModel):
    name: str
    binary_path: str


_MAX_CUSTOM_HISTORY = 10


def _write_customs(customs: list[dict]) -> None:
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["custom_engines"] = customs
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _find_custom(customs: list[dict], name: str) -> dict:
    entry = next((c for c in customs if c.get("name") == name), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"未登记该自定义引擎: {name}")
    return entry


@router.post("/engine/custom/update")
async def engine_update_custom(body: CustomEngineUpdateBody):
    """Re-point a custom engine at another binary, remembering the old one.

    This is deliberately a *registration* swap, not a file copy: a custom engine
    is often a whole runtime directory (exe + its DLLs), and half-replacing it
    is worse than the old version still sitting where it was. Rollback restores
    the previous registration, so "上一版" means "上一个你登记的这个文件".
    """
    name = body.name.strip()
    new_path = _validate_custom_engine_path(body.binary_path)
    try:
        cfg = yaml.safe_load(open("config.yaml", encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取配置失败: {e}")
    customs = cfg.get("custom_engines", []) or []
    entry = _find_custom(customs, name)
    old_path = str(entry.get("binary_path") or "")
    if Path(old_path).resolve() == new_path:
        return {"status": "already_current", "name": name, "binary_path": str(new_path)}

    # ---- guardian: ask user before changing config ----
    req = await config_guardian.submit(
        endpoint="POST /api/engine/custom/update",
        description=f"更换自定义引擎 {name} 的二进制：{old_path or '（未登记）'} → {new_path}",
        parameters={"name": name, "from": old_path, "to": str(new_path)},
    )
    approved = await config_guardian.wait_for_approval(req)
    if not approved:
        raise HTTPException(status_code=403, detail="用户已拒绝此配置变更")
    # ---- end guardian ----

    history = list(entry.get("history") or [])
    if old_path:
        ver = await local_version(Path(old_path))
        history.insert(0, {
            "binary_path": old_path,
            "version": ver.get("raw") or "unknown",
            "at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
    entry["history"] = history[:_MAX_CUSTOM_HISTORY]
    entry["binary_path"] = str(new_path)
    try:
        _write_customs(customs)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"保存自定义引擎失败: {e}")
    engine_registry.invalidate()
    return {"status": "updated", "name": name, "binary_path": str(new_path),
            "history": entry["history"]}


@router.post("/engine/custom/rollback")
async def engine_rollback_custom(name: str = Query(...)):
    """Restore the previously registered binary for a custom engine."""
    name = name.strip()
    try:
        cfg = yaml.safe_load(open("config.yaml", encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取配置失败: {e}")
    customs = cfg.get("custom_engines", []) or []
    entry = _find_custom(customs, name)
    history = list(entry.get("history") or [])
    if not history:
        raise HTTPException(status_code=404, detail="没有可回退的历史登记")
    target = history[0]
    target_path = Path(str(target.get("binary_path") or ""))
    if not target_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"上一版引擎文件已不在原处，无法回退：{target_path}",
        )

    # ---- guardian: ask user before changing config ----
    req = await config_guardian.submit(
        endpoint="POST /api/engine/custom/rollback",
        description=f"回退自定义引擎 {name}：{entry.get('binary_path')} → {target_path}",
        parameters={"name": name, "from": entry.get("binary_path"), "to": str(target_path)},
    )
    approved = await config_guardian.wait_for_approval(req)
    if not approved:
        raise HTTPException(status_code=403, detail="用户已拒绝此配置变更")
    # ---- end guardian ----

    old_path = str(entry.get("binary_path") or "")
    current_ver = await local_version(Path(old_path)) if old_path else {}
    history.pop(0)
    if old_path:
        history.insert(0, {
            "binary_path": old_path,
            "version": current_ver.get("raw") or "unknown",
            "at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
    entry["history"] = history[:_MAX_CUSTOM_HISTORY]
    entry["binary_path"] = str(target_path)
    try:
        _write_customs(customs)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"保存自定义引擎失败: {e}")
    engine_registry.invalidate()
    return {"status": "rolled_back", "name": name, "binary_path": str(target_path),
            "history": entry["history"]}


@router.get("/engine/update-status")
async def engine_update_status(slot: str = Query("ternary")):
    """Local version of a slot's binary + its configured update source."""
    cfg = _slot_cfg(slot)
    exe = slot_binary(cfg)
    installed = exe.exists()
    ver = await local_version(exe)

    fallback_info = None
    if not installed:
        info = engine_registry.get_engine(slot)
        if info is not None and info.fallback:
            fb_ver = await local_version(Path(info.binary_path))
            fallback_info = {"binary_path": info.binary_path, "local": fb_ver}

    return {
        "slot": slot,
        "local": ver,
        "installed": installed,
        "fallback": fallback_info,
        "update_repo": cfg.update_repo,
        "update_kind": cfg.update_kind,
        "bundled_dir": str(engine_dir()),
        "bundled": bundled_binary().exists(),
        "configured_bundled": is_bundled(exe),
        "cuda_runtime_available": cuda_runtime_available(settings.cuda_path),
        "backups": [
            {"dir": str(b), "version": (await local_version(b / "llama-server.exe")).get("raw", "unknown"),
             "build": (await local_version(b / "llama-server.exe")).get("build"),
             "mtime": b.stat().st_mtime}
            for b in _find_backups(slot_install_dir(cfg.update_subdir))
        ],
    }


@router.post("/engine/check-update")
async def engine_check_update(slot: str = Query("ternary")):
    """Query the slot's configured GitHub source for the newest release."""
    cfg = _slot_cfg(slot)
    if not cfg.update_repo:
        raise HTTPException(status_code=400, detail="该引擎槽未配置更新源")
    try:
        latest = await check_latest(cfg.update_repo, cfg.update_kind)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"检查更新失败: {e}")
    exe = slot_binary(cfg)
    ver = await local_version(exe)
    local_build = ver.get("build")
    latest_build = latest.get("build")
    # Build numbers are only comparable within the same source kind; if the
    # local binary can't be parsed (e.g. a dev/prism build while checking the
    # official source), report the remote as available for install.
    has_update = bool(latest_build) and (
        local_build is None or latest_build > local_build
    )
    return {"slot": slot, "local": ver, "latest": latest, "has_update": has_update}


@router.post("/engine/update")
async def engine_update(slot: str = Query("ternary")):
    """Unload every model, download the slot's newest build, install it."""
    cfg = _slot_cfg(slot)
    if not cfg.update_repo:
        raise HTTPException(status_code=400, detail="该引擎槽未配置更新源")
    exe = slot_binary(cfg)
    ver = await local_version(exe)
    latest = await check_latest(cfg.update_repo, cfg.update_kind)
    if not latest.get("asset_url"):
        raise HTTPException(status_code=404, detail="官方未发布 Windows 预编译包，无法自动更新")
    latest_build = latest.get("build")
    if latest_build and ver.get("build") and latest_build <= ver["build"] \
            and cfg.update_subdir == "":
        return {"status": "already_latest", "local": ver, "latest": latest}

    # Pre-update CUDA runtime detection: if the system lacks cudart DLLs,
    # bundle them alongside the engine so the updated build can start.
    need_cudart = not cuda_runtime_available(settings.cuda_path)
    cudart_url = None
    if need_cudart and latest.get("cudart_asset_url"):
        cudart_url = latest["cudart_asset_url"]

    desc = f"更新 {slot} 引擎：{ver.get('raw', '?')} → {latest.get('tag', '?')}（将卸载所有已加载模型）"
    if cudart_url:
        desc += f"；检测到系统缺少 CUDA 运行时，将一并下载 {latest.get('cudart_asset_name', 'cudart')} 捆绑包"

    # ---- guardian: ask user before changing config (installs new engine) ----
    req = await config_guardian.submit(
        endpoint="POST /api/engine/update",
        description=desc,
        parameters={"slot": slot, "current_version": ver.get("raw"), "new_version": latest.get("tag"),
                     "cudart_bundled": bool(cudart_url)},
    )
    approved = await config_guardian.wait_for_approval(req)
    if not approved:
        raise HTTPException(status_code=403, detail="用户已拒绝此配置变更")
    # ---- end guardian ----

    await lifecycle.unload_all()  # release exe/dll file locks first
    dest = slot_install_dir(cfg.update_subdir)
    try:
        result = await download_and_install(latest["tag"], latest["asset_url"], dest,
                                            cudart_asset_url=cudart_url)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"更新失败: {e}")

    new_binary = dest / "llama-server.exe"
    if new_binary.exists():
        _rewrite_slot_path(slot, new_binary)
    engine_registry.invalidate()
    new_ver = await local_version(new_binary if new_binary.exists() else exe)
    return {"status": "updated", "slot": slot, "result": result,
            "local": new_ver, "latest": latest}


def _find_backups(dest: Path) -> list[Path]:
    """Return backup directories for *dest*, newest first.

    Matches the naming convention used by ``download_and_install``:
    ``{dest.parent}/{dest.name}_backup_{tag}``.
    """
    parent = dest.parent
    prefix = f"{dest.name}_backup_"
    if not parent.exists():
        return []
    backups = sorted(
        (p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return backups


@router.get("/engine/rollback-info")
async def engine_rollback_info(slot: str = Query("ternary")):
    """List available backups for a slot (for the GUI rollback button)."""
    cfg = _slot_cfg(slot)
    dest = slot_install_dir(cfg.update_subdir)
    backups = _find_backups(dest)
    items = []
    for b in backups:
        exe = b / "llama-server.exe"
        ver = await local_version(exe) if exe.exists() else {"raw": "unknown", "build": None}
        items.append({"dir": str(b), "version": ver.get("raw", "unknown"),
                      "build": ver.get("build"), "mtime": b.stat().st_mtime})
    return {"slot": slot, "backups": items}


@router.post("/engine/rollback")
async def engine_rollback(slot: str = Query("ternary")):
    """Roll back a slot's engine to the most recent backup."""
    cfg = _slot_cfg(slot)
    dest = slot_install_dir(cfg.update_subdir)
    backups = _find_backups(dest)
    if not backups:
        raise HTTPException(status_code=404, detail="没有可用的回退备份")

    latest_backup = backups[0]
    backup_exe = latest_backup / "llama-server.exe"
    if not backup_exe.exists():
        raise HTTPException(status_code=404, detail=f"备份中未找到引擎文件: {latest_backup}")

    backup_ver = await local_version(backup_exe)
    current_exe = dest / "llama-server.exe"
    current_ver = await local_version(current_exe) if current_exe.exists() else {"raw": "not-found"}

    # ---- guardian: ask user before changing config ----
    req = await config_guardian.submit(
        endpoint="POST /api/engine/rollback",
        description=f"回退 {slot} 引擎：{current_ver.get('raw', '?')} → {backup_ver.get('raw', '?')}（将卸载所有已加载模型）",
        parameters={"slot": slot, "from": current_ver.get("raw"), "to": backup_ver.get("raw"),
                    "backup_dir": str(latest_backup)},
    )
    approved = await config_guardian.wait_for_approval(req)
    if not approved:
        raise HTTPException(status_code=403, detail="用户已拒绝此配置变更")
    # ---- end guardian ----

    await lifecycle.unload_all()

    try:
        # Move current files aside, then copy backup into place.
        scratch = dest.parent / f"{dest.name}_rollback_scratch"
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir()

        # Move current engine files to scratch.
        current_files = [p for p in dest.iterdir() if p.is_file()
                         and not p.name.startswith("_")]
        for f in current_files:
            try:
                shutil.move(str(f), str(scratch / f.name))
            except PermissionError:
                # Restore anything already moved.
                for m in scratch.iterdir():
                    shutil.move(str(m), str(dest / m.name))
                shutil.rmtree(scratch, ignore_errors=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"文件被占用，无法替换 {f.name} — 请先卸载所有模型并停止服务后再回退",
                )

        # Copy backup files into dest.
        for f in latest_backup.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(dest / f.name))

        # Verify the rolled-back binary runs.
        rolled_ver = await local_version(dest / "llama-server.exe")
        if rolled_ver.get("build") is None:
            # Rollback failed — restore from scratch.
            for f in list(dest.iterdir()):
                if f.is_file() and not f.name.startswith("_"):
                    f.unlink(missing_ok=True)
            for m in scratch.iterdir():
                shutil.move(str(m), str(dest / m.name))
            shutil.rmtree(scratch, ignore_errors=True)
            raise HTTPException(status_code=500, detail="回退后的引擎无法运行，已恢复原状")

        # Clean up scratch.
        shutil.rmtree(scratch, ignore_errors=True)

        # Update slot binary_path in config.
        new_binary = dest / "llama-server.exe"
        _rewrite_slot_path(slot, new_binary)
        engine_registry.invalidate()

        logger.info(f"Rolled back {slot} engine to {rolled_ver.get('raw')} from {latest_backup.name}")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"回退失败: {e}")

    return {
        "status": "rolled_back",
        "slot": slot,
        "from": current_ver.get("raw"),
        "to": rolled_ver.get("raw"),
        "backup_dir": str(latest_backup),
    }
