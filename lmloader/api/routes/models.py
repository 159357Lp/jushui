"""Model management routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from lmloader.infra.database import db
from lmloader.infra.scanner import scanner
from lmloader.infra.events import event_log
from lmloader.engine.lifecycle import lifecycle
from lmloader.engine.router import router as engine_router

router = APIRouter()


class LoadModelRequest(BaseModel):
    model_name: str
    keep_previous: bool = False


class SwitchModelRequest(BaseModel):
    model_name: str
    keep_previous: bool = False


class VisionRequest(BaseModel):
    model_name: str
    enabled: bool


class RunConfigBody(BaseModel):
    ctx_size: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    threads: Optional[int] = None
    flash_attn: Optional[bool] = None
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    parallel: Optional[int] = None
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    mtp_enabled: Optional[bool] = None
    spec_draft_max: Optional[int] = None


# Run-config clamping lives in the engine router so BOTH the save path (here)
# and the load path (router.build_launch_config) share one implementation and
# can never drift apart. Re-exported for backwards compatibility.
from lmloader.engine.router import (
    CACHE_TYPES,
    MAX_PARALLEL,
    MAX_SPEC_DRAFT,
    clamp_run_config,
)


def _defaults_for_engine_slot(model) -> dict:
    """Engine-slot defaults, used as the baseline when no run config is saved."""
    from lmloader.config import settings as cfg
    slot = "ternary" if engine_router.select_engine_slot(model) == "ternary" else "primary"
    d = getattr(cfg.engines, slot).defaults
    return {
        "ctx_size": d.ctx_size,
        "n_gpu_layers": d.n_gpu_layers,
        "threads": d.threads,
        "flash_attn": d.flash_attn,
        "cache_type_k": d.cache_type_k,
        "cache_type_v": d.cache_type_v,
        "parallel": d.parallel,
        "batch_size": d.batch_size,
        "ubatch_size": d.ubatch_size,
        "mtp_enabled": d.mtp_enabled,
        "spec_draft_max": d.spec_draft_max,
    }


@router.get("/models/{model_name}/config")
async def get_model_run_config(model_name: str):
    model = await db.get_model_by_name(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    saved = await db.get_run_config(model_name)
    defaults = _defaults_for_engine_slot(model)
    info = model.to_dict()
    if saved is None:
        return {"model_name": model_name, "saved": False, "config": defaults, "model": info}
    return {"model_name": model_name, "saved": True, "config": saved, "model": info}


@router.post("/models/{model_name}/config")
async def save_model_run_config(model_name: str, body: RunConfigBody):
    model = await db.get_model_by_name(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    saved = await db.get_run_config(model_name)
    defaults = _defaults_for_engine_slot(model)
    merged = dict(saved or defaults)
    for key in ("ctx_size", "n_gpu_layers", "threads", "flash_attn",
                "cache_type_k", "cache_type_v", "parallel",
                "batch_size", "ubatch_size", "mtp_enabled", "spec_draft_max"):
        val = getattr(body, key, None)
        if val is not None:
            merged[key] = val
    merged = clamp_run_config(model, merged)
    await db.save_run_config(model_name, merged)
    return {"model_name": model_name, "saved": True, "config": merged}


@router.delete("/models/{model_name}/config")
async def delete_model_run_config(model_name: str):
    # Without this check an unknown model name also answered 200
    # {"saved": false}, so the caller could not tell "removed" from
    # "there was nothing / no such model".
    model = await db.get_model_by_name(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    await db.delete_run_config(model_name)
    return {"model_name": model_name, "saved": False}


@router.get("/models")
async def list_models(status: Optional[str] = None):
    models = await db.list_models(status)
    saved_names = await db.list_run_config_names()
    items = []
    for m in models:
        d = m.to_dict()
        d["is_vision"] = engine_router.is_vision_encoder(m)
        d["has_vision"] = not d["is_vision"] and engine_router.find_mmproj(m) is not None
        d["has_run_config"] = m.name in saved_names
        items.append(d)
    return {"models": items}


@router.get("/models/loaded")
async def get_loaded_models():
    loaded = await lifecycle.get_loaded_models()
    return {"loaded": loaded}


@router.get("/models/{model_id}")
async def get_model(model_id: int):
    model = await db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    d = model.to_dict()
    d["is_vision"] = engine_router.is_vision_encoder(model)
    d["has_vision"] = not d["is_vision"] and engine_router.find_mmproj(model) is not None
    return d


@router.post("/models/scan")
async def scan_models():
    result = await scanner.scan_and_register()
    return result


@router.post("/models/load")
async def load_model(request: LoadModelRequest):
    model = await db.get_model_by_name(request.model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if engine_router.is_vision_encoder(model):
        event_log.warning(
            "lifecycle",
            f"视觉模型 {model.name} 不能单独加载，请通过主模型的眼睛按钮挂载",
            model=model.name,
        )
        raise HTTPException(
            status_code=400,
            detail="Vision encoder (CLIP/mmproj) cannot be loaded standalone; it is mounted automatically when you load its paired text model",
        )

    try:
        success = await lifecycle.load_model(model, request.keep_previous)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load model")

    return {"status": "loaded", "model": request.model_name}


@router.post("/models/unload")
async def unload_model(model_name: str):
    success = await lifecycle.unload_model(model_name)
    if not success:
        raise HTTPException(status_code=400, detail="Model not loaded or failed to unload")

    return {"status": "unloaded", "model": model_name}


@router.post("/models/switch")
async def switch_model(request: SwitchModelRequest):
    model = await db.get_model_by_name(request.model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if engine_router.is_vision_encoder(model):
        event_log.warning(
            "lifecycle",
            f"视觉模型 {model.name} 不能单独加载，请通过主模型的眼睛按钮挂载",
            model=model.name,
        )
        raise HTTPException(
            status_code=400,
            detail="Vision encoder (CLIP/mmproj) cannot be loaded standalone; it is mounted automatically when you load its paired text model",
        )

    success = await lifecycle.switch_model(model, request.keep_previous)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to switch model")

    return {"status": "switched", "model": request.model_name}


@router.post("/models/vision")
async def toggle_vision(request: VisionRequest):
    """Toggle the vision encoder on an already-loaded model (restarts it)."""
    success = await lifecycle.set_vision(request.model_name, request.enabled)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Model not loaded, or vision toggle failed",
        )
    loaded = await lifecycle.get_loaded_models()
    return {"status": "ok", "model": request.model_name, "enabled": request.enabled, "loaded": loaded}


@router.delete("/models/{model_id}")
async def delete_model(model_id: int):
    model = await db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.status == "loaded":
        await lifecycle.unload_model(model.name)

    await db.delete_model(model_id)
    return {"status": "deleted"}
