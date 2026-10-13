"""Ollama-compatible API routes."""

import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Optional
import time
import httpx

from lmloader.infra.database import db
from lmloader.infra.model_names import resolve_model, normalize_spec
from lmloader.infra.guardian import config_guardian
from lmloader.engine.lifecycle import lifecycle
from lmloader.logger import logger

router = APIRouter()


class OllamaChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaChatMessage]
    stream: bool = True
    options: Optional[dict] = None
    format: Optional[str] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    stream: bool = True
    options: Optional[dict] = None


async def _find_and_load_model(name: str):
    """Find model by spec and auto-load if not loaded. FIXED: B3"""
    model = await resolve_model(name)
    if not model:
        raise HTTPException(status_code=404, detail=f"model '{name}' not found")

    instance = await lifecycle.get_model_instance(model.name)
    if instance:
        instance.touch_activity()
        return instance

    if model.status != "loaded":
        try:
            # lifecycle.load_model expects a ModelRecord, not a name string
            await lifecycle.load_model(model)
            instance = await lifecycle.get_model_instance(model.name)
            if instance:
                instance.touch_activity()
                return instance
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to auto-load model: {e}")

    raise HTTPException(status_code=500, detail=f"Model '{name}' could not be loaded")


def _build_ollama_model_entry(model_record) -> dict:
    display_name = model_record.name
    if not display_name.endswith(".gguf") and ":" not in display_name:
        display_name = f"{display_name}:latest"

    return {
        "name": display_name,
        "model": display_name,
        "modified_at": model_record.updated_at,
        "size": model_record.size,
        "digest": f"sha256:{model_record.id}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": model_record.architecture,
            "families": [model_record.architecture],
            "parameter_size": model_record.parameters,
            "quantization_level": model_record.quantization,
        },
    }


@router.get("/api/tags")
async def list_tags():
    models = await db.list_models()
    loaded = await lifecycle.get_loaded_models()
    loaded_names = {item["model"]["name"] for item in loaded}

    return {
        "models": [
            _build_ollama_model_entry(m)
            for m in models
        ]
    }


@router.post("/api/generate")
async def generate(request: OllamaGenerateRequest):
    instance = await _find_and_load_model(request.model)

    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})

    body = {
        "model": instance.model.name,
        "messages": messages,
        "stream": request.stream,
        "temperature": (request.options or {}).get("temperature", 0.7),
    }

    if request.stream:
        return StreamingResponse(
            _stream_generate(instance, body),
            media_type="application/x-ndjson",
        )

    response = await lifecycle.proxy_request(
        instance.model.name, "POST", "/v1/chat/completions", json_data=body
    )
    if not response or response.status_code != 200:
        raise HTTPException(status_code=502, detail="Engine request failed")

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    return {
        "model": instance.model.name,
        "created_at": int(time.time()),
        "response": content,
        "done": True,
        "total_duration": 0,
        "eval_count": data.get("usage", {}).get("completion_tokens", 0),
    }


async def _stream_generate(instance, body: dict):
    url = f"{instance.base_url}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    yield json.dumps({"error": error_body.decode()}) + "\n"
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunk = {
                                "model": instance.model.name,
                                "created_at": int(time.time()),
                                "response": content,
                                "done": False,
                            }
                            yield json.dumps(chunk) + "\n"
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"
        return

    yield json.dumps({
        "model": instance.model.name,
        "created_at": int(time.time()),
        "response": "",
        "done": True,
        "total_duration": 0,
    }) + "\n"


@router.post("/api/chat")
async def chat(request: OllamaChatRequest):
    instance = await _find_and_load_model(request.model)  # FIXED: B3

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    body = {
        "model": instance.model.name,
        "messages": messages,
        "stream": request.stream,
        "temperature": (request.options or {}).get("temperature", 0.7),
    }

    if request.stream:
        return StreamingResponse(
            _stream_chat(instance, body),
            media_type="application/x-ndjson",
        )

    response = await lifecycle.proxy_request(
        instance.model.name, "POST", "/v1/chat/completions", json_data=body
    )
    if not response or response.status_code != 200:
        raise HTTPException(status_code=502, detail="Engine request failed")

    data = response.json()
    msg = data["choices"][0]["message"]

    return {
        "model": instance.model.name,
        "created_at": int(time.time()),
        "message": {"role": msg["role"], "content": msg["content"]},
        "done": True,
        "total_duration": 0,
        "eval_count": data.get("usage", {}).get("completion_tokens", 0),
    }


async def _stream_chat(instance, body: dict):
    url = f"{instance.base_url}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    yield json.dumps({"error": error_body.decode()}) + "\n"
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunk = {
                                "model": instance.model.name,
                                "created_at": int(time.time()),
                                "message": {"role": "assistant", "content": content},
                                "done": False,
                            }
                            yield json.dumps(chunk) + "\n"
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"
        return

    yield json.dumps({
        "model": instance.model.name,
        "created_at": int(time.time()),
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "total_duration": 0,
    }) + "\n"


@router.post("/api/pull")
async def pull_model(name: str = "", stream: bool = True):
    from lmloader.infra.downloader import downloader

    async def progress_stream():
        async for event in downloader.download_with_progress(name):
            yield json.dumps(event) + "\n"

    if stream:
        return StreamingResponse(progress_stream(), media_type="application/x-ndjson")

    result = await downloader.download(name)
    return result


class DownloadConfigBody(BaseModel):
    # ``None`` means "field not supplied — leave the stored value alone".
    # ``""`` is an explicit clear. The distinction matters: the settings page
    # posts both keys on every save, but any other caller that omits a key
    # (or posts an empty body) must not silently wipe the stored config.
    proxy: Optional[str] = None
    ollama_dir: Optional[str] = None


@router.get("/api/download/config")
async def get_download_config():
    from lmloader.infra.ollama_lib import ollama_root
    detected = ollama_root()
    return {
        "proxy": settings_proxy(),
        "ollama_dir": settings_ollama_dir(),
        "detected_ollama_root": str(detected) if detected else "",
    }


@router.post("/api/download/config")
async def set_download_config(body: DownloadConfigBody):
    import yaml
    if body.proxy is None and body.ollama_dir is None:
        # Nothing to do. Previously this fell through and wrote both keys as
        # empty strings, erasing the user's proxy / ollama_dir without any
        # indication that a change had been made.
        raise HTTPException(
            status_code=400,
            detail="未提供任何字段：proxy / ollama_dir 至少需要一项（显式传空字符串才会清空）",
        )

    # ---- guardian: ask user before changing config ----
    changes = {}
    if body.proxy is not None:
        changes["proxy"] = body.proxy.strip()
    if body.ollama_dir is not None:
        changes["ollama_dir"] = body.ollama_dir.strip()
    req = await config_guardian.submit(
        endpoint="POST /api/download/config",
        description=f"修改下载配置：{changes}",
        parameters=changes,
    )
    approved = await config_guardian.wait_for_approval(req)
    if not approved:
        raise HTTPException(status_code=403, detail="用户已拒绝此配置变更")
    # ---- end guardian ----

    try:
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        changed = []
        if body.proxy is not None:
            new_proxy = body.proxy.strip()
            if cfg.get("proxy", "") != new_proxy:
                cfg["proxy"] = new_proxy
                changed.append("proxy")
        if body.ollama_dir is not None:
            new_dir = body.ollama_dir.strip()
            if cfg.get("ollama_dir", "") != new_dir:
                cfg["ollama_dir"] = new_dir
                changed.append("ollama_dir")
        if changed:
            with open("config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            from lmloader.config import settings as _s
            _s._config.proxy = cfg.get("proxy", "")
            _s._config.ollama_dir = cfg.get("ollama_dir", "")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"保存下载配置失败: {e}")
    return {
        "status": "ok",
        "changed": changed,
        "proxy": cfg.get("proxy", ""),
        "ollama_dir": cfg.get("ollama_dir", ""),
    }


def settings_proxy() -> str:
    from lmloader.config import settings as _s
    return _s.proxy


def settings_ollama_dir() -> str:
    from lmloader.config import settings as _s
    return _s.ollama_dir


@router.get("/api/ollama/search")
async def ollama_search(q: str = "", limit: int = 30):
    """Search the Ollama model library.

    ollama.com publishes no JSON search API (the registry answers
    `tags/list` with 404), so this scrapes the server-rendered search page.
    """
    from lmloader.infra import ollama_search as ollama_search_mod

    query = (q or "").strip()
    if not query:
        return {"query": "", "models": []}
    try:
        models = await asyncio.to_thread(ollama_search_mod.search_models, query, limit)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ollama 搜索失败（HTTP {e.response.status_code}）")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"无法访问 ollama.com: {e}")
    return {"query": query, "models": models}


@router.get("/api/ollama/tags")
async def ollama_tags(model: str = ""):
    """List the available tags (quantizations) of one Ollama model."""
    from lmloader.infra import ollama_search as ollama_search_mod

    name = (model or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="model is required (e.g. qwen2.5)")
    try:
        tags = await asyncio.to_thread(ollama_search_mod.list_tags, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=404 if e.response.status_code == 404 else 502,
                            detail=f"未找到模型 {name}" if e.response.status_code == 404
                            else f"Ollama 请求失败（HTTP {e.response.status_code}）")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"无法访问 ollama.com: {e}")
    return {"model": name, "tags": tags}


@router.get("/api/ollama/installed")
async def ollama_installed():
    """Model refs already present in the local Ollama store."""
    from lmloader.infra import ollama_search as ollama_search_mod

    refs = await asyncio.to_thread(ollama_search_mod.installed_refs)
    return {"models": refs}


# --- active pull registry (for cancellation) -------------------------------

_pull_tokens: dict[str, Any] = {}


def _pull_key(name: str, mode: str) -> str:
    return f"{mode}:{name.strip().lower()}"


@router.get("/api/ollama/pulls")
async def list_active_pulls():
    return {"active": sorted(_pull_tokens.keys())}


@router.post("/api/ollama/pull/cancel")
async def cancel_pull(name: str = "", mode: str = "builtin"):
    """Cancel a running pull. Idempotent: cancelling an idle pull is a no-op."""
    from lmloader.infra.ollama_pull import CancelToken

    key = _pull_key(name, mode)
    token = _pull_tokens.get(key)
    if token is None:
        return {"cancelled": False, "reason": "no active pull for this model", "name": name, "mode": mode}
    token.cancel()
    return {"cancelled": True, "name": name, "mode": mode}


@router.post("/api/ollama/pull")
async def pull_ollama_model(name: str = "", mode: str = "builtin"):
    """Pull a model from the Ollama registry.

    mode=builtin: direct registry v2 download into the Ollama store (default);
    mode=cli:     drive the installed `ollama pull` CLI.
    Streams ndjson progress; on completion the new model is auto-registered.
    Can be stopped via POST /api/ollama/pull/cancel (see _pull_tokens).
    """
    from lmloader.infra import ollama_pull
    from lmloader.infra.ollama_search import is_valid_spec

    if not name:
        raise HTTPException(status_code=400, detail="name is required (e.g. qwen2.5:7b)")
    if not is_valid_spec(name):
        raise HTTPException(status_code=400, detail=f"非法的模型名: {name}")
    if mode not in ("builtin", "cli"):
        raise HTTPException(status_code=400, detail="mode must be 'builtin' or 'cli'")

    key = _pull_key(name, mode)
    if key in _pull_tokens:
        raise HTTPException(status_code=409, detail=f"{name} 正在拉取中")

    token = ollama_pull.CancelToken()
    _pull_tokens[key] = token

    async def progress_stream():
        try:
            if mode == "cli":
                gen = ollama_pull.pull_via_cli(name, token)
            else:
                gen = ollama_pull.pull_with_progress(name, token)
            async for event in gen:
                if event.get("status") == "completed":
                    # register the freshly pulled blob(s) without a full disk scan
                    try:
                        from lmloader.infra.scanner import ModelScanner
                        await ModelScanner()._scan_ollama()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"post-pull ollama registration failed: {e}")
                yield json.dumps(event) + "\n"
        finally:
            _pull_tokens.pop(key, None)

    return StreamingResponse(progress_stream(), media_type="application/x-ndjson")


@router.delete("/api/delete")
async def delete_model(model: str = ""):
    model_record = await resolve_model(model)  # FIXED: B2
    if not model_record:
        raise HTTPException(status_code=404, detail=f"model '{model}' not found")

    instance = await lifecycle.get_model_instance(model_record.name)
    if instance:
        await lifecycle.unload_model(model_record.name)

    await db.delete_model(model_record.id)
    return {"status": "ok"}


@router.post("/api/show")
async def show_model(model: str = ""):
    model_record = await resolve_model(model)  # FIXED: B2
    if not model_record:
        raise HTTPException(status_code=404, detail=f"model '{model}' not found")

    return {
        "modelfile": f"FROM {model_record.path}",
        "parameters": "\n".join([
            f"stop <|end|>",
        ]),
        "template": "{{ .Prompt }}",
        "details": {
            "format": "gguf",
            "family": model_record.architecture,
            "families": [model_record.architecture],
            "parameter_size": model_record.parameters,
            "quantization_level": model_record.quantization,
        },
        "model_info": model_record.metadata,
    }
