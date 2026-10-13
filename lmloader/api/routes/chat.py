"""OpenAI-compatible routes — raw passthrough proxy to managed llama-server.

Every request body is forwarded verbatim (tools, response_format, top_p,
seed, stop, max_completion_tokens ... are preserved instead of being filtered
by a strict pydantic schema). Covers chat/completions, completions, embeddings
and the newer responses API; the target model is just-in-time loaded.
"""

import json

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from lmloader.engine.lifecycle import lifecycle
from lmloader.infra.events import event_log
from lmloader.infra.model_names import resolve_model

router = APIRouter()


async def _ensure_instance(model_name: str):
    """Resolve a model by name and just-in-time load it."""
    instance = await lifecycle.get_model_instance(model_name)
    if instance:
        return instance
    model = await resolve_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"model '{model_name}' not found")
    try:
        await lifecycle.load_model(model)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to auto-load model: {e}")
    instance = await lifecycle.get_model_instance(model.name)
    if not instance:
        raise HTTPException(status_code=500, detail="model failed to start")
    return instance


async def _read_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(body, dict) or "model" not in body:
        raise HTTPException(status_code=400, detail="'model' is required")
    return body


def apply_thinking_default(model_name: str, body: dict) -> dict:
    """Force thinking OFF for models whose reasoning block is unreliable.

    Bonsai binary/ternary builds ship the Qwen3.5 chat template, where an
    undefined ``enable_thinking`` falls through to the *thinking* branch at
    ``reasoning_effort='xhigh'``. On these low-bit weights the reasoning block
    degenerates into long, repetitive stalls (tens of seconds of blank output)
    and structured garbage, so override it unless the caller set it explicitly.
    """
    if "bonsai" not in (model_name or "").lower():
        return body
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
        return body  # caller decided explicitly; don't fight it
    if not isinstance(kwargs, dict):
        kwargs = {}
    kwargs["enable_thinking"] = False
    body["chat_template_kwargs"] = kwargs
    return body


async def _stream(url: str, body: dict):
    async def generate():
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            async with client.stream("POST", url, json=body) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    yield f"data: {json.dumps({'error': error_body.decode('utf-8', 'replace')})}\n\n"
                    return
                async for line in response.aiter_lines():
                    if line:
                        # SSE events must be terminated by a blank line.
                        yield f"{line}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _passthrough(request: Request, upstream_path: str, kind: str):
    body = await _read_body(request)
    model_name = body["model"]
    instance = await _ensure_instance(model_name)
    # Any real request resets the idle-unload timer for this model.
    instance.touch_activity()

    n_msgs = len(body.get("messages") or [])
    if kind == "chat":
        event_log.info("lifecycle",
                       f"[{model_name}] Running chat completion on conversation "
                       f"with {n_msgs} message{'s' if n_msgs != 1 else ''}.")
    elif kind == "embeddings":
        inp = body.get("input")
        n_items = len(inp) if isinstance(inp, list) else 1
        event_log.info("lifecycle", f"[{model_name}] Running embedding on {n_items} input(s).")
    elif kind == "completions":
        event_log.info("lifecycle", f"[{model_name}] Running text completion.")
    elif kind == "responses":
        event_log.info("lifecycle", f"[{model_name}] Running responses API request.")

    if kind == "chat":
        apply_thinking_default(model_name, body)

    if body.get("stream"):
        return await _stream(f"{instance.base_url}{upstream_path}", body)

    response = await lifecycle.proxy_request(
        model_name, "POST", upstream_path, json_data=body
    )
    if not response:
        raise HTTPException(status_code=502, detail="llama-server request failed")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code,
                            detail=f"llama-server error: {response.text}")
    return response.json()


@router.post("/chat/completions")
async def create_chat_completion(request: Request):
    return await _passthrough(request, "/v1/chat/completions", "chat")


@router.post("/completions")
async def create_completion(request: Request):
    return await _passthrough(request, "/v1/completions", "completions")


@router.post("/embeddings")
async def create_embedding(request: Request):
    return await _passthrough(request, "/v1/embeddings", "embeddings")


@router.post("/responses")
async def create_responses(request: Request):
    # Only present on llama.cpp builds that implement it; older engines 404.
    return await _passthrough(request, "/v1/responses", "responses")


@router.get("/models")
async def list_models():
    from lmloader.infra.database import db

    all_models = await db.list_models()
    loaded = await lifecycle.get_loaded_models()
    loaded_map = {item["model"]["name"]: item for item in loaded}

    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "created": int(m.created_at.timestamp()) if hasattr(m.created_at, 'timestamp') else 0,
                "owned_by": "llama-server",
                "loaded": m.name in loaded_map,
                "metadata": {
                    "architecture": m.architecture,
                    "quantization": m.quantization,
                    "parameters": m.parameters,
                    "size": m.size,
                    "status": m.status,
                    **(
                        {
                            "port": loaded_map[m.name]["port"],
                            "pid": loaded_map[m.name]["pid"],
                            "engine_slot": loaded_map[m.name].get("engine_slot", "primary"),
                        }
                        if m.name in loaded_map
                        else {}
                    ),
                },
            }
            for m in all_models
        ],
    }
