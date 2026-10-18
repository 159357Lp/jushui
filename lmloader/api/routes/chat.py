"""OpenAI-compatible routes — raw passthrough proxy to managed llama-server.

Every request body is forwarded verbatim (tools, response_format, top_p,
seed, stop, max_completion_tokens ... are preserved instead of being filtered
by a strict pydantic schema). Covers chat/completions, completions, embeddings
and the newer responses API; the target model is just-in-time loaded.
"""

import json

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

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


async def _refuse_decision_model(model_name: str, upstream_path: str) -> None:
    """Reject chat-style endpoints for decision models, before loading them.

    A decision model has no chat template: its vocabulary is the answer tokens,
    so sampling it through /v1/chat/completions streams letter soup ("AAHHBBFF
    ...") instead of an error, and it keeps burning the whole context as one
    non-causal batch while doing so. Both shapes are refused here — a Decision
    3.0 package and a GGUF carrying ``<arch>.decision.type`` — and the check
    runs off the stored metadata so the model is never even started for a
    request it can only answer with garbage.
    """
    from lmloader.engine.router import router as engine_router
    from lmloader.infra.database import db

    record = await db.get_model_by_name(model_name)
    if record is None:
        return  # an alias we cannot resolve here; let the loader decide
    decision_type = str((record.metadata or {}).get("decision_type") or "")
    if not engine_router.is_decision_model(record):
        return
    label = f"Decision 3.0 决策模型（{decision_type}）" if decision_type else "Decision 3.0 决策模型"
    raise HTTPException(
        status_code=400,
        detail=(
            f"'{model_name}' 是{label}，不是聊天模型，不支持 {upstream_path}；"
            "请改用 POST /v1/systemone（state 情境 + questions 问题集 → 每个选项的概率）"
        ),
    )


async def _passthrough(request: Request, upstream_path: str, kind: str):
    body = await _read_body(request)
    model_name = body["model"]
    await _refuse_decision_model(model_name, upstream_path)
    instance = await _ensure_instance(model_name)
    if instance.launch_config.kind == "d3":
        # Reached only when the request named the model by something other than
        # its DB name, so the metadata check above could not identify it.
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{model_name}' 是 Decision 3.0 决策模型，不支持 {upstream_path}；"
                "请改用 POST /v1/systemone（questions 问题集 → 每个选项的概率）"
            ),
        )
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


@router.post("/systemone")
async def create_system_one(request: Request):
    """Decision 3.0 passthrough — the body is forwarded to d3_server verbatim.

    ``{"model", "state", "questions", "images", "videos"}`` in, the model's
    System One reply (``{"model", "answers", "usage"}``) out. Nothing is
    translated or invented here: d3 has no chat API and faking one would hide
    that it is a router/decision model.
    """
    body = await _read_body(request)
    model_name = body["model"]
    instance = await _ensure_instance(model_name)
    # Two shapes of decision model reach this endpoint: a Decision 3.0 package
    # (Python d3_server, kind "d3") and a decision-type GGUF served by a
    # llama.cpp build that has /v1/systemone. Both take the body verbatim;
    # anything else (an ordinary chat GGUF) genuinely has no such endpoint, so
    # it is refused instead of being forwarded into a 404.
    from lmloader.infra.database import db

    record = await db.get_model_by_name(model_name)
    is_decision_gguf = bool((record.metadata or {}).get("decision_type"))
    if instance.launch_config.kind != "d3" and not is_decision_gguf:
        raise HTTPException(
            status_code=400,
            detail=f"'{model_name}' 不是决策模型（引擎 {instance.launch_config.kind}），无 /v1/systemone",
        )
    instance.touch_activity()
    n_q = len(body.get("questions") or {})
    event_log.info(
        "lifecycle",
        f"[{model_name}] Running System One decision on {n_q} question(s)"
        f"{', with media' if body.get('images') or body.get('videos') else ''}.",
    )
    response = await lifecycle.proxy_request(
        model_name, "POST", "/v1/systemone", json_data=body, timeout=600.0
    )
    if not response:
        raise HTTPException(status_code=502, detail="d3 server request failed")
    # d3 reports bad requests as 422 {"detail": ...}; keep its verdict intact.
    # Raising HTTPException with that dict would nest a second "detail" around
    # it, so the upstream status and body are returned verbatim instead.
    if response.status_code != 200:
        try:
            return JSONResponse(status_code=response.status_code,
                                content=response.json())
        except Exception:
            return JSONResponse(status_code=response.status_code,
                                content={"detail": response.text})
    return response.json()


@router.get("/models")
async def list_models():
    from lmloader.engine.router import router as engine_router
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
                    # Spelled out for every entry, never only for the ones it
                    # matches: an OpenAI-only client has to be able to tell a
                    # decision model (``/v1/systemone`` only) from a chat model
                    # before offering it, and a missing key is not an answer.
                    "is_decision_model": engine_router.is_decision_model(m),
                    **(
                        {
                            "port": loaded_map[m.name]["port"],
                            "pid": loaded_map[m.name]["pid"],
                            "engine_slot": loaded_map[m.name].get("engine_slot", "primary"),
                            "engine_kind": loaded_map[m.name].get("engine_kind", "llama"),
                        }
                        if m.name in loaded_map
                        else {}
                    ),
                },
            }
            for m in all_models
        ],
    }
