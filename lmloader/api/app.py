"""FastAPI application."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from lmloader import __version__
from lmloader.api.security import (
    SecurityHeadersMiddleware,
    TokenAuthMiddleware,
    unauthenticated_network_warning,
)

from lmloader.config import settings
from lmloader.infra.database import db
from lmloader.infra.events import event_log
from lmloader.api.routes import models, engine, chat, ollama, monitor, events, config_routes


# Endpoints worth showing in Developer Logs — the externally consumable model
# APIs plus state-changing management actions. High-frequency GUI polling
# (events, monitor, model list, loaded list, engine status, docs...) is skipped.
_LOGGED_PREFIXES = (
    "/v1/",
    "/api/tags",
    "/api/chat",
    "/api/generate",
    "/api/show",
    "/api/pull",
    "/api/delete",
    "/api/ollama/",
)
_LOGGED_EXACT = {
    "/api/models/load", "/api/models/unload", "/api/models/switch",
    "/api/models/vision", "/api/models/scan", "/api/engine/update",
}
# Non-streaming JSON listings whose full response body is logged (LM Studio
# style "Returning {...}").
_BODY_CAPTURE_PATHS = {"/v1/models", "/api/tags"}
# POST endpoints whose non-streaming JSON response is logged as the generated
# prediction / completion (SSE streaming responses are never buffered).
_PREDICTION_CAPTURE_PATHS = {
    "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
    "/v1/responses", "/api/chat", "/api/generate",
}
_MAX_LOG_BYTES = 6000


def _is_logged(path: str) -> bool:
    if path in _LOGGED_EXACT:
        return True
    return any(path.startswith(p) for p in _LOGGED_PREFIXES)


def _pretty(body_bytes: bytes) -> str | None:
    try:
        return json.dumps(json.loads(body_bytes.decode("utf-8", "replace")),
                          ensure_ascii=False, indent=2)
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init()
    from lmloader.engine.lifecycle import lifecycle
    # Reap engines left behind by a previous crashed run, then start the
    # idle-unload watchdog (both no-ops when nothing applies).
    await lifecycle.start_watchdog()
    _install_hard_exit_cleanup(lifecycle)
    _emit_startup_banner()
    yield
    await lifecycle.unload_all()


def _install_hard_exit_cleanup(lifecycle) -> None:
    """Synchronously kill every engine when the backend console is closed or
    the interpreter exits hard (CTRL_CLOSE_EVENT gives only ~5s and the event
    loop is already dead there, so this must NOT be async). The job object is
    the final kernel-level net; this is the userspace fast path."""
    from lmloader.infra.proc_util import install_exit_cleanup, kill_process_tree

    def _sync_kill_engines() -> None:
        for inst in list(lifecycle.instances.values()):
            try:
                kill_process_tree(inst.process.pid)
            except Exception:
                pass

    install_exit_cleanup(_sync_kill_engines)


def _emit_startup_banner():
    host = settings.server.host
    port = settings.server.port
    base = f"http://{host}:{port}"
    event_log.info("server", f"掬水 server listening on {base}")
    if getattr(settings.server, "api_token", ""):
        event_log.info("server", "  鉴权已启用：所有请求需带 Authorization: Bearer <token>")
    else:
        warning = unauthenticated_network_warning(
            host, getattr(settings.server, "allow_unauthenticated_network", False)
        )
        if warning:
            for line in warning.splitlines():
                event_log.warning("server", line)
    event_log.info("server", "Supported endpoints:")
    event_log.info("server", "  OpenAI-compatible")
    for method, path in (
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/responses"),
    ):
        event_log.info("server", f"   {method:4} {base}{path}")
    event_log.info("server", "  Ollama-compatible")
    for method, path in (
        ("GET", "/api/tags"),
        ("POST", "/api/chat"),
        ("POST", "/api/generate"),
        ("POST", "/api/show"),
        ("POST", "/api/pull"),
        ("DELETE", "/api/delete"),
    ):
        event_log.info("server", f"   {method:6} {base}{path}")
    event_log.info("server", "  Management")
    for method, path in (
        ("POST", "/api/models/load"),
        ("POST", "/api/models/unload"),
        ("POST", "/api/models/switch"),
        ("POST", "/api/models/vision"),
        ("POST", "/api/models/scan"),
        ("POST", "/v1/systemone"),
    ):
        event_log.info("server", f"   {method:4} {base}{path}")
    event_log.info("server", f"Logs are saved into {Path.cwd() / 'logs'}")
    event_log.info("server", "Server started. Just-in-time model loading active.")


app = FastAPI(
    title="掬水",
    description="Desktop model loader powered by llama-server",
    version=__version__,
    lifespan=lifespan,
)


class DeveloperLogsMiddleware:
    """Pure-ASGI traffic logger for Developer Logs.

    A function middleware (Starlette BaseHTTPMiddleware) buffers proxied SSE
    streams through an anyio channel and swallows chat-completion streams
    entirely (0 bytes reach the client). This pure-ASGI version forwards every
    http.response.body chunk immediately, so streaming responses pass through
    untouched; only non-stream JSON bodies are captured for logging.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not _is_logged(path):
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        qs = scope.get("query_string", b"").decode("latin1")
        target = f"{method} to {path}" + (f"?{qs}" if qs else "")

        # Wrap receive so the request JSON body can be logged, then replayed.
        body_parts: list[bytes] = []

        async def logging_receive():
            message = await receive()
            if message.get("type") == "http.request":
                body_parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    raw = b"".join(body_parts)
                    if raw and method in ("POST", "DELETE", "PUT"):
                        pretty = _pretty(raw)
                        shown = pretty if pretty else raw.decode("utf-8", "replace")
                        event_log.info("http",
                                       f"Received request: {target} with body {shown[:_MAX_LOG_BYTES]}")
                    else:
                        event_log.info("http", f"Received request: {target}")
            return message

        state = {"status": 0, "ctype": "", "capture": False, "chunks": b""}

        async def send_wrapper(message):
            mtype = message.get("type")
            if mtype == "http.response.start":
                state["status"] = message.get("status", 0)
                headers = message.get("headers", [])
                ctype = ""
                for k, v in headers:
                    if k.lower() == b"content-type":
                        ctype = v.decode("latin1", "replace")
                state["ctype"] = ctype
                if method == "GET" and path in _BODY_CAPTURE_PATHS:
                    state["capture"] = True
                elif (method == "POST" and path in _PREDICTION_CAPTURE_PATHS
                      and "text/event-stream" not in ctype
                      and "application/json" in ctype):
                    state["capture"] = True
                await send(message)
            elif mtype == "http.response.body":
                chunk = message.get("body", b"")
                if state["capture"] and "text/event-stream" not in state["ctype"]:
                    state["chunks"] += chunk
                await send(message)  # forward immediately — never buffer SSE
                if not message.get("more_body", False):
                    if state["capture"]:
                        pretty = _pretty(state["chunks"])
                        if pretty is not None:
                            event_log.info("http", "Returning " + pretty[:_MAX_LOG_BYTES]
                                           if method == "GET" and path in _BODY_CAPTURE_PATHS
                                           else "Generated prediction: " + pretty[:_MAX_LOG_BYTES])
                        else:
                            event_log.record(
                                "warning" if state["status"] >= 400 else "info",
                                "http", f"-> {state['status']}")
                    else:
                        event_log.record(
                            "warning" if state["status"] >= 400 else "info",
                            "http", f"-> {state['status']}")
            else:
                await send(message)

        await self.app(scope, logging_receive, send_wrapper)


app.add_middleware(DeveloperLogsMiddleware)

# Security middleware. Starlette applies add_middleware bottom-up, so the token
# gate listed first ends up OUTERMOST and rejects a request before it reaches
# the developer-log middleware (an unauthenticated caller should not even be
# logged as a normal API consumer).
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    TokenAuthMiddleware,
    token_provider=lambda: getattr(settings.server, "api_token", ""),
)
if settings.server.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.server.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(engine.router, prefix="/api", tags=["engine"])
app.include_router(config_routes.router, prefix="/api", tags=["config"])
app.include_router(chat.router, prefix="/v1", tags=["chat"])
app.include_router(ollama.router, tags=["ollama"])
app.include_router(monitor.router, prefix="/api", tags=["monitor"])
app.include_router(events.router, prefix="/api", tags=["events"])

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>掬水</h1><p>API running. Visit /docs for API documentation.</p>"


@app.get("/health")
async def health(response: Response):
    response.headers["X-LM-Loader-Pid"] = str(os.getpid())
    return {"status": "ok"}


@app.post("/api/shutdown")
async def shutdown():
    async def _delayed_shutdown():
        await asyncio.sleep(0.5)
        from lmloader.engine.lifecycle import lifecycle
        # Graceful unload, but never let it hang the shutdown: the kill-on-close
        # job object reaps any engine still alive once this process exits, so we
        # always reach os._exit() even if a graceful unload stalls.
        try:
            await asyncio.wait_for(lifecycle.unload_all(), timeout=8.0)
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass
        finally:
            os._exit(0)

    asyncio.create_task(_delayed_shutdown())
    return {"status": "shutting_down"}
