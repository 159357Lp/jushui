"""HTTP API client for communicating with lmloader backend."""

import json
import threading
from typing import Optional

import httpx
from PySide6.QtCore import QObject, Signal, QThread

# Upper bound on a single answer. llama-server defaults to -1 (unbounded), so
# without this a repetition loop streams forever and holds the GPU.
DEFAULT_MAX_TOKENS = 8192


class _RespHolder:
    """Thread-safe slot holding the live streaming response.

    ``chat_stream_sse`` runs in the worker thread and publishes the response
    here; the GUI thread reads it to abort the stream (stop button, repetition
    guard). A plain attribute would race, hence the lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resp = None

    def set(self, resp) -> None:
        with self._lock:
            self._resp = resp

    def get(self):
        with self._lock:
            return self._resp


def _http_detail(e: "httpx.HTTPStatusError") -> str:
    """Extract the backend's FastAPI ``detail`` message from a failed response."""
    try:
        data = e.response.json()
        if isinstance(data, dict) and data.get("detail"):
            return str(data["detail"])
    except Exception:  # noqa: BLE001
        pass
    return str(e)


_pool_lock = threading.Lock()
_pooled: dict[tuple, httpx.Client] = {}


def _client_for(url: str, headers: Optional[dict]) -> httpx.Client:
    """One pooled client per (origin, headers) pair.

    Constructing an httpx Client costs ~200ms of CPU on Windows (it builds an
    SSL context every time), and the pages poll endpoints several times a
    second, so per-call clients pin a core while the app does nothing else.
    httpx.Client is thread-safe, which the QThread workers rely on.
    """
    url_parts = httpx.URL(url)
    key = (url_parts.scheme, url_parts.host, url_parts.port,
           tuple(sorted((headers or {}).items())))
    with _pool_lock:
        client = _pooled.get(key)
        if client is None:
            client = httpx.Client(timeout=10.0)
            _pooled[key] = client
        return client


def http_request(method: str, url: str, **kw):
    return _client_for(url, kw.get("headers")).request(method, url, **kw)


def http_get(url, **kw):
    return http_request("GET", url, **kw)


def http_post(url, **kw):
    return http_request("POST", url, **kw)


def http_delete(url, **kw):
    return http_request("DELETE", url, **kw)


class LMLoaderAPI:
    def __init__(self, base_url: str = "http://127.0.0.1:8900", token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.token = token or ""

    def _headers(self) -> dict:
        """Authorization header, present only when a token is configured.

        The backend rejects requests without a valid token as soon as
        server.api_token is set, so the GUI has to send it on every call.
        """
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def health(self) -> bool:
        try:
            resp = http_get(f"{self.base_url}/health", headers=self._headers(), timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[dict]:
        try:
            resp = http_get(f"{self.base_url}/api/models", headers=self._headers(), timeout=10.0)
            resp.raise_for_status()
            return resp.json().get("models", [])
        except Exception:
            return []

    def get_loaded_models(self) -> list[dict]:
        """Loaded models, or ``[]`` if the query failed.

        ``[]`` is ambiguous on purpose for callers that only need "show
        something reasonable" — see ``fetch_loaded_models_or_none`` for the
        variant that keeps success and failure apart.
        """
        out = self.fetch_loaded_models_or_none()
        return [] if out is None else out

    def fetch_loaded_models_or_none(self) -> list[dict] | None:
        """Loaded models, or ``None`` when the request itself failed.

        This distinction matters: a caller that polls the loaded set must not
        treat "could not reach the backend" as "nothing is loaded", or a single
        timeout flips every badge to 未加载. ``get_loaded_models`` collapses the
        two cases into ``[]``, which is fine for a one-shot display but wrong
        for anything reconciling state.
        """
        try:
            resp = http_get(f"{self.base_url}/api/models/loaded", headers=self._headers(), timeout=10.0)
            resp.raise_for_status()
            return resp.json().get("loaded", [])
        except Exception:
            return None

    def load_model(self, model_name: str, keep_previous: bool = False) -> dict:
        resp = http_post(
            f"{self.base_url}/api/models/load",
            json={"model_name": model_name, "keep_previous": keep_previous},
            headers=self._headers(),
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()

    def unload_model(self, model_name: str) -> dict:
        resp = http_post(
            f"{self.base_url}/api/models/unload",
            params={"model_name": model_name},
            headers=self._headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    def toggle_vision(self, model_name: str, enabled: bool) -> dict:
        resp = http_post(
            f"{self.base_url}/api/models/vision",
            json={"model_name": model_name, "enabled": enabled},
            headers=self._headers(),
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()

    def switch_model(self, model_name: str, keep_previous: bool = False) -> dict:
        resp = http_post(
            f"{self.base_url}/api/models/switch",
            json={"model_name": model_name, "keep_previous": keep_previous},
            headers=self._headers(),
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()

    def scan_models(self) -> dict:
        resp = http_post(f"{self.base_url}/api/models/scan", headers=self._headers(), timeout=60.0)
        resp.raise_for_status()
        return resp.json()

    def get_engine(self) -> dict:
        try:
            resp = http_get(f"{self.base_url}/api/engine", headers=self._headers(), timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

    def get_monitor(self) -> dict:  # FIXED: G3
        try:
            resp = http_get(f"{self.base_url}/api/monitor", headers=self._headers(), timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

    def get_events(self, since_id: int = 0, level: str = "all") -> dict:
        try:
            resp = http_get(
                f"{self.base_url}/api/events",
                params={"since_id": since_id, "level": level, "limit": 500},
                headers=self._headers(),
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"events": [], "last_id": since_id}

    def clear_events(self) -> dict:
        resp = http_post(f"{self.base_url}/api/events/clear", headers=self._headers(), timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def chat_stream(self, model: str, messages: list[dict], temperature: float = 0.7,
                    thinking: bool = True, max_tokens: int | None = None):
        """Non-streaming chat completion."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if not thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        resp = http_post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def chat_stream_sse(self, model: str, messages: list[dict], temperature: float = 0.7,
                        thinking: bool = True, max_tokens: int | None = None,
                        resp_holder: "_RespHolder | None" = None):
        """Streaming chat — yields (kind, text) where kind is content/reasoning.

        thinking=False passes chat_template_kwargs.enable_thinking=false to the
        engine (Qwen3-style templates then skip the chain-of-thought block).

        max_tokens is mandatory in practice: llama-server's own default is -1
        (unbounded), so a model that falls into a repetition loop would stream
        forever and burn VRAM until the user manually stops it. Callers pass
        DEFAULT_MAX_TOKENS; omitting it reproduces that runaway.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if not thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                # Hand the live response to the caller so it can close the stream
                # from another thread (the repetition guard, or the stop button).
                if resp_holder is not None:
                    resp_holder.set(resp)
                try:
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                            if isinstance(data, dict) and "error" in data:
                                # An error event has no choices, so without this
                                # the loop would end silently on an empty answer
                                # and the chat would render "(空回复)" instead of
                                # what the engine actually said.
                                raise RuntimeError(_engine_error_text(data["error"]))
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            reasoning = delta.get("reasoning_content")
                            content = delta.get("content")
                            if reasoning:
                                yield ("reasoning", reasoning)
                            if content:
                                yield ("content", content)
                        except json.JSONDecodeError:
                            continue
                finally:
                    if resp_holder is not None:
                        resp_holder.set(None)

    # ---- engine update / pluggable engines ----

    def engine_update_status(self, slot: str = "ternary") -> dict:
        try:
            resp = http_get(
                f"{self.base_url}/api/engine/update-status",
                params={"slot": slot}, headers=self._headers(), timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

    def check_engine_update(self, slot: str = "ternary") -> dict:
        try:
            resp = http_post(
                f"{self.base_url}/api/engine/check-update",
                params={"slot": slot}, headers=self._headers(), timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(_http_detail(e)) from e

    def update_engine(self, slot: str = "ternary") -> dict:
        try:
            resp = http_post(
                f"{self.base_url}/api/engine/update",
                params={"slot": slot}, headers=self._headers(), timeout=1200.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(_http_detail(e)) from e

    def rollback_engine(self, slot: str = "ternary") -> dict:
        try:
            resp = http_post(
                f"{self.base_url}/api/engine/rollback",
                params={"slot": slot}, headers=self._headers(), timeout=300.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(_http_detail(e)) from e

    def engine_discover(self) -> dict:
        try:
            resp = http_get(f"{self.base_url}/api/engine/discover", headers=self._headers(), timeout=15.0)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"discovered": [], "scan_dirs": []}

    def set_engine_scan_dirs(self, scan_dirs: list[str]) -> dict:
        resp = http_post(f"{self.base_url}/api/engine/scan-dirs", json={"scan_dirs": scan_dirs}, headers=self._headers(), timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    def add_custom_engine(self, name: str, kind: str, binary_path: str) -> dict:
        resp = http_post(
            f"{self.base_url}/api/engine/custom",
            json={"name": name, "kind": kind, "binary_path": binary_path},
            headers=self._headers(),
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()

    # ---- per-model run config (LM Studio style) ----

    def get_model_run_config(self, model_name: str) -> dict:
        resp = http_get(
            f"{self.base_url}/api/models/{model_name}/config", headers=self._headers(), timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    def save_model_run_config(self, model_name: str, cfg: dict) -> dict:
        resp = http_post(
            f"{self.base_url}/api/models/{model_name}/config",
            json=cfg, headers=self._headers(), timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_model_run_config(self, model_name: str) -> dict:
        resp = http_delete(
            f"{self.base_url}/api/models/{model_name}/config", headers=self._headers(), timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    # ---- download settings (proxy / ollama store) ----

    def get_download_config(self) -> dict:
        resp = http_get(f"{self.base_url}/api/download/config", headers=self._headers(), timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def set_download_config(self, cfg: dict) -> dict:
        resp = http_post(
            f"{self.base_url}/api/download/config", json=cfg, headers=self._headers(), timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    # ---- config approval (ConfigGuardian) ----

    def get_pending_config_changes(self) -> list[dict]:
        """Poll for pending config-change requests waiting for user approval."""
        try:
            resp = http_get(
                f"{self.base_url}/api/config/pending",
                headers=self._headers(), timeout=5.0,
            )
            resp.raise_for_status()
            return resp.json().get("pending", [])
        except Exception:
            return []

    def approve_config_change(self, request_id: str) -> dict:
        resp = http_post(
            f"{self.base_url}/api/config/approve/{request_id}",
            headers=self._headers(), timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    def reject_config_change(self, request_id: str) -> dict:
        resp = http_post(
            f"{self.base_url}/api/config/reject/{request_id}",
            headers=self._headers(), timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()


class TaskWorker(QObject):
    """Generic background worker for API calls."""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


def _engine_error_text(raw: object) -> str:
    """Unwrap the engine's error payload.

    The 掬水 proxy reports an upstream failure as an SSE ``data:`` line inside a
    200 stream, and the value it puts in ``error`` is the engine's whole response
    body — itself JSON, double-encoded as a string. Show the inner ``message``
    (``request (15535 tokens) exceeds the available context size (2048 tokens)``)
    instead of a wall of escaped braces, but never lose the text: an error that
    is not JSON is passed through verbatim.
    """
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    try:
        inner = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(inner, dict):
        inner = inner.get("error", inner)
        if isinstance(inner, dict):
            for key in ("message", "detail", "code"):
                value = inner.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return text


class ChatStreamWorker(QObject):
    """Background worker for streaming chat."""
    # kind: "content" (final answer) or "reasoning" (model chain-of-thought)
    token_received = Signal(str, str)
    finished = Signal()
    error = Signal(str)
    # Emitted with a diagnostic dict when the repetition guard trips.
    repetition_detected = Signal(object)

    def __init__(self, api: LMLoaderAPI, model: str, messages: list[dict], temperature: float,
                 thinking: bool = True, max_tokens: int | None = None):
        super().__init__()
        self._api = api
        self._model = model
        self._messages = messages
        self._temperature = temperature
        self._thinking = thinking
        self._max_tokens = max_tokens
        self._resp_holder = _RespHolder()
        self._stop = threading.Event()

    def stop(self):
        """Abort the in-flight stream. Safe to call from the GUI thread.

        Closing the response makes the worker's ``iter_lines`` raise, which the
        run loop turns into a normal (silent) finish rather than an error dialog.
        """
        self._stop.set()
        resp = self._resp_holder.get()
        if resp is not None:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass

    def run(self):
        try:
            for kind, token in self._api.chat_stream_sse(
                self._model, self._messages, self._temperature, self._thinking,
                self._max_tokens, self._resp_holder,
            ):
                if self._stop.is_set():
                    break
                self.token_received.emit(kind, token)
            if self._stop.is_set():
                # Deliberately no finished() emit: the GUI already reset its
                # streaming state when the user asked to stop.
                return
            self.finished.emit()
        except Exception as e:
            if self._stop.is_set():
                return
            self.error.emit(str(e))


class _ResultBridge(QObject):
    """Lives in the main (GUI) thread; forwards worker results back to it.

    Worker signals are connected to this bridge's signals: because the bridge
    has main-thread affinity, cross-thread emits are queued and re-emitted in
    the GUI thread, so UI callbacks never run in a worker thread.
    """
    result_ready = Signal(object)
    error_ready = Signal(str)


def run_in_thread(fn, on_result, on_error):
    """Run a function in a background thread, calling on_result/on_error in the MAIN thread.

    The QThread's Python wrapper is kept alive by a mutual reference with the
    bridge until the thread finishes. Without this, GC may delete the C++ QThread
    while its thread is still running, which hard-crashes PySide6 (0xC0000409).
    """
    worker = TaskWorker(fn)
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    bridge = _ResultBridge()  # created in the calling (main) thread
    worker.finished.connect(bridge.result_ready)  # cross-thread -> queued to main thread
    worker.error.connect(bridge.error_ready)
    bridge.result_ready.connect(on_result)         # main thread -> direct call
    bridge.error_ready.connect(on_error)

    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)

    def _on_thread_finished():
        # Thread has exited; release the mutual refs so wrappers can be GC'd.
        bridge._thread = None
        thread._bridge = None

    thread.finished.connect(_on_thread_finished)
    thread.finished.connect(thread.deleteLater)

    # Keep the Python wrappers alive until the thread finishes.
    bridge._thread = thread
    thread._bridge = bridge
    thread._worker = worker
    thread.start()
    return thread
