# -*- coding: utf-8 -*-
"""An engine error must never reach the chat as "(空回复)".

Reproduces the Bonsai-2 incident: the model's saved run config pinned ctx_size
to 2048, a 15,535-token conversation was refused by llama-server, and the 掬水
proxy forwarded that refusal as an SSE ``data: {"error": …}`` line inside an
HTTP 200 stream. The stream parser only looked at ``choices[0].delta``, so the
worker saw zero tokens, no exception, and the chat painted an empty reply.
"""

import json

import httpx
import pytest

import lmloader.gui.api_client as ac


def _sse(*payloads):
    """Render proxy-style SSE lines: each payload is one ``data:`` event."""
    return [f"data: {json.dumps(p, ensure_ascii=False)}" for p in payloads]


class _Response:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Client:
    """Stands in for httpx.Client: hands back a scripted SSE stream."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, method, url, json=None, headers=None):
        return _Response(self._lines)


def _patch(monkeypatch, lines):
    monkeypatch.setattr(ac.httpx, "Client", lambda timeout=None, **kw: _Client(lines))


def _drive(monkeypatch, lines):
    """Consume chat_stream_sse against a scripted SSE stream."""
    _patch(monkeypatch, lines)
    api = ac.LMLoaderAPI()
    return list(api.chat_stream_sse("m", [{"role": "user", "content": "hi"}],
                                    0.7, True, ac.DEFAULT_MAX_TOKENS))


OVERFLOW = json.dumps({
    "error": {
        "code": 400,
        "message": "request (15535 tokens) exceeds the available context size "
                   "(2048 tokens), try increasing it",
        "type": "exceed_context_size_error",
        "n_prompt_tokens": 15535,
        "n_ctx": 2048,
    }
}, ensure_ascii=False)


def test_context_overflow_raises_instead_of_yielding_nothing(monkeypatch):
    """The proxy double-encodes the engine body as a string; the message survives."""
    with pytest.raises(RuntimeError) as excinfo:
        _drive(monkeypatch, _sse({"error": OVERFLOW}))
    assert "exceeds the available context size (2048 tokens)" in str(excinfo.value)


def test_normal_deltas_still_stream(monkeypatch):
    """Control: the error guard must not touch a healthy stream."""
    got = _drive(monkeypatch, _sse(
        {"choices": [{"delta": {"content": "你好"}}]},
        {"choices": [{"delta": {"reasoning_content": "想"}}]},
    ))
    assert got == [("content", "你好"), ("reasoning", "想")]


def test_error_dict_form_is_unwrapped(monkeypatch):
    """Some engines put the object inline instead of stringified."""
    with pytest.raises(RuntimeError) as excinfo:
        _drive(monkeypatch, _sse({"error": {"message": "CUDA out of memory"}}))
    assert str(excinfo.value) == "CUDA out of memory"


def test_plain_text_error_is_not_mangled(monkeypatch):
    """A body that is not JSON gets through verbatim rather than crashing the parse."""
    with pytest.raises(RuntimeError) as excinfo:
        _drive(monkeypatch, _sse({"error": "server closed the connection"}))
    assert str(excinfo.value) == "server closed the connection"


def test_empty_error_value_falls_back_to_the_raw_payload(monkeypatch):
    """No message/detail/code: better a blob than a blank bubble."""
    with pytest.raises(RuntimeError) as excinfo:
        _drive(monkeypatch, _sse({"error": {"code": 503}}))
    assert "503" in str(excinfo.value)


@pytest.fixture(scope="module")
def qapp():
    """Signals need a QCoreApplication, but not the GUI or a platform plugin."""
    from PySide6.QtCore import QCoreApplication
    return QCoreApplication.instance() or QCoreApplication([])


def test_worker_turns_the_error_event_into_a_signal(monkeypatch, qapp):
    """Closes the loop: ChatStreamWorker must emit error(), not finished().

    This is what makes the chat render the engine's words — the page only paints
    "(空回复)" from the silent-success path.
    """
    events = []
    worker = ac.ChatStreamWorker(
        ac.LMLoaderAPI(), "m", [{"role": "user", "content": "hi"}], 0.7, True)
    worker.error.connect(lambda msg: events.append(("error", msg)))
    worker.finished.connect(lambda: events.append(("finished", None)))
    _patch(monkeypatch, _sse({"error": OVERFLOW}))
    worker.run()
    assert [kind for kind, _ in events] == ["error"]
    assert "exceeds the available context size" in events[0][1]


def test_user_stop_still_silences_the_aborted_stream(monkeypatch, qapp):
    """The stop button closes the response, so the error path must stay quiet."""
    events = []
    worker = ac.ChatStreamWorker(
        ac.LMLoaderAPI(), "m", [{"role": "user", "content": "hi"}], 0.7, True)
    worker.error.connect(lambda msg: events.append(msg))

    class Aborted(_Response):
        """The first token arrives, then the user hits stop and the read fails."""
        def iter_lines(self):
            worker.stop()
            yield f"data: {json.dumps({'choices': [{'delta': {'content': '半句'}}]})}"
            raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

    monkeypatch.setattr(ac.httpx, "Client", lambda timeout=None, **kw: _Client([]))
    monkeypatch.setattr(_Client, "stream", lambda self, *a, **kw: Aborted([]))
    worker.run()
    assert events == []
