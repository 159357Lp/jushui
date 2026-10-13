"""The load/unload endpoints take the model name in DIFFERENT places.

Hit while writing probe_models_poll_live.py: `load` wants a JSON body,
`unload` wants a query parameter, and mixing them up returns a 422 whose
detail ("loc": ["query", "model_name"]) points at the real cause only if you
read it carefully. A probe that fails to clean up after itself then leaves an
engine running — the exact failure the user reported twice.

These tests read the real route signatures rather than calling the app, so they
stay fast and need no server.
"""

import ast
import inspect
import pathlib

import pytest

ROUTES = pathlib.Path(__file__).resolve().parent.parent / "lmloader" / "api" / "routes" / "models.py"


def _func(name):
    tree = ast.parse(ROUTES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {ROUTES.name}")


def test_load_takes_a_pydantic_body():
    """load(request: LoadModelRequest) -> model_name comes from the JSON body."""
    node = _func("load_model")
    args = [a.arg for a in node.args.args]
    assert args == ["request"], (
        "load must take a single Pydantic body; if this changes, the probe and "
        "any external client posting to /api/models/load must change with it"
    )


def test_unload_takes_a_query_param():
    """unload(model_name: str) -> a bare str param is a QUERY param in FastAPI.

    This is the asymmetry that caused a 422: sending model_name as JSON to
    unload fails, because FastAPI looks for it in the query string.
    """
    node = _func("unload_model")
    args = node.args.args
    assert len(args) == 1 and args[0].arg == "model_name"
    assert args[0].annotation is None or getattr(args[0].annotation, "id", "") == "str", (
        "unload's parameter must stay a plain str (=> query param). Annotating it "
        "with a BaseModel would silently move it to the request body and break "
        "every existing caller."
    )
    # A body model would show up as the annotation being a ClassDef, not Name.
    assert not isinstance(args[0].annotation, ast.ClassDef), (
        "a ClassDef annotation means unload switched to a request body — that "
        "changes the HTTP contract"
    )


def test_unload_has_no_pydantic_body_param():
    node = _func("unload_model")
    assert node.args.kwonlyargs == []
    assert node.args.vararg is None
    assert node.args.kwarg is None


def test_the_two_endpoints_disagree_on_placement():
    """Explicit statement of the trap, so deleting it looks deliberate."""
    load_node = _func("load_model")
    unload_node = _func("unload_model")
    assert load_node.args.args[0].arg == "request"
    assert unload_node.args.args[0].arg == "model_name"
    assert load_node.args.args[0].arg != unload_node.args.args[0].arg, (
        "if both now take the same argument, one of the two call styles in "
        "probe_models_poll_live.py must be updated"
    )


def test_probe_uses_the_right_style_for_each_call():
    """Lock the probe itself: load->body, unload->query."""
    probe = pathlib.Path(__file__).resolve().parent.parent / "probe_models_poll_live.py"
    text = probe.read_text(encoding="utf-8")
    assert '/api/models/load", {"model_name"' in text, "load must send a JSON body"
    assert "/api/models/unload?model_name=" in text, (
        "unload must pass model_name in the query string"
    )
    assert '/api/models/unload", {"model_name"' not in text, (
        "unload with a JSON body returns 422 and leaves the engine running"
    )