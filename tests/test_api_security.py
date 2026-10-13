"""Tests for the API hardening added after the full-stack audit (P4).

Covers three independent pieces:

* bearer-token auth on the control plane, inert while no token is configured;
* security headers on every response;
* the custom-engine ``binary_path`` confinement — without it, that endpoint is
  "spawn any executable on this machine".
"""

import pytest
from fastapi.testclient import TestClient

from lmloader.api.security import (
    SECURITY_HEADERS,
    generate_token,
    is_loopback_host,
    redact_token,
    unauthenticated_network_warning,
)


TOKEN = "test-token-abcdef123456"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="session")
def app_client():
    """One TestClient for the whole session.

    The app's lifespan starts the database and runs engine discovery, which
    costs ~10s. Per-test clients made the file take minutes, so the client is
    session-scoped and the auth state is toggled per test instead (the token
    provider reads settings on every request).
    """
    import lmloader.config as config
    from lmloader.api.app import app

    config.settings._config.server.api_token = ""
    with TestClient(app) as c:
        yield c
    config.settings._config.server.api_token = ""


@pytest.fixture
def client(app_client):
    """Session client with auth guaranteed off before each test."""
    import lmloader.config as config

    config.settings._config.server.api_token = ""
    yield app_client
    config.settings._config.server.api_token = ""


@pytest.fixture
def isolated_config():
    """Restore config.yaml after a test that legitimately writes to it."""
    import os
    import shutil

    backup = "config.yaml.security_test_backup"
    shutil.copy("config.yaml", backup)
    try:
        yield
    finally:
        shutil.copy(backup, "config.yaml")
        os.remove(backup)


def _with_token(token):
    import lmloader.config as config
    config.settings._config.server.api_token = token


# --------------------------------------------------------------- auth: off

def test_no_token_configured_keeps_the_api_open(client):
    """The default install is a loopback desktop tool; adding auth must not
    lock the user out of their own backend."""
    for path in ("/health", "/api/models", "/api/engine", "/api/monitor"):
        assert client.get(path).status_code == 200, path


def test_security_headers_are_always_present(client):
    response = client.get("/health")
    for name, value in SECURITY_HEADERS.items():
        assert response.headers.get(name) == value, f"{name} missing"


# --------------------------------------------------------------- auth: on

@pytest.mark.parametrize("method,path,payload", [
    ("get", "/api/models", None),
    ("get", "/api/engine", None),
    ("get", "/api/monitor", None),
    ("get", "/api/events", None),
    ("get", "/api/download/config", None),
    ("post", "/api/models/load", {"model_name": "x"}),
    ("post", "/api/shutdown", None),
])
def test_control_plane_requires_the_token(client, method, path, payload):
    _with_token(TOKEN)
    try:
        response = getattr(client, method)(path, **({"json": payload} if payload else {}))
        assert response.status_code == 401, f"{method.upper()} {path} was reachable"
        # FastAPI-shaped error so the GUI's _http_detail() can render it.
        assert isinstance(response.json().get("detail"), str)
    finally:
        _with_token("")


def test_health_stays_public(client):
    """The GUI polls /health before it knows whether a token is configured."""
    _with_token(TOKEN)
    try:
        assert client.get("/health").status_code == 200
    finally:
        _with_token("")


@pytest.mark.parametrize("header,expected", [
    (f"Bearer {TOKEN}", 200),
    (f"bearer {TOKEN}", 200),          # scheme is case-insensitive per RFC 7235
    ("Bearer wrong-token", 401),
    (f"Token {TOKEN}", 401),            # wrong scheme
    (TOKEN, 401),                       # no scheme at all
    ("", 401),
])
def test_token_variants(client, header, expected):
    _with_token(TOKEN)
    try:
        response = client.get("/api/models", headers={"Authorization": header})
        assert response.status_code == expected, f"header={header!r}"
    finally:
        _with_token("")


def test_valid_token_unlocks_the_api(client):
    _with_token(TOKEN)
    try:
        for path in ("/api/models", "/api/engine", "/api/monitor", "/api/events"):
            assert client.get(path, headers=AUTH).status_code == 200, path
    finally:
        _with_token("")


def test_gui_client_sends_the_token():
    """If the GUI omits the header, enabling auth breaks the whole desktop app
    — so every HTTP call must carry it."""
    import ast
    import inspect

    from lmloader.gui.api_client import LMLoaderAPI

    source = inspect.getsource(inspect.getmodule(LMLoaderAPI))
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("get", "post", "delete", "put", "stream"):
            continue
        receiver = ast.unparse(node.func.value)
        if not (receiver == "httpx" or receiver.endswith("client")):
            continue
        calls.append(node)

    assert calls, "no HTTP calls found — the AST scan is stale"
    missing = [node.lineno for node in calls
               if not any(kw.arg == "headers" for kw in node.keywords)]
    assert not missing, f"these calls send no Authorization header: {missing}"

    assert LMLoaderAPI("http://x", token="abc")._headers() == {"Authorization": "Bearer abc"}
    assert LMLoaderAPI("http://x")._headers() == {}


# ------------------------------------------------------- engine path limit

@pytest.mark.parametrize("binary_path", [
    r"C:\Windows\System32\cmd.exe",
    r"C:\Windows\notepad.exe",
    "../../../../Windows/System32/calc.exe",
    "/usr/bin/python3",
])
def test_custom_engine_rejects_binaries_outside_the_allowed_roots(client, binary_path):
    response = client.post("/api/engine/custom",
                           headers=AUTH,
                           json={"name": "evil", "kind": "llama",
                                 "binary_path": binary_path})
    assert response.status_code == 400, binary_path
    assert "引擎目录" in response.json().get("detail", "")


def test_custom_engine_rejects_a_missing_file_inside_an_allowed_root(client):
    from lmloader.infra.engine_store import engine_dir

    response = client.post("/api/engine/custom",
                           headers=AUTH,
                           json={"name": "ghost", "kind": "llama",
                                 "binary_path": str(engine_dir() / "not-here.exe")})
    assert response.status_code == 400
    assert "不存在" in response.json().get("detail", "")


@pytest.fixture
def isolated_config():
    """Restore config.yaml after a test that legitimately writes to it."""
    import os
    import shutil

    backup = "config.yaml.security_test_backup"
    shutil.copy("config.yaml", backup)
    try:
        yield
    finally:
        shutil.copy(backup, "config.yaml")
        os.remove(backup)


def test_custom_engine_accepts_a_real_bundled_binary(client, isolated_config):
    from lmloader.infra.engine_store import engine_dir

    binary = engine_dir().resolve() / "llama-server.exe"
    if not binary.is_file():
        pytest.skip("bundled engine not present in this environment")
    response = client.post("/api/engine/custom",
                           headers=AUTH,
                           json={"name": "probe-ok", "kind": "llama",
                                 "binary_path": str(binary)})
    assert response.status_code == 200, response.text


# ------------------------------------------------------ host classification

@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True),
    ("localhost", True),
    ("::1", True),
    ("127.0.0.2", True),
    ("0.0.0.0", False),
    ("192.168.1.5", False),
    ("", False),
    ("my-laptop.local", False),   # unclassifiable -> assume the worst
])
def test_loopback_classification(host, expected):
    assert is_loopback_host(host) is expected


def test_warning_only_fires_off_loopback():
    assert unauthenticated_network_warning("127.0.0.1") is None
    warning = unauthenticated_network_warning("0.0.0.0")
    assert warning and "api_token" in warning


def test_acknowledged_exposure_downgrades_the_warning():
    """allow_unauthenticated_network must not silence the warning entirely —
    it shortens it to a note that still states the risk."""
    loud = unauthenticated_network_warning("0.0.0.0")
    quiet = unauthenticated_network_warning("0.0.0.0", acknowledged=True)
    assert quiet and len(quiet) < len(loud)
    assert "allow_unauthenticated_network" in quiet
    assert "api_token" in quiet
    assert unauthenticated_network_warning("127.0.0.1", acknowledged=True) is None


def test_startup_banner_mentions_auth_when_enabled():
    """Enabling a token should be visible at startup, not discovered later."""
    import inspect

    from lmloader.api.app import _emit_startup_banner

    source = inspect.getsource(_emit_startup_banner)
    assert "api_token" in source
    assert "unauthenticated_network_warning" in source


def test_token_helpers():
    assert len(generate_token()) >= 32
    assert redact_token("failed with sekret-value", "sekret-value") == "failed with ***"
    assert redact_token("nothing to hide", "") == "nothing to hide"