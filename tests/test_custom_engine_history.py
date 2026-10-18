"""自定义引擎的登记历史：换二进制要留痕，回退要换得回去。

自定义引擎没有安装目录可回退（它不在 engines/ 里，掬水 也不该搬用户的文件），
所以「更新/回退」只能是对 config.yaml 里那条登记的换入换出。这里盯的就是这个
换入换出会不会丢历史、会不会把已经不存在的文件登记回去。
"""

from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """一个只认 tmp_path 的服务实例：配置和二进制都在临时目录里。"""
    import lmloader.config as config
    import lmloader.api.routes.engine as engine_routes

    config.settings._config.server.api_token = ""

    engines_dir = tmp_path / "engines"
    engines_dir.mkdir()
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"engine_scan_dirs": [str(engines_dir)]}, allow_unicode=True),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # 路径围栏本身在 test_api_security 里判；这里只关心历史链，所以让临时目录合法。
    monkeypatch.setattr(engine_routes, "_allowed_engine_roots", lambda: [engines_dir])
    monkeypatch.setattr(engine_routes, "local_version",
                        AsyncMock(return_value={"build": 1, "raw": "b-test"}))
    monkeypatch.setattr(engine_routes.config_guardian, "wait_for_approval",
                        AsyncMock(return_value=True))
    from lmloader.api.app import app

    with TestClient(app) as c:
        yield c


def _make_binary(tmp_path, name: str) -> str:
    binary = tmp_path / "engines" / name
    binary.write_bytes(b"MZ")
    return str(binary)


def _customs(tmp_path) -> list[dict]:
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8")) or {}
    return cfg.get("custom_engines", []) or []


def _register(client, tmp_path, name="rwkv-runner"):
    v1 = _make_binary(tmp_path, "runner-v1.exe")
    resp = client.post("/api/engine/custom",
                       json={"name": name, "kind": "rwkv", "binary_path": v1})
    assert resp.status_code == 200, resp.text
    return name, v1


def test_swap_then_rollback_round_trips_the_registration(client, tmp_path):
    name, v1 = _register(client, tmp_path)
    v2 = _make_binary(tmp_path, "runner-v2.exe")

    resp = client.post("/api/engine/custom/update", json={"name": name, "binary_path": v2})
    assert resp.status_code == 200, resp.text
    entry = _customs(tmp_path)[0]
    assert entry["binary_path"] == v2
    assert [h["binary_path"] for h in entry["history"]] == [v1]
    assert entry["history"][0]["version"] == "b-test"

    resp = client.post("/api/engine/custom/rollback", params={"name": name})
    assert resp.status_code == 200, resp.text
    entry = _customs(tmp_path)[0]
    assert entry["binary_path"] == v1
    assert [h["binary_path"] for h in entry["history"]] == [v2]


def test_swap_to_the_same_binary_changes_nothing(client, tmp_path):
    name, v1 = _register(client, tmp_path)
    resp = client.post("/api/engine/custom/update", json={"name": name, "binary_path": v1})
    assert resp.status_code == 200
    assert resp.json()["status"] == "already_current"
    assert _customs(tmp_path)[0].get("history", []) == []


def test_rollback_without_history_is_404(client, tmp_path):
    name, _ = _register(client, tmp_path)
    resp = client.post("/api/engine/custom/rollback", params={"name": name})
    assert resp.status_code == 404
    assert "历史" in resp.json()["detail"]


def test_rollback_refuses_a_deleted_previous_binary(client, tmp_path):
    name, v1 = _register(client, tmp_path)
    v2 = _make_binary(tmp_path, "runner-v2.exe")
    assert client.post("/api/engine/custom/update",
                       json={"name": name, "binary_path": v2}).status_code == 200
    (tmp_path / "engines" / "runner-v1.exe").unlink()

    resp = client.post("/api/engine/custom/rollback", params={"name": name})
    assert resp.status_code == 404
    assert "不在原处" in resp.json()["detail"]
    # 拒绝必须落在改配置之前
    assert _customs(tmp_path)[0]["binary_path"] == v2


def test_unknown_engine_name_is_404(client, tmp_path):
    resp = client.post("/api/engine/custom/update",
                       json={"name": "ghost", "binary_path": _make_binary(tmp_path, "g.exe")})
    assert resp.status_code == 404
    resp = client.post("/api/engine/custom/rollback", params={"name": "ghost"})
    assert resp.status_code == 404


def test_denied_approval_leaves_the_registration_untouched(client, tmp_path, monkeypatch):
    import lmloader.api.routes.engine as engine_routes

    name, v1 = _register(client, tmp_path)
    monkeypatch.setattr(engine_routes.config_guardian, "wait_for_approval",
                        AsyncMock(return_value=False))
    v2 = _make_binary(tmp_path, "runner-v2.exe")
    resp = client.post("/api/engine/custom/update", json={"name": name, "binary_path": v2})
    assert resp.status_code == 403
    assert _customs(tmp_path)[0]["binary_path"] == v1


def test_history_is_capped_and_ordered(client, tmp_path):
    name, _ = _register(client, tmp_path)
    for i in range(12):
        path = _make_binary(tmp_path, f"runner-{i}.exe")
        assert client.post("/api/engine/custom/update",
                           json={"name": name, "binary_path": path}).status_code == 200
    history = _customs(tmp_path)[0]["history"]
    # 历史里放的是「被挤掉」的那些：最新一条是 runner-10（runner-11 仍在任），
    # 最早的 v1 和 runner-0 已经被 10 条上限挤出去。
    assert len(history) == 10
    assert history[0]["binary_path"].endswith("runner-10.exe")
    assert history[-1]["binary_path"].endswith("runner-1.exe")


def test_re_registering_a_name_keeps_its_history(client, tmp_path):
    name, v1 = _register(client, tmp_path)
    v2 = _make_binary(tmp_path, "runner-v2.exe")
    assert client.post("/api/engine/custom/update",
                       json={"name": name, "binary_path": v2}).status_code == 200

    v3 = _make_binary(tmp_path, "runner-v3.exe")
    assert client.post("/api/engine/custom",
                       json={"name": name, "kind": "rwkv", "binary_path": v3}).status_code == 200
    entry = _customs(tmp_path)[0]
    assert entry["binary_path"] == v3
    # 被 v3 挤掉的 v2 必须排在旧历史前面，否则回退会跳过它
    assert [h["binary_path"] for h in entry["history"]] == [v2, v1]


def test_list_endpoint_exposes_the_history_the_gui_needs(client, tmp_path):
    name, v1 = _register(client, tmp_path)
    v2 = _make_binary(tmp_path, "runner-v2.exe")
    client.post("/api/engine/custom/update", json={"name": name, "binary_path": v2})
    body = client.get("/api/engine/custom").json()
    entry = body["custom_engines"][0]
    assert entry["binary_path"] == v2
    assert entry["history"][0]["binary_path"] == v1
