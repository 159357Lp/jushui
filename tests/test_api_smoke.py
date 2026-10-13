"""Smoke tests for FastAPI endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi.testclient import TestClient


@pytest.fixture
def mock_db():
    with patch("lmloader.api.app.db") as mock:
        mock.init = AsyncMock()
        mock.list_models = AsyncMock(return_value=[])
        yield mock


@pytest.fixture
def mock_lifecycle():
    with patch("lmloader.api.app.lifecycle") as mock:
        mock.get_loaded_models = AsyncMock(return_value=[])
        yield mock


@pytest.fixture
def client(mock_db):
    from lmloader.api.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_api_tags(client):
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data


def test_api_monitor(client):
    resp = client.get("/api/monitor")
    assert resp.status_code == 200
    data = resp.json()
    assert "cpu" in data
    assert "memory" in data
    assert "gpu" in data
