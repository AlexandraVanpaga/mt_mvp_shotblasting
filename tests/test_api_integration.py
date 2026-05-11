"""HTTP integration tests for the translation API (FastAPI TestClient, stubbed MT)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.translate_cache import translate_cache


def test_translate_applies_glossary_and_postedit(api_client: TestClient) -> None:
    payload = {
        "text": "Order one red widget today.",
        "apply_glossary": True,
        "apply_postedit": True,
        "include_debug": False,
    }
    r = api_client.post("/api/v1/translate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["glossary_applied"] is True
    assert body["postedit_applied"] is True
    assert body["from_cache"] is False
    assert "widget rojo" in body["translation"].lower()
    assert "red widget" not in body["translation"].lower()


def test_translate_skips_glossary_when_disabled(api_client: TestClient) -> None:
    payload = {
        "text": "Order one red widget today.",
        "apply_glossary": False,
        "apply_postedit": False,
        "include_debug": False,
    }
    r = api_client.post("/api/v1/translate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["glossary_applied"] is False
    assert body["postedit_applied"] is False
    assert "red widget" in body["translation"].lower()


def test_translate_cache_second_request_is_hit(api_client: TestClient) -> None:
    payload = {
        "text": "Second red widget cache line.",
        "apply_glossary": True,
        "apply_postedit": True,
        "include_debug": False,
    }
    first = api_client.post("/api/v1/translate", json=payload)
    assert first.status_code == 200
    assert first.json()["from_cache"] is False

    second = api_client.post("/api/v1/translate", json=payload)
    assert second.status_code == 200
    assert second.json()["from_cache"] is True
    assert second.json()["translation"] == first.json()["translation"]


def test_translate_include_debug_skips_cache_and_returns_debug(api_client: TestClient) -> None:
    payload = {
        "text": "Debug red widget path.",
        "apply_glossary": True,
        "apply_postedit": False,
        "include_debug": True,
    }
    r = api_client.post("/api/v1/translate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["from_cache"] is False
    assert body["debug"] is not None
    assert "protected_source" in body["debug"]
    assert "__GLS" in (body["debug"]["protected_source"] or "")


def test_translate_empty_text_returns_422(api_client: TestClient) -> None:
    r = api_client.post(
        "/api/v1/translate",
        json={
            "text": "",
            "apply_glossary": True,
            "apply_postedit": False,
            "include_debug": False,
        },
    )
    assert r.status_code == 422


def test_health_returns_ok(api_client: TestClient) -> None:
    r = api_client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "cuda_available" in data
    assert data["mt_engine_configured"] == "ctranslate2"
    assert isinstance(data.get("translate_cache_entries"), int)


def test_clear_translate_cache(api_client: TestClient) -> None:
    translate_cache.clear()
    api_client.post(
        "/api/v1/translate",
        json={
            "text": "Warm cache for clear test.",
            "apply_glossary": False,
            "apply_postedit": False,
            "include_debug": False,
        },
    )
    assert translate_cache.size() >= 1

    r = api_client.post("/api/v1/translate/cache/clear")
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "ok"
    assert out["entries_removed"] >= 1
    assert translate_cache.size() == 0
