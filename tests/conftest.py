"""Shared fixtures for API integration tests (dependency overrides, no real MT weights)."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import create_app
from app.core.config import Settings
from app.services.glossary import Glossary
from app.services.postedit import PostEditor
from app.services.translate_cache import translate_cache


class FakeMTEngine:
    """Deterministic MT: returns the protected source string unchanged."""

    def translate(self, text: str) -> str:
        return text


@pytest.fixture
def glossary_entries() -> list[dict]:
    return [{"source": "red widget", "target": "widget rojo", "notes": None}]


@pytest.fixture
def api_client(tmp_path: Path, glossary_entries: list[dict]) -> Generator[TestClient, None, None]:
    glossary_path = tmp_path / "glossary.json"
    glossary_path.write_text(
        json.dumps({"entries": glossary_entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    postedit_prompt_path = tmp_path / "postedit.md"
    postedit_prompt_path.write_text(
        "Preserve meaning and glossary Spanish terms.\n",
        encoding="utf-8",
    )

    test_settings = Settings(
        glossary_path=glossary_path,
        postedit_prompt_path=postedit_prompt_path,
        postedit_use_qwen=False,
        mt_engine="ctranslate2",
        ct2_model_dir=tmp_path / "ct2_weights",
        _env_file=None,
    )

    glossary = Glossary(glossary_path)
    posteditor = PostEditor(postedit_prompt_path, qwen=None)

    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: test_settings
    app.dependency_overrides[deps.get_mt_engine] = lambda: FakeMTEngine()
    app.dependency_overrides[deps.get_glossary] = lambda: glossary
    app.dependency_overrides[deps.get_posteditor] = lambda: posteditor

    translate_cache.clear()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    translate_cache.clear()
