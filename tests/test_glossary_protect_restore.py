"""Tests for glossary source protection (placeholders) and MT output restoration."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.glossary import Glossary


def _write_glossary(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "glossary.json"
    path.write_text(
        json.dumps({"entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_protect_source_replaces_phrase_with_placeholder(tmp_path: Path) -> None:
    path = _write_glossary(
        tmp_path,
        [{"source": "blast cabinet", "target": "cabina de granallado", "notes": None}],
    )
    g = Glossary(path)
    out, ph = g.protect_source("We need a blast cabinet today.")
    assert "__GLS0__" in out
    assert "blast cabinet" not in out.lower()
    assert ph["__GLS0__"] == "cabina de granallado"


def test_protect_source_is_case_insensitive(tmp_path: Path) -> None:
    path = _write_glossary(
        tmp_path,
        [{"source": "Steel Grit", "target": "GRANO DE ACERO", "notes": None}],
    )
    g = Glossary(path)
    out, ph = g.protect_source("steel grit and STEEL GRIT")
    assert out.lower().count("steel grit") == 0
    assert "__GLS0__" in out
    assert ph["__GLS0__"] == "GRANO DE ACERO"


def test_protect_source_prefers_longer_match_first(tmp_path: Path) -> None:
    path = _write_glossary(
        tmp_path,
        [
            {"source": "cab", "target": "SHORT", "notes": None},
            {"source": "blast cabinet", "target": "LONG TARGET", "notes": None},
        ],
    )
    g = Glossary(path)
    out, ph = g.protect_source("Use blast cabinet only.")
    # Sorted by source length descending: "blast cabinet" before "cab"
    assert "SHORT" not in out
    assert "__GLS0__" in out
    assert ph["__GLS0__"] == "LONG TARGET"
    assert ph["__GLS1__"] == "SHORT"


def test_enforce_placeholders_restores_targets(tmp_path: Path) -> None:
    path = _write_glossary(
        tmp_path,
        [{"source": "nozzle", "target": "boquilla", "notes": None}],
    )
    g = Glossary(path)
    _, ph = g.protect_source("nozzle")
    restored = g.enforce_placeholders("La __GLS0__ nueva.", ph)
    assert "boquilla" in restored
    assert "__GLS0__" not in restored


def test_enforce_placeholders_tolerates_missing_spaces_around_token(tmp_path: Path) -> None:
    path = _write_glossary(
        tmp_path,
        [{"source": "hose", "target": "manguera", "notes": None}],
    )
    g = Glossary(path)
    _, ph = g.protect_source("hose")
    # Simulate MT gluing placeholders to neighbors
    restored = g.enforce_placeholders("La__GLS0__rota", ph)
    assert "manguera" in restored
    assert "__GLS0__" not in restored


def test_enforce_placeholders_unknown_token_left_unchanged(tmp_path: Path) -> None:
    path = _write_glossary(
        tmp_path,
        [{"source": "a", "target": "b", "notes": None}],
    )
    g = Glossary(path)
    ph = {"__GLS0__": "b"}
    out = g.enforce_placeholders("x __GLS99__ y", ph)
    assert "__GLS99__" in out


def test_protect_then_restore_round_trip_without_mt(tmp_path: Path) -> None:
    path = _write_glossary(
        tmp_path,
        [
            {"source": "control valve", "target": "válvula de control", "notes": None},
            {"source": "metering valve", "target": "válvula dosificadora", "notes": None},
        ],
    )
    g = Glossary(path)
    source = "Replace control valve and metering valve."
    protected, ph = g.protect_source(source)
    fake_mt = protected.replace("__GLS0__", "__GLS0__").replace("__GLS1__", "__GLS1__")
    restored = g.enforce_placeholders(fake_mt, ph)
    assert "válvula de control" in restored
    assert "válvula dosificadora" in restored
    assert "__GLS" not in restored


def test_protect_source_collapses_extra_whitespace(tmp_path: Path) -> None:
    path = _write_glossary(
        tmp_path,
        [{"source": "x", "target": "y", "notes": None}],
    )
    g = Glossary(path)
    out, _ = g.protect_source("  a  x  b  ")
    assert "  " not in out.strip()
    assert out == "a __GLS0__ b"


def test_enforce_placeholders_strips_and_collapses_spaces(tmp_path: Path) -> None:
    path = _write_glossary(
        tmp_path,
        [{"source": "k", "target": "kk", "notes": None}],
    )
    g = Glossary(path)
    ph = {"__GLS0__": "kk"}
    out = g.enforce_placeholders("  __GLS0__  __GLS0__  ", ph)
    assert out == "kk kk"
