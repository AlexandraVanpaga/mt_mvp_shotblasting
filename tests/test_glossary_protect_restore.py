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


def test_enforce_placeholders_tolerates_one_trailing_underscore(tmp_path: Path) -> None:
    """Marian sometimes outputs ``__GLS62_`` (single trailing underscore) next to
    punctuation. The restoration must still recover the canonical target."""
    path = _write_glossary(
        tmp_path,
        [{"source": "panblast", "target": "Panblast", "notes": None}],
    )
    g = Glossary(path)
    _, ph = g.protect_source("panblast")
    # Simulate MT output where '__' got truncated to '_' next to '.' and '@'
    restored = g.enforce_placeholders("contacto@ __GLS0_ .com www. __GLS0_ .com", ph)
    assert "Panblast" in restored
    assert "__GLS" not in restored
    assert restored.count("Panblast") == 2


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


# ─── reassert_targets_after_edit ───────────────────────────────────────────────


def _gloss_with(tmp_path: Path, source: str, target: str) -> Glossary:
    path = _write_glossary(tmp_path, [{"source": source, "target": target, "notes": None}])
    return Glossary(path)


def test_reassert_no_op_when_reference_equals_edited(tmp_path: Path) -> None:
    g = _gloss_with(tmp_path, "Remote control valve", "Válvula neumática")
    ref = "Conecte la Válvula neumática al pezón."
    assert g.reassert_targets_after_edit(ref, ref) == ref


def test_reassert_fixes_accent_drift(tmp_path: Path) -> None:
    """Qwen drops the acute on ``á`` and capitalises ``Neumática`` — must snap back."""
    g = _gloss_with(tmp_path, "Remote control valve", "Válvula neumática")
    ref = "Conecte la Válvula neumática al pezón roscado."
    edited = "Conecte la Valvula Neumática al pezón roscado."
    assert g.reassert_targets_after_edit(ref, edited) == ref


def test_reassert_fixes_multiple_drifted_occurrences_in_one_row(tmp_path: Path) -> None:
    g = _gloss_with(tmp_path, "Remote control valve", "Válvula neumática")
    ref = "La Válvula neumática de entrada y la Válvula neumática de salida."
    # Qwen drifted BOTH occurrences (the v1 bug only fixed the first).
    edited = "La Valvula Neumatica de entrada y la VALVULA NEUMATICA de salida."
    out = g.reassert_targets_after_edit(ref, edited)
    assert out.count("Válvula neumática") == 2
    assert "Valvula" not in out
    assert "VALVULA" not in out


def test_reassert_handles_canonical_plus_drifted_in_one_row(tmp_path: Path) -> None:
    """First occurrence is already canonical; second drifted. Both must be canonical at the end."""
    g = _gloss_with(tmp_path, "Remote control valve", "Válvula neumática")
    ref = "La Válvula neumática y la Válvula neumática otra."
    edited = "La Válvula neumática y la Valvula Neumatica otra."
    out = g.reassert_targets_after_edit(ref, edited)
    assert out.count("Válvula neumática") == 2


def test_reassert_falls_back_to_reference_when_qwen_drops_term(tmp_path: Path) -> None:
    """When Qwen completely rephrases the term away, we prefer MT-only correctness."""
    g = _gloss_with(tmp_path, "Remote control valve", "Válvula neumática")
    ref = "Conecte la Válvula neumática al pezón roscado."
    edited = "Conecte la válvula remota de escape al pezón roscado."  # Qwen rephrased
    assert g.reassert_targets_after_edit(ref, edited) == ref


def test_reassert_does_not_touch_terms_not_in_reference(tmp_path: Path) -> None:
    """If the term wasn't in the pre-edit reference, nothing should change."""
    g = _gloss_with(tmp_path, "Remote control valve", "Válvula neumática")
    ref = "Un texto sin el término."
    edited = "Un texto modificado por Qwen."
    assert g.reassert_targets_after_edit(ref, edited) == edited


def test_reassert_keeps_other_qwen_edits_when_glossary_preserved(tmp_path: Path) -> None:
    """Qwen-improved phrasing must survive when the glossary term is intact."""
    g = _gloss_with(tmp_path, "Remote control valve", "Válvula neumática")
    ref = "Conecte la Válvula neumática a el pezón."
    edited = "Conecte la Válvula neumática al pezón roscado de salida."  # better phrasing
    assert g.reassert_targets_after_edit(ref, edited) == edited


def test_reassert_with_multiple_glossary_entries_one_dropped(tmp_path: Path) -> None:
    """If ANY canonical term is dropped, fall back to reference (glossary > fluency)."""
    path = _write_glossary(
        tmp_path,
        [
            {"source": "Remote control valve", "target": "Válvula neumática", "notes": None},
            {"source": "Blast pot", "target": "Tolva", "notes": None},
        ],
    )
    g = Glossary(path)
    ref = "Desmonte la Válvula neumática del Tolva."
    # Qwen kept Tolva but rephrased Válvula neumática away
    edited = "Desmonte la válvula remota del Tolva."
    assert g.reassert_targets_after_edit(ref, edited) == ref


def test_reassert_ignores_very_short_targets(tmp_path: Path) -> None:
    """Single-char targets should not trigger any rewrite logic."""
    path = _write_glossary(
        tmp_path,
        [{"source": "x", "target": "y", "notes": None}],
    )
    g = Glossary(path)
    assert g.reassert_targets_after_edit("y here", "edited text without y") == "edited text without y"


# ── v3 regression: word-bounded protection (Spartan must not match Spartans) ──
def test_protect_source_word_bounded_no_partial_match(tmp_path: Path) -> None:
    """Short verbatim entries must not bleed into longer adjacent words."""
    path = _write_glossary(
        tmp_path,
        [{"source": "Spartan", "target": "Spartan", "notes": None}],
    )
    g = Glossary(path)
    out, ph = g.protect_source("The Spartans defeated the Spartan helmet team.")
    # The standalone "Spartan" is replaced; the plural "Spartans" stays put.
    assert "Spartans" in out
    assert out.count("__GLS0__") == 1
    assert ph["__GLS0__"] == "Spartan"


def test_protect_source_verbatim_brand_round_trips(tmp_path: Path) -> None:
    """A verbatim brand name survives MT untouched after protect + enforce."""
    path = _write_glossary(
        tmp_path,
        [{"source": "Spartan", "target": "Spartan", "notes": None}],
    )
    g = Glossary(path)
    protected, ph = g.protect_source("Use the Spartan helmet now.")
    # Simulate MT keeping the placeholder verbatim and translating the rest.
    fake_mt = protected.replace("Use the", "Use el").replace("helmet now", "casco ahora")
    restored = g.enforce_placeholders(fake_mt, ph)
    assert "Spartan" in restored
    assert "__GLS" not in restored
