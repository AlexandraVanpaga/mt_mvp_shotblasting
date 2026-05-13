"""Tests for the ALL-CAPS sentence-case preprocessor."""

from __future__ import annotations

from app.services.text_case import (
    is_mostly_uppercase,
    postprocess_after_mt,
    preprocess_for_mt,
    to_sentence_case,
)


# ── is_mostly_uppercase ────────────────────────────────────────────────────


def test_detects_long_allcaps_warning() -> None:
    text = "NOTE: NEVER LIFT AND/OR CARRY THE SUPPLIED AIR RESPIRATOR HELMET."
    assert is_mostly_uppercase(text) is True


def test_mixed_case_paragraph_not_flagged() -> None:
    text = "Always wear the Helmet before operating the Remote control valve."
    assert is_mostly_uppercase(text) is False


def test_very_short_acronym_header_not_flagged() -> None:
    # 7-char headers are below the 8-alpha floor and pass through untouched.
    assert is_mostly_uppercase("NPT TIP") is False
    assert is_mostly_uppercase("PANBLAST") is True  # exactly 8 — flagged on purpose
    # 10-letter headers like "NPT FITTING" do trigger; sentence-casing them
    # to "Npt fitting" is a no-op for translation quality, so we accept it.
    assert is_mostly_uppercase("NPT FITTING") is True


def test_exactly_at_threshold() -> None:
    text = "SOME TEXT here"
    letters = sum(1 for c in text if c.isalpha())
    upper = sum(1 for c in text if c.isupper())
    ratio = upper / letters
    assert ratio > 0.5  # sanity
    if ratio >= 0.6:
        assert is_mostly_uppercase(text) is True


def test_lower_threshold_override() -> None:
    text = "Lower CASE Mixed"
    assert is_mostly_uppercase(text, threshold=0.4) is True
    assert is_mostly_uppercase(text, threshold=0.8) is False


def test_pure_digits_or_symbols_not_flagged() -> None:
    assert is_mostly_uppercase("12345 67890 !!!") is False


# ── to_sentence_case ───────────────────────────────────────────────────────


def test_sentence_case_simple() -> None:
    out = to_sentence_case("NEVER LIFT THE HELMET.")
    assert out == "Never lift the helmet."


def test_sentence_case_multiple_sentences() -> None:
    out = to_sentence_case("WARNING. DO NOT LIFT. CHECK SEALS.")
    assert out == "Warning. Do not lift. Check seals."


def test_sentence_case_preserves_punctuation_and_digits() -> None:
    out = to_sentence_case("TIGHTEN TO 25 N·M.")
    assert out == "Tighten to 25 n·m."


def test_sentence_case_keeps_glossary_placeholders_intact() -> None:
    # placeholders have no alphabetic chars except 'GLS', which lowercases
    # cleanly — but they must survive as a single token so the glossary
    # restore step still finds them.
    src = "NEVER LIFT THE __GLS76__ WITHOUT __GLS22__."
    out = to_sentence_case(src)
    assert "__gls76__" in out
    assert "__gls22__" in out
    # First word is sentence-cased
    assert out.startswith("Never lift")


def test_sentence_case_handles_exclamation_and_question() -> None:
    out = to_sentence_case("WARNING! IS IT TIGHT? CHECK AGAIN.")
    assert out == "Warning! Is it tight? Check again."


# ── preprocess_for_mt + postprocess_after_mt round-trip ───────────────────


def test_preprocess_flags_uppercase_and_returns_lowered_text() -> None:
    src = "WARNING: NEVER LIFT THE HELMET ASSEMBLY BY THE BREATHING TUBE."
    rewritten, was_upper = preprocess_for_mt(src)
    assert was_upper is True
    assert rewritten.startswith("Warning:")
    assert "helmet" in rewritten


def test_preprocess_leaves_mixed_case_untouched() -> None:
    src = "Always wear the Helmet."
    rewritten, was_upper = preprocess_for_mt(src)
    assert was_upper is False
    assert rewritten == src


def test_postprocess_upper_when_was_upper() -> None:
    es = "nunca levante el casco."
    assert postprocess_after_mt(es, True) == "NUNCA LEVANTE EL CASCO."


def test_postprocess_no_op_when_was_mixed_case() -> None:
    es = "Siempre use el casco."
    assert postprocess_after_mt(es, False) == "Siempre use el casco."


def test_full_round_trip_via_uppercase_helmet_warning() -> None:
    # Simulate the full caller pattern: preprocess EN, "translate", postprocess ES.
    src = "NEVER LIFT THE HELMET BY THE BREATHING TUBE."
    rewritten, was_upper = preprocess_for_mt(src)
    fake_translation = "nunca levante el casco por el tubo respirador."
    final = postprocess_after_mt(fake_translation, was_upper)
    assert was_upper is True
    assert final == "NUNCA LEVANTE EL CASCO POR EL TUBO RESPIRADOR."
