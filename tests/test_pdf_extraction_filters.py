"""Tests for the PDF-extraction noise filters in ``scripts/pdfs_to_csv.py``."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pdfs_to_csv import (  # noqa: E402
    is_caps_parts_dump,
    is_parts_dump,
    is_titlecase_parts_dump,
)


# ── ALL-CAPS parts-list flavor (already shipped in v2; regression-fence it) ──
def test_caps_parts_dump_detects_long_glued_uppercase_block() -> None:
    line = (
        "WINDOW FRAME SEALING BAND STRAP HANDLE BREATHING TUBE ASSEMBLY "
        "INNER CAPE OUTER CAPE INTERMEDIATE LENS"
    )
    assert is_caps_parts_dump(line) is True


def test_caps_parts_dump_keeps_legitimate_uppercase_warning() -> None:
    line = (
        "WARNING DO NOT USE THE SUPPLIED AIR RESPIRATOR HELMET IF THE FLOW "
        "INDICATOR IS SHOWING RED"
    )
    assert is_caps_parts_dump(line) is False


def test_caps_parts_dump_rejects_short_lines() -> None:
    assert is_caps_parts_dump("WINDOW FRAME") is False
    assert is_caps_parts_dump("AIR FLOW INDICATOR") is False


# ── Title-Case parts-list flavor (new in v3) ────────────────────────────────
def test_titlecase_parts_dump_detects_explicit_parts_list_block() -> None:
    line = (
        "PARTS LIST Air Cooling Controller And Belt Assembly Item Stock Code "
        "Description Waist Belt And Buckle Assembly Mounting Bracket Reducing "
        "Bush Male Quick Disconnect Coupling Heat Shield"
    )
    assert is_titlecase_parts_dump(line) is True


def test_titlecase_parts_dump_detects_titan_listing() -> None:
    line = (
        "Titan II Supplied Air Respirator Parts Listing Item Stock Code "
        "Description Titan II Respirator Helmet With Standard Cape Titan II "
        "Respirator Helmet With Standard Cape And Air Flow Controller"
    )
    assert is_titlecase_parts_dump(line) is True


def test_titlecase_parts_dump_keeps_normal_titlecase_heading() -> None:
    # Real product heading, Title Case but short and without catalog keywords.
    line = "Supplied Air Respirator Helmet With Standard Cape Assembly Diagram"
    assert is_titlecase_parts_dump(line) is False


def test_titlecase_parts_dump_keeps_instructional_titlecase_sentence() -> None:
    line = (
        "Ensure That The Supplied Air Respirator Helmet With Cape Assembly Is "
        "Properly Mounted Before You Connect The Hose To The Stock Code Item"
    )
    # Contains "Stock", "Code", "Item" (3 catalog keywords) BUT also "Ensure",
    # "Is", "Before", "You" — instructional words, so we keep it.
    assert is_titlecase_parts_dump(line) is False


def test_titlecase_parts_dump_keeps_sentence_with_period() -> None:
    line = (
        "PARTS LIST Air Cooling Controller And Belt Assembly. Item Stock Code "
        "Description Waist Belt And Buckle Assembly Mounting Bracket"
    )
    # Internal period → not a glued block.
    assert is_titlecase_parts_dump(line) is False


# ── Unified entry point ─────────────────────────────────────────────────────
def test_is_parts_dump_combines_both_flavors() -> None:
    assert is_parts_dump(
        "PARTS LIST Air Cooling Controller And Belt Assembly Item Stock Code "
        "Description Waist Belt And Buckle Assembly Mounting Bracket"
    )
    assert is_parts_dump(
        "WINDOW FRAME SEALING BAND STRAP HANDLE BREATHING TUBE ASSEMBLY "
        "INNER CAPE OUTER CAPE INTERMEDIATE LENS"
    )
    assert not is_parts_dump(
        "WARNING DO NOT USE THE SUPPLIED AIR RESPIRATOR HELMET IF THE FLOW "
        "INDICATOR IS SHOWING RED"
    )
