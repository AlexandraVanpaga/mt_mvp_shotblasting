"""Audit glossary application across translated_data_final/all_segments.csv.

For every translated row, this script answers three questions:

  1. Which glossary English sources occur in `source_en`?
     (longest-first, non-overlapping — mirrors `Glossary.protect_source`)
  2. For each match, does the canonical Spanish target appear in `target_es`?
     (loose phrase match: any whitespace between words, case-insensitive —
     mirrors `Glossary._accent_loose_pattern`)
  3. Did the English source phrase LEAK into `target_es` untranslated?
     (only flagged when the source is multi-word ≥2 tokens, to avoid
     false positives on globally valid Spanish words like "Spartan")

Outputs (under `evaluation/`):
  * `glossary_audit.csv`        — per-row details, one line per translated row
  * `glossary_term_stats.csv`   — per-term: occurrences, applied, missed, hit-rate
  * `glossary_pdf_stats.csv`    — per-PDF: rows, terms-present, terms-applied
  * stdout summary              — overall hit-rate, top-missed terms, leaks

Usage (from project root):
  python scripts/evaluate_glossary.py
  python scripts/evaluate_glossary.py --input translated_data_final/all_segments.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Stdout on Windows defaults to cp1252; force UTF-8 so we can print arrows and
# Spanish accents in the summary table without UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WS_RE = re.compile(r"\s+")


def load_glossary(path: Path) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for e in raw.get("entries", []):
        src = (e.get("source") or "").strip()
        tgt = (e.get("target") or "").strip()
        if not src or not tgt:
            continue
        key = src.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((src, tgt))
    out.sort(key=lambda e: len(e[0]), reverse=True)
    return out


def loose_phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Whitespace-tolerant, case-insensitive phrase regex (same approach as Glossary)."""
    words = [re.escape(w) for w in phrase.split() if w]
    return re.compile(r"\s+".join(words) if words else re.escape(phrase), flags=re.IGNORECASE)


def word_boundary_pattern(phrase: str) -> re.Pattern[str]:
    """Anchored word-boundary version (used in source detection to avoid in-word hits)."""
    words = [re.escape(w) for w in phrase.split() if w]
    if not words:
        return re.compile(re.escape(phrase), flags=re.IGNORECASE)
    body = r"\s+".join(words)
    return re.compile(rf"(?<!\w){body}(?!\w)", flags=re.IGNORECASE)


def detect_terms_in_source(source_en: str, glossary: list[tuple[str, str]]) -> list[tuple[str, str, int]]:
    """Longest-first, non-overlapping match. Returns [(en, es, count), …]."""
    masked = source_en
    hits: list[tuple[str, str, int]] = []
    for en, es in glossary:
        pattern = word_boundary_pattern(en)
        matches = list(pattern.finditer(masked))
        if not matches:
            continue
        hits.append((en, es, len(matches)))
        # Replace each match with a non-letter mask so shorter substrings don't re-hit.
        spans = sorted({(m.start(), m.end()) for m in matches}, reverse=True)
        for s, e in spans:
            masked = masked[:s] + ("·" * (e - s)) + masked[e:]
    return hits


def contains_target(text: str, target: str) -> bool:
    if not target.strip():
        return False
    return bool(loose_phrase_pattern(target).search(text))


def english_leak(text: str, source_en: str) -> bool:
    if len(source_en.split()) < 2:
        return False  # single-word leaks are too noisy to flag automatically
    return bool(word_boundary_pattern(source_en).search(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "translated_data_final" / "all_segments.csv",
        help="CSV with at least source_en, target_es columns",
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=PROJECT_ROOT / "glossary" / "en_es_shotblasting.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: not found: {args.input}", file=sys.stderr)
        return 1

    glossary = load_glossary(args.glossary)
    print(f"[load] glossary entries (unique sources): {len(glossary)}")
    print(f"[load] input: {args.input}")

    with args.input.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    translated = [r for r in rows if r.get("target_es", "").strip() and not r.get("error", "").strip()]
    print(f"[load] rows: {len(rows)} total, {len(translated)} with translations to audit")
    if not translated:
        print("Nothing to audit (no rows have target_es populated).", file=sys.stderr)
        return 1

    # Per-term counters.
    term_occurrences: dict[str, int] = defaultdict(int)
    term_applied: dict[str, int] = defaultdict(int)
    term_leaked: dict[str, int] = defaultdict(int)
    term_targets: dict[str, str] = {en: es for en, es in glossary}

    # Per-PDF counters.
    pdf_rows: dict[str, int] = defaultdict(int)
    pdf_with_terms: dict[str, int] = defaultdict(int)
    pdf_term_occ: dict[str, int] = defaultdict(int)
    pdf_term_app: dict[str, int] = defaultdict(int)

    audit_rows: list[dict[str, str]] = []
    rows_with_terms = 0
    rows_all_applied = 0

    for r in translated:
        src = r["source_en"]
        tgt = r["target_es"]
        pdf_key = f"{r.get('category', '')}/{r.get('pdf', '')}"
        pdf_rows[pdf_key] += 1

        hits = detect_terms_in_source(src, glossary)
        if not hits:
            audit_rows.append(
                {
                    "id": r.get("id", ""),
                    "category": r.get("category", ""),
                    "pdf": r.get("pdf", ""),
                    "page": r.get("page", ""),
                    "terms_in_source": "",
                    "terms_applied": "",
                    "terms_missed": "",
                    "english_leaks": "",
                    "row_hit_rate": "",
                }
            )
            continue

        rows_with_terms += 1
        pdf_with_terms[pdf_key] += 1

        applied: list[str] = []
        missed: list[str] = []
        leaks: list[str] = []
        row_occ = 0
        row_app = 0

        for en, es, count in hits:
            term_occurrences[en] += count
            row_occ += count
            pdf_term_occ[pdf_key] += count
            if contains_target(tgt, es):
                term_applied[en] += count
                row_app += count
                pdf_term_app[pdf_key] += count
                applied.append(f"{en}→{es}")
            else:
                missed.append(f"{en}→{es}")
            if english_leak(tgt, en):
                term_leaked[en] += 1
                leaks.append(en)

        if row_occ and row_occ == row_app:
            rows_all_applied += 1

        audit_rows.append(
            {
                "id": r.get("id", ""),
                "category": r.get("category", ""),
                "pdf": r.get("pdf", ""),
                "page": r.get("page", ""),
                "terms_in_source": " | ".join(f"{en} (×{c})" for en, _, c in hits),
                "terms_applied": " | ".join(applied),
                "terms_missed": " | ".join(missed),
                "english_leaks": " | ".join(leaks),
                "row_hit_rate": f"{row_app}/{row_occ}",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Per-row audit.
    audit_csv = args.out_dir / "glossary_audit.csv"
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        cols = [
            "id",
            "category",
            "pdf",
            "page",
            "terms_in_source",
            "terms_applied",
            "terms_missed",
            "english_leaks",
            "row_hit_rate",
        ]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(audit_rows)

    # Per-term stats.
    term_stats_rows: list[dict[str, str]] = []
    for en, es in glossary:
        occ = term_occurrences.get(en, 0)
        app = term_applied.get(en, 0)
        leaked = term_leaked.get(en, 0)
        if occ == 0 and leaked == 0:
            continue
        term_stats_rows.append(
            {
                "source_en": en,
                "target_es": es,
                "occurrences_in_source": str(occ),
                "applied_in_target": str(app),
                "missed_in_target": str(occ - app),
                "english_leak_rows": str(leaked),
                "hit_rate": f"{(app / occ * 100):.1f}%" if occ else "n/a",
            }
        )
    term_stats_rows.sort(key=lambda r: int(r["occurrences_in_source"]), reverse=True)
    term_csv = args.out_dir / "glossary_term_stats.csv"
    with term_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "source_en",
                "target_es",
                "occurrences_in_source",
                "applied_in_target",
                "missed_in_target",
                "english_leak_rows",
                "hit_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(term_stats_rows)

    # Per-PDF stats.
    pdf_csv = args.out_dir / "glossary_pdf_stats.csv"
    with pdf_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "pdf",
                "rows_audited",
                "rows_with_terms",
                "term_occurrences",
                "terms_applied",
                "hit_rate",
            ],
        )
        writer.writeheader()
        for key in sorted(pdf_rows.keys()):
            occ = pdf_term_occ[key]
            app = pdf_term_app[key]
            writer.writerow(
                {
                    "pdf": key,
                    "rows_audited": pdf_rows[key],
                    "rows_with_terms": pdf_with_terms[key],
                    "term_occurrences": occ,
                    "terms_applied": app,
                    "hit_rate": f"{(app / occ * 100):.1f}%" if occ else "n/a",
                }
            )

    # ── Summary ─────────────────────────────────────────────────────────────
    total_occ = sum(term_occurrences.values())
    total_app = sum(term_applied.values())
    total_leaks = sum(term_leaked.values())
    print()
    print("==== glossary application summary ====")
    print(f"  rows audited:              {len(translated):,}")
    print(f"  rows with >=1 source term: {rows_with_terms:,}  ({rows_with_terms * 100 / len(translated):.1f}% of audited)")
    print(f"  rows fully applied:        {rows_all_applied:,}  ({rows_all_applied * 100 / max(rows_with_terms, 1):.1f}% of rows-with-terms)")
    print(f"  total term occurrences:    {total_occ:,}")
    print(f"  applied in target:         {total_app:,}  ({total_app * 100 / max(total_occ, 1):.1f}%)")
    print(f"  missed in target:          {total_occ - total_app:,}")
    print(f"  english leak events:       {total_leaks:,}  (multi-word source still in target)")

    if term_stats_rows:
        missed_sorted = [r for r in term_stats_rows if int(r["missed_in_target"]) > 0]
        missed_sorted.sort(key=lambda r: int(r["missed_in_target"]), reverse=True)
        if missed_sorted:
            print("\n  top missed terms:")
            for r in missed_sorted[:10]:
                print(
                    f"    {r['source_en']:42.42s} → {r['target_es']:36.36s}"
                    f"  missed {r['missed_in_target']:>3}/{r['occurrences_in_source']:>3}  "
                    f"({100 - float(r['hit_rate'].rstrip('%')):.0f}% miss)"
                )

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p.resolve())

    print("\n  outputs:")
    print(f"    {_rel(audit_csv)}")
    print(f"    {_rel(term_csv)}")
    print(f"    {_rel(pdf_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
