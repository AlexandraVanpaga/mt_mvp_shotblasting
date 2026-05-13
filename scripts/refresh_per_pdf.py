"""Re-split `translated_data_final/all_segments.csv` into per-PDF CSVs.

The main batch translator (`scripts/translate_csv.py`) only writes per-PDF files
at the very end of its run. While translation is still going, run this script
to refresh the per-PDF mirror so you can inspect what's been translated so far
in `translated_data_final/<category>/<pdf_stem>.csv`.

Usage:
  python scripts/refresh_per_pdf.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "translated_data_final" / "all_segments.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "translated_data_final"

OUTPUT_COLUMNS = [
    "id",
    "category",
    "pdf",
    "page",
    "segment_idx",
    "source_en",
    "target_es",
    "char_count",
    "glossary_applied",
    "postedit_applied",
    "error",
]


def main() -> int:
    src = DEFAULT_INPUT
    out_dir = DEFAULT_OUT_DIR
    if not src.is_file():
        print(f"ERROR: not found: {src}", file=sys.stderr)
        return 1

    with src.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for col in OUTPUT_COLUMNS:
            r.setdefault(col, "")

    buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        buckets.setdefault((r["category"], r["pdf"]), []).append(r)

    total_done = 0
    for (category, pdf), group in buckets.items():
        done = sum(1 for r in group if r.get("target_es", "").strip())
        total_done += done
        stem = Path(pdf).stem
        dest = out_dir / category / f"{stem}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(group)
        tmp.replace(dest)
        print(f"  {category}/{pdf}: {done}/{len(group)} translated")

    print(f"\nTotal translated: {total_done}/{len(rows)} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
