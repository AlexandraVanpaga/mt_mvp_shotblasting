"""Produce client-facing per-PDF CSVs with just two columns: source / translation.

Reads a full translation run (e.g. ``translated_data_final/``) and emits, for
every original PDF, a side-by-side CSV under ``resulting_files/<category>/``:

    source_en,target_es
    "Always wear the helmet ...","Lleve siempre el casco ..."
    ...

This is the artifact a non-technical reviewer can open in Excel or Google
Sheets to spot-check the final translation per PDF, without the extra
metadata columns (id, page, segment_idx, glossary_applied, …) that live in
the pipeline-internal ``translated_data_final/<category>/<pdf>.csv`` files.

Usage:
  python scripts/build_resulting_files.py                       # default paths
  python scripts/build_resulting_files.py --input translated_data_marian_v4 \\
                                          --output resulting_files_marian
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT_COLUMNS = ["source_en", "target_es"]


def _read_all_segments(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_two_column(rows: list[dict[str, str]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "source_en": r.get("source_en", ""),
                    "target_es": r.get("target_es", ""),
                }
            )
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "translated_data_final" / "all_segments.csv",
        help="Path to a translation run's all_segments.csv (default: translated_data_final/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "resulting_files",
        help="Destination directory for per-PDF 2-column CSVs (default: resulting_files/).",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        default=True,
        help="Drop rows with an empty target_es (defaults to True).",
    )
    parser.add_argument(
        "--keep-empty",
        dest="skip_empty",
        action="store_false",
        help="Keep rows with empty target_es (untranslated segments).",
    )
    parser.add_argument(
        "--all-segments-name",
        default="all_segments.csv",
        help="Filename for the consolidated 2-column CSV (default: all_segments.csv).",
    )
    args = parser.parse_args()

    src = args.input.resolve()
    out_dir = args.output.resolve()

    if not src.is_file():
        print(f"ERROR: input file not found: {src}", file=sys.stderr)
        return 1

    rows = _read_all_segments(src)
    if args.skip_empty:
        rows = [r for r in rows if r.get("target_es", "").strip()]

    if not rows:
        print(f"ERROR: no translated rows in {src}", file=sys.stderr)
        return 2

    print(f"[input] {len(rows)} translated segment(s) from {src.relative_to(PROJECT_ROOT)}")

    buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        category = (r.get("category") or "uncategorized").strip()
        pdf = (r.get("pdf") or "").strip()
        if not pdf:
            continue
        buckets.setdefault((category, pdf), []).append(r)

    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for (category, pdf), group in sorted(buckets.items()):
        stem = Path(pdf).stem
        dest = out_dir / category / f"{stem}.csv"
        _write_two_column(group, dest)
        written += 1

    consolidated = out_dir / args.all_segments_name
    _write_two_column(rows, consolidated)

    rel = out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir
    print(f"[write] {written} per-PDF CSV(s) under {rel}/<category>/")
    print(f"[write] consolidated: {consolidated.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
