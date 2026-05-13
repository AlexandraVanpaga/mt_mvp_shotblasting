"""Filter `translated_data_final/all_segments.csv` down to rows with English source.

Why
===
PanBlast PDFs are multilingual (English alongside Hungarian, Russian, German,
French, etc.). Our MT pipeline is en->es, so non-English source rows are
out-of-distribution and drag the corpus-level quality metrics down without
representing real translation failures.

How
===
Uses Facebook's fastText `lid.176.bin` (176-language identifier). The model
file is expected at `models/lid.176.bin`; if missing, the script prints the
download URL and exits.

For every row, we predict the top language label and confidence. A row is
kept iff:
  * predicted label == `en`, AND
  * confidence >= `--min-confidence` (default 0.50), AND
  * source contains at least `--min-letters` alphabetic chars (default 3) -
    very short fragments are unreliable for LID and are kept by default.

Outputs (next to the input):
  * <input_stem>.en.csv      - kept rows, same columns + lang/confidence
  * <input_stem>.non_en.csv  - rejected rows, same columns + lang/confidence
                               (for manual inspection)
  * stdout                   - per-language histogram of rejections

Usage:
  python scripts/filter_english.py
  python scripts/filter_english.py --input translated_data_final/all_segments.csv \
      --min-confidence 0.5
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LID_MODEL_PATH = PROJECT_ROOT / "models" / "lid.176.bin"
LID_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

_NL_RE = re.compile(r"[\r\n]+")


def _normalise_for_lid(text: str) -> str:
    """fastText `predict` rejects newlines; flatten them, strip, lowercase.

    Lowercasing matters: ``lid.176.bin`` mis-classifies uppercase non-English
    fragments (e.g. Dutch ``AFDICHTBAND BUITENCAPE DRUKKNOPEN``) as borderline
    English; the lowercase form predicts ``nl`` confidently.
    """
    return _NL_RE.sub(" ", text).strip().lower()


def _count_letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def load_lid():
    import fasttext

    fasttext.FastText.eprint = lambda x: None  # silence the noisy banner
    if not LID_MODEL_PATH.is_file():
        print(
            f"ERROR: language-id model not found at {LID_MODEL_PATH}\n"
            f"Download it once with:\n"
            f"  Invoke-WebRequest -Uri '{LID_DOWNLOAD_URL}' -OutFile '{LID_MODEL_PATH}'",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"[lid] loading {LID_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    return fasttext.load_model(str(LID_MODEL_PATH))


def predict_batch(model, texts: list[str]) -> list[tuple[str, float]]:
    """Return (lang_code, confidence) per input text. Empty/blank -> ('', 0.0)."""
    cleaned = [_normalise_for_lid(t) for t in texts]
    labels, probs = model.predict(cleaned, k=1)
    out: list[tuple[str, float]] = []
    for lbl, pr, src in zip(labels, probs, cleaned, strict=True):
        if not src:
            out.append(("", 0.0))
            continue
        code = lbl[0].replace("__label__", "") if lbl else ""
        conf = float(pr[0]) if len(pr) else 0.0
        out.append((code, conf))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "translated_data_final" / "all_segments.csv",
    )
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument(
        "--min-letters",
        type=int,
        default=3,
        help="Rows with fewer alphabetic chars in source are KEPT (too short for reliable LID).",
    )
    parser.add_argument(
        "--out-kept",
        type=Path,
        default=None,
        help="Output path for English-source rows (default: <input_stem>.en.csv).",
    )
    parser.add_argument(
        "--out-rejected",
        type=Path,
        default=None,
        help="Output path for non-English source rows (default: <input_stem>.non_en.csv).",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: not found: {args.input}", file=sys.stderr)
        return 1

    out_kept = args.out_kept or args.input.with_suffix(".en.csv")
    out_rejected = args.out_rejected or args.input.with_suffix(".non_en.csv")

    with args.input.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    print(f"[load] {args.input.relative_to(PROJECT_ROOT)}  rows={len(rows):,}")

    sources = [r.get("source_en", "") for r in rows]

    model = load_lid()
    preds: list[tuple[str, float]] = []
    for start in range(0, len(sources), args.batch_size):
        chunk = sources[start : start + args.batch_size]
        preds.extend(predict_batch(model, chunk))
        if start and start % (args.batch_size * 8) == 0:
            print(f"[lid] processed {start:,}/{len(sources):,}")

    extra = ["lang", "lang_confidence"]
    out_cols = fieldnames + [c for c in extra if c not in fieldnames]
    out_kept.parent.mkdir(parents=True, exist_ok=True)
    out_rejected.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0
    kept_short_unreliable = 0
    rejection_langs: Counter[str] = Counter()
    rejection_low_conf = 0

    with (
        out_kept.open("w", encoding="utf-8-sig", newline="") as fk,
        out_rejected.open("w", encoding="utf-8-sig", newline="") as fr,
    ):
        w_kept = csv.DictWriter(fk, fieldnames=out_cols)
        w_rej = csv.DictWriter(fr, fieldnames=out_cols)
        w_kept.writeheader()
        w_rej.writeheader()
        for row, (code, conf) in zip(rows, preds, strict=True):
            row = dict(row)
            row["lang"] = code
            row["lang_confidence"] = f"{conf:.4f}"
            src = row.get("source_en", "") or ""
            letters = _count_letters(src)
            keep_short = letters < args.min_letters
            is_english = code == "en" and conf >= args.min_confidence
            if keep_short:
                kept_short_unreliable += 1
                w_kept.writerow(row)
                kept += 1
                continue
            if is_english:
                w_kept.writerow(row)
                kept += 1
            else:
                if code == "en":
                    rejection_low_conf += 1
                else:
                    rejection_langs[code or "<empty>"] += 1
                w_rej.writerow(row)
                rejected += 1

    total = kept + rejected
    print()
    print("==== language filter summary ====")
    print(f"  rows total           : {total:,}")
    print(f"  kept (English source): {kept:,}  ({kept * 100 / max(total, 1):.1f}%)")
    print(f"    of which kept-because-too-short-for-LID: {kept_short_unreliable:,}")
    print(f"  rejected             : {rejected:,}  ({rejected * 100 / max(total, 1):.1f}%)")
    if rejection_low_conf:
        print(f"    of which 'en' but below min_confidence={args.min_confidence}: {rejection_low_conf:,}")
    if rejection_langs:
        print("  top rejected source-languages:")
        for code, count in rejection_langs.most_common(15):
            print(f"    {code:<6} {count:>5}  ({count * 100 / max(rejected, 1):.1f}% of rejects)")

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p.resolve())

    print("\n  outputs:")
    print(f"    {_rel(out_kept)}     ({kept:,} rows)")
    print(f"    {_rel(out_rejected)} ({rejected:,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
