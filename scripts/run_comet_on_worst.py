"""Run COMET-QE (`Unbabel/wmt20-comet-qe-da`) on the N lowest-LaBSE pairs.

This is the third independent quality signal on top of LaBSE + BERTScore.
It is reference-less ("QE" = quality estimation): COMET-QE scores each
(source, mt) pair on its own. Note on scale: `wmt20-comet-qe-da` returns
DA-z-score-like values (typical news-domain ranges roughly -1.5 .. +1.0,
higher = better). A typical fluent MT pair scores around 0.0 to +0.3, weak
translations go to -0.3, and anything below -1.0 is almost always
genuinely broken. This is NOT the same scale as COMET-Kiwi (which is in
[0,1]); do not compare across models.

We focus on the worst N rows by LaBSE because:
  * That's where the actionable failures live; the great-majority of the
    corpus already scores well by LaBSE/BERTScore.
  * COMET-QE on full 1.4k corpus on RTX 3060 is feasible (~3-4 min) but the
    long tail is what's interesting for triage.

Input
=====
`results_final/quality/quality_scores.csv` (output of evaluate_quality.py).

Output
======
`results_final/quality/comet_qe_worst.csv` with:
  id, category, pdf, page, char_count, labse_cos, bertscore_xlmr_f1,
  comet_qe, source_en, target_es

Usage
=====
  python scripts/run_comet_on_worst.py
  python scripts/run_comet_on_worst.py --top 300 --device cpu
  python scripts/run_comet_on_worst.py --rank-by bertscore_xlmr_f1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import mean, median, pstdev

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fnum(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results_final" / "quality" / "quality_scores.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results_final" / "quality" / "comet_qe_worst.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "wmt20-comet-qe-da" / "checkpoints" / "model.ckpt",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=200,
        help="How many lowest-ranked pairs to score (default 200).",
    )
    parser.add_argument(
        "--rank-by",
        default="labse_cos",
        choices=["labse_cos", "bertscore_xlmr_f1"],
        help="Which existing metric to rank by (ascending).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        default=None,
        help='"cuda" or "cpu". Default: auto.',
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: not found: {args.input}", file=sys.stderr)
        return 1
    if not args.checkpoint.is_file():
        print(
            f"ERROR: COMET checkpoint not found: {args.checkpoint}\n"
            f"Run `python scripts/download_comet_qe.py` first.",
            file=sys.stderr,
        )
        return 1

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    gpus = 1 if device == "cuda" else 0
    print(f"[device] {device}")

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p.resolve())

    with args.input.open("r", encoding="utf-8-sig", newline="") as fh:
        all_rows = list(csv.DictReader(fh))
    print(f"[load] {_rel(args.input)}  rows={len(all_rows):,}")

    scored = []
    for r in all_rows:
        v = _fnum(r.get(args.rank_by, ""))
        if v is None:
            continue
        if not r.get("source_en", "").strip() or not r.get("target_es", "").strip():
            continue
        scored.append((v, r))
    scored.sort(key=lambda x: x[0])
    top_n = min(args.top, len(scored))
    worst = scored[:top_n]
    print(f"[select] worst {top_n:,} by {args.rank_by} (range {worst[0][0]:.4f} .. {worst[-1][0]:.4f})")

    from comet import load_from_checkpoint

    print(f"[comet] loading {_rel(args.checkpoint)}")
    model = load_from_checkpoint(str(args.checkpoint))

    data = [{"src": r["source_en"], "mt": r["target_es"]} for _, r in worst]
    print(f"[comet] scoring {len(data)} pairs on {device} (batch={args.batch_size})")
    out = model.predict(data, batch_size=args.batch_size, gpus=gpus, progress_bar=True)
    comet_scores = list(getattr(out, "scores", out))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "id",
        "category",
        "pdf",
        "page",
        "char_count",
        "comet_qe",
        "labse_cos",
        "bertscore_xlmr_f1",
        "source_en",
        "target_es",
    ]
    with args.out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for (rank_val, r), comet_score in zip(worst, comet_scores, strict=True):
            row = {
                "id": r.get("id", ""),
                "category": r.get("category", ""),
                "pdf": r.get("pdf", ""),
                "page": r.get("page", ""),
                "char_count": r.get("char_count", ""),
                "comet_qe": f"{float(comet_score):.4f}",
                "labse_cos": r.get("labse_cos", ""),
                "bertscore_xlmr_f1": r.get("bertscore_xlmr_f1", ""),
                "source_en": r.get("source_en", ""),
                "target_es": r.get("target_es", ""),
            }
            writer.writerow(row)

    floats = [float(s) for s in comet_scores]
    print()
    print(f"==== COMET-QE on worst-{top_n} by {args.rank_by} ====")
    print(f"  mean    : {mean(floats):.4f}")
    print(f"  median  : {median(floats):.4f}")
    print(f"  std     : {pstdev(floats):.4f}")
    print(f"  min     : {min(floats):.4f}")
    print(f"  max     : {max(floats):.4f}")

    # Bucketise on the wmt20-comet-qe-da DA-z-score scale.
    bands = [
        (-1.00, "broken (<-1.0)"),
        (-0.50, "very weak (-1.0..-0.5)"),
        (-0.20, "weak (-0.5..-0.2)"),
        (0.10, "ok (-0.2..0.1)"),
        (10.0, "good (>=0.1)"),
    ]
    counts = {label: 0 for _, label in bands}
    for f in floats:
        for upper, label in bands:
            if f < upper:
                counts[label] += 1
                break
    print("  buckets :")
    for _, label in bands:
        c = counts[label]
        print(f"    {label:<22s} {c:>4} ({c * 100 / len(floats):5.1f}%)")

    # Show worst 10
    pairs = sorted(zip(comet_scores, worst, strict=True), key=lambda x: x[0])
    print("\n  worst 10 by COMET-QE:")
    for sc, (rank_val, r) in pairs[:10]:
        src = r["source_en"][:90].replace("\n", " ")
        tgt = r["target_es"][:90].replace("\n", " ")
        print(f"    comet={float(sc):.3f}  labse={r.get('labse_cos','')}  "
              f"{r.get('pdf','')}:{r.get('page','')}")
        print(f"      EN: {src!r}")
        print(f"      ES: {tgt!r}")

    print(f"\n  output: {_rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
