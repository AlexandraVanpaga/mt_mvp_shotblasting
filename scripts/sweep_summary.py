"""N-way comparison of MT engine sweep results.

Reads ``quality/quality_summary.json``, ``glossary/glossary_term_stats.csv`` and
``quality/comet_qe_worst.csv`` from each of the supplied result directories and
prints (a) a Markdown table with all engines side-by-side and (b) a "winner"
recommendation based on a simple convex score:

    score = 0.5 * LaBSE_mean + 0.3 * BERTScore_F1_mean + 0.2 * COMET_QE_norm

where COMET_QE_norm = sigmoid(COMET_QE_mean) so the three components share the
``[0, 1]`` range. Glossary hit-rate is reported but **not** used in the score —
both Marian and NLLB feed the same glossary placeholder step, so the score
already reflects MT quality rather than glossary plumbing.

Usage:

  python scripts/sweep_summary.py \\
    --label marian   --dir results_sweep/marian_v4 \\
    --label nllb600m --dir results_sweep/nllb600m  \\
    --label nllb13b  --dir results_sweep/nllb1_3b  \\
    --out results_sweep/sweep_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _metric(summary: dict, code: str, key: str) -> float | None:
    m = (summary.get("metrics") or {}).get(code) or {}
    val = m.get(key)
    return None if val is None else float(val)


def _gloss_hit_rate(path: Path) -> tuple[int, int, float]:
    rows = _read_csv(path)
    occ = sum(int(r["occurrences_in_source"]) for r in rows)
    app = sum(int(r["applied_in_target"]) for r in rows)
    rate = (app / occ) if occ else 0.0
    return occ, app, rate


def _comet_mean(path: Path) -> tuple[int, float | None]:
    rows = _read_csv(path)
    vals: list[float] = []
    for r in rows:
        try:
            vals.append(float(r["comet_qe"]))
        except (KeyError, ValueError):
            continue
    return len(vals), (mean(vals) if vals else None)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _score(labse: float | None, bert: float | None, comet: float | None) -> float | None:
    if labse is None or bert is None:
        return None
    comet_n = _sigmoid(comet) if comet is not None else 0.5
    return 0.5 * labse + 0.3 * bert + 0.2 * comet_n


def _fmt(v: float | None, digits: int = 4) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def build_report(variants: list[tuple[str, Path]]) -> tuple[str, str | None]:
    rows: list[dict] = []
    for label, root in variants:
        summary = _safe_load_json(root / "quality" / "quality_summary.json")
        labse = _metric(summary, "labse_cos", "mean")
        labse_ci = _metric(summary, "labse_cos", "ci95_halfwidth")
        bert = _metric(summary, "bertscore_xlmr_f1", "mean")
        bert_ci = _metric(summary, "bertscore_xlmr_f1", "ci95_halfwidth")
        gloss_occ, gloss_app, gloss_rate = _gloss_hit_rate(
            root / "glossary" / "glossary_term_stats.csv"
        )
        comet_n, comet_m = _comet_mean(root / "quality" / "comet_qe_worst.csv")
        rows.append(
            {
                "label": label,
                "root": root,
                "labse": labse,
                "labse_ci": labse_ci,
                "bert": bert,
                "bert_ci": bert_ci,
                "gloss_occ": gloss_occ,
                "gloss_app": gloss_app,
                "gloss_rate": gloss_rate,
                "comet_n": comet_n,
                "comet_mean": comet_m,
                "score": _score(labse, bert, comet_m),
            }
        )

    out: list[str] = []
    push = out.append
    push("# MT engine sweep - summary")
    push("")
    push("Score = 0.5 * LaBSE + 0.3 * BERTScore_F1 + 0.2 * sigmoid(COMET-QE).")
    push("")

    # ── Quality metrics ──
    push("## Corpus-level quality (means)")
    push("")
    push("| Engine | LaBSE cosine | BERTScore F1 | COMET-QE (mean / n) | Score |")
    push("|---|---:|---:|---:|---:|")
    for r in rows:
        push(
            f"| {r['label']} | "
            f"{_fmt(r['labse'])} (±{_fmt(r['labse_ci'])}) | "
            f"{_fmt(r['bert'])} (±{_fmt(r['bert_ci'])}) | "
            f"{_fmt(r['comet_mean'])} / {r['comet_n']} | "
            f"{_fmt(r['score'])} |"
        )
    push("")

    # ── Glossary ──
    push("## Glossary application")
    push("")
    push("| Engine | Term occurrences | Applied in target | Hit rate |")
    push("|---|---:|---:|---:|")
    for r in rows:
        push(
            f"| {r['label']} | {r['gloss_occ']} | {r['gloss_app']} | "
            f"{r['gloss_rate'] * 100:.1f}% |"
        )
    push("")

    # ── Winner ──
    scored = [r for r in rows if r["score"] is not None]
    winner_label: str | None = None
    if scored:
        scored.sort(key=lambda r: r["score"], reverse=True)
        winner = scored[0]
        winner_label = winner["label"]
        push("## Winner")
        push("")
        push(f"**{winner_label}** — score {winner['score']:.4f}")
        push("")
        push("Ranking:")
        push("")
        for r in scored:
            push(f"- `{r['label']}` → score {r['score']:.4f}")
        push("")

    return "\n".join(out) + "\n", winner_label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Human label (e.g. 'marian', 'nllb600m'). Repeat for each engine.",
    )
    parser.add_argument(
        "--dir",
        action="append",
        type=Path,
        required=True,
        help="Result directory for this label. Repeat in the same order as --label.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the Markdown summary (default: print to stdout only).",
    )
    args = parser.parse_args()

    if len(args.label) != len(args.dir):
        print("ERROR: --label and --dir must come in pairs.", file=sys.stderr)
        return 1

    variants: list[tuple[str, Path]] = list(zip(args.label, args.dir))
    text, winner = build_report(variants)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[done] wrote {args.out}")
    if winner is not None:
        print(f"[winner] {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
