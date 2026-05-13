"""Produce evaluation plots for ``results_final/``.

Reads the CSV outputs of ``evaluate_glossary.py`` and ``evaluate_quality.py`` and
writes PNG plots under ``results_final/plots/``. Intentionally minimal: matplotlib only,
no seaborn, no notebook — so it runs reliably in CI / headless.

Plots produced
==============
Quality:
  q1_labse_distribution.png        — histogram + mean + p05/median/p95 vert lines
  q2_bertscore_distribution.png    — same, for BERTScore F1
  q3_labse_vs_bertscore.png        — scatter, both metrics on the same rows
  q4_per_pdf_mean.png              — bar chart, mean LaBSE per PDF (sorted)
  q5_score_vs_length.png           — scatter LaBSE vs char_count
  q6_quality_buckets.png           — stacked bar of quality buckets per PDF

Glossary:
  g1_per_term_hit_rate.png         — bar chart, top-N glossary terms by occurrences
  g2_per_pdf_hit_rate.png          — bar chart, glossary application per PDF
  g3_row_status_pie.png            — pie: rows with no terms / fully applied / partial

Usage:
  python scripts/make_eval_plots.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def quality_distribution(
    values: list[float], title: str, xlabel: str, dest: Path, color: str
) -> None:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    p05, med, p95 = (float(x) for x in np.quantile(arr, [0.05, 0.5, 0.95]))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(arr, bins=60, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    for x, label, c, ls in [
        (mean, f"mean = {mean:.3f}", "#222222", "--"),
        (med, f"median = {med:.3f}", "#0a7", "-"),
        (p05, f"p05 = {p05:.3f}", "#a33", ":"),
        (p95, f"p95 = {p95:.3f}", "#a33", ":"),
    ]:
        ax.axvline(x, color=c, linestyle=ls, linewidth=1.4, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rows")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def scatter(
    x: list[float],
    y: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    dest: Path,
    diag: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=6, alpha=0.35, color="#1f6dad", edgecolors="none")
    if diag:
        lo = min(min(x), min(y))
        hi = max(max(x), max(y))
        ax.plot([lo, hi], [lo, hi], color="#aa3333", linewidth=1, linestyle="--", label="y = x")
        ax.legend(loc="upper left")
    # Pearson correlation
    a = np.array(x); b = np.array(y)
    if len(a) > 1 and a.std() > 0 and b.std() > 0:
        r = float(np.corrcoef(a, b)[0, 1])
        ax.text(
            0.04,
            0.96,
            f"Pearson r = {r:.3f}\nn = {len(x):,}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def per_pdf_mean_bar(
    rows: list[dict[str, str]], metric_col: str, title: str, xlabel: str, dest: Path, color: str
) -> None:
    by_pdf: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = fnum(r.get(metric_col, ""))
        if v is None:
            continue
        by_pdf[r["pdf"]].append(v)
    pairs = sorted(by_pdf.items(), key=lambda kv: np.mean(kv[1]))
    labels = [p[: -4] if p.endswith(".pdf") else p for p, _ in pairs]
    means = [float(np.mean(vs)) for _, vs in pairs]
    counts = [len(vs) for _, vs in pairs]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, means, color=color, edgecolor="white")
    overall = float(np.mean(means))
    ax.axvline(overall, color="#222", linestyle="--", linewidth=1.2, label=f"corpus mean = {overall:.3f}")
    for bar, m, n in zip(bars, means, counts, strict=True):
        ax.text(m + 0.005, bar.get_y() + bar.get_height() / 2, f"{m:.3f}  (n={n:,})", va="center", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(min(means) - 0.05, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def quality_buckets_per_pdf(rows: list[dict[str, str]], metric_col: str, cutoffs: list[tuple[str, float, str]], title: str, dest: Path) -> None:
    by_pdf: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = fnum(r.get(metric_col, ""))
        if v is None:
            continue
        by_pdf[r["pdf"]].append(v)
    pdfs = sorted(by_pdf.keys(), key=lambda p: np.mean(by_pdf[p]))
    bucket_labels = [c[0] for c in cutoffs]
    bucket_uppers = [c[1] for c in cutoffs]
    bucket_colors = [c[2] for c in cutoffs]
    matrix = np.zeros((len(pdfs), len(cutoffs)), dtype=float)
    for i, p in enumerate(pdfs):
        vals = by_pdf[p]
        for v in vals:
            for j, upper in enumerate(bucket_uppers):
                if v < upper:
                    matrix[i, j] += 1
                    break
            else:
                matrix[i, -1] += 1
        if len(vals):
            matrix[i] = matrix[i] / len(vals) * 100.0
    labels = [p[: -4] if p.endswith(".pdf") else p for p in pdfs]
    fig, ax = plt.subplots(figsize=(10, 7))
    left = np.zeros(len(pdfs))
    for j, (label, color) in enumerate(zip(bucket_labels, bucket_colors, strict=True)):
        ax.barh(labels, matrix[:, j], left=left, color=color, edgecolor="white", label=label)
        left += matrix[:, j]
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of segments")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def per_term_hit_rate(term_csv: Path, dest: Path, top_n: int = 25) -> None:
    rows = read_csv(term_csv)
    rows = [r for r in rows if int(r["occurrences_in_source"]) > 0]
    rows.sort(key=lambda r: int(r["occurrences_in_source"]), reverse=True)
    rows = rows[:top_n]
    labels = [
        (r["source_en"] if len(r["source_en"]) <= 36 else r["source_en"][:33] + "...")
        for r in rows
    ]
    occ = [int(r["occurrences_in_source"]) for r in rows]
    applied = [int(r["applied_in_target"]) for r in rows]
    missed = [o - a for o, a in zip(occ, applied, strict=True)]
    hit_rate = [a / o for o, a in zip(occ, applied, strict=True)]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, max(6, 0.32 * len(labels))))
    ax.barh(y, applied, color="#3aaa5b", edgecolor="white", label="applied")
    ax.barh(y, missed, left=applied, color="#d04848", edgecolor="white", label="missed")
    for i, (o, a) in enumerate(zip(occ, applied, strict=True)):
        ax.text(o + 0.5, i, f"{a}/{o}  ({a / o * 100:.0f}%)", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("occurrences in source")
    ax.set_title(f"Glossary application — top {len(rows)} terms by occurrence")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def per_pdf_glossary_hit_rate(pdf_csv: Path, dest: Path) -> None:
    rows = read_csv(pdf_csv)
    rows = [r for r in rows if int(r["term_occurrences"]) > 0]
    for r in rows:
        r["__rate"] = float(int(r["terms_applied"]) / int(r["term_occurrences"]))
    rows.sort(key=lambda r: r["__rate"])
    labels = [r["pdf"].split("/", 1)[-1].replace(".pdf", "") for r in rows]
    rates = [r["__rate"] for r in rows]
    occ = [int(r["term_occurrences"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, [r * 100 for r in rates], color="#2b8cbe", edgecolor="white")
    overall = float(np.mean(rates)) * 100
    ax.axvline(overall, color="#222", linestyle="--", linewidth=1.2, label=f"corpus mean = {overall:.1f}%")
    for bar, rate, n in zip(bars, rates, occ, strict=True):
        ax.text(rate * 100 + 0.5, bar.get_y() + bar.get_height() / 2, f"{rate * 100:.0f}%  (n={n})", va="center", fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("glossary terms correctly applied (%)")
    ax.set_title("Glossary application by PDF")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def row_status_pie(audit_csv: Path, dest: Path) -> None:
    rows = read_csv(audit_csv)
    no_terms = 0
    fully = 0
    partial = 0
    for r in rows:
        rate = r.get("row_hit_rate", "")
        if not rate:
            no_terms += 1
            continue
        try:
            num, den = rate.split("/")
            num, den = int(num), int(den)
        except Exception:
            continue
        if den == 0:
            continue
        if num == den:
            fully += 1
        else:
            partial += 1
    labels = ["no glossary terms", "fully applied", "partially applied"]
    sizes = [no_terms, fully, partial]
    colors = ["#cccccc", "#3aaa5b", "#d04848"]
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=[f"{lbl}\n({s:,})" for lbl, s in zip(labels, sizes, strict=True)],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    ax.set_title("Per-row glossary application status")
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=PROJECT_ROOT / "results_final" / "quality" / "quality_scores.csv",
    )
    parser.add_argument(
        "--glossary-dir", type=Path, default=PROJECT_ROOT / "results_final" / "glossary"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "results_final" / "plots"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] quality scores: {args.quality_csv}")
    q_rows = read_csv(args.quality_csv)
    labse = [v for r in q_rows if (v := fnum(r.get("labse_cos", ""))) is not None]
    bert = [v for r in q_rows if (v := fnum(r.get("bertscore_xlmr_f1", ""))) is not None]
    char_count = [int(r["char_count"]) for r in q_rows if r.get("char_count", "").isdigit()]
    print(f"[load] labse={len(labse):,}  bertscore={len(bert):,}  with-length={len(char_count):,}")

    if labse:
        quality_distribution(
            labse,
            title="LaBSE cosine similarity (EN source vs ES target)",
            xlabel="LaBSE cosine similarity",
            dest=args.out_dir / "q1_labse_distribution.png",
            color="#3aaa5b",
        )
        per_pdf_mean_bar(
            q_rows,
            metric_col="labse_cos",
            title="Mean LaBSE cosine similarity by PDF",
            xlabel="mean LaBSE",
            dest=args.out_dir / "q4_per_pdf_mean_labse.png",
            color="#3aaa5b",
        )
        quality_buckets_per_pdf(
            q_rows,
            metric_col="labse_cos",
            cutoffs=[
                ("poor (<0.7)", 0.7, "#d04848"),
                ("ok (0.7-0.8)", 0.8, "#e8a25b"),
                ("good (0.8-0.9)", 0.9, "#7fc497"),
                ("great (>=0.9)", 1.01, "#1f8a3a"),
            ],
            title="LaBSE quality buckets by PDF (% of segments)",
            dest=args.out_dir / "q6_buckets_per_pdf.png",
        )
    if bert:
        quality_distribution(
            bert,
            title="BERTScore F1 with xlm-roberta-large (EN source as pseudo-ref)",
            xlabel="BERTScore F1",
            dest=args.out_dir / "q2_bertscore_distribution.png",
            color="#5b8fc7",
        )
    if labse and bert and len(labse) == len(bert):
        scatter(
            bert, labse,
            xlabel="BERTScore F1 (xlm-roberta-large)",
            ylabel="LaBSE cosine similarity",
            title="LaBSE vs BERTScore — same rows",
            dest=args.out_dir / "q3_labse_vs_bertscore.png",
            diag=True,
        )
    if labse and char_count and len(labse) == len(char_count):
        scatter(
            char_count, labse,
            xlabel="segment length (characters)",
            ylabel="LaBSE cosine similarity",
            title="Translation quality vs segment length",
            dest=args.out_dir / "q5_score_vs_length.png",
        )

    term_csv = args.glossary_dir / "glossary_term_stats.csv"
    pdf_csv = args.glossary_dir / "glossary_pdf_stats.csv"
    audit_csv = args.glossary_dir / "glossary_audit.csv"
    if term_csv.is_file():
        print(f"[load] {term_csv}")
        per_term_hit_rate(term_csv, args.out_dir / "g1_per_term_hit_rate.png")
    if pdf_csv.is_file():
        print(f"[load] {pdf_csv}")
        per_pdf_glossary_hit_rate(pdf_csv, args.out_dir / "g2_per_pdf_hit_rate.png")
    if audit_csv.is_file():
        print(f"[load] {audit_csv}")
        row_status_pie(audit_csv, args.out_dir / "g3_row_status_pie.png")

    print(f"\n[done] plots saved under {args.out_dir.resolve()}")
    for p in sorted(args.out_dir.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
