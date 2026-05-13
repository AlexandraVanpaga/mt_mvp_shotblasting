"""Side-by-side comparison of two evaluation runs.

Generic tool — useful whenever you iterate on the pipeline and want a diff
between the current ``results_final/`` and a previously archived run.

Each ``--prev-dir`` / ``--curr-dir`` is expected to contain the standard
evaluation layout produced by ``scripts/evaluate_quality.py`` +
``scripts/evaluate_glossary.py`` + ``scripts/run_comet_on_worst.py``:

  <dir>/
    quality/quality_summary.json
    quality/comet_qe_worst.csv
    glossary/glossary_term_stats.csv
    glossary/glossary_audit.csv

The script prints a Markdown comparison table to stdout AND writes the same
text to ``<curr-dir>/comparison_<prev_label>_<curr_label>.md`` (override with
``--out``). If any input file is missing the row falls back to ``"—"`` rather
than aborting so partial runs still produce a useful report.

Example:

  python scripts/compare_versions.py \
    --prev-dir results_final_backup --prev-label baseline \
    --curr-dir results_final         --curr-label new
"""

from __future__ import annotations

import argparse
import csv
import json
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


# COMET-QE banding (DA-z-score scale). Anything below -1.0 is essentially
# unintelligible MT; -0.2 .. 0.1 is "publishable with light human review".
_COMET_BANDS = [
    (-1.0, "broken"),
    (-0.5, "very_weak"),
    (-0.2, "weak"),
    (0.1, "ok"),
    (10.0, "good"),
]


def _comet_stats(rows: list[dict[str, str]]) -> dict[str, float | int]:
    vals: list[float] = []
    for r in rows:
        try:
            vals.append(float(r["comet_qe"]))
        except (KeyError, ValueError):
            continue
    if not vals:
        return {"n": 0}
    counts = {label: 0 for _, label in _COMET_BANDS}
    for v in vals:
        for upper, label in _COMET_BANDS:
            if v < upper:
                counts[label] += 1
                break
    return {"n": len(vals), "mean": mean(vals), **counts}


def _glossary_summary(path: Path) -> dict[str, int | float]:
    if not path.is_file():
        return {}
    rows = _read_csv(path)
    occ = sum(int(r["occurrences_in_source"]) for r in rows)
    app = sum(int(r["applied_in_target"]) for r in rows)
    rate = (app / occ) if occ else 0.0
    full_miss = sum(
        1 for r in rows
        if int(r["occurrences_in_source"]) > 0 and int(r["applied_in_target"]) == 0
    )
    return {"occ": occ, "applied": app, "hit_rate": rate, "fully_missed_terms": full_miss}


def _row_status(audit_csv: Path) -> dict[str, int]:
    rows = _read_csv(audit_csv)
    no_terms = fully = partial = 0
    for r in rows:
        rate = r.get("row_hit_rate", "")
        if not rate:
            no_terms += 1
            continue
        try:
            num, den = (int(x) for x in rate.split("/"))
        except ValueError:
            continue
        if den == 0:
            continue
        if num == den:
            fully += 1
        else:
            partial += 1
    return {"no_terms": no_terms, "fully": fully, "partial": partial}


def _fmt(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _delta(a: object, b: object) -> str:
    if a is None or b is None:
        return "—"
    try:
        return f"{float(b) - float(a):+.4f}"
    except (TypeError, ValueError):
        return "—"


def _metric(summary: dict, metric: str, key: str) -> float | None:
    m = (summary.get("metrics") or {}).get(metric) or {}
    return m.get(key)


def build_report(
    prev_dir: Path,
    curr_dir: Path,
    prev_label: str,
    curr_label: str,
) -> str:
    p_sum = _safe_load_json(prev_dir / "quality" / "quality_summary.json")
    c_sum = _safe_load_json(curr_dir / "quality" / "quality_summary.json")
    p_gloss = _glossary_summary(prev_dir / "glossary" / "glossary_term_stats.csv")
    c_gloss = _glossary_summary(curr_dir / "glossary" / "glossary_term_stats.csv")
    p_rows = _row_status(prev_dir / "glossary" / "glossary_audit.csv")
    c_rows = _row_status(curr_dir / "glossary" / "glossary_audit.csv")
    p_cmt = _comet_stats(_read_csv(prev_dir / "quality" / "comet_qe_worst.csv"))
    c_cmt = _comet_stats(_read_csv(curr_dir / "quality" / "comet_qe_worst.csv"))

    out: list[str] = []
    push = out.append
    push(f"# {prev_label} → {curr_label} comparison")
    push("")
    push(
        f"Side-by-side metrics for `{prev_dir.name}` (previous) vs "
        f"`{curr_dir.name}` (current)."
    )
    push("")

    # ── Corpus size ──
    push("## Corpus size")
    push("")
    push(f"| Stage | {prev_label} | {curr_label} |")
    push("|---|---:|---:|")
    push(
        f"| Segments evaluated | {p_sum.get('sample_size', '—')} | "
        f"{c_sum.get('sample_size', '—')} |"
    )
    push("")

    # ── Quality metrics ──
    push("## Quality metrics — corpus means (95 % CI half-width)")
    push("")
    push(f"| Metric | {prev_label} | {curr_label} | Δ |")
    push("|---|---:|---:|---:|")
    for code, label in [
        ("labse_cos", "LaBSE cosine"),
        ("bertscore_xlmr_f1", "BERTScore F1"),
    ]:
        pm = _metric(p_sum, code, "mean")
        cm = _metric(c_sum, code, "mean")
        pc = _metric(p_sum, code, "ci95_halfwidth")
        cc = _metric(c_sum, code, "ci95_halfwidth")
        push(
            f"| {label} | {_fmt(pm)} (±{_fmt(pc)}) | "
            f"{_fmt(cm)} (±{_fmt(cc)}) | {_delta(pm, cm)} |"
        )
    push(
        f"| COMET-QE | {_fmt(p_cmt.get('mean'))} (n={p_cmt.get('n')}) | "
        f"{_fmt(c_cmt.get('mean'))} (n={c_cmt.get('n')}) | "
        f"{_delta(p_cmt.get('mean'), c_cmt.get('mean'))} |"
    )
    push("")

    # ── Glossary ──
    push("## Glossary application")
    push("")
    push(f"| Stat | {prev_label} | {curr_label} | Δ |")
    push("|---|---:|---:|---:|")
    for k, label in [
        ("occ", "Term occurrences in source"),
        ("applied", "Applied in target"),
        ("hit_rate", "Term-level hit rate"),
        ("fully_missed_terms", "Terms with 100 % miss"),
    ]:
        pv = p_gloss.get(k)
        cv = c_gloss.get(k)
        if isinstance(pv, float):
            pvs = f"{pv * 100:.1f}%"
            cvs = f"{cv * 100:.1f}%" if cv is not None else "—"
            dv = (
                f"{(cv - pv) * 100:+.1f} pp"
                if pv is not None and cv is not None
                else "—"
            )
        else:
            pvs = _fmt(pv)
            cvs = _fmt(cv)
            dv = _delta(pv, cv) if pv is not None and cv is not None else "—"
        push(f"| {label} | {pvs} | {cvs} | {dv} |")
    push("")

    push("## Per-row glossary status")
    push("")
    push(f"| Status | {prev_label} | {curr_label} |")
    push("|---|---:|---:|")
    for k, label in [
        ("no_terms", "No glossary terms"),
        ("fully", "All terms applied"),
        ("partial", "Partially applied"),
    ]:
        push(f"| {label} | {p_rows.get(k, '—')} | {c_rows.get(k, '—')} |")
    push("")

    # ── COMET-QE buckets ──
    if p_cmt.get("n") and c_cmt.get("n"):
        push("## COMET-QE bucket distribution")
        push("")
        push(f"| Bucket | {prev_label} | {curr_label} | Δ |")
        push("|---|---:|---:|---:|")
        for k, label in [
            ("broken", "broken (<-1.0)"),
            ("very_weak", "very weak (-1.0 .. -0.5)"),
            ("weak", "weak (-0.5 .. -0.2)"),
            ("ok", "ok (-0.2 .. 0.1)"),
            ("good", "good (>=0.1)"),
        ]:
            pn = p_cmt["n"]
            cn = c_cmt["n"]
            pp = p_cmt.get(k, 0) * 100 / pn
            cp = c_cmt.get(k, 0) * 100 / cn
            push(
                f"| {label} | {p_cmt.get(k, 0)} ({pp:.1f}%) | "
                f"{c_cmt.get(k, 0)} ({cp:.1f}%) | {cp - pp:+.1f} pp |"
            )
        push("")

    return "\n".join(out) + "\n"


def _slug(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label.strip().lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prev-dir", type=Path, required=True)
    parser.add_argument("--curr-dir", type=Path, required=True)
    parser.add_argument("--prev-label", default=None)
    parser.add_argument("--curr-label", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    prev_label = args.prev_label or args.prev_dir.name
    curr_label = args.curr_label or args.curr_dir.name
    out_path = args.out or (
        args.curr_dir / f"comparison_{_slug(prev_label)}_{_slug(curr_label)}.md"
    )

    text = build_report(args.prev_dir, args.curr_dir, prev_label, curr_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    try:
        rel = out_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_path
    print(f"[done] wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
