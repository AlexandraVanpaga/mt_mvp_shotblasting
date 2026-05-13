"""One-shot driver: run glossary + LaBSE + BERTScore + COMET-QE for a translation run.

Chains the four evaluation scripts in order:

  1. ``scripts/evaluate_glossary.py``   → ``<out>/glossary/``
  2. ``scripts/evaluate_quality.py``    → ``<out>/quality/``    (LaBSE + BERTScore F1)
  3. ``scripts/run_comet_on_worst.py``  → ``<out>/quality/comet_qe_worst.csv``
  4. ``scripts/make_eval_plots.py``     → ``<out>/plots/``

Usage:

  python scripts/run_full_eval.py
  python scripts/run_full_eval.py --input translated_data_final/all_segments.csv \
      --out-dir results_final
  python scripts/run_full_eval.py --no-comet --no-plots   # quick smoke pass

If you ever iterate on the pipeline and want to diff two runs, use
``scripts/compare_versions.py`` directly:

  python scripts/compare_versions.py --prev-dir results_final \
      --curr-dir results_final_v2 --prev-label final --curr-label final-v2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _run(args: list[str]) -> int:
    """Subprocess wrapper that echoes the command and aborts on failure."""
    print()
    print(">>> " + " ".join(str(a) for a in args))
    print()
    return subprocess.call([sys.executable, *args], cwd=str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "translated_data_final" / "all_segments.csv",
        help="Translated CSV to evaluate (default: translated_data_final/all_segments.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results_final",
        help="Where to write glossary/, quality/, plots/ (default: results_final).",
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=PROJECT_ROOT / "glossary" / "en_es_shotblasting.json",
    )
    parser.add_argument(
        "--top-comet",
        type=int,
        default=None,
        help="How many rows to run COMET-QE on. Default = all rows in --input.",
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Skip the COMET-QE step (saves ~30 s but loses the most-trusted metric).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the matplotlib plots step.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    glossary_out = args.out_dir / "glossary"
    quality_out = args.out_dir / "quality"
    plots_out = args.out_dir / "plots"

    rc = _run(
        [
            str(SCRIPTS / "evaluate_glossary.py"),
            "--input",
            str(args.input),
            "--glossary",
            str(args.glossary),
            "--out-dir",
            str(glossary_out),
        ]
    )
    if rc != 0:
        return rc

    rc = _run(
        [
            str(SCRIPTS / "evaluate_quality.py"),
            "--input",
            str(args.input),
            "--out-dir",
            str(quality_out),
            "--sample",
            "all",
            "--no-comet",
        ]
    )
    if rc != 0:
        return rc

    if not args.no_comet:
        # Default to the full corpus if --top-comet is unset (the script
        # silently caps at the number of available rows when --top exceeds it).
        top = args.top_comet if args.top_comet is not None else 100_000
        rc = _run(
            [
                str(SCRIPTS / "run_comet_on_worst.py"),
                "--input",
                str(quality_out / "quality_scores.csv"),
                "--out",
                str(quality_out / "comet_qe_worst.csv"),
                "--top",
                str(top),
            ]
        )
        if rc != 0:
            return rc

    if not args.no_plots:
        rc = _run(
            [
                str(SCRIPTS / "make_eval_plots.py"),
                "--quality-csv",
                str(quality_out / "quality_scores.csv"),
                "--glossary-dir",
                str(glossary_out),
                "--out-dir",
                str(plots_out),
            ]
        )
        if rc != 0:
            return rc

    print()
    print(f"[done] evaluation artefacts in {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
