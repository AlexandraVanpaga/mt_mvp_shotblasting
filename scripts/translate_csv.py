"""Batch-translate `data/csv_final/all_segments.csv` through the full EN→ES pipeline.

Pipeline order is identical to the live HTTP route — both routes call
:func:`app.services.translation.run_pipeline`:

  1. ALL-CAPS sentence-case the source if it is mostly upper.
  2. Glossary placeholder protection of the English source.
  3. Machine translation via the configured engine (CT2 Marian / HF Marian / NLLB).
  4. Restore canonical Spanish targets from the placeholders.
  5. Optional Qwen 2.5 Instruct post-edit + glossary re-assertion + cleanup.
  6. UPPER-case the output if the source was ALL-CAPS.

Outputs are written to `translated_data_final/`:
  * `all_segments.csv` — every input row with `target_es` populated.
  * `<category>/<pdf_stem>.csv` — per-PDF CSVs mirroring the input layout.

The job is **resumable**: rerunning the script reuses any rows whose `target_es`
is already populated in `translated_data_final/all_segments.csv`. Output is flushed to
disk every `--flush-every` rows (default 25), so a hard kill leaves the partial
file intact and re-running picks up exactly where it stopped.

Usage:
  python scripts/translate_csv.py                       # full pipeline, full corpus
  python scripts/translate_csv.py --no-postedit         # MT-only fast pass
  python scripts/translate_csv.py --limit 50            # smoke test on first 50 rows
  python scripts/translate_csv.py --no-glossary --no-postedit  # raw MT only
  python scripts/translate_csv.py --engine nllb \\
      --mt-model facebook/nllb-200-distilled-1.3B       # try a different engine
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.deps import (  # noqa: E402
    _resolve_mt_engine,
    _resolve_qwen,
    get_glossary,
)
from app.core.config import Settings  # noqa: E402
from app.services.glossary import Glossary  # noqa: E402
from app.services.postedit import PostEditor  # noqa: E402
from app.services.translation import MTEngine, run_pipeline  # noqa: E402


INPUT_COLUMNS = [
    "id",
    "category",
    "pdf",
    "page",
    "segment_idx",
    "source_en",
    "target_es",
    "char_count",
]
OUTPUT_COLUMNS = INPUT_COLUMNS + ["glossary_applied", "postedit_applied", "error"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    for r in rows:
        for col in OUTPUT_COLUMNS:
            r.setdefault(col, "")
    return rows


def write_rows(rows: list[dict[str, str]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(dest)


def translate_one(
    text: str,
    *,
    glossary: Glossary,
    engine: MTEngine,
    posteditor: PostEditor | None,
    apply_glossary: bool,
) -> tuple[str, bool, bool]:
    """Return ``(translation, postedit_applied, was_uppercase)``.

    Thin wrapper over :func:`app.services.translation.run_pipeline` so the batch
    job and the live HTTP route share the *exact* same translation logic.
    """
    result = run_pipeline(
        text,
        glossary=glossary,
        engine=engine,
        posteditor=posteditor,
        apply_glossary=apply_glossary,
    )
    return result.translation, result.postedit_applied, result.was_uppercase


def split_by_pdf(rows: list[dict[str, str]], out_dir: Path) -> None:
    """Write per-PDF mirrored CSVs to <out_dir>/<category>/<pdf_stem>.csv."""
    buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        key = (r["category"], r["pdf"])
        buckets.setdefault(key, []).append(r)
    for (category, pdf), group in buckets.items():
        stem = Path(pdf).stem
        dest = out_dir / category / f"{stem}.csv"
        write_rows(group, dest)


def format_eta(secs_left: float) -> str:
    if secs_left < 0 or secs_left != secs_left:  # NaN guard
        return "?"
    h, rem = divmod(int(secs_left), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "csv_final" / "all_segments.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "translated_data_final",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Translate at most N rows (skips count toward limit only if untranslated)."
    )
    parser.add_argument("--no-glossary", action="store_true", help="Skip glossary protection.")
    parser.add_argument("--no-postedit", action="store_true", help="Skip Qwen post-edit (MT-only).")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Flush partial results to disk every N rows (default: 25).",
    )
    parser.add_argument(
        "--start-fresh",
        action="store_true",
        help="Ignore any existing translated_data_final/all_segments.csv and re-translate everything.",
    )
    parser.add_argument(
        "--engine",
        choices=["ctranslate2", "marian_hf", "nllb"],
        default=None,
        help="Override MT_MVP_MT_ENGINE for this run (e.g. compare Marian vs NLLB).",
    )
    parser.add_argument(
        "--mt-model",
        default=None,
        help="Override MT_MVP_MT_MODEL_NAME (e.g. facebook/nllb-200-distilled-1.3B).",
    )
    args = parser.parse_args()

    src_csv: Path = args.input.resolve()
    out_dir: Path = args.output_dir.resolve()
    combined_out = out_dir / "all_segments.csv"

    if not src_csv.is_file():
        print(f"ERROR: input not found: {src_csv}", file=sys.stderr)
        return 1

    cfg = Settings()
    if args.engine is not None:
        cfg.mt_engine = args.engine
    if args.mt_model is not None:
        cfg.mt_model_name = args.mt_model
    if args.no_postedit:
        cfg.postedit_use_qwen = False

    print(f"[config] mt_engine={cfg.mt_engine} model={cfg.mt_model_name}")
    print(
        f"[config] glossary={'on' if not args.no_glossary else 'off'}  "
        f"postedit={'on (Qwen ' + cfg.postedit_qwen_model + ')' if (cfg.postedit_use_qwen and not args.no_postedit) else 'off'}"
    )

    print("[load] glossary…")
    glossary = get_glossary(cfg)
    print(f"[load] glossary entries: {len(glossary.entries)}")

    print(f"[load] MT engine ({cfg.mt_engine})…")
    engine = _resolve_mt_engine(cfg)

    posteditor: PostEditor | None = None
    if cfg.postedit_use_qwen and not args.no_postedit:
        print("[load] Qwen post-edit model (this may take a minute on first run)…")
        qwen = _resolve_qwen(cfg)
        posteditor = PostEditor(cfg.postedit_prompt_path, qwen=qwen)
    elif not args.no_postedit:
        # Glossary-only post-edit (cleanup + spacing) with no neural step.
        posteditor = PostEditor(cfg.postedit_prompt_path, qwen=None)

    rows = read_rows(src_csv)
    print(f"[input] {len(rows)} segment(s) from {src_csv.relative_to(PROJECT_ROOT)}")

    # Merge existing translated rows for resume.
    if combined_out.is_file() and not args.start_fresh:
        existing = {r["id"]: r for r in read_rows(combined_out)}
        carried = 0
        for r in rows:
            prior = existing.get(r["id"])
            if prior and prior.get("target_es", "").strip():
                r["target_es"] = prior["target_es"]
                r["glossary_applied"] = prior.get("glossary_applied", "")
                r["postedit_applied"] = prior.get("postedit_applied", "")
                r["error"] = prior.get("error", "")
                carried += 1
        if carried:
            print(f"[resume] reusing {carried} previously-translated row(s) from {combined_out.name}")

    to_do = [r for r in rows if not r.get("target_es", "").strip()]
    if args.limit is not None:
        to_do = to_do[: args.limit]
    total = len(to_do)
    print(f"[plan] rows to translate this run: {total}")
    if total == 0:
        print("[done] nothing to do.")
        write_rows(rows, combined_out)
        split_by_pdf(rows, out_dir)
        print(f"[write] {combined_out.relative_to(PROJECT_ROOT)}")
        return 0

    apply_glossary = not args.no_glossary
    started = time.time()
    errors = 0
    since_flush = 0
    # Cross-row translation cache. The PanBlast corpus has ~25% duplicate EN
    # paragraphs across PDFs (safety boilerplate, parts captions). Reusing the
    # cached output cuts a full run from ~45 min to ~32 min on our box without
    # changing the per-PDF outputs (results are identical to the un-cached run).
    cache: dict[tuple[str, bool], tuple[str, bool, bool]] = {}
    cache_hits = 0
    allcaps_rewrites = 0

    for i, row in enumerate(to_do, start=1):
        src = row["source_en"].strip()
        if not src:
            row["target_es"] = ""
            row["error"] = "empty_source"
            errors += 1
            continue
        cache_key = (src, apply_glossary)
        try:
            cached = cache.get(cache_key)
            if cached is not None:
                translation, postedit_used, was_upper = cached
                cache_hits += 1
            else:
                translation, postedit_used, was_upper = translate_one(
                    src,
                    glossary=glossary,
                    engine=engine,
                    posteditor=posteditor,
                    apply_glossary=apply_glossary,
                )
                cache[cache_key] = (translation, postedit_used, was_upper)
            if was_upper:
                allcaps_rewrites += 1
            row["target_es"] = translation
            row["glossary_applied"] = "true" if apply_glossary else "false"
            row["postedit_applied"] = "true" if postedit_used else "false"
            row["error"] = ""
        except Exception as exc:  # noqa: BLE001 - keep batch alive
            row["target_es"] = ""
            row["glossary_applied"] = "true" if apply_glossary else "false"
            row["postedit_applied"] = ""
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors += 1

        since_flush += 1
        if since_flush >= args.flush_every or i == total:
            write_rows(rows, combined_out)
            since_flush = 0

        if i == 1 or i % 10 == 0 or i == total:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = (total - i) / rate if rate > 0 else 0.0
            sys.stdout.write(
                f"\r[run] {i}/{total} ({i * 100 / total:5.1f}%)  "
                f"{rate:5.2f} seg/s  elapsed {format_eta(elapsed)}  eta {format_eta(remaining)}  "
                f"err {errors}    "
            )
            sys.stdout.flush()

    sys.stdout.write("\n")
    print(f"[write] {combined_out.relative_to(PROJECT_ROOT)}")
    split_by_pdf(rows, out_dir)
    print(f"[write] per-PDF CSVs under {out_dir.relative_to(PROJECT_ROOT)}/<category>/")

    elapsed = time.time() - started
    print(
        f"[done] translated {total - errors}/{total} this run "
        f"({errors} error(s)) in {format_eta(elapsed)}"
    )
    if cache_hits:
        print(
            f"[cache] duplicate-EN cache hits: {cache_hits} "
            f"({cache_hits * 100 / max(total, 1):.1f}% of this run)"
        )
    if allcaps_rewrites:
        print(
            f"[allcaps] sentence-cased {allcaps_rewrites} segment(s) "
            f"({allcaps_rewrites * 100 / max(total, 1):.1f}% of this run); "
            "MT output is upper-cased back at the end of the pipeline."
        )
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
