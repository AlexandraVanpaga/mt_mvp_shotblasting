"""Reference-less translation quality evaluation: COMET-Kiwi + BERTScore + LaBSE.

What it does
============
Reads `translated_data_final/all_segments.csv`, samples a test set, and scores every
(source_en, target_es) pair with up to three complementary reference-free
metrics:

  * COMET-Kiwi (reference-free QE) — `Unbabel/wmt22-cometkiwi-da`
    State-of-the-art reference-less MT quality estimation. Score in roughly
    [0..1] (higher = better). Requires HuggingFace authentication: run
    `huggingface-cli login` and accept the license at
    https://huggingface.co/Unbabel/wmt22-cometkiwi-da .

  * LaBSE cross-lingual cosine similarity — `sentence-transformers/LaBSE`
    Google's Language-Agnostic BERT, trained specifically for cross-lingual
    sentence-level similarity (109 languages). Cosine of L2-normalised
    [CLS] embeddings of source and target. Ungated. Higher = more semantic
    equivalence between EN source and ES target. Empirically a stronger
    reference-free MT-quality proxy than vanilla BERTScore.

  * Cross-lingual BERTScore F1 — `xlm-roberta-large`
    Token-level multilingual similarity between source and target.
    Standard BERTScore is reference-based; we use the source as a
    pseudo-reference. Interpretation is relative, not absolute.

All metrics run in batches on GPU when CUDA is available, otherwise CPU.

Test-set size (math is documented; --sample sets the actual count)
==================================================================
We want a tight 95% confidence interval on the mean score:

    half-width E = 1.96 * sigma / sqrt(n)   =>   n = (1.96 * sigma / E)^2

Empirical sigma for COMET-Kiwi on a single MT system is typically 0.08-0.13.
Using sigma = 0.12 (conservative):

    +/- 0.02  ->   138 samples
    +/- 0.01  ->   553 samples
    +/- 0.005 -> 2,212 samples

With our population N = 4,201 the finite-population correction trims those a
bit (n_eff = n / (1 + (n-1)/N)). The default `--sample 500` therefore lands at
roughly +/- 0.01 CI on the mean, which is plenty to compare pipeline variants
(e.g. with vs without Qwen post-edit). For the full corpus pass `--sample all`.

Outputs (under `evaluation/`)
=============================
  * `quality_scores.csv`     — per-row metrics on the sampled set
  * `quality_summary.json`   — aggregate stats (mean, std, percentiles, buckets)
  * stdout summary           — same numbers, plus worst / best examples

Usage
=====
  pip install -r requirements-eval.txt           # one-time
  huggingface-cli login                          # needed for COMET-Kiwi (gated)
  python scripts/evaluate_quality.py             # sample 500
  python scripts/evaluate_quality.py --sample 1000
  python scripts/evaluate_quality.py --sample all
  python scripts/evaluate_quality.py --no-comet  # skip COMET (no HF auth)
  python scripts/evaluate_quality.py --no-labse  # skip LaBSE
  python scripts/evaluate_quality.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from statistics import median, pstdev, quantiles

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMET_DEFAULT = "Unbabel/wmt22-cometkiwi-da"
BERTSCORE_DEFAULT = "xlm-roberta-large"
# Layer 17 of xlm-roberta-large is the BERTScore-recommended layer for multilingual.
BERTSCORE_LAYER = 17
LABSE_DEFAULT = "sentence-transformers/LaBSE"


def _read_translated(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [
        r
        for r in rows
        if r.get("target_es", "").strip() and not r.get("error", "").strip()
    ]


def _sample(rows: list[dict[str, str]], n: int | None, seed: int) -> list[dict[str, str]]:
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def _summary(values: list[float], name: str) -> dict[str, float | int]:
    if not values:
        return {"name": name, "n": 0}
    q = quantiles(values, n=20) if len(values) >= 20 else []
    return {
        "name": name,
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "p05": float(q[0]) if q else float(min(values)),
        "p25": float(q[4]) if q else float(min(values)),
        "median": float(median(values)),
        "p75": float(q[14]) if q else float(max(values)),
        "p95": float(q[18]) if q else float(max(values)),
        "max": float(max(values)),
    }


def _ci_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    s = pstdev(values)
    return 1.96 * s / math.sqrt(len(values))


def _bucketize(values: list[float], cutoffs: list[tuple[str, float]]) -> dict[str, int]:
    out: dict[str, int] = {label: 0 for label, _ in cutoffs}
    for v in values:
        for label, upper in cutoffs:
            if v < upper:
                out[label] += 1
                break
        else:
            out[cutoffs[-1][0]] += 1
    return out


def run_comet(
    pairs: list[tuple[str, str]],
    model_name: str,
    device: str,
    batch_size: int,
) -> list[float]:
    """Reference-free QE with COMET-Kiwi. Each pair is (source_en, target_es)."""
    from comet import download_model, load_from_checkpoint  # type: ignore

    print(f"[comet] downloading / loading {model_name} (first run may take minutes)...")
    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    data = [{"src": src, "mt": mt} for src, mt in pairs]
    gpus = 1 if device == "cuda" else 0
    print(f"[comet] scoring {len(data)} pairs on {device} (batch={batch_size})")
    out = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=True)
    # COMET 2.x returns a Prediction object with .scores (list) and .system_score.
    scores = list(getattr(out, "scores", out))
    return [float(s) for s in scores]


def run_labse(
    sources: list[str],
    targets: list[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> list[float]:
    """Cross-lingual sentence similarity (cosine) via LaBSE pooled+normalised."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    print(f"[labse] loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    @torch.inference_mode()
    def embed(texts: list[str]) -> "torch.Tensor":
        out_chunks: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            # LaBSE uses the pooled [CLS] output then L2-normalises.
            emb = outputs.pooler_output
            emb = F.normalize(emb, p=2, dim=1)
            out_chunks.append(emb.cpu())
        return torch.cat(out_chunks, dim=0)

    print(f"[labse] embedding {len(sources)} source + {len(targets)} target on {device} (batch={batch_size})")
    src = embed(sources)
    tgt = embed(targets)
    cos = (src * tgt).sum(dim=1).tolist()
    # Free GPU memory before BERTScore (xlm-roberta-large is heavy).
    del model
    del tok
    if device == "cuda":
        import torch as _torch

        _torch.cuda.empty_cache()
    return [float(v) for v in cos]


def run_bertscore(
    sources: list[str],
    targets: list[str],
    model_type: str,
    device: str,
    batch_size: int,
) -> list[float]:
    """Cross-lingual reference-free similarity (F1) via multilingual BERTScore."""
    from bert_score import score  # type: ignore

    print(f"[bert ] scoring {len(targets)} pairs with {model_type} on {device}")
    _, _, f1 = score(
        cands=targets,
        refs=sources,
        model_type=model_type,
        num_layers=BERTSCORE_LAYER,
        device=device,
        batch_size=batch_size,
        lang="es",
        rescale_with_baseline=False,
        verbose=False,
    )
    return [float(v) for v in f1.tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "translated_data_final" / "all_segments.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "evaluation"
    )
    parser.add_argument(
        "--sample",
        default="500",
        help='Sample size: integer or "all". Default 500 (~+/-0.01 CI for sigma=0.12).',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--no-labse", action="store_true")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--comet-model", default=COMET_DEFAULT)
    parser.add_argument("--bertscore-model", default=BERTSCORE_DEFAULT)
    parser.add_argument("--labse-model", default=LABSE_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help='"cuda" or "cpu" (default: auto). Use cpu while batch translation is running.',
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: not found: {args.input}", file=sys.stderr)
        return 1

    # Resolve device.
    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"[device] {device}")
    if device == "cuda":
        try:
            import torch

            free, total = torch.cuda.mem_get_info()
            print(f"[device] CUDA mem free: {free / 1e9:.1f} / {total / 1e9:.1f} GB")
            if free < 3 * 1024**3:
                print("[device] WARNING: <3 GB free on GPU. If another job is running, pass --device cpu.")
        except Exception:
            pass

    print(f"[load] {args.input}")
    rows = _read_translated(args.input)
    print(f"[load] translated rows available: {len(rows):,}")
    if not rows:
        print("Nothing to evaluate.", file=sys.stderr)
        return 1

    if args.sample.lower() in {"all", "full", "0", "-1"}:
        sample_n: int | None = None
    else:
        try:
            sample_n = int(args.sample)
        except ValueError:
            print(f"ERROR: --sample must be integer or 'all', got {args.sample!r}", file=sys.stderr)
            return 1
    sampled = _sample(rows, sample_n, args.seed)
    print(f"[sample] using {len(sampled):,} rows (seed={args.seed})")

    sources = [r["source_en"] for r in sampled]
    targets = [r["target_es"] for r in sampled]

    comet_scores: list[float] = []
    bert_scores: list[float] = []
    labse_scores: list[float] = []

    if not args.no_comet:
        t0 = time.time()
        try:
            comet_scores = run_comet(
                list(zip(sources, targets, strict=True)),
                model_name=args.comet_model,
                device=device,
                batch_size=args.batch_size,
            )
            print(f"[comet] done in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"[comet] FAILED: {exc!r}")
            print(
                "[comet] If the model is gated, run `huggingface-cli login` and accept the\n"
                "        license at https://huggingface.co/Unbabel/wmt22-cometkiwi-da .\n"
                "        As a fallback, retry with --comet-model Unbabel/wmt20-comet-qe-da"
            )

    if not args.no_labse:
        t0 = time.time()
        try:
            labse_scores = run_labse(
                sources, targets,
                model_name=args.labse_model,
                device=device,
                batch_size=args.batch_size,
            )
            print(f"[labse] done in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"[labse] FAILED: {exc!r}")

    if not args.no_bertscore:
        t0 = time.time()
        try:
            bert_scores = run_bertscore(
                sources, targets,
                model_type=args.bertscore_model,
                device=device,
                batch_size=args.batch_size,
            )
            print(f"[bert ] done in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"[bert ] FAILED: {exc!r}")

    # Per-row CSV.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = args.out_dir / "quality_scores.csv"
    cols = [
        "id",
        "category",
        "pdf",
        "page",
        "char_count",
        "comet_kiwi",
        "labse_cos",
        "bertscore_xlmr_f1",
        "source_en",
        "target_es",
    ]
    with detail_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for i, r in enumerate(sampled):
            writer.writerow(
                {
                    "id": r.get("id", ""),
                    "category": r.get("category", ""),
                    "pdf": r.get("pdf", ""),
                    "page": r.get("page", ""),
                    "char_count": r.get("char_count", ""),
                    "comet_kiwi": f"{comet_scores[i]:.4f}" if comet_scores else "",
                    "labse_cos": f"{labse_scores[i]:.4f}" if labse_scores else "",
                    "bertscore_xlmr_f1": f"{bert_scores[i]:.4f}" if bert_scores else "",
                    "source_en": r["source_en"],
                    "target_es": r["target_es"],
                }
            )

    # Aggregate.
    summary: dict[str, object] = {
        "input": str(args.input),
        "sample_size": len(sampled),
        "sample_seed": args.seed,
        "device": device,
        "comet_model": args.comet_model if comet_scores else None,
        "labse_model": args.labse_model if labse_scores else None,
        "bertscore_model": args.bertscore_model if bert_scores else None,
        "metrics": {},
    }

    if comet_scores:
        s = _summary(comet_scores, "comet_kiwi")
        s["ci95_halfwidth"] = _ci_half_width(comet_scores)
        s["buckets"] = _bucketize(
            comet_scores,
            cutoffs=[("poor (<0.5)", 0.5), ("ok (0.5-0.7)", 0.7), ("good (0.7-0.85)", 0.85), ("great (>=0.85)", 1.01)],
        )
        summary["metrics"]["comet_kiwi"] = s
    if labse_scores:
        s = _summary(labse_scores, "labse_cos")
        s["ci95_halfwidth"] = _ci_half_width(labse_scores)
        s["buckets"] = _bucketize(
            labse_scores,
            cutoffs=[("poor (<0.7)", 0.7), ("ok (0.7-0.8)", 0.8), ("good (0.8-0.9)", 0.9), ("great (>=0.9)", 1.01)],
        )
        summary["metrics"]["labse_cos"] = s
    if bert_scores:
        s = _summary(bert_scores, "bertscore_xlmr_f1")
        s["ci95_halfwidth"] = _ci_half_width(bert_scores)
        s["buckets"] = _bucketize(
            bert_scores,
            cutoffs=[("low (<0.7)", 0.7), ("mid (0.7-0.85)", 0.85), ("high (0.85-0.95)", 0.95), ("very-high (>=0.95)", 1.01)],
        )
        summary["metrics"]["bertscore_xlmr_f1"] = s

    summary_path = args.out_dir / "quality_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("==== quality evaluation summary ====")
    for name, s in summary["metrics"].items():
        print(f"\n  {name}:")
        print(f"    n           : {s['n']}")
        print(f"    mean        : {s['mean']:.4f}  (95% CI +/- {s['ci95_halfwidth']:.4f})")
        print(f"    std         : {s['std']:.4f}")
        print(f"    p05 / median / p95 : {s['p05']:.4f} / {s['median']:.4f} / {s['p95']:.4f}")
        print(f"    min / max   : {s['min']:.4f} / {s['max']:.4f}")
        print(f"    buckets     :")
        for label, count in s["buckets"].items():
            pct = count * 100 / s["n"]
            print(f"        {label:<22s} {count:>4} ({pct:5.1f}%)")

    # Worst / best examples (rank by COMET if available, else LaBSE, else BERTScore).
    if comet_scores:
        scored = sorted(zip(comet_scores, sampled, strict=True), key=lambda x: x[0])
    elif labse_scores:
        scored = sorted(zip(labse_scores, sampled, strict=True), key=lambda x: x[0])
    elif bert_scores:
        scored = sorted(zip(bert_scores, sampled, strict=True), key=lambda x: x[0])
    else:
        scored = []

    if scored:
        print("\n  worst 5:")
        for sc, r in scored[:5]:
            print(f"    [{sc:.3f}] {r['source_en'][:80]!r}")
            print(f"            -> {r['target_es'][:80]!r}")
        print("\n  best 5:")
        for sc, r in reversed(scored[-5:]):
            print(f"    [{sc:.3f}] {r['source_en'][:80]!r}")
            print(f"            -> {r['target_es'][:80]!r}")

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p.resolve())

    print("\n  outputs:")
    print(f"    {_rel(detail_csv)}")
    print(f"    {_rel(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
