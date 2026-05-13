"""Pre-download Marian + Qwen weights into the HuggingFace cache.

Without this step the *first* `/translate` request lazily downloads ~5.5 GB of
weights (mostly Qwen2.5-3B-Instruct), causing a multi-minute delay. Running
this script once primes ``$HF_HOME`` (or the default ``~/.cache/huggingface``)
so subsequent container starts are instant.

Examples
--------
    # Local (Windows / Linux):
    python scripts/prewarm_hf_cache.py

    # Inside Docker (uses the `prewarm` profile in docker-compose.yml):
    docker compose --profile prewarm run --rm prewarm

    # Skip Qwen (e.g. CPU-only deployment without Qwen post-editing):
    python scripts/prewarm_hf_cache.py --no-qwen
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings


def _human(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def prewarm_marian(model_name: str) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"[1/2] Marian: {model_name}")
    t0 = time.time()
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print(f"      done in {_human(time.time() - t0)}")


def prewarm_qwen(model_name: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[2/2] Qwen:   {model_name}")
    t0 = time.time()
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
    print(f"      done in {_human(time.time() - t0)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--marian-model",
        default=settings.mt_model_name,
        help=f"Marian model id (default: {settings.mt_model_name})",
    )
    parser.add_argument(
        "--qwen-model",
        default=settings.postedit_qwen_model,
        help=f"Qwen model id (default: {settings.postedit_qwen_model})",
    )
    parser.add_argument(
        "--no-qwen",
        action="store_true",
        help="Skip Qwen download (only Marian).",
    )
    args = parser.parse_args()

    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    print(f"HF_HOME = {hf_home}")
    print("=" * 60)

    t_start = time.time()
    prewarm_marian(args.marian_model)
    if not args.no_qwen:
        prewarm_qwen(args.qwen_model)
    else:
        print("[2/2] Qwen:   skipped (--no-qwen)")

    print("=" * 60)
    print(f"[done] HF cache primed in {_human(time.time() - t_start)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
