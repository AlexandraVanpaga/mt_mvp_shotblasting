"""Pre-download NLLB-200 weights to the HuggingFace cache without loading onto GPU.

Used during the v4 model sweep to overlap weight download with the in-flight
Marian translation run (Marian uses the GPU; HF downloads are pure I/O so
they don't contend).

Usage:

  python scripts/prefetch_nllb.py --model facebook/nllb-200-distilled-600M
  python scripts/prefetch_nllb.py --model facebook/nllb-200-distilled-1.3B
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="HuggingFace model id to pre-download.",
    )
    parser.add_argument(
        "--allow-bin",
        action="store_true",
        help="Also fetch *.bin checkpoint shards (otherwise prefer safetensors only).",
    )
    args = parser.parse_args()

    # Without --allow-bin we still pull *.bin because NLLB upstream may not
    # ship safetensors. The point of this script is to overlap network I/O
    # with the live GPU run; the load-time torch≥2.6 check happens later
    # when the real NLLB engine starts.
    allow_patterns: list[str] | None = None
    print(f"[prefetch] snapshot_download {args.model} -> HF_HOME cache...")
    snapshot_path = snapshot_download(
        repo_id=args.model,
        allow_patterns=allow_patterns,
    )
    print(f"[prefetch] cached at: {snapshot_path}")
    print("[prefetch] done — engine boot will skip the network on first run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
