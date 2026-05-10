"""
Convert a Hugging Face Marian checkpoint to CTranslate2 format.

Example:
  python scripts/convert_marian_to_ct2.py \\
    --model Helsinki-NLP/opus-mt-en-es \\
    --output-dir models/opus-mt-en-es-ct2

Requires: pip install ctranslate2 transformers
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="HF Marian → CTranslate2 converter")
    p.add_argument("--model", default="Helsinki-NLP/opus-mt-en-es", help="HF model id or local path")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models") / "opus-mt-en-es-ct2",
        help="Directory for model.bin (created if missing)",
    )
    p.add_argument(
        "--quantization",
        default=None,
        help="Optional: int8, int8_float32, float16, bfloat16, float32 (see ct2-transformers-converter --help)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a non-empty output_dir (required if it already exists)",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "ctranslate2.converters.transformers",
        "--model",
        args.model,
        "--output_dir",
        str(args.output_dir.resolve()),
    ]
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])
    if args.force:
        cmd.append("--force")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done. Point MT_MVP_CT2_MODEL_DIR (or default models/opus-mt-en-es-ct2) at this folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
