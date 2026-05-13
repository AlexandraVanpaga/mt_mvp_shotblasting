"""Download `Unbabel/wmt20-comet-qe-da` from HuggingFace and stage it locally.

Implementation note (Windows)
=============================
COMET's bundled `download_model(...)` first tries `huggingface_hub.snapshot_download`
with the default cache (symlink-based). On vanilla Windows without Developer
Mode this fails with `OSError: [WinError 1314] A required privilege is not
held by the client` and COMET then incorrectly falls back to the legacy S3
registry, raising `KeyError("Model not supported by COMET")`.

We bypass that entire path by calling `snapshot_download` ourselves with
`local_dir=models/<leaf>` and `local_dir_use_symlinks=False`, then loading
the checkpoint with `load_from_checkpoint` directly. Result: the model lives
inside the project at `models/wmt20-comet-qe-da/checkpoints/model.ckpt` and
no longer depends on the user's HF cache.

Usage:
    python scripts/download_comet_qe.py
    python scripts/download_comet_qe.py --model Unbabel/wmt20-comet-qe-da
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Silence the noisy "your machine does not support symlinks" warning - we
# explicitly disable symlinks below.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from app.hf_env import load_root_dotenv

    load_root_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Unbabel/wmt20-comet-qe-da")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Local destination dir. Default: models/<model-name-leaf>/",
    )
    args = parser.parse_args()

    leaf = args.model.split("/", 1)[-1]
    dest_root = args.dest or (PROJECT_ROOT / "models" / leaf)
    dest_root.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"[download] {args.model} -> {dest_root.relative_to(PROJECT_ROOT)}")
    snapshot_download(
        repo_id=args.model,
        local_dir=str(dest_root),
        local_dir_use_symlinks=False,
    )

    local_ckpt = dest_root / "checkpoints" / "model.ckpt"
    if local_ckpt.is_file():
        print(f"[ok] local checkpoint ready: {local_ckpt.relative_to(PROJECT_ROOT)}")
        print(f"[ok] size: {local_ckpt.stat().st_size / 1e6:.1f} MB")
    else:
        # Some COMET repos place the checkpoint at the repo root rather than under checkpoints/.
        candidates = list(dest_root.rglob("*.ckpt"))
        if candidates:
            local_ckpt = candidates[0]
            print(f"[ok] checkpoint found at: {local_ckpt.relative_to(PROJECT_ROOT)}")
        else:
            print(f"ERROR: no .ckpt file found under {dest_root}", file=sys.stderr)
            return 1

    print("[verify] loading checkpoint from local path...")
    from comet import load_from_checkpoint

    _ = load_from_checkpoint(str(local_ckpt))
    print("[verify] load_from_checkpoint OK")
    print(f"\n  use it from python:\n    from comet import load_from_checkpoint\n    model = load_from_checkpoint(r\"{local_ckpt}\")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
