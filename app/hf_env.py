"""Load ``<project>/.env`` into ``os.environ`` for unprefixed keys (e.g. HF tokens).

``Settings`` only reads ``MT_MVP_*`` from ``.env``. Hugging Face libraries expect
``HUGGINGFACE_HUB_TOKEN`` / ``HF_TOKEN`` in the environment — call
:func:`load_root_dotenv` early in CLI entrypoints that use HF.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_root_dotenv() -> None:
    path = _PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)
