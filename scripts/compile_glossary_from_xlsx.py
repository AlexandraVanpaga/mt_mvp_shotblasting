"""
Compile `glossary/en_es_shotblasting.json` from `raw_glossary.xlsx`.

Expected layout: two columns, no header row — English (col A), Spanish (col B).
Run from project root: `python scripts/compile_glossary_from_xlsx.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    xlsx = root / "raw_glossary.xlsx"
    out = root / "glossary" / "en_es_shotblasting.json"

    if not xlsx.is_file():
        print(f"Missing {xlsx}", file=sys.stderr)
        return 1

    df = pd.read_excel(xlsx, header=None, names=["source", "target"])
    df["source"] = df["source"].astype(str).map(lambda s: s.strip())
    df["target"] = df["target"].astype(str).map(lambda s: s.strip())
    df = df[(df["source"] != "nan") & (df["target"] != "nan") & (df["source"] != "") & (df["target"] != "")]

    # Same English twice: keep the more informative Spanish (longer), else last row wins.
    chosen: dict[str, str] = {}
    for src, tgt in zip(df["source"], df["target"], strict=True):
        if src not in chosen or len(tgt) > len(chosen[src]):
            chosen[src] = tgt

    entries = [{"source": s, "target": t, "notes": None} for s, t in chosen.items()]
    entries.sort(key=lambda e: len(e["source"]), reverse=True)

    payload = {
        "locale_pair": "en-es",
        "domain": "shotblasting_equipment",
        "source_file": "raw_glossary.xlsx",
        "entries": entries,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
