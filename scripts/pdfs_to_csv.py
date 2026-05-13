"""Extract every PDF under `data/` into CSVs of EN segments ready for OPUS-MT.

Pipeline (designed for technical shotblasting manuals):
  1. Walk `data/<category>/*.pdf`.
  2. Extract text page-by-page with PyMuPDF (best reading order on technical PDFs).
  3. Detect & strip running headers / footers (lines repeated on >=50% of pages).
  4. Per page: drop noise (page numbers, isolated drawing labels, pure numerics,
     lone punctuation), then split paragraphs into sentence-ish segments.
  5. Drop CAPS-only parts-list dumps that the PDF extractor glued onto one line
     (these were the bottom of the COMET-QE distribution in v1 of the corpus).
  6. Drop segments whose source language is not English. Uses fastText
     ``lid.176.bin`` at ``models/lid.176.bin`` (download once via the link in
     ``scripts/filter_english.py``). Pass ``--no-lang-filter`` to keep every
     segment regardless of language. Without the LID model the script still
     works, just without language filtering.
  7. Write one CSV per PDF mirroring the category layout, plus a combined
     `data/csv_final/all_segments.csv` for batch translation.

The CSV schema is built around OPUS-MT (Helsinki-NLP MarianMT, ~512-token limit):
  id, category, pdf, page, segment_idx, source_en, target_es, char_count

`target_es` is left blank — fill it in by running the segments through the OPUS
engine in `app/services/`. `id` is a deterministic hash, so re-running this
script over updated PDFs preserves row identity for previously-translated rows.

Usage (from the project root):

    python scripts/pdfs_to_csv.py
    python scripts/pdfs_to_csv.py --data-dir data --out-dir data/csv_final
    python scripts/pdfs_to_csv.py --no-lang-filter        # keep every language
    python scripts/pdfs_to_csv.py --lang-min-confidence 0.40
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import pymupdf  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fall back for older PyMuPDF installs
    import fitz as pymupdf  # type: ignore[import-not-found,no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LID_MODEL_DEFAULT = PROJECT_ROOT / "models" / "lid.176.bin"

CSV_COLUMNS = [
    "id",
    "category",
    "pdf",
    "page",
    "segment_idx",
    "source_en",
    "target_es",
    "char_count",
]

# Segments longer than this will be re-split (OPUS-MT truncates beyond ~512 tokens).
MAX_SEGMENT_CHARS = 800
# Drop segments with fewer than this many letters (catches "!", "—", part numbers).
MIN_LETTERS = 3

# "Page 3", "Page 3 of 12", "3 / 12", "- 3 -" etc.
PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s+\d+(?:\s*(?:of|/)\s*\d+)?|\d+\s*/\s*\d+|[-–—]\s*\d+\s*[-–—]|\d{1,3})\s*$",
    re.IGNORECASE,
)
# Lines that are only digits / punctuation / spaces.
NUMERIC_NOISE_RE = re.compile(r"^[\d\s.,;:!?()\[\]{}/\\\-–—_=+*#°'\"]+$")
# Bare part numbers like "ZVP-PC-0027-01", "BAC-VA-PB-0060". No spaces, all uppercase/digits.
PART_NUMBER_RE = re.compile(r"^[A-Z]{2,5}(?:-[A-Z0-9]+){1,5}$")
# Sentence splitter: break on .!? followed by whitespace and an upper/paren/digit start.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\d])")
# Whitespace normalisation (collapse runs of any whitespace, including NBSP and ZWSP).
WS_RE = re.compile(r"[\s\u00a0\u200b\u200c\u200d\ufeff]+")

# ── Parts-list detection ──────────────────────────────────────────────────────
# When PDF extractors flatten a parts-list block (or an exploded-view table)
# into one line, we get strings like
#   ``WINDOW FRAME SEALING BAND STRAP HANDLE BREATHING TUBE ASSEMBLY ...``      [all caps]
#   ``PARTS LIST Air Cooling Controller And Belt Assembly Item Stock Code ...`` [Title Case]
# 80+ characters of consecutive nouns with no verb and no internal punctuation.
# These dominated the COMET-QE bottom of the v1+v2 distributions (mean ≈ -1.3).
# We drop them at extraction time. There are two flavors:
#
#   (a) ``is_caps_parts_dump`` — predominantly uppercase. v2 already used this.
#   (b) ``is_titlecase_parts_dump`` — Title-Case-heavy noun dumps that escaped
#       (a) because they were not all-uppercase. v3 adds this.
#
# We MUST NOT drop legitimate all-caps warnings like
#   ``WARNING - DO NOT USE THE SUPPLIED AIR RESPIRATOR HELMET IF THE AIR FLOW...``
# so we whitelist any segment containing at least one instruction/auxiliary word.
CAPS_INSTRUCTION_WORDS = frozenset(
    {
        # Modals & auxiliaries
        "DO", "DONT", "DOES", "DID", "IS", "ARE", "WAS", "WERE", "HAS", "HAVE",
        "BE", "BEEN", "BEING", "CAN", "COULD", "WILL", "WOULD", "SHALL", "SHOULD",
        "MAY", "MIGHT", "MUST", "OUGHT",
        # Imperative verbs commonly seen in safety / install instructions
        "USE", "USED", "USING", "USES",
        "READ", "FOLLOW", "AVOID", "ENSURE", "CHECK", "VERIFY", "INSPECT",
        "CONNECT", "DISCONNECT", "REMOVE", "PLACE", "REPLACE", "REPAIR",
        "INSTALL", "ATTACH", "TIGHTEN", "LOOSEN", "MAKE", "MAKES", "MADE",
        "PROVIDE", "PROTECT", "OPEN", "CLOSE", "TURN", "PRESS", "PULL", "PUSH",
        "WEAR", "WEARING", "STOP", "START", "STARTED", "CONTINUE",
        # Negation / quantifiers / conjunctions
        "NOT", "NO", "NEVER", "ALWAYS", "ONLY", "BOTH", "EACH",
        "IF", "WHEN", "WHILE", "BEFORE", "AFTER", "UNTIL", "UNLESS",
        "AND", "OR", "BUT", "WITH", "WITHOUT", "FROM", "TO", "AT", "BY",
        # Safety vocabulary
        "WARNING", "WARNINGS", "CAUTION", "DANGER", "NOTE", "NOTICE", "IMPORTANT",
        "ATTENTION", "REQUIRED", "REQUIRES", "NEEDED",
    }
)

# Catalog / parts-list keyword set — when many of these co-occur in a single
# segment with no internal sentence punctuation, it's almost certainly a flat
# table that PyMuPDF glued back into one line.
CATALOG_KEYWORDS = frozenset(
    {
        "PARTS", "LIST", "LISTING",
        "STOCK", "CODE", "DESCRIPTION", "DESC",
        "ITEM", "ITEMS", "QTY", "QUANTITY", "REF",
        "ASSEMBLY", "ASSY", "ASSEMBLE", "ASSEMBLED",
        "EXPLODED", "VIEW", "DIAGRAM",
        "MODEL", "PART", "NUMBER", "NO",
        "KIT", "SET",
    }
)

# Subset of CAPS_INSTRUCTION_WORDS that — even in a flat Title-Case noun dump —
# strongly indicates a real sentence (not a glued parts table). We deliberately
# exclude bridge words like AND / OR / WITH / TO because parts tables routinely
# contain them between catalog rows ("Air Cooling Controller And Belt Assembly").
STRICT_INSTRUCTION_WORDS = frozenset(
    {
        # Modals & auxiliaries
        "DO", "DONT", "DOES", "DID", "IS", "ARE", "WAS", "WERE", "HAS", "HAVE",
        "BE", "BEEN", "BEING", "CAN", "COULD", "WILL", "WOULD", "SHALL", "SHOULD",
        "MAY", "MIGHT", "MUST", "OUGHT",
        # Imperative verbs (excludes CONNECT / DISCONNECT — those frequently
        # appear as adjectives in catalog phrases like "Quick Disconnect
        # Coupling", "Connect Hose Fitting", and we'd lose real parts dumps).
        "USE", "USED", "USING", "USES",
        "READ", "FOLLOW", "AVOID", "ENSURE", "CHECK", "VERIFY", "INSPECT",
        "REMOVE", "REPLACE", "REPAIR",
        "INSTALL", "TIGHTEN", "LOOSEN", "MAKE", "MAKES", "MADE",
        "PROVIDE", "PROTECT", "TURN", "PRESS", "PULL", "PUSH",
        "WEAR", "WEARING", "STOP", "START", "STARTED", "CONTINUE",
        # Negation
        "NOT", "NEVER",
        # Safety vocabulary
        "WARNING", "WARNINGS", "CAUTION", "DANGER", "NOTE", "NOTICE", "IMPORTANT",
        "ATTENTION", "REQUIRED", "REQUIRES", "NEEDED",
    }
)


def is_caps_parts_dump(line: str) -> bool:
    """Detect an ALL-CAPS parts-list line glued by the PDF extractor.

    True iff the line is long (>= 30 letters), mostly uppercase (>= 85%),
    has many tokens (>= 5), no internal ``.!?``, and no instruction word.
    """
    letters = [ch for ch in line if ch.isalpha()]
    if len(letters) < 30:
        return False
    upper_letters = sum(1 for ch in letters if ch.isupper())
    if upper_letters / len(letters) < 0.85:
        return False
    tokens = re.findall(r"[A-Za-z']+", line)
    if len(tokens) < 5:
        return False
    inner = line[1:-1]
    if any(ch in inner for ch in ".!?"):
        return False
    upper_tokens = {t.upper() for t in tokens}
    if upper_tokens & CAPS_INSTRUCTION_WORDS:
        return False  # legitimate warning / imperative
    return True


def is_titlecase_parts_dump(line: str) -> bool:
    """Detect a Title-Case parts-list line glued by the PDF extractor.

    True iff the line is:
      * long (>= 80 chars and >= 12 word tokens),
      * Title-Case dominant (>= 60% of words start with an uppercase letter),
      * has no internal ``.!?`` and at most one ``,``,
      * carries two or more catalog/parts keywords (``PARTS``, ``LIST``,
        ``STOCK``, ``DESCRIPTION``, ``ITEM``, ``ASSEMBLY``, ``EXPLODED``…).

    The catalog-keyword requirement keeps us from accidentally dropping
    legitimate Title-Case headings or product names.
    """
    if len(line) < 80:
        return False
    tokens = re.findall(r"[A-Za-z']+", line)
    if len(tokens) < 12:
        return False
    if any(ch in line[1:-1] for ch in ".!?"):
        return False
    if line.count(",") > 1:
        return False
    title_or_caps = sum(1 for t in tokens if t[0].isupper())
    if title_or_caps / len(tokens) < 0.6:
        return False
    upper_tokens = {t.upper() for t in tokens}
    if len(upper_tokens & CATALOG_KEYWORDS) < 2:
        return False
    # Don't drop if it looks like an instructional sentence wearing Title Case.
    # We deliberately use the *strict* subset here (excludes AND / OR / WITH /
    # TO) because catalog tables routinely contain those between rows.
    if upper_tokens & STRICT_INSTRUCTION_WORDS:
        return False
    return True


def is_parts_dump(line: str) -> bool:
    """Either flavor of glued parts-list line."""
    return is_caps_parts_dump(line) or is_titlecase_parts_dump(line)


@dataclass
class Segment:
    category: str
    pdf: str
    page: int
    segment_idx: int
    source_en: str

    def as_row(self) -> dict[str, str | int]:
        sid = hashlib.sha1(
            f"{self.category}|{self.pdf}|{self.page}|{self.segment_idx}|{self.source_en}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        return {
            "id": sid,
            "category": self.category,
            "pdf": self.pdf,
            "page": self.page,
            "segment_idx": self.segment_idx,
            "source_en": self.source_en,
            "target_es": "",
            "char_count": len(self.source_en),
        }


def normalise(line: str) -> str:
    return WS_RE.sub(" ", line).strip()


def is_noise(line: str) -> bool:
    # Require at least MIN_LETTERS alphabetic characters — kills "!", "—", "1.", etc.,
    # without depending on byte length (zero-width / invisible chars can fool len()).
    if sum(ch.isalpha() for ch in line) < MIN_LETTERS:
        return True
    if PAGE_NUMBER_RE.match(line):
        return True
    if NUMERIC_NOISE_RE.match(line):
        return True
    if PART_NUMBER_RE.match(line):
        return True
    if is_parts_dump(line):
        return True
    return False


# ── fastText language ID (lazy) ──────────────────────────────────────────────
class LanguageFilter:
    """Lazy-loading wrapper around fastText ``lid.176.bin``.

    Created with ``enabled=False`` if the user passes ``--no-lang-filter`` or
    the LID model is not on disk; in that case ``is_english`` always returns
    True so the rest of the pipeline behaves as before.
    """

    def __init__(self, enabled: bool, model_path: Path, min_confidence: float) -> None:
        self._enabled = enabled
        self._model_path = model_path
        self._min_confidence = min_confidence
        self._model = None
        self.rejected: Counter[str] = Counter()
        self.kept_total = 0
        self.kept_short = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import fasttext  # type: ignore[import-not-found]
        except ImportError:
            print(
                "[lid] WARNING: `fasttext` not installed — language filter disabled. "
                "Install `fasttext-wheel` from requirements-eval.txt to enable.",
                file=sys.stderr,
            )
            self._enabled = False
            return
        fasttext.FastText.eprint = lambda x: None  # silence the noisy banner
        if not self._model_path.is_file():
            print(
                f"[lid] WARNING: model not found at {self._model_path} — language filter disabled. "
                f"Download once with:\n"
                f"  Invoke-WebRequest "
                f"-Uri 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin' "
                f"-OutFile '{self._model_path}'",
                file=sys.stderr,
            )
            self._enabled = False
            return
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self._model = fasttext.load_model(str(self._model_path))

    def is_english(self, text: str) -> bool:
        """Always True if the filter is disabled. Always True for very short fragments
        (LID is unreliable on them; we keep them and let `is_noise` decide).

        Two v3 hardenings vs. the v2 implementation:
          * we **lowercase before predict** — empirically `lid.176.bin` mis-classifies
            uppercase Dutch / German / French fragments as ``en@0.31`` and lets them
            slip through, but the lowercase form is correctly ``nl@0.79``;
          * the default ``--lang-min-confidence`` is raised to ``0.55`` (was 0.30)
            so we reject the borderline ``en@0.3-0.5`` band where LID was guessing.
        """
        if not self._enabled:
            return True
        letters = sum(1 for ch in text if ch.isalpha())
        if letters < 8:
            # Too short for reliable LID; defer to noise / dedup elsewhere.
            self.kept_short += 1
            return True
        self._load()
        if self._model is None:
            return True  # load failed, behave as disabled
        cleaned = WS_RE.sub(" ", text).strip().lower()
        labels, probs = self._model.predict(cleaned, k=1)
        code = (labels[0] if labels else "").replace("__label__", "")
        conf = float(probs[0]) if probs.size else 0.0
        if code == "en" and conf >= self._min_confidence:
            self.kept_total += 1
            return True
        self.rejected[code or "<empty>"] += 1
        return False


def extract_page_lines(doc: "pymupdf.Document") -> list[list[str]]:
    """Return per-page lists of normalised, non-empty raw lines."""
    pages: list[list[str]] = []
    for page in doc:
        raw = page.get_text("text") or ""
        lines = [normalise(ln) for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]
        pages.append(lines)
    return pages


def repeated_headers_footers(pages: list[list[str]]) -> set[str]:
    """Lines that appear at the top OR bottom of >=50% of pages are treated as chrome."""
    if len(pages) < 3:
        return set()
    top_counter: Counter[str] = Counter()
    bot_counter: Counter[str] = Counter()
    for lines in pages:
        if not lines:
            continue
        for ln in lines[:2]:
            top_counter[ln] += 1
        for ln in lines[-2:]:
            bot_counter[ln] += 1
    threshold = max(2, len(pages) // 2)
    chrome = {ln for ln, c in top_counter.items() if c >= threshold}
    chrome |= {ln for ln, c in bot_counter.items() if c >= threshold}
    return chrome


def split_long(segment: str) -> list[str]:
    """Split a too-long segment on sentence boundaries, then commas, then hard chunks."""
    if len(segment) <= MAX_SEGMENT_CHARS:
        return [segment]

    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(segment) if p.strip()]
    out: list[str] = []
    for part in parts:
        if len(part) <= MAX_SEGMENT_CHARS:
            out.append(part)
            continue
        # Fall back to comma splits.
        sub = [s.strip() for s in re.split(r",\s+", part) if s.strip()]
        buf = ""
        for chunk in sub:
            candidate = f"{buf}, {chunk}" if buf else chunk
            if len(candidate) > MAX_SEGMENT_CHARS and buf:
                out.append(buf)
                buf = chunk
            else:
                buf = candidate
        if buf:
            out.append(buf)
    # Hard fallback so we never emit anything > MAX_SEGMENT_CHARS.
    final: list[str] = []
    for piece in out:
        if len(piece) <= MAX_SEGMENT_CHARS:
            final.append(piece)
        else:
            for i in range(0, len(piece), MAX_SEGMENT_CHARS):
                final.append(piece[i : i + MAX_SEGMENT_CHARS].strip())
    return [p for p in final if p]


def paragraphs_to_segments(lines: list[str]) -> list[str]:
    """Group consecutive lines into paragraphs, then sentence-split each paragraph.

    The grouping rule: a paragraph ends at a sentence terminator. Bullet/list
    fragments without trailing punctuation are concatenated with the next line,
    which matches how PDF extraction tends to break wrapped lines.
    """
    paragraphs: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            paragraphs.append(" ".join(buf))
            buf.clear()

    for ln in lines:
        buf.append(ln)
        if ln.endswith((".", "!", "?")):
            flush()
    flush()

    segments: list[str] = []
    for para in paragraphs:
        for sent in SENTENCE_SPLIT_RE.split(para):
            sent = sent.strip()
            if sent and not is_noise(sent):
                segments.extend(s for s in split_long(sent) if not is_noise(s))
    return segments


def segments_for_pdf(
    pdf_path: Path, category: str, lang_filter: LanguageFilter | None = None
) -> list[Segment]:
    with pymupdf.open(pdf_path) as doc:
        pages = extract_page_lines(doc)
    chrome = repeated_headers_footers(pages)

    out: list[Segment] = []
    seen_global: set[str] = set()  # dedupe across the whole PDF
    for page_no, lines in enumerate(pages, start=1):
        cleaned = [ln for ln in lines if ln not in chrome and not is_noise(ln)]
        page_segments = paragraphs_to_segments(cleaned)

        idx = 0
        for seg in page_segments:
            key = seg.lower()
            if key in seen_global:
                continue
            seen_global.add(key)
            if lang_filter is not None and not lang_filter.is_english(seg):
                continue
            out.append(
                Segment(
                    category=category,
                    pdf=pdf_path.name,
                    page=page_no,
                    segment_idx=idx,
                    source_en=seg,
                )
            )
            idx += 1
    return out


def write_csv(rows: Iterable[dict[str, str | int]], dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    # utf-8-sig => Excel opens it correctly on Windows; newline="" => no blank lines.
    with dest.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Folder containing <category>/*.pdf (default: ./data)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where CSVs are written (default: <data-dir>/csv_final)",
    )
    parser.add_argument(
        "--no-lang-filter",
        action="store_true",
        help="Keep every segment regardless of language (skip fastText LID).",
    )
    parser.add_argument(
        "--lang-model",
        type=Path,
        default=LID_MODEL_DEFAULT,
        help=f"Path to fastText LID model (default: {LID_MODEL_DEFAULT.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--lang-min-confidence",
        type=float,
        default=0.55,
        help='Minimum confidence to accept "__label__en" (default: 0.55).',
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    out_dir: Path = (args.out_dir or (data_dir / "csv_final")).resolve()
    if not data_dir.is_dir():
        print(f"Missing data dir: {data_dir}", file=sys.stderr)
        return 1

    pdfs = sorted(p for p in data_dir.rglob("*.pdf") if out_dir not in p.parents)
    if not pdfs:
        print(f"No PDFs found under {data_dir}", file=sys.stderr)
        return 1

    lang_filter = LanguageFilter(
        enabled=not args.no_lang_filter,
        model_path=args.lang_model.resolve(),
        min_confidence=args.lang_min_confidence,
    )
    if lang_filter.enabled:
        print(
            f"[lid] language filter ENABLED  model={args.lang_model.name}"
            f"  min_conf={args.lang_min_confidence}"
        )
    else:
        print("[lid] language filter DISABLED")

    print(f"Extracting {len(pdfs)} PDF(s) -> {out_dir}")
    combined: list[dict[str, str | int]] = []
    per_pdf_counts: list[tuple[str, int, int]] = []

    for pdf in pdfs:
        category = pdf.parent.relative_to(data_dir).as_posix() or "_root"
        segs = segments_for_pdf(pdf, category, lang_filter=lang_filter)
        rows = [s.as_row() for s in segs]
        rel_csv = out_dir / category / (pdf.stem + ".csv")
        written = write_csv(rows, rel_csv)
        combined.extend(rows)
        try:
            page_count = sum(1 for _ in pymupdf.open(pdf))
        except Exception:
            page_count = -1
        per_pdf_counts.append((f"{category}/{pdf.name}", page_count, written))
        print(f"  {category}/{pdf.name}: {page_count} pages -> {written} segments")

    combined_csv = out_dir / "all_segments.csv"
    write_csv(combined, combined_csv)

    total_chars = sum(int(r["char_count"]) for r in combined)
    print()
    print(f"Wrote {len(combined)} segments to {combined_csv.relative_to(data_dir.parent)}")
    print(f"Total source characters: {total_chars:,}")
    print(f"Per-PDF CSVs: {out_dir.relative_to(data_dir.parent)}/<category>/<stem>.csv")

    if lang_filter.enabled:
        total_rej = sum(lang_filter.rejected.values())
        kept_lid = lang_filter.kept_total
        kept_short = lang_filter.kept_short
        scanned = kept_lid + kept_short + total_rej
        if scanned:
            print()
            print(
                f"[lid] scanned segments: {scanned:,}  "
                f"kept: {kept_lid + kept_short:,}  rejected: {total_rej:,}"
            )
            if kept_short:
                print(f"[lid]   kept-too-short-for-LID: {kept_short:,}")
            if total_rej:
                print("[lid] top rejected source-languages:")
                for code, count in lang_filter.rejected.most_common(10):
                    print(f"[lid]   {code:<6} {count:>5}  ({count * 100 / total_rej:.1f}% of rejects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
