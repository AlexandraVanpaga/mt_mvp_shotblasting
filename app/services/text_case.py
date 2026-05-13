"""Sentence-case preprocessor for long ALL-CAPS sentences.

Marian (and to a lesser extent NLLB) was trained on mixed-case text and
mis-translates segments like ``NOTE: NEVER LIFT THE HELMET ASSEMBLY ...``
— it treats tokens like ``LIFT`` / ``MAY`` / ``OCCUR`` as proper-noun
fragments and either transliterates them (``LIFT -> LÍNEA``) or copies
them through verbatim. The fix that lands cleanly without retraining is
to sentence-case the input before MT, then UPPER-case the output if the
source was uppercase.

Detection threshold (``--upper-ratio``) is tuned on the PanBlast corpus:
60 % of alphabetic chars being upper, with a floor of 8 alphabetic chars
to avoid catching short ACRONYM HEADERS like ``NPT FITTING``.
"""

from __future__ import annotations

import re

_LOWER_RE = re.compile(r"[a-záéíóúüñ]")
_SENTENCE_BOUNDARY = re.compile(r"([.!?]\s+)([a-záéíóúüñ])")
_FIRST_ALPHA = re.compile(r"^(\W*)([a-záéíóúüñ])")


def is_mostly_uppercase(text: str, threshold: float = 0.6, min_alpha: int = 8) -> bool:
    """Return True iff text is dominated by upper-case letters.

    ``threshold`` is the share of *alphabetic* chars that must be upper-case;
    ``min_alpha`` is a floor so short headers ("NPT FITTING") don't trigger
    the preprocessor — they are usually safe for Marian as-is.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < min_alpha:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= threshold


def to_sentence_case(text: str) -> str:
    """Lowercase the whole text, then upper-case the first letter and the
    first letter after every ``.!?`` sentence boundary.

    Glossary placeholders (``__GLS{i}__``) survive because they contain
    digits / underscores only — they are unaffected by ``.lower()``.
    """
    lower = text.lower()
    result = _FIRST_ALPHA.sub(lambda m: m.group(1) + m.group(2).upper(), lower, count=1)
    result = _SENTENCE_BOUNDARY.sub(lambda m: m.group(1) + m.group(2).upper(), result)
    return result


def preprocess_for_mt(text: str, threshold: float = 0.6, min_alpha: int = 8) -> tuple[str, bool]:
    """Return ``(rewritten, was_mostly_upper)``.

    Caller is responsible for re-applying upper-case to the MT output if
    ``was_mostly_upper`` is True (see :func:`postprocess_after_mt`).
    """
    if is_mostly_uppercase(text, threshold=threshold, min_alpha=min_alpha):
        return to_sentence_case(text), True
    return text, False


def postprocess_after_mt(text: str, was_mostly_upper: bool) -> str:
    """Restore ALL-CAPS register on the MT output when needed."""
    return text.upper() if was_mostly_upper else text
