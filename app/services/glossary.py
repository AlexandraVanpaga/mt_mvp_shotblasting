from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    target: str
    notes: str | None


# Placeholder regex is loose to cover MT-side tokenisation drift. We have
# observed Marian to occasionally drop one trailing underscore (``__GLS62_``
# instead of ``__GLS62__``) when the placeholder sits next to punctuation like
# ``@`` or a literal ``.``. We accept 1 or 2 trailing underscores; the capture
# group is the numeric index, so the lookup key is always normalised to the
# canonical ``__GLS{i}__`` form.
_PLACEHOLDER_RE = re.compile(r"__GLS(\d+)_{1,2}")


def _fold_accents(text: str) -> str:
    """Strip combining diacritics. ``"Válvula" -> "Valvula"``, ``"señal" -> "senal"``."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


# Accent equivalence classes for case+accent-insensitive matching. Covers the
# vowels Spanish post-edit drift can realistically hit (we have seen Qwen lose
# acute accents on ``á é í ó ú`` and the tilde on ``ñ``). The compiled patterns
# below are wrapped with ``re.IGNORECASE``, so we only need to enumerate one case.
_ACCENT_CLASS = {
    "a": "[aáàâä]",
    "e": "[eéèêë]",
    "i": "[iíìîï]",
    "o": "[oóòôö]",
    "u": "[uúùûü]",
    "n": "[nñ]",
    "y": "[yý]",
    "c": "[cç]",
}


def _accent_loose_pattern(phrase: str) -> re.Pattern[str] | None:
    """Whitespace-, case-, AND accent-insensitive phrase regex.

    Built so ``Válvula neumática`` matches ``Valvula Neumatica`` and friends.
    """
    words = phrase.split()
    if not words:
        return None
    parts: list[str] = []
    for word in words:
        chars: list[str] = []
        for ch in word:
            # Fold first: the character may already be the accented form
            # (e.g. ``á`` in ``Válvula``); the class table is keyed on the BASE.
            base = _fold_accents(ch).lower()
            cls = _ACCENT_CLASS.get(base)
            if cls is not None:
                chars.append(cls)
            else:
                chars.append(re.escape(ch))
        parts.append("".join(chars))
    return re.compile(r"\s+".join(parts), flags=re.IGNORECASE)


class Glossary:
    """Terminology only — loaded from `glossary/`, separate from `prompts/`."""

    def __init__(self, path: Path) -> None:
        self._path = path
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = [GlossaryEntry(**e) for e in raw.get("entries", [])]
        self._entries = sorted(entries, key=lambda e: len(e.source), reverse=True)

    @property
    def entries(self) -> tuple[GlossaryEntry, ...]:
        return tuple(self._entries)

    def protect_source(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace longest glossary sources with ``__GLS{i}__`` placeholders before MT.

        The match is **word-bounded** (``(?<!\\w)src(?!\\w)``), which matters once
        the glossary contains short verbatim entries like ``Titan`` → ``Titan``:
        without word boundaries, ``Titan`` would also be carved out of
        ``Titaniate`` or ``Titans``. All hand-curated sources start and end on
        an alphanumeric character, so this rule is backwards-compatible.
        """
        placeholders: dict[str, str] = {}
        out = text
        for i, e in enumerate(self._entries):
            token = f"__GLS{i}__"
            placeholders[token] = e.target
            pattern = re.compile(rf"(?<!\w){re.escape(e.source)}(?!\w)", flags=re.IGNORECASE)
            out = pattern.sub(f" {token} ", out)
        out = re.sub(r"[ \t]{2,}", " ", out).strip()
        return out, placeholders

    def enforce_placeholders(self, text: str, placeholders: dict[str, str]) -> str:
        """Restore glossary targets; tolerates MT eating spaces and one trailing
        underscore around ``__GLS{i}__`` tokens."""

        def _sub(m: re.Match[str]) -> str:
            idx = m.group(1)
            canonical = f"__GLS{idx}__"
            tgt = placeholders.get(canonical)
            if tgt is None:
                return m.group(0)
            return f" {tgt.strip()} "

        out = _PLACEHOLDER_RE.sub(_sub, text)
        out = re.sub(r"[ \t]{2,}", " ", out)
        return out.strip()

    def enforce_phrases_in_target(self, text: str) -> str:
        """If English glossary phrases leaked into Spanish, swap to preferred targets."""
        out = text
        for e in self._entries:
            pattern = re.compile(rf"\b{re.escape(e.source)}\b", flags=re.IGNORECASE)
            out = pattern.sub(e.target, out)
        return out

    def reassert_targets_after_edit(self, reference: str, edited: str) -> str:
        """Restore canonical glossary Spanish targets after neural post-edit.

        Two failure modes were observed in v1 of the pipeline and are both fixed here:

        1. **Accent / case drift.** Qwen rewrote ``Válvula neumática`` to
           ``Valvula Neumática`` (no acute on ``Vá-``, capital ``N``). The v1
           reassertion used a whitespace-loose but accent-sensitive pattern and
           silently failed. We now match accent- and case-insensitive across the
           classes in ``_ACCENT_CLASS`` and rewrite every drifted occurrence to
           the exact canonical form.

        2. **Total term drop.** Qwen completely paraphrased ``Remote Control
           Valve`` to ``válvula remota de escape``, removing every trace of the
           canonical target. The v1 reassertion had nothing to anchor on and
           the canonical form was lost forever. We now detect that case (target
           was present in ``reference`` but is missing from ``edited`` even
           under accent folding) and **fall back to the pre-edit reference**
           for the whole row, preferring glossary fidelity over Qwen fluency.
        """
        if reference == edited:
            return edited
        if not self._entries:
            return edited

        ref_folded = _fold_accents(reference).lower()
        out = edited
        dropped_canonicals: list[str] = []

        for e in sorted(self._entries, key=lambda x: len(x.target), reverse=True):
            if len(e.target) < 2:
                continue
            target_folded = _fold_accents(e.target).lower()
            if target_folded not in ref_folded:
                continue

            pattern = _accent_loose_pattern(e.target)
            if pattern is None:
                continue

            # Rewrite every drifted occurrence to the canonical accented form.
            pos = 0
            guard = 0
            while guard < 64:
                guard += 1
                m = pattern.search(out, pos)
                if not m:
                    break
                if m.group(0) == e.target:
                    pos = m.end()  # already canonical at this spot; check for more
                    continue
                out = out[: m.start()] + e.target + out[m.end() :]
                pos = m.start() + len(e.target)

            # Detect total drop: even with accent folding the term is gone now.
            out_folded = _fold_accents(out).lower()
            if target_folded not in out_folded:
                dropped_canonicals.append(e.target)

        if dropped_canonicals:
            # Qwen erased one or more canonical glossary terms. Trust the
            # MT-only pre-edit output over the silently-truncated Qwen edit.
            return reference
        return out

    def fix_spacing_around_targets(self, text: str) -> str:
        """
        Insert missing spaces when catalog Spanish targets are glued to adjacent words
        (e.g. ``El`` + ``Junta…`` → ``El Junta…``, ``Titan`` + ``debe`` → ``Titan debe``).
        Only applies to **multi-word** targets to limit false splits.
        """
        out = text
        targets = sorted(
            {e.target for e in self._entries if " " in e.target.strip() and len(e.target.strip()) >= 6},
            key=len,
            reverse=True,
        )
        for t in targets:
            et = re.escape(t)
            out = re.sub(rf"(?<=[\wáéíóúñÁÉÍÓÚÑ0-9])({et})", r" \1", out, flags=re.IGNORECASE)
            out = re.sub(rf"({et})(?=[\wáéíóúñÁÉÍÓÚÑ0-9])", r"\1 ", out, flags=re.IGNORECASE)
        return out
