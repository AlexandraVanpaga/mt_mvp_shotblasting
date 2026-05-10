from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    target: str
    notes: str | None


_PLACEHOLDER_RE = re.compile(r"(__GLS\d+__)")


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
        """Replace longest glossary sources with ``__GLS{i}__`` placeholders before MT."""
        placeholders: dict[str, str] = {}
        out = text
        for i, e in enumerate(self._entries):
            token = f"__GLS{i}__"
            placeholders[token] = e.target
            pattern = re.compile(re.escape(e.source), flags=re.IGNORECASE)
            out, n = pattern.subn(f" {token} ", out)
            if n:
                pass
        out = re.sub(r"[ \t]{2,}", " ", out).strip()
        return out, placeholders

    def enforce_placeholders(self, text: str, placeholders: dict[str, str]) -> str:
        """Restore glossary targets; tolerates MT eating spaces around ``__GLS*__``."""

        def _sub(m: re.Match[str]) -> str:
            tok = m.group(1)
            tgt = placeholders.get(tok)
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
        """
        After neural post-edit, restore canonical glossary **Spanish** targets that
        appeared in ``reference`` but were altered or split by the LLM.
        """
        if reference == edited:
            return edited
        out = edited
        for e in sorted(self._entries, key=lambda x: len(x.target), reverse=True):
            if len(e.target) < 2:
                continue
            if e.target.lower() not in reference.lower():
                continue
            loose = _loose_phrase_pattern(e.target)
            if loose is None:
                continue
            guard = 0
            while guard < 64:
                guard += 1
                m = loose.search(out)
                if not m:
                    break
                if m.group(0) == e.target:
                    break
                out = out[: m.start()] + e.target + out[m.end() :]
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


def _loose_phrase_pattern(target: str) -> re.Pattern[str] | None:
    words = [re.escape(w) for w in target.split() if w]
    if not words:
        return None
    return re.compile(r"\s+".join(words), flags=re.IGNORECASE)
