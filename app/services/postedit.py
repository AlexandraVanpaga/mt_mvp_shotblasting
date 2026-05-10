from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from app.services.glossary import Glossary
from app.services.qwen_postedit import QwenPostEditService


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).replace(" \n", "\n").strip()


@lru_cache(maxsize=8)
def _load_postedit_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


class PostEditor:
    """
    Post-editing: optional Qwen instruct pass, then glossary lock + cleanup.

    Qwen runs first; canonical glossary targets from the pre-Qwen Spanish are
    re-applied so the LLM cannot drift terminology or glue words.
    """

    def __init__(
        self,
        prompt_path: Path,
        *,
        qwen: QwenPostEditService | None = None,
    ) -> None:
        self.prompt_path = prompt_path
        self.instructions = _load_postedit_prompt(str(prompt_path.resolve()))
        self._qwen = qwen

    def edit(self, source_en: str, target_es: str, glossary: Glossary) -> str:
        reference = target_es
        text = target_es
        if self._qwen is not None:
            text = self._qwen.refine(source_en, text, self.instructions)
        text = glossary.reassert_targets_after_edit(reference, text)
        text = glossary.enforce_phrases_in_target(text)
        text = glossary.fix_spacing_around_targets(text)
        return _normalize_whitespace(text)
