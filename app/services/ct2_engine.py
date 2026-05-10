from __future__ import annotations

from pathlib import Path

import ctranslate2
import torch
from transformers import MarianTokenizer


class Ctranslate2Engine:
    """
    Marian-style EN→ES via CTranslate2 (converted checkpoint on disk).

    Tokenization matches HuggingFace Marian using ``tokenizer_model_name``; the
    transformer weights are loaded from ``model_dir`` (``model.bin``).
    """

    def __init__(
        self,
        model_dir: Path | str,
        tokenizer_model_name: str,
        device: str | None = None,
        compute_type: str = "default",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.tokenizer_model_name = tokenizer_model_name
        self.compute_type = compute_type

        if not (self.model_dir / "model.bin").is_file():
            raise FileNotFoundError(
                f"CTranslate2 weights not found at {self.model_dir / 'model.bin'}. "
                "Convert a Marian checkpoint, e.g.: "
                "`python scripts/convert_marian_to_ct2.py --output-dir models/opus-mt-en-es-ct2`"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = device
        self._translator = ctranslate2.Translator(
            str(self.model_dir),
            device=device,
            compute_type=compute_type,
        )
        self._tokenizer = MarianTokenizer.from_pretrained(tokenizer_model_name)

    def translate(self, text: str) -> str:
        ids = self._tokenizer.encode(
            text,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        source_tokens = self._tokenizer.convert_ids_to_tokens(ids)
        results = self._translator.translate_batch([source_tokens])
        hyp = results[0].hypotheses[0]
        out_ids = self._tokenizer.convert_tokens_to_ids(hyp)
        return self._tokenizer.decode(out_ids, skip_special_tokens=True).strip()
