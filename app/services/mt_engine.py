from __future__ import annotations

import torch
from transformers import MarianMTModel, MarianTokenizer


class MarianEngine:  # HuggingFace path (no CTranslate2)
    """Lightweight EN→ES Marian baseline; swap `mt_model_name` for a domain-adapted checkpoint later."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name).to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        batch = self._tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        gen = self._model.generate(**batch, max_length=512)
        return self._tokenizer.decode(gen[0], skip_special_tokens=True).strip()
