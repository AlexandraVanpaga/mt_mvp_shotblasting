from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

from app.api.schemas import TranslateRequest
from app.core.config import Settings


class TranslateResponseCache:
    """
    Thread-safe LRU cache with TTL for translation responses.

    - ``maxsize`` (default 1024): inserting when full evicts the least-recently-used
      entry (LRU via ``OrderedDict`` + ``popitem(last=False)``).
    - ``ttl_seconds`` (default 86_400 = 24h): entries are removed once their expiry
      time passes; expiry is checked on every ``get`` / ``set``.
    """

    def __init__(self, maxsize: int = 1024, ttl_seconds: int = 86_400) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._od: OrderedDict[tuple[Any, ...], tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def _purge_expired(self) -> None:
        now = time.time()
        for k in list(self._od.keys()):
            expires_at, _ = self._od[k]
            if now > expires_at:
                del self._od[k]

    def get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        with self._lock:
            self._purge_expired()
            if key not in self._od:
                return None
            expires_at, payload = self._od.pop(key)
            if time.time() > expires_at:
                return None
            self._od[key] = (expires_at, payload)
            return dict(payload)

    def set(self, key: tuple[Any, ...], value: dict[str, Any]) -> None:
        with self._lock:
            self._purge_expired()
            self._od.pop(key, None)
            self._od[key] = (time.time() + self._ttl, dict(value))
            while len(self._od) > self._maxsize:
                self._od.popitem(last=False)

    def clear(self) -> int:
        """Remove all entries. Returns how many entries were dropped."""
        with self._lock:
            n = len(self._od)
            self._od.clear()
            return n

    def size(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._od)


translate_cache = TranslateResponseCache(maxsize=1024, ttl_seconds=86_400)


def _mt_cache_token(cfg: Settings) -> tuple:
    if cfg.mt_engine == "marian_hf":
        return ("marian_hf", cfg.mt_model_name)
    bin_path = cfg.ct2_model_dir / "model.bin"
    try:
        ct2_mtime_ns = bin_path.stat().st_mtime_ns
    except OSError:
        ct2_mtime_ns = 0
    return (
        "ctranslate2",
        str(cfg.ct2_model_dir.resolve()),
        cfg.ct2_compute_type,
        ct2_mtime_ns,
    )


def build_translate_cache_key(body: TranslateRequest, cfg: Settings) -> tuple:
    """Hashable key covering inputs and config that affect the translation output."""
    try:
        glossary_mtime_ns = cfg.glossary_path.stat().st_mtime_ns
    except OSError:
        glossary_mtime_ns = 0
    try:
        prompt_mtime_ns = cfg.postedit_prompt_path.stat().st_mtime_ns
    except OSError:
        prompt_mtime_ns = 0
    return (
        body.text,
        bool(body.apply_glossary),
        bool(body.apply_postedit),
        _mt_cache_token(cfg),
        cfg.mt_model_name,
        bool(cfg.postedit_use_qwen),
        bool(cfg.postedit_force_cpu),
        cfg.postedit_qwen_model,
        int(cfg.postedit_max_new_tokens),
        int(cfg.postedit_qwen_max_input_tokens),
        str(cfg.glossary_path.resolve()),
        glossary_mtime_ns,
        str(cfg.postedit_prompt_path.resolve()),
        prompt_mtime_ns,
    )
