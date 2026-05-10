"""Core EN→ES translation pass (glossary + MT + post-edit), shared by the HTTP router."""

from __future__ import annotations

import torch

from app.api.schemas import TranslateRequest, TranslateResponse
from app.core.config import Settings
from app.services.ct2_engine import Ctranslate2Engine
from app.services.glossary import Glossary
from app.services.mt_engine import MarianEngine
from app.services.postedit import PostEditor


def run_translate(
    body: TranslateRequest,
    cfg: Settings,
    glossary: Glossary,
    engine: Ctranslate2Engine | MarianEngine,
    posteditor: PostEditor,
) -> TranslateResponse:
    """One full translation (no HTTP cache logic)."""
    protected = body.text
    placeholders: dict[str, str] = {}
    if body.apply_glossary:
        protected, placeholders = glossary.protect_source(body.text)

    raw_mt = engine.translate(protected)
    after_glossary = (
        glossary.enforce_placeholders(raw_mt, placeholders) if body.apply_glossary else raw_mt
    )

    final = after_glossary
    postedit_flag = False
    if body.apply_postedit:
        final = posteditor.edit(source_en=body.text, target_es=after_glossary, glossary=glossary)
        postedit_flag = True

    debug = None
    if body.include_debug:
        debug = _build_debug_payload(
            body=body,
            cfg=cfg,
            posteditor=posteditor,
            protected=protected,
            raw_mt=raw_mt,
        )

    return TranslateResponse(
        translation=final,
        glossary_applied=body.apply_glossary,
        postedit_applied=postedit_flag,
        from_cache=False,
        debug=debug,
    )


def _build_debug_payload(
    *,
    body: TranslateRequest,
    cfg: Settings,
    posteditor: PostEditor,
    protected: str,
    raw_mt: str,
) -> dict:
    resolved_mt_device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stages: list[str] = [
        "cache_bypassed_for_debug",
        "glossary_protect" if body.apply_glossary else "glossary_skipped",
        f"mt_{cfg.mt_engine}",
        "glossary_restore_placeholders" if body.apply_glossary else "glossary_restore_skipped",
        "postedit" if body.apply_postedit else "postedit_skipped",
    ]
    return {
        "pipeline_order_on_miss": [
            "cache_check",
            "glossary_protect_source",
            "machine_translate",
            "glossary_restore_placeholders",
            "postedit",
        ],
        "stages_executed_this_request": stages,
        "mt_engine": cfg.mt_engine,
        "mt_model": cfg.mt_model_name,
        "mt_resolved_device": resolved_mt_device,
        "cuda_available": torch.cuda.is_available(),
        "ct2_model_dir": str(cfg.ct2_model_dir) if cfg.mt_engine == "ctranslate2" else None,
        "ct2_compute_type": cfg.ct2_compute_type if cfg.mt_engine == "ctranslate2" else None,
        "glossary_path": str(cfg.glossary_path),
        "postedit_prompt_path": str(cfg.postedit_prompt_path),
        "postedit_prompt_chars": len(posteditor.instructions),
        "postedit_use_qwen": cfg.postedit_use_qwen,
        "postedit_qwen_model": cfg.postedit_qwen_model if cfg.postedit_use_qwen else None,
        "postedit_max_new_tokens": cfg.postedit_max_new_tokens if cfg.postedit_use_qwen else None,
        "postedit_qwen_max_input_tokens": cfg.postedit_qwen_max_input_tokens
        if cfg.postedit_use_qwen
        else None,
        "protected_source": protected if body.apply_glossary else None,
        "after_mt_before_placeholder_restore": raw_mt,
        "cache_hit": False,
    }
