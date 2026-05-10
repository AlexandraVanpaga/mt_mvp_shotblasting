from __future__ import annotations

import torch
from fastapi import APIRouter, Depends

from app.api.deps import get_glossary, get_mt_engine, get_posteditor, get_settings
from app.api.schemas import TranslateRequest, TranslateResponse
from app.core.config import Settings
from app.services.ct2_engine import Ctranslate2Engine
from app.services.glossary import Glossary
from app.services.mt_engine import MarianEngine
from app.services.postedit import PostEditor
from app.services.translate_cache import build_translate_cache_key, translate_cache
from app.services.translation import run_translate


router = APIRouter(tags=["translate"])


@router.post("/translate", response_model=TranslateResponse)
def translate(
    body: TranslateRequest,
    cfg: Settings = Depends(get_settings),
    glossary: Glossary = Depends(get_glossary),
    engine: Ctranslate2Engine | MarianEngine = Depends(get_mt_engine),
    posteditor: PostEditor = Depends(get_posteditor),
) -> TranslateResponse:
    if not body.include_debug:
        cache_key = build_translate_cache_key(body, cfg)
        hit = translate_cache.get(cache_key)
        if hit is not None:
            return TranslateResponse.model_validate({**hit, "from_cache": True})

    out = run_translate(body, cfg, glossary, engine, posteditor)

    if not body.include_debug:
        translate_cache.set(build_translate_cache_key(body, cfg), out.model_dump(mode="python"))

    return out


@router.get("/health")
def health(cfg: Settings = Depends(get_settings)) -> dict:
    """Lightweight readiness + GPU hints (does not load MT or Qwen weights)."""
    cuda = torch.cuda.is_available()
    resolved = cfg.device or ("cuda" if cuda else "cpu")
    out: dict = {
        "status": "ok",
        "cuda_available": cuda,
        "mt_engine_configured": cfg.mt_engine,
        "mt_resolved_device": resolved,
        "postedit_qwen_configured": cfg.postedit_use_qwen,
        "translate_pipeline_on_cache_miss": [
            "cache_check",
            "glossary_protect_source",
            "machine_translate",
            "glossary_restore_placeholders",
            "postedit",
        ],
    }
    if cuda:
        out["cuda_device_name"] = torch.cuda.get_device_name(0)
    out["translate_cache_entries"] = translate_cache.size()
    return out


@router.post("/translate/cache/clear")
def clear_translate_cache() -> dict[str, int | str]:
    """Drop all in-memory translation cache entries (Swagger-friendly). Restarting Uvicorn also clears the cache."""
    removed = translate_cache.clear()
    return {"status": "ok", "entries_removed": removed}
