from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MT_MVP_", env_file=".env", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[2]
    glossary_path: Path = project_root / "glossary" / "en_es_shotblasting.json"
    postedit_prompt_path: Path = project_root / "prompts" / "postedit_en_es.md"

    # MT: `ctranslate2` (default) uses converted weights under `ct2_model_dir`;
    # `marian_hf` uses PyTorch Marian in-process.
    mt_engine: Literal["ctranslate2", "marian_hf"] = "ctranslate2"
    mt_model_name: str = "Helsinki-NLP/opus-mt-en-es"  # tokenizer id (and HF weights when marian_hf)
    ct2_model_dir: Path = project_root / "models" / "opus-mt-en-es-ct2"
    ct2_compute_type: str = "default"  # e.g. int8, float16, int8_float32 (see CTranslate2 docs)

    device: str | None = None  # "cuda", "cpu", or None for auto

    # Neural post-editing (Qwen2.5 Instruct). Disable with MT_MVP_POSTEDIT_USE_QWEN=false if VRAM is tight.
    postedit_use_qwen: bool = True
    # Lighter / faster: Qwen/Qwen2.5-3B-Instruct; stronger: Qwen/Qwen2.5-7B-Instruct
    postedit_qwen_model: str = "Qwen/Qwen2.5-3B-Instruct"
    postedit_max_new_tokens: int = 256
    postedit_qwen_max_input_tokens: int = 2048


settings = Settings()
