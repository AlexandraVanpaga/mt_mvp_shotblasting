from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MT_MVP_", env_file=".env", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[2]
    glossary_path: Path = project_root / "glossary" / "en_es_shotblasting.json"
    postedit_prompt_path: Path = project_root / "prompts" / "postedit_en_es.md"

    # MT backend selection.
    #   - ``ctranslate2``: Marian opus-mt-en-es converted to CT2 (default — winner
    #                      of the v4 engine sweep, see ``results_final/report.md``);
    #   - ``marian_hf``:   same Marian checkpoint via PyTorch / transformers;
    #   - ``nllb``:        Meta NLLB-200 (distilled-600M / distilled-1.3B / 3.3B).
    # Override via env: ``MT_MVP_MT_ENGINE=nllb`` and
    # ``MT_MVP_MT_MODEL_NAME=facebook/nllb-200-distilled-1.3B``.
    mt_engine: Literal["ctranslate2", "marian_hf", "nllb"] = "ctranslate2"
    mt_model_name: str = "Helsinki-NLP/opus-mt-en-es"
    ct2_model_dir: Path = project_root / "models" / "opus-mt-en-es-ct2"
    ct2_compute_type: str = "default"  # e.g. int8, float16, int8_float32 (see CTranslate2 docs)

    # NLLB-specific (only consulted when mt_engine == "nllb").
    nllb_src_lang: str = "eng_Latn"
    nllb_tgt_lang: str = "spa_Latn"
    nllb_num_beams: int = 4
    # ``auto`` picks bf16 on CUDA and fp32 on CPU; override to ``fp16`` / ``bf16`` / ``fp32``.
    nllb_dtype: str = "auto"

    device: str | None = None  # "cuda", "cpu", or None for auto

    # Qwen loads multi-GB weights on first /translate and often OOMs single mid-size GPUs
    # when MT already uses VRAM. Default off for local dev; enable with
    # ``MT_MVP_POSTEDIT_USE_QWEN=true`` (see ``docker-compose`` ``api-gpu``).
    postedit_use_qwen: bool = True
    # Keep MT on GPU while running Qwen on CPU (avoids OOM on single mid-size cards, e.g. 12 GB + 3B Qwen).
    postedit_force_cpu: bool = False
    # Lighter / faster: Qwen/Qwen2.5-3B-Instruct; stronger: Qwen/Qwen2.5-7B-Instruct
    postedit_qwen_model: str = "Qwen/Qwen2.5-3B-Instruct"
    postedit_max_new_tokens: int = 256
    postedit_qwen_max_input_tokens: int = 2048

    # Mount Gradio at ``/gradio`` inside the FastAPI app (``MT_MVP_ENABLE_GRADIO=1`` in ``.env`` or shell).
    enable_gradio: bool = False


settings = Settings()
