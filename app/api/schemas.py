from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    apply_glossary: bool = True
    apply_postedit: bool = True
    include_debug: bool = False


class TranslateResponse(BaseModel):
    translation: str
    glossary_applied: bool
    postedit_applied: bool
    from_cache: bool = False
    debug: dict | None = None
