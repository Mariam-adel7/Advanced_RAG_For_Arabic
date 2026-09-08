from __future__ import annotations
import os
from dataclasses import dataclass, field

def _env(name: str, default, cast=str):
    value = os.environ.get(name)
    return cast(value) if value is not None else default

@dataclass
class Config:
    # -- chunking --------------------------------------------------------- #
    chunking_strategy: str = _env("CHUNKING_STRATEGY", "parent_child")  # fixed_size | recursive_character | structural
    chunk_size: int = _env("CHUNK_SIZE", 1000, int)
    chunk_overlap: int = _env("CHUNK_OVERLAP", 150, int)
    parent_size: int = _env("PARENT_SIZE", 2400, int)
    child_size: int = _env("CHILD_SIZE", 700, int)
    child_overlap: int = _env("CHILD_OVERLAP", 100, int)

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_candidates: int = 15

    # -- embedding / vector store ------------------------------------------ #
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-m3")
    vector_store_backend: str = _env("VECTOR_STORE_BACKEND", "auto")   # auto | numpy | faiss

    # -- retrieval --------------------------------------------------------- #
    retrieval_strategy: str = _env("RETRIEVAL_STRATEGY", "hybrid")     # semantic | keyword | hybrid
    top_k: int = _env("TOP_K", 5, int)
    hybrid_fusion_method: str = _env("HYBRID_FUSION_METHOD", "rrf")    # rrf | weighted
    hybrid_alpha: float = _env("HYBRID_ALPHA", 0.5, float)

    # -- augmentation -------------------------------------------------------#
    max_context_chars: int = _env("MAX_CONTEXT_CHARS", 12000, int)

    # -- generation ---------------------------------------------------------#
    llm_backend: str = _env("LLM_BACKEND", "gemini")                     # gemini | mock
    llm_model: str = _env("RAG_LLM_MODEL", "gemini-3.5-flash-lite")
    gemini_api_key: str | None = os.environ.get("GEMINI_API_KEY")
    max_answer_tokens: int = _env("MAX_ANSWER_TOKENS", 1024, int)

    # -- PDF / OCR ----------------------------------------------------------
    ocr_enabled: bool = _env("OCR_ENABLED", "true").lower() in {"1","true","yes"}

    # -- advanced RAG -------------------------------------------------------
    rerank_bonuses: dict = field(default_factory=lambda: {
    "table_bonus": _env("RERANK_TABLE_BONUS", 0.04, float),
    "formula_bonus": _env("RERANK_FORMULA_BONUS", 0.10, float),
    "keyword_bonus_step": _env("RERANK_KEYWORD_STEP", 0.02, float),
    "keyword_bonus_max": _env("RERANK_KEYWORD_MAX", 0.08, float),
})

    confidence_weights: dict = field(default_factory=lambda: {
    "top_score_weight": _env("CONF_TOP_WEIGHT", 0.55, float),
    "avg_top_weight": _env("CONF_AVG_WEIGHT", 0.20, float),
    "margin_weight": _env("CONF_MARGIN_WEIGHT", 0.10, float),
    "agreement_weight": _env("CONF_AGREE_WEIGHT", 0.15, float),
})
    allowed_metadata_fields: list = field(
    default_factory=lambda: _env(
        "ALLOWED_METADATA_FIELDS",
        "document,procedure,section,version,issue_date,issue_year,review_date,review_year,department,year_gt",
        str
    ).split(",")
)
    enable_llm_judge: bool = _env("ENABLE_LLM_JUDGE", "true").lower() in {"1","true","yes"}
    evaluation_output: str = _env("EVALUATION_OUTPUT", "./evaluation_results.json")

    # -- storage -------------------------------------------------------------#
    store_path: str = _env("STORE_PATH", "./rag_store.pkl")


DEFAULT_CONFIG = Config()
