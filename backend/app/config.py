import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Set

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# Load environment variables from a local .env file if present
load_dotenv()


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_env_choice(name: str, default: str, allowed: Set[str]) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    candidate = value.strip().lower()
    return candidate if candidate in allowed else default


class Settings(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    chat_model: str = Field(default=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"))
    embedding_model: str = Field(default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    cohere_api_key: str = Field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))

    top_k: int = Field(default_factory=lambda: max(_get_env_int("RAG_TOP_K", 4), 1))
    max_tokens: int = Field(default_factory=lambda: max(_get_env_int("OPENAI_MAX_TOKENS", 512), 1))
    temperature: float = Field(default_factory=lambda: _get_env_float("OPENAI_TEMPERATURE", 0.3))

    enable_memory: bool = Field(default_factory=lambda: _get_env_bool("ENABLE_MEMORY", True))
    memory_backend: str = Field(
        default_factory=lambda: _get_env_choice("MEMORY_BACKEND", "sqlite", {"sqlite", "postgres"}),
    )
    memory_dsn: str = Field(default_factory=lambda: os.getenv("MEMORY_DSN", "sqlite:///backend/memory.db"))
    history_window: int = Field(default_factory=lambda: max(_get_env_int("HISTORY_WINDOW", 6), 0))
    summary_every_n_turns: int = Field(default_factory=lambda: max(_get_env_int("SUMMARY_EVERY_N_TURNS", 6), 1))
    summary_max_tokens: int = Field(default_factory=lambda: max(_get_env_int("SUMMARY_MAX_TOKENS", 800), 100))

    enable_rerank: bool = Field(default_factory=lambda: _get_env_bool("ENABLE_RERANK", True))
    rerank_provider: str = Field(
        default_factory=lambda: _get_env_choice("RERANK_PROVIDER", "cohere", {"cohere", "local", "none"}),
    )
    rerank_top_n: int = Field(default_factory=lambda: _get_env_int("RERANK_TOP_N", 0))

    enable_hybrid: bool = Field(default_factory=lambda: _get_env_bool("ENABLE_HYBRID", True))
    hybrid_alpha: float = Field(default_factory=lambda: _get_env_float("HYBRID_ALPHA", 0.5))

    vector_backend: str = Field(
        default_factory=lambda: _get_env_choice("VECTOR_BACKEND", "faiss", {"faiss", "pgvector"}),
    )
    pg_dsn: str = Field(default_factory=lambda: os.getenv("POSTGRES_DSN", ""))

    chunker: str = Field(
        default_factory=lambda: _get_env_choice("CHUNKER", "token", {"token", "char"}),
    )
    chunk_size_tokens: int = Field(default_factory=lambda: max(_get_env_int("CHUNK_SIZE_TOKENS", 600), 100))
    chunk_overlap_tokens: int = Field(default_factory=lambda: max(_get_env_int("CHUNK_OVERLAP_TOKENS", 100), 0))
    max_input_tokens: int = Field(default_factory=lambda: max(_get_env_int("MAX_INPUT_TOKENS", 6000), 1000))

    data_dir: Path = Field(default=Path(__file__).resolve().parent.parent / "data")
    vector_store_dir: Path = Field(default=Path(__file__).resolve().parent.parent / "vector_store")

    loaders: Dict[str, bool] = Field(
        default_factory=lambda: {
            "enable_pdf": _get_env_bool("ENABLE_PDF_LOADER", True),
            "enable_docx": _get_env_bool("ENABLE_DOCX_LOADER", True),
            "enable_web": _get_env_bool("ENABLE_WEB_LOADER", True),
        },
    )

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def _validate_ranges(self) -> "Settings":
        # Temperature in [0, 2]
        try:
            self.temperature = max(0.0, min(float(self.temperature), 2.0))
        except Exception:
            self.temperature = 0.3

        # History window must be non-negative and not excessive
        try:
            window = int(self.history_window)
        except Exception:
            window = 6
        self.history_window = max(window, 0)

        # Summary cadence
        try:
            cadence = int(self.summary_every_n_turns)
        except Exception:
            cadence = 6
        self.summary_every_n_turns = max(cadence, 1)

        # Summary token cap
        try:
            summary_tokens = int(self.summary_max_tokens)
        except Exception:
            summary_tokens = 800
        self.summary_max_tokens = max(summary_tokens, 100)

        # Hybrid alpha in [0, 1]
        try:
            self.hybrid_alpha = max(0.0, min(float(self.hybrid_alpha), 1.0))
        except Exception:
            self.hybrid_alpha = 0.5

        # Rerank top-n defaults to min(top_k, 10) when unset/invalid
        default_top_n = min(int(self.top_k) if isinstance(self.top_k, (int, float)) else 4, 10)
        try:
            rtop = int(self.rerank_top_n)
        except Exception:
            rtop = 0
        self.rerank_top_n = min(rtop, default_top_n) if rtop > 0 else default_top_n

        # Chunk sizes and overlaps (by tokens)
        try:
            csize = int(self.chunk_size_tokens)
        except Exception:
            csize = 600
        try:
            cover = int(self.chunk_overlap_tokens)
        except Exception:
            cover = 100
        if cover >= csize:
            cover = max(0, csize // 4)
        self.chunk_size_tokens = max(csize, 100)
        self.chunk_overlap_tokens = max(cover, 0)

        # Max tokens caps
        try:
            self.max_tokens = int(self.max_tokens) if int(self.max_tokens) > 0 else 512
        except Exception:
            self.max_tokens = 512
        try:
            self.max_input_tokens = (
                int(self.max_input_tokens) if int(self.max_input_tokens) > 0 else 6000
            )
        except Exception:
            self.max_input_tokens = 6000

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
