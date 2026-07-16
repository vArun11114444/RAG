"""
Centralised configuration — loaded once, injected everywhere.
All secrets come from environment variables; sensible defaults for dev.

MIGRATION CHANGES:
  - Removed: CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION
  - Added:   QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
  - Added:   SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET
  - Added:   OPENAI_BASE_URL (enables OpenRouter)
  - Updated: NEO4J_URI supports neo4j+s:// (Aura)
"""
from __future__ import annotations
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "Enterprise RAG"
    ENV: Literal["dev", "staging", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"

    # ── LLM (OpenRouter or OpenAI) ────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://openrouter.ai/api/v1"   # OpenRouter default
    OPENAI_MODEL: str = "openai/gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.0

    # ── Qdrant Cloud (replaces ChromaDB) ─────────────────────────────────────
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "documents"

    # ── BM25 ─────────────────────────────────────────────────────────────────
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    BM25_TOP_K: int = 20

    # ── Hybrid retrieval ──────────────────────────────────────────────────────
    VECTOR_TOP_K: int = 20
    HYBRID_TOP_K: int = 10
    RRF_K: int = 60
    QUERY_EXPANSION_MAX: int = 3

    # ── Neo4j Aura ────────────────────────────────────────────────────────────
    NEO4J_URI: str = "neo4j+s://your-aura-instance.databases.neo4j.io"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""
    NEO4J_DATABASE: str = "neo4j"
    GRAPH_HOP_LIMIT: int = 2

    # ── Supabase Storage (replaces local filesystem) ──────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_BUCKET: str = "rag-documents"

    # ── Verification ──────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.6
    HALLUCINATION_NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
    CONTRADICTION_THRESHOLD: float = 0.75

    # ── LangSmith ─────────────────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "enterprise-rag"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # ── Prometheus ────────────────────────────────────────────────────────────
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090

    # ── Security ──────────────────────────────────────────────────────────────
    SECURITY_ENABLED: bool = True
    SECURITY_INJECTION_THRESHOLD: float = 0.7
    SECURITY_BLOCK_ON_PII: bool = False
    SECURITY_PII_SCORE_THRESHOLD: float = 0.4
    SECURITY_MAX_FILE_SIZE_MB: int = 50
    SECURITY_AUDIT_LOG_ENABLED: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
