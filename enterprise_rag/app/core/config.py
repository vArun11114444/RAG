"""
app/core/config.py — services-layer settings.
MIGRATION: ChromaDB removed, Qdrant + Supabase + OpenRouter added.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM — OpenRouter
    openai_api_key: str = ""
    openai_base_url: str = "https://openrouter.ai/api/v1"
    anthropic_api_key: str = ""
    llm_provider: str = "openrouter"
    llm_model: str = "openai/gpt-4o-mini"

    # Qdrant Cloud
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "documents"

    # Neo4j Aura
    neo4j_uri: str = "neo4j+s://your-aura-instance.databases.neo4j.io"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_bucket: str = "rag-documents"

    # Observability
    langsmith_api_key: str = ""
    langsmith_project: str = "enterprise-rag"
    langchain_tracing_v2: str = "false"
    log_level: str = "INFO"
    prometheus_port: int = 9090

    # Retrieval tuning
    bm25_top_k: int = 20
    dense_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 10
    verification_threshold: float = 0.65

    # Security
    security_enabled: bool = True
    security_injection_threshold: float = 0.7
    security_block_on_pii: bool = False
    security_pii_score_threshold: float = 0.4
    security_max_file_size_mb: int = 50
    security_audit_log_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
