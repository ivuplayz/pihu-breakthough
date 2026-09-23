"""Configuration manager for Pihu-BreakThough.

Safely loads environment variables, providing sensible defaults without
crashing if external API keys or database URLs are absent.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

# Safely attempt to load .env without crashing if python-dotenv is missing
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _safe_int(val: str | None, default: int) -> int:
    """Safely parse integer environment variable, defaulting if unset or empty."""
    if not val or not val.strip():
        return default
    try:
        return int(val.strip())
    except (ValueError, TypeError):
        return default


class Config:
    """Application configuration for Pihu-BreakThough."""

    APP_NAME: str = "Pihu-BreakThough"
    VERSION: str = "1.0.0"
    STAGE: str = "Stage 9"

    # Server settings
    ENV: str = os.getenv("FLASK_ENV") or "production"
    DEBUG: bool = (os.getenv("FLASK_DEBUG") or "0").lower() in ("1", "true", "yes")
    PORT: int = _safe_int(os.getenv("PORT"), 5000)
    SECRET_KEY: str = (
        os.getenv("SECRET_KEY") or "pihu-breakthough-dev-insecure-secret-key"
    )

    # Maximum payload size (4 MB default to accommodate multimodal payloads within Vercel edge limits)
    MAX_CONTENT_LENGTH: int = _safe_int(
        os.getenv("MAX_CONTENT_LENGTH"), 4 * 1024 * 1024
    )


    # Database settings (Neon PostgreSQL) - Optional in Stage 1
    DATABASE_URL: str | None = os.getenv("DATABASE_URL")

    # LLM Provider settings (Future stages) - Optional in Stage 1
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    HF_API_KEY: str | None = os.getenv("HF_API_KEY")
    HF_MODEL: str = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @classmethod
    def is_database_configured(cls) -> bool:
        """Check if a database URL has been supplied."""
        return bool(cls.DATABASE_URL and cls.DATABASE_URL.strip())

    @classmethod
    def get_configured_providers(cls) -> List[str]:
        """List provider identifiers that currently have credentials/endpoints configured."""
        configured: List[str] = ["stage1-deterministic"]
        if cls.GEMINI_API_KEY and cls.GEMINI_API_KEY.strip():
            configured.append("gemini")
        if cls.GROQ_API_KEY and cls.GROQ_API_KEY.strip():
            configured.append("groq")
        if cls.HF_API_KEY and cls.HF_API_KEY.strip():
            configured.append("huggingface")
        if cls.OLLAMA_BASE_URL and cls.OLLAMA_BASE_URL.strip():
            configured.append("ollama")
        return configured

    @classmethod
    def to_safe_dict(cls) -> Dict[str, Any]:
        """Return a dictionary of configuration with sensitive values masked."""
        return {
            "app_name": cls.APP_NAME,
            "version": cls.VERSION,
            "stage": cls.STAGE,
            "env": cls.ENV,
            "debug": cls.DEBUG,
            "port": cls.PORT,
            "database_configured": cls.is_database_configured(),
            "configured_providers": cls.get_configured_providers(),
            "max_content_length_bytes": cls.MAX_CONTENT_LENGTH,
        }
