"""
ncite.config

Centralized configuration loaded from environment variables.
Validates required keys at import time — fail fast, not at first API call.

Place a `.env` file in the project root (see .env.example).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Required ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Optional ──────────────────────────────────────────────────────────
NCBI_API_KEY: str | None = os.environ.get("NCBI_API_KEY") or None
OPENALEX_EMAIL: str = os.environ.get("OPENALEX_EMAIL", "research@ncite.org")
CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
PUBMED_MAX_PER_QUERY: int = int(os.environ.get("PUBMED_MAX_PER_QUERY", "2000"))
PUBMED_MAX_RESULTS: int = PUBMED_MAX_PER_QUERY  # backwards compat alias

# ── Derived ───────────────────────────────────────────────────────────
PUBMED_RATE_LIMIT: int = 10 if NCBI_API_KEY else 3


def require_api_key() -> str:
    """Return the Anthropic key or raise with a helpful message."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example → .env and fill in your key."
        )
    return ANTHROPIC_API_KEY
