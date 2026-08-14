"""
Runtime configuration: reads secrets/API keys from the environment.

Local runs pick these up from a `.env` file (via python-dotenv, if
installed); GitHub Actions runs inject them directly as env vars.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from config.constants import (
    DEFAULT_AI_PROVIDER,
    DEFAULT_MODEL_NAME,
    DEFAULT_GROQ_BASE_URL,
    REQUIRED_ENV_VARS,
)
from utils.logger import get_logger

# --------------------------------------------------------------------------
# Optional local .env support. On GitHub Actions, env vars are injected
# directly via `os.environ`, so a missing dotenv package/file is not fatal.
# --------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

log = get_logger()


@dataclass(frozen=True)
class Settings:
    """Validated runtime secrets/config required for a single run."""
    telegram_bot_token: str
    telegram_chat_id: str
    tavily_api_key: str
    groq_api_key: str
    ai_provider: str
    model_name: str
    groq_base_url: str


def load_settings() -> Settings:
    """Read and validate all required environment variables."""
    raw = {var: os.environ.get(var, "").strip() for var in REQUIRED_ENV_VARS}
    missing = [var for var, val in raw.items() if not val]
    if missing:
        log.critical("Missing required environment variable(s): %s", ", ".join(missing))
        sys.exit(1)

    # Optional, non-secret overrides — fall back to the defaults in
    # config/constants.py when unset or blank, so no new env vars are
    # required for the agent to keep working out of the box.
    ai_provider = os.environ.get("AI_PROVIDER", "").strip() or DEFAULT_AI_PROVIDER
    model_name = os.environ.get("MODEL_NAME", "").strip() or DEFAULT_MODEL_NAME
    groq_base_url = os.environ.get("GROQ_BASE_URL", "").strip() or DEFAULT_GROQ_BASE_URL

    return Settings(
        telegram_bot_token=raw["TELEGRAM_BOT_TOKEN"],
        telegram_chat_id=raw["TELEGRAM_CHAT_ID"],
        tavily_api_key=raw["TAVILY_API_KEY"],
        groq_api_key=raw["GROQ_API_KEY"],
        ai_provider=ai_provider,
        model_name=model_name,
        groq_base_url=groq_base_url,
    )
