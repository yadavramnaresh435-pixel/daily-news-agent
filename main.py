#!/usr/bin/env python3
"""
Hindu Research Daily Intelligence & Content Agent
===================================================

Fetches fresh, real-world news across four research niches using Tavily,
summarizes each result into a factual bullet + content-writing hook using
an AI model via the OpenRouter REST API, and delivers a formatted daily
digest to a Telegram chat.

Designed to run as a scheduled GitHub Actions job, but also runs locally
via a `.env` file (python-dotenv).

This module is intentionally a thin orchestrator: it wires the
config/services/utils modules together in the correct order and contains
no business logic of its own. See:
  - config/     -> static + runtime configuration
  - services/   -> Tavily search, OpenRouter summarization, Telegram delivery
  - utils/      -> logging and small generic helpers

Author: AI Systems Engineering
"""

from __future__ import annotations

from tavily import TavilyClient

from config.constants import CATEGORIES
from config.settings import load_settings
from services.ai_service import (
    CategoryDigest,
    OpenRouterClient,
    format_full_message,
    summarize_with_gemini,
)
from services.tavily_service import run_tavily_search
from services.telegram_service import deliver_digest
from utils.logger import get_logger

log = get_logger()


def generate_digest(tavily_client: TavilyClient, ai_client: OpenRouterClient) -> list[CategoryDigest]:
    """Run search + summarization for every category, isolating failures."""
    digests: list[CategoryDigest] = []

    for category in CATEGORIES:
        digest = CategoryDigest(category=category)
        try:
            digest.results = run_tavily_search(tavily_client, category)
            digest.summary_html = summarize_with_gemini(ai_client, category, digest.results)
        except Exception as exc:  # noqa: BLE001 - absolute last line of defense per category
            log.error("Unexpected failure processing '%s': %s", category.name, exc)
            digest.error = str(exc)
            digest.summary_html = "⚠️ This category could not be processed today due to an internal error."
        digests.append(digest)

    return digests


def main() -> None:
    log.info("Starting Hindu Research Daily Intelligence Agent run.")

    settings = load_settings()
    log.info("AI provider: %s (model: %s)", settings.ai_provider, settings.model_name)

    tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    ai_client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.model_name,
    )

    digests = generate_digest(tavily_client, ai_client)
    full_message = format_full_message(digests)

    log.info("Digest assembled (%d characters). Sending to Telegram...", len(full_message))
    deliver_digest(settings.telegram_bot_token, settings.telegram_chat_id, full_message)

    log.info("Run complete.")


if __name__ == "__main__":
    main()
