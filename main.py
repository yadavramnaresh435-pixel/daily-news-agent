#!/usr/bin/env python3
"""
Hindu Research Daily Intelligence & Content Agent
===================================================

Fetches fresh, real-world news across four research niches using Tavily,
summarizes each result into a factual bullet + content-writing hook using
an AI model via the Groq REST API, and delivers a formatted daily
digest to a Telegram chat.

Designed to run as a scheduled GitHub Actions job, but also runs locally
via a `.env` file (python-dotenv).

This module is intentionally a thin orchestrator: it wires the
config/services/utils modules together in the correct order and contains
no business logic of its own. See:
  - config/     -> static + runtime configuration
  - services/   -> Tavily search, Groq summarization, Telegram delivery
  - utils/      -> logging and small generic helpers

Phase 1 additions (reliability only — no architecture changes):
  - Empty/meaningless articles are filtered out before they reach Telegram.
  - Duplicate articles (same URL, same title, or the same story covered by
    multiple outlets) are collapsed to the richest source.
  - If Groq summarization fails even after its built-in retries, a short
    fallback digest is generated directly from the raw snippets instead of
    dropping the category.
  - A single malformed article can never crash a category or the run.
  - Stage-by-stage counts are logged so GitHub Actions runs are easy to
    audit at a glance.

Author: AI Systems Engineering
"""

from __future__ import annotations

import re

from tavily import TavilyClient

from config.constants import CATEGORIES
from config.settings import load_settings
from services.ai_service import (
    CategoryDigest,
    GroqClient,
    format_full_message,
    summarize_with_gemini,
)
from services.tavily_service import SearchResult, run_tavily_search
from services.telegram_service import deliver_digest
from utils.logger import get_logger

log = get_logger()

# Phrases that indicate Tavily returned a "result" with no real content.
# Anything matching these (case-insensitive) is treated the same as an
# empty snippet and skipped.
_EMPTY_CONTENT_MARKERS = (
    "no significant information",
    "no information available",
    "no information is present",
    "no content available",
    "no relevant information",
)

# Marker used by services.ai_service.summarize_with_gemini's own fallback
# path (raised only after its internal retries are exhausted). We detect it
# here to build a nicer, snippet-based fallback and to count it for logs.
_AI_FAILURE_MARKER = "⚠️ AI summary unavailable"

# Minimum snippet length (chars) to be considered a "meaningful" article.
_MIN_CONTENT_LENGTH = 20


def _is_meaningful_result(result: SearchResult) -> bool:
    """False for a result with no usable snippet/description/content."""
    content = (result.content or "").strip()
    if not content or content.lower() in {"none", "null", "n/a"}:
        return False
    if len(content) < _MIN_CONTENT_LENGTH:
        return False
    lowered = content.lower()
    if any(marker in lowered for marker in _EMPTY_CONTENT_MARKERS):
        return False
    return True


def _filter_meaningful_results(
    results: list[SearchResult], category_name: str
) -> tuple[list[SearchResult], int]:
    """
    Drop results with no usable snippet/content. Each result is checked in
    isolation — one malformed item is skipped and logged rather than
    aborting the whole category.
    """
    meaningful: list[SearchResult] = []
    skipped = 0
    for result in results:
        try:
            if _is_meaningful_result(result):
                meaningful.append(result)
            else:
                skipped += 1
        except Exception as exc:  # noqa: BLE001 - one bad article must never stop the run
            skipped += 1
            log.warning("Skipping malformed article in '%s': %s", category_name, exc)
    return meaningful, skipped


def _normalize_title(title: str) -> str:
    """Lowercase, punctuation-stripped title used for duplicate matching."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _dedupe_results(results: list[SearchResult]) -> tuple[list[SearchResult], int]:
    """
    Remove duplicate articles: same URL, same title, or the same event
    reported by multiple outlets under a near-identical headline. When a
    duplicate is found, the source with the richer (longer) snippet is
    kept, as a simple proxy for "highest quality source".
    """
    original_count = len(results)

    # Pass 1 — exact URL duplicates.
    by_url: dict[str, SearchResult] = {}
    url_order: list[str] = []
    for result in results:
        key = (result.url or "").strip().lower()
        existing = by_url.get(key)
        if existing is None:
            by_url[key] = result
            url_order.append(key)
        elif len(result.content) > len(existing.content):
            by_url[key] = result
    stage1 = [by_url[key] for key in url_order]

    # Pass 2 — same/near-same title duplicates (same story, different site).
    by_title: dict[str, SearchResult] = {}
    title_order: list[str] = []
    for result in stage1:
        norm_title = _normalize_title(result.title)
        # Very short/empty normalized titles are too ambiguous to merge on
        # safely, so treat each as its own unique bucket instead.
        key = norm_title if len(norm_title) > 8 else f"__unique__:{result.url}"
        existing = by_title.get(key)
        if existing is None:
            by_title[key] = result
            title_order.append(key)
        elif len(result.content) > len(existing.content):
            by_title[key] = result
    deduped = [by_title[key] for key in title_order]

    duplicates_removed = original_count - len(deduped)
    return deduped, duplicates_removed


def _build_snippet_fallback(results: list[SearchResult]) -> str:
    """
    Short, manual digest built straight from the raw snippets when Groq
    summarization fails even after its internal retries. Keeps the run
    alive and the links intact instead of dropping the category.
    """
    if not results:
        return "⚠️ AI summary unavailable and no raw sources to fall back on for this category."

    blocks = []
    for result in results:
        snippet = (result.content or "").strip()
        if len(snippet) > 200:
            snippet = snippet[:200].rsplit(" ", 1)[0] + "…"
        blocks.append(f"<b>• {result.title}</b>\n{snippet}\n🔗 <a href=\"{result.url}\">Source</a>")

    return "⚠️ AI summary unavailable — showing raw snippets:\n\n" + "\n\n".join(blocks)


def generate_digest(
    tavily_client: TavilyClient, ai_client: GroqClient
) -> tuple[list[CategoryDigest], dict[str, int]]:
    """
    Run search + filtering + summarization for every category, isolating
    failures at both the category level and the individual-article level.
    Returns the digests plus aggregate stats for logging.
    """
    digests: list[CategoryDigest] = []
    stats = {
        "found": 0,
        "skipped_empty": 0,
        "duplicates_removed": 0,
        "after_filtering": 0,
        "ai_summaries_generated": 0,
        "fallback_summaries_used": 0,
    }

    for category in CATEGORIES:
        digest = CategoryDigest(category=category)
        try:
            raw_results = run_tavily_search(tavily_client, category)
            stats["found"] += len(raw_results)

            meaningful, skipped_empty = _filter_meaningful_results(raw_results, category.name)
            stats["skipped_empty"] += skipped_empty

            deduped, duplicates_removed = _dedupe_results(meaningful)
            stats["duplicates_removed"] += duplicates_removed
            stats["after_filtering"] += len(deduped)

            digest.results = deduped

            # summarize_with_gemini already retries internally on transient
            # Groq failures and never raises; if it still can't produce a
            # summary after retries, it returns a marked fallback string.
            summary = summarize_with_gemini(ai_client, category, digest.results)
            if summary.startswith(_AI_FAILURE_MARKER):
                stats["fallback_summaries_used"] += 1
                summary = _build_snippet_fallback(digest.results)
            else:
                stats["ai_summaries_generated"] += 1

            digest.summary_html = summary
        except Exception as exc:  # noqa: BLE001 - absolute last line of defense per category
            log.error("Unexpected failure processing '%s': %s", category.name, exc)
            digest.error = str(exc)
            digest.summary_html = "⚠️ This category could not be processed today due to an internal error."
        digests.append(digest)

    return digests, stats


def main() -> None:
    log.info("Starting Hindu Research Daily Intelligence Agent run.")

    settings = load_settings()
    log.info("AI provider: %s (model: %s)", settings.ai_provider, settings.model_name)

    tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    ai_client = GroqClient(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        model=settings.model_name,
    )

    log.info("Searching articles...")
    digests, stats = generate_digest(tavily_client, ai_client)

    log.info("Found %d articles", stats["found"])
    log.info("After relevance filtering: %d", stats["after_filtering"])
    log.info("Duplicates removed: %d", stats["duplicates_removed"])
    log.info("Skipped empty articles: %d", stats["skipped_empty"])
    log.info("AI summaries generated: %d", stats["ai_summaries_generated"])
    log.info("Fallback summaries used: %d", stats["fallback_summaries_used"])

    full_message = format_full_message(digests)
    log.info("Digest assembled (%d characters). Sending to Telegram...", len(full_message))

    try:
        deliver_digest(settings.telegram_bot_token, settings.telegram_chat_id, full_message)
        log.info("Telegram message sent successfully")
    except Exception as exc:  # noqa: BLE001 - a delivery failure must not crash the run
        log.error("Telegram delivery failed: %s", exc)

    log.info("Run complete.")


if __name__ == "__main__":
    main()
