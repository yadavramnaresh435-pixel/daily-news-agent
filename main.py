#!/usr/bin/env python3
"""
Hindu Research Daily Intelligence & Content Agent
===================================================

Fetches fresh, real-world news across four research niches using Tavily,
summarizes each result into a factual bullet + content-writing hook using
Gemini 2.5 Flash, and delivers a formatted daily digest to a Telegram chat.

Designed to run as a scheduled GitHub Actions job, but also runs locally
via a `.env` file (python-dotenv).

Author: AI Systems Engineering
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

# --------------------------------------------------------------------------
# Optional local .env support. On GitHub Actions, env vars are injected
# directly via `os.environ`, so a missing dotenv package/file is not fatal.
# --------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types as genai_types
from tavily import TavilyClient

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("hindu_research_agent")


# ==========================================================================
# Configuration
# ==========================================================================

@dataclass(frozen=True)
class Category:
    """A single research niche to search and summarize."""
    emoji: str
    name: str
    query: str


CATEGORIES: list[Category] = [
    Category(
        emoji="🏛️",
        name="Archaeology & Ancient Discoveries",
        query="latest ancient temple discoveries archaeological excavations India Hindu history news",
    ),
    Category(
        emoji="📜",
        name="Vedic & Manuscript Research",
        query="Sanskrit research studies Indology ancient Indian science manuscripts research papers",
    ),
    Category(
        emoji="🚩",
        name="Global Cultural & Temple Events",
        query="major Hindu diaspora events global temple inaugurations cultural milestones today",
    ),
    Category(
        emoji="💡",
        name="Daily Research Hook / Topic Idea",
        query="ancient Indian philosophy wisdom insights Bhagavad Gita Upanishads research analysis",
    ),
]

GEMINI_MODEL = "gemini-2.5-flash"
TAVILY_MAX_RESULTS = 4
TAVILY_SEARCH_DAYS = 1  # freshness window in days
TELEGRAM_CHUNK_LIMIT = 4000  # keep a safety margin under Telegram's 4096 cap
TELEGRAM_API_ROOT = "https://api.telegram.org"

SYSTEM_INSTRUCTION = """\
You are the lead content strategist and research analyst for hinduresearch.com, \
a serious academic-leaning research portal covering Hindu history, archaeology, \
Vedic studies, and global cultural affairs.

You will be given a category name and a list of raw news search results \
(title, URL, and snippet) for that category.

Your task, for EACH source article/result provided:
1. Extract the core factual discovery or event in 1-2 concise, information-dense bullet points.
2. Suggest ONE practical "Content Writing Hook" — a scroll-stopping angle or headline idea \
that hinduresearch.com could use to turn this into an article, thread, or video script.
3. Always include the original Source Title and the direct URL exactly as given. Never invent, \
alter, or guess a URL. Never fabricate facts not present in the provided snippets.

Formatting rules (strict):
- Output must use Telegram HTML formatting ONLY (supported tags: <b>, <i>, <u>, <a href="">, <code>). \
Do not use Markdown asterisks or square-bracket links.
- Do not use <br> or unsupported HTML tags.
- For each result, format exactly like this pattern (repeat per result):

<b>• [Fact bullet 1]</b>
[Fact bullet 2 if applicable]
💡 <i>Content Hook: [your suggested hook]</i>
🔗 <a href="URL">Source Title</a>

- Separate each result block with a blank line.
- If NO usable results are provided for a category, respond with exactly:
No significant fresh updates found for this category today.
- Keep the entire response concise — this will be delivered inside a Telegram message with \
character limits. Do not add greetings, preambles, or closing remarks. Output ONLY the \
formatted bullets described above.
"""


# ==========================================================================
# Data structures
# ==========================================================================

@dataclass
class SearchResult:
    title: str
    url: str
    content: str


@dataclass
class CategoryDigest:
    category: Category
    results: list[SearchResult] = field(default_factory=list)
    summary_html: str = ""
    error: Optional[str] = None


# ==========================================================================
# Step 1: Tavily Search
# ==========================================================================

def run_tavily_search(client: TavilyClient, category: Category) -> list[SearchResult]:
    """
    Query Tavily for a single category. Returns an empty list (never raises)
    on failure or zero results, so one bad category can't crash the run.
    """
    try:
        response = client.search(
            query=category.query,
            topic="news",
            days=TAVILY_SEARCH_DAYS,
            max_results=TAVILY_MAX_RESULTS,
            include_answer=False,
        )
    except Exception as exc:  # noqa: BLE001 - we want to swallow ALL search errors
        log.error("Tavily search failed for '%s': %s", category.name, exc)
        return []

    raw_results = response.get("results", []) if isinstance(response, dict) else []
    if not raw_results:
        log.warning("Tavily returned zero results for '%s'.", category.name)
        return []

    results: list[SearchResult] = []
    for item in raw_results:
        title = (item.get("title") or "Untitled Source").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        if not url:
            continue  # a bullet without a verifiable source link is useless here
        results.append(SearchResult(title=title, url=url, content=content))

    log.info("Tavily: %d usable result(s) for '%s'.", len(results), category.name)
    return results


# ==========================================================================
# Step 2: Gemini Summarization
# ==========================================================================

def build_user_prompt(category: Category, results: list[SearchResult]) -> str:
    """Serialize raw search results into a prompt for Gemini."""
    lines = [f"Category: {category.name}", ""]
    for i, r in enumerate(results, start=1):
        lines.append(f"Result {i}:")
        lines.append(f"Title: {r.title}")
        lines.append(f"URL: {r.url}")
        lines.append(f"Snippet: {r.content[:800]}")  # cap snippet length for token safety
        lines.append("")
    return "\n".join(lines)


def summarize_with_gemini(
    client: genai.Client, category: Category, results: list[SearchResult]
) -> str:
    """
    Summarize one category's search results via Gemini 2.5 Flash.
    Falls back to a graceful placeholder string on any API failure.
    """
    if not results:
        return "No significant fresh updates found for this category today."

    prompt = build_user_prompt(category, results)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.4,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Empty response text from Gemini.")
        return text
    except Exception as exc:  # noqa: BLE001 - never let one bad summary kill the run
        log.error("Gemini summarization failed for '%s': %s", category.name, exc)
        # Fallback: build a minimal manual digest so links are never lost.
        fallback_lines = []
        for r in results:
            fallback_lines.append(f"🔗 <a href=\"{r.url}\">{r.title}</a>")
        return "⚠️ AI summary unavailable — raw sources:\n" + "\n".join(fallback_lines)


# ==========================================================================
# Step 3: Assemble the Digest
# ==========================================================================

def generate_digest(tavily_client: TavilyClient, gemini_client: genai.Client) -> list[CategoryDigest]:
    """Run search + summarization for every category, isolating failures."""
    digests: list[CategoryDigest] = []

    for category in CATEGORIES:
        digest = CategoryDigest(category=category)
        try:
            digest.results = run_tavily_search(tavily_client, category)
            digest.summary_html = summarize_with_gemini(gemini_client, category, digest.results)
        except Exception as exc:  # noqa: BLE001 - absolute last line of defense per category
            log.error("Unexpected failure processing '%s': %s", category.name, exc)
            digest.error = str(exc)
            digest.summary_html = "⚠️ This category could not be processed today due to an internal error."
        digests.append(digest)

    return digests


def format_full_message(digests: list[CategoryDigest]) -> str:
    """Build the final HTML-formatted Telegram message from all category digests."""
    from datetime import datetime, timezone

    date_str = datetime.now(timezone.utc).strftime("%d %B %Y")
    header = f"🕉️ <b>Hindu Research Daily Intelligence Digest</b>\n📅 {date_str} (UTC)\n"

    sections = [header]
    for digest in digests:
        cat = digest.category
        section = f"\n{cat.emoji} <b>{cat.name}</b>\n{digest.summary_html}\n"
        sections.append(section)

    sections.append("\n<i>— Generated automatically for hinduresearch.com —</i>")
    return "\n".join(sections)


# ==========================================================================
# Step 4: Telegram Delivery
# ==========================================================================

def split_message(text: str, limit: int = TELEGRAM_CHUNK_LIMIT) -> list[str]:
    """
    Split a long message into chunks under Telegram's character limit.
    Splits on paragraph/line boundaries where possible to avoid breaking
    HTML tags (e.g. an unclosed <a href="..."> tag) mid-way.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""

    for line in text.split("\n"):
        # +1 accounts for the newline that will rejoin lines
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current.rstrip())
            current = line
            # Edge case: a single line longer than the limit on its own.
            while len(current) > limit:
                chunks.append(current[:limit])
                current = current[limit:]
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current.rstrip())

    return chunks


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> bool:
    """Send a single message chunk to Telegram. Returns True on success."""
    url = f"{TELEGRAM_API_ROOT}/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, data=payload, timeout=15)
        if response.status_code == 200:
            return True
        log.error(
            "Telegram API error (status %s): %s",
            response.status_code,
            response.text[:500],
        )
        return False
    except requests.RequestException as exc:
        log.error("Telegram request failed: %s", exc)
        return False


def deliver_digest(bot_token: str, chat_id: str, full_text: str) -> None:
    """Split (if needed) and deliver the digest, logging per-chunk outcomes."""
    chunks = split_message(full_text)
    log.info("Delivering digest in %d chunk(s).", len(chunks))

    for i, chunk in enumerate(chunks, start=1):
        success = send_telegram_message(bot_token, chat_id, chunk)
        status = "sent" if success else "FAILED"
        log.info("Chunk %d/%d %s (%d chars).", i, len(chunks), status, len(chunk))
        if len(chunks) > 1:
            time.sleep(0.5)  # gentle pacing to avoid Telegram rate limits


# ==========================================================================
# Configuration loading & validation
# ==========================================================================

REQUIRED_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TAVILY_API_KEY",
    "GEMINI_API_KEY",
]


def load_config() -> dict[str, str]:
    """Read and validate all required environment variables."""
    config = {var: os.environ.get(var, "").strip() for var in REQUIRED_ENV_VARS}
    missing = [var for var, val in config.items() if not val]
    if missing:
        log.critical("Missing required environment variable(s): %s", ", ".join(missing))
        sys.exit(1)
    return config


# ==========================================================================
# Entry point
# ==========================================================================

def main() -> None:
    log.info("Starting Hindu Research Daily Intelligence Agent run.")

    config = load_config()

    tavily_client = TavilyClient(api_key=config["TAVILY_API_KEY"])
    gemini_client = genai.Client(api_key=config["GEMINI_API_KEY"])

    digests = generate_digest(tavily_client, gemini_client)
    full_message = format_full_message(digests)

    log.info("Digest assembled (%d characters). Sending to Telegram...", len(full_message))
    deliver_digest(config["TELEGRAM_BOT_TOKEN"], config["TELEGRAM_CHAT_ID"], full_message)

    log.info("Run complete.")


if __name__ == "__main__":
    main()
