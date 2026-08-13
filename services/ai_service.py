"""
AI summarization service.

Turns raw Tavily search results into the formatted Telegram HTML digest
text, and assembles all per-category digests into the final message body.

Provider: Groq (REST API — see `GroqClient` below). Previously backed by
the Gemini SDK / OpenRouter; the public functions below are unchanged so
no other module needed modification when the provider was swapped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from config.constants import (
    Category,
    GROQ_MAX_RETRIES,
    GROQ_RETRY_BACKOFF_SECONDS,
    GROQ_TIMEOUT_SECONDS,
    SYSTEM_INSTRUCTION,
)
from services.tavily_service import SearchResult
from utils.helpers import current_utc_date_str
from utils.logger import get_logger

log = get_logger()


@dataclass
class CategoryDigest:
    category: Category
    results: list[SearchResult] = field(default_factory=list)
    summary_html: str = ""
    error: Optional[str] = None


class GroqClient:
    """
    Reusable, minimal client for Groq's OpenAI-compatible chat-completions
    REST API.

    Handles auth headers, request timeout, and retrying transient failures
    with a short backoff. Instantiated once in main.py and passed into
    `summarize_with_gemini` for every category, the same way the old
    Gemini SDK client was.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = GROQ_TIMEOUT_SECONDS,
        max_retries: int = GROQ_MAX_RETRIES,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, system_instruction: str, user_prompt: str, temperature: float = 0.4) -> str:
        """
        Send a single chat-completion request to OpenRouter, retrying up to
        `max_retries` times on any failure (timeout, network error, bad
        status, malformed/empty response). Raises the last error if every
        attempt fails, so the caller can apply its own fallback behavior.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                text = (data["choices"][0]["message"]["content"] or "").strip()
                if not text:
                    raise ValueError("Empty response content from OpenRouter.")
                return text
            except Exception as exc:  # noqa: BLE001 - retry on any transient error
                last_exc = exc
                log.warning(
                    "Groq request attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(GROQ_RETRY_BACKOFF_SECONDS * attempt)

        raise last_exc  # retries exhausted — let the caller apply its fallback


def build_user_prompt(category: Category, results: list[SearchResult]) -> str:
    """Serialize raw search results into a prompt for the AI model."""
    lines = [f"Category: {category.name}", ""]
    for i, r in enumerate(results, start=1):
        lines.append(f"Result {i}:")
        lines.append(f"Title: {r.title}")
        lines.append(f"URL: {r.url}")
        lines.append(f"Snippet: {r.content[:800]}")  # cap snippet length for token safety
        lines.append("")
    return "\n".join(lines)


def summarize_with_gemini(
    client: GroqClient, category: Category, results: list[SearchResult]
) -> str:
    """
    Summarize one category's search results via the configured Groq
    model. Falls back to a graceful placeholder string on any API failure
    (including after retries are exhausted).

    Function name kept as `summarize_with_gemini` for interface stability —
    this is the AI service's public entry point and no other module needs
    to change — even though the underlying provider is now Groq.
    """
    if not results:
        return "No significant fresh updates found for this category today."

    prompt = build_user_prompt(category, results)

    try:
        return client.chat(system_instruction=SYSTEM_INSTRUCTION, user_prompt=prompt, temperature=0.4)
    except Exception as exc:  # noqa: BLE001 - never let one bad summary kill the run
        log.error("Groq summarization failed for '%s': %s", category.name, exc)
        # Fallback: build a minimal manual digest so links are never lost.
        fallback_lines = []
        for r in results:
            fallback_lines.append(f"🔗 <a href=\"{r.url}\">{r.title}</a>")
        return "⚠️ AI summary unavailable — raw sources:\n" + "\n".join(fallback_lines)


def format_full_message(digests: list[CategoryDigest]) -> str:
    """Build the final HTML-formatted Telegram message from all category digests."""
    date_str = current_utc_date_str("%d %B %Y")
    header = f"🕉️ <b>Hindu Research Daily Intelligence Digest</b>\n📅 {date_str} (UTC)\n"

    sections = [header]
    for digest in digests:
        cat = digest.category
        section = f"\n{cat.emoji} <b>{cat.name}</b>\n{digest.summary_html}\n"
        sections.append(section)

    sections.append("\n<i>— Generated automatically for hinduresearch.com —</i>")
    return "\n".join(sections)
