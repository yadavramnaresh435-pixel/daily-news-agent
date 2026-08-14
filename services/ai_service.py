"""
AI summarization service.

Turns raw Tavily search results into the formatted Telegram HTML digest
text, and assembles all per-category digests into the final message body.

Provider: Groq (REST API — see `GroqClient` below). Previously backed by
the Gemini SDK / OpenRouter; the public functions below are unchanged so
no other module needed modification when the provider was swapped.
"""

from __future__ import annotations

import json
import re
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



@dataclass
class EditorialReview:
    """Structured AI editorial assessment used by the ranking engine."""

    decision: bool | None
    confidence: float
    research_value: float
    reason: str


EDITORIAL_REVIEW_SYSTEM_INSTRUCTION = """\
You are the final editorial gatekeeper for hinduresearch.com, a serious,
academic-leaning research portal covering Indian civilisation, Hindu history,
archaeology, heritage, Sanskrit, manuscripts, ancient science and related research.

Evaluate ONE candidate source article using only the supplied title and source snippet.
Assess:
- Is this genuinely news or a concrete new development?
- Is it important enough for a serious research portal?
- Does it contain original/new information rather than generic discussion?
- Is it useful for a future HinduResearch.com article?
- Does it contain evidence or specific information rather than promotion, opinion or
  generic spirituality?
- What is its long-term research value?

HIGH research value includes archaeological discoveries and excavations, newly digitized
or translated manuscripts/Sanskrit texts, temple restoration or conservation, ASI and
government heritage initiatives, epigraphy, ancient astronomy or mathematics, Ayurveda
research, museum acquisitions, archival discoveries, scholarly publications and related
substantial heritage research.

LOW research value includes festival greetings, motivational content, generic spirituality,
opinion-only articles, commercial/promotional pages, travel promotion, religious marketing,
social-media updates, thin SEO pages and content without a concrete new development.

Do not judge a source solely by organisation size. A small research organisation or
independent historian can receive a high score when the evidence and subject are genuine.

Return ONLY this JSON object, with no Markdown or extra text:
{"decision":"KEEP|REJECT","confidence":0-100,"research_value":0-100,"reason":"one concise sentence"}

Confidence means how strongly the supplied evidence supports the editorial decision.
Very low-confidence material should normally be rejected. Do not invent facts or evidence.
"""


def _parse_editorial_response(response: str) -> EditorialReview:
    """Parse strict JSON while tolerating a fenced JSON response from the model."""
    raw = (response or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("Editorial response did not contain a JSON object.")
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Editorial response JSON must be an object.")

    decision_raw = str(data.get("decision", "")).strip().upper()
    if decision_raw.startswith("KEEP"):
        decision: bool | None = True
    elif decision_raw.startswith("REJECT"):
        decision = False
    else:
        decision = None

    def score(key: str) -> float:
        try:
            return max(0.0, min(100.0, float(data.get(key, 0))))
        except (TypeError, ValueError):
            return 0.0

    confidence = score("confidence")
    research_value = score("research_value")
    reason = str(data.get("reason", "")).strip() or "Editorial evaluation could not be parsed."
    return EditorialReview(decision, confidence, research_value, reason[:300])


def evaluate_article_with_groq(
    client: GroqClient, category: Category, result: SearchResult
) -> EditorialReview:
    """Return a structured editorial assessment; one failure remains non-fatal."""
    prompt = (
        f"Category: {category.name}\n"
        f"Title: {result.title}\n"
        f"URL: {result.url}\n"
        f"Snippet: {(result.content or '')[:2200]}\n"
    )
    try:
        response = client.chat(
            system_instruction=EDITORIAL_REVIEW_SYSTEM_INSTRUCTION,
            user_prompt=prompt,
            temperature=0.0,
        )
        review = _parse_editorial_response(response)
        if review.decision is None:
            log.warning("Unparseable editorial decision for '%s': %r", result.title, response)
        return review
    except Exception as exc:  # noqa: BLE001 - one article must never stop the run
        log.warning("Editorial evaluation failed for '%s': %s", result.title, exc)
        return EditorialReview(None, 0.0, 0.0, "Editorial evaluation unavailable.")


def review_article_with_groq(
    client: GroqClient, category: Category, result: SearchResult
) -> tuple[bool | None, str]:
    """Backward-compatible wrapper retained for existing callers."""
    review = evaluate_article_with_groq(client, category, result)
    return review.decision, review.reason


RESEARCH_DIGEST_SYSTEM_INSTRUCTION = """\
You are the senior research editor for hinduresearch.com, a serious, academic-leaning
portal covering Indian civilisation, Hindu history, archaeology, heritage, Sanskrit,
manuscripts, ancient science and related research.

Prepare a concise Telegram HTML research digest from the supplied shortlisted sources.
Use only facts supported by the supplied material. Never invent facts, dates, quotations,
institutions, interpretations or URLs.

For EVERY accepted source produce:
<b>• [specific factual finding]</b>
<b>Key Takeaways</b>
• [most important factual takeaway]
• [second important factual takeaway]
• [third important factual takeaway]
<b>Why This Matters</b>
[1-2 natural, specific sentences explaining significance for Indian civilisation, heritage
or historical research]
💡 <i>Research Hook: [specific future HinduResearch.com article angle]</i>
[Only when genuinely useful: 🔎 <i>Future Research: [specific follow-up direction]</i>]
<i>Editorial Confidence: [score]/100</i>
🔗 <a href="URL">Source Title</a>

Editorial rules:
- Key Takeaways must contain exactly 3 concise facts and must not copy the source wording.
- Why This Matters must explain a concrete significance, not generic statements such as
  "this is important for preserving our heritage" without saying why.
- Research Hook must resemble a real research article idea, not a generic topic or slogan.
- When historical memory is supplied and genuinely related, add a short
  <b>Historical Context</b> section explaining the connection to the earlier report.
  If the evidence shows an evolving excavation, manuscript project, restoration,
  inscription study or other ongoing subject, explicitly identify it as a
  <b>Continuing Story</b>. Do not imply continuity when the supplied evidence does not support it.
- Do not restate an earlier report as today's discovery. Focus on what is new today.
- Future Research is optional. Use it for concrete next questions: related inscriptions,
  nearby archaeological sites, texts to compare, archives, datasets, conservation records,
  or unresolved historical questions.
- If multiple supplied sources concern the same event, synthesize their non-duplicative
  facts. Prefer official/institutional evidence for the core claim and use another source
  only when it contributes a genuinely new fact.
- Avoid clickbait, generic AI language, repetition, promotional language and unsupported
  certainty.
- Telegram HTML only: <b>, <i>, <u>, <a href="">, <code>. No Markdown and no <br>.
- Preserve source URLs exactly as supplied.
- Separate article blocks with one blank line.
"""



def summarize_with_gemini(
    client: GroqClient,
    category: Category,
    results: list[SearchResult],
    editorial_reviews: dict[str, EditorialReview] | None = None,
    historical_context: dict[str, list[dict]] | None = None,
) -> str:
    """Summarize shortlisted sources with the existing Groq client.

    The optional fourth argument is backward-compatible and lets Phase 3 pass
    editorial evidence into the final writing prompt. The optional fifth
    argument adds best-effort historical memory for Phase 4.
    """
    if not results:
        return "No significant fresh updates found for this category today."

    lines = [f"Category: {category.name}", ""]
    for i, r in enumerate(results, start=1):
        review = (editorial_reviews or {}).get(r.url)
        lines.extend([
            f"Result {i}:",
            f"Title: {r.title}",
            f"URL: {r.url}",
            f"Snippet: {(r.content or '')[:1400]}",
        ])
        if review is not None:
            decision = "KEEP" if review.decision is True else "REVIEW"
            lines.append(
                f"Editorial assessment: {decision}; confidence={review.confidence:.0f}; "
                f"research_value={review.research_value:.0f}; reason={review.reason}"
            )

        prior_reports = (historical_context or {}).get(r.url, [])
        if prior_reports:
            lines.append("Historical memory / possible continuing story:")
            for prior in prior_reports:
                lines.append(
                    f"- Previously reported: {prior.get('reported_at', '')}; "
                    f"Topic: {prior.get('topic', '')}; Title: {prior.get('title', '')}; "
                    f"URL: {prior.get('url', '')}"
                )
                lines.append(f"  Prior evidence: {(prior.get('content') or '')[:700]}")
            lines.append("Use this only to explain a factual connection. Do not repeat the prior report as today's new finding.")

        lines.append("")

    try:
        return client.chat(
            system_instruction=RESEARCH_DIGEST_SYSTEM_INSTRUCTION,
            user_prompt="\n".join(lines),
            temperature=0.25,
        )
    except Exception as exc:  # noqa: BLE001 - never let one bad summary kill the run
        log.error("Groq summarization failed for '%s': %s", category.name, exc)
        fallback_lines = [f"🔗 <a href=\"{r.url}\">{r.title}</a>" for r in results]
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
