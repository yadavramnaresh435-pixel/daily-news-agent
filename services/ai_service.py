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


# ==========================================================================
# Phase 6.0 — structured per-article content for hinduresearch.com
#
# The Telegram digest above is a single free-form HTML blob per category, so
# it doesn't give the website structured per-article fields. This mirrors
# the existing evaluate_article_with_groq pattern (one article in, one
# strict-JSON object out) instead of inventing a new call style.
# ==========================================================================

@dataclass
class WebsiteArticle:
    """Structured website content with an extensible multilingual payload.

    The original English fields remain unchanged for Phase 6.0 compatibility.
    ``translations`` is an additive extension keyed by BCP-47 language code
    (currently ``en`` and ``hi``), so future languages can be added without
    changing the article schema shape again.
    """

    title: str
    summary: str
    key_takeaways: list[str]
    why_this_matters: str
    research_hook: str
    translations: dict[str, dict[str, object]] = field(default_factory=dict)


WEBSITE_CONTENT_SYSTEM_INSTRUCTION = """\
You are the research editor preparing ONE article for direct publication on
the hinduresearch.com website (not Telegram). Using only the supplied title
and source snippet, produce structured content for a serious, academic-
leaning research portal covering Indian civilisation, Hindu history,
archaeology, heritage, Sanskrit, manuscripts and ancient science.

Return ONLY this JSON object, with no Markdown or extra text:
{"title":"a clear factual website headline, rewritten not copied","summary":"2-3 sentence factual summary in your own words","key_takeaways":["fact 1","fact 2","fact 3"],"why_this_matters":"1-2 sentences on significance for Indian civilisation, heritage or historical research","research_hook":"a specific future hinduresearch.com research article angle"}

Rules:
- Use only facts supported by the supplied material. Never invent facts, dates, quotations or institutions.
- key_takeaways must contain exactly 3 concise, non-duplicative facts.
- Do not copy the source wording verbatim; rewrite everything in your own words.
- Avoid promotional language, clickbait and generic statements.
"""


def _parse_website_content_response(response: str, fallback_title: str) -> WebsiteArticle:
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
            raise ValueError("Website content response did not contain a JSON object.")
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Website content response JSON must be an object.")

    title = str(data.get("title") or fallback_title).strip() or fallback_title
    summary = str(data.get("summary") or "").strip()
    takeaways_raw = data.get("key_takeaways")
    key_takeaways = (
        [str(x).strip() for x in takeaways_raw if str(x).strip()]
        if isinstance(takeaways_raw, list)
        else []
    )
    why_this_matters = str(data.get("why_this_matters") or "").strip()
    research_hook = str(data.get("research_hook") or "").strip()
    return WebsiteArticle(title, summary, key_takeaways[:3], why_this_matters, research_hook)


HINDI_TRANSLATION_SYSTEM_INSTRUCTION = """\
You are the Hindi-language editor for hinduresearch.com.
Translate the supplied English research brief into clear, natural, high-quality
modern Hindi suitable for a serious research portal. Preserve factual meaning,
names, dates, institutions, titles of works, URLs and technical terminology.
Do not add facts, interpretation, claims or citations that are absent from the
English source. Use Devanagari for normal Hindi prose, while retaining
well-established proper nouns or technical terms in their standard English/Sanskrit
form where that improves clarity.

Return ONLY this JSON object, with no Markdown or extra text:
{"title":"Hindi title","summary":"Hindi summary","key_takeaways":["Hindi fact 1","Hindi fact 2","Hindi fact 3"],"why_this_matters":"Hindi significance","research_hook":"Hindi research angle"}

Rules:
- Translate meaning, not word-for-word syntax.
- Keep exactly 3 concise key takeaways when the English source has 3.
- Do not invent or omit factual qualifiers.
- Keep the tone neutral, scholarly and readable.
"""


def _parse_translation_response(response: str) -> dict[str, object]:
    raw = (response or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("Hindi translation response did not contain a JSON object.")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Hindi translation response JSON must be an object.")

    def clean_text(key: str) -> str:
        return str(data.get(key) or "").strip()

    takeaways_raw = data.get("key_takeaways")
    takeaways = (
        [str(x).strip() for x in takeaways_raw if str(x).strip()]
        if isinstance(takeaways_raw, list) else []
    )
    return {
        "title": clean_text("title"),
        "summary": clean_text("summary"),
        "key_takeaways": takeaways[:3],
        "why_this_matters": clean_text("why_this_matters"),
        "research_hook": clean_text("research_hook"),
    }


def translate_website_content_to_hindi(client: GroqClient, content: WebsiteArticle) -> WebsiteArticle:
    """Translate an already-generated English brief with the same Groq model.

    This deliberately reuses the English result instead of asking the model to
    independently author a second article, keeping both language versions
    factually aligned. On failure, English remains available and the caller can
    publish a safe fallback rather than breaking the workflow.
    """
    source = {
        "title": content.title,
        "summary": content.summary,
        "key_takeaways": content.key_takeaways,
        "why_this_matters": content.why_this_matters,
        "research_hook": content.research_hook,
    }
    try:
        response = client.chat(
            system_instruction=HINDI_TRANSLATION_SYSTEM_INSTRUCTION,
            user_prompt=json.dumps(source, ensure_ascii=False),
            temperature=0.2,
        )
        hindi = _parse_translation_response(response)
        required = ("title", "summary", "why_this_matters", "research_hook")
        if any(not str(hindi.get(key) or "").strip() for key in required):
            raise ValueError("Hindi translation omitted one or more required fields.")
        if len(hindi.get("key_takeaways", [])) < min(3, len(content.key_takeaways)):
            raise ValueError("Hindi translation omitted required key takeaways.")
        content.translations["hi"] = hindi
    except Exception as exc:  # noqa: BLE001 - translation is additive and non-fatal
        log.warning("Hindi translation failed for website article '%s': %s", content.title, exc)
    content.translations["en"] = {
        "title": content.title,
        "summary": content.summary,
        "key_takeaways": list(content.key_takeaways),
        "why_this_matters": content.why_this_matters,
        "research_hook": content.research_hook,
    }
    return content


def generate_website_content(
    client: GroqClient,
    category: Category,
    result: SearchResult,
    review: "EditorialReview | None" = None,
) -> WebsiteArticle:
    """Generate English once, then translate that exact brief into Hindi.

    AI remains responsible only for structured content. HTML is still rendered
    deterministically from templates by the website publisher.
    """
    prompt = (
        f"Category: {category.name}\n"
        f"Title: {result.title}\n"
        f"URL: {result.url}\n"
        f"Snippet: {(result.content or '')[:2200]}\n"
    )
    if review is not None:
        prompt += (
            f"Editorial notes: confidence={review.confidence:.0f}; "
            f"research_value={review.research_value:.0f}; reason={review.reason}\n"
        )
    try:
        response = client.chat(
            system_instruction=WEBSITE_CONTENT_SYSTEM_INSTRUCTION,
            user_prompt=prompt,
            temperature=0.3,
        )
        content = _parse_website_content_response(response, fallback_title=result.title)
        return translate_website_content_to_hindi(client, content)
    except Exception as exc:  # noqa: BLE001 - one article must never block website publishing
        log.warning("Website content generation failed for '%s': %s", result.title, exc)
        snippet = (result.content or "").strip()
        content = WebsiteArticle(
            title=result.title,
            summary=snippet[:400] if snippet else "Summary unavailable.",
            key_takeaways=[],
            why_this_matters="",
            research_hook="",
        )
        # Preserve a valid English payload even when AI generation itself fails.
        content.translations["en"] = {
            "title": content.title,
            "summary": content.summary,
            "key_takeaways": [],
            "why_this_matters": "",
            "research_hook": "",
        }
        return content


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

    Function name kept as `summarize_with_gemini` for interface stability —
    this is the AI service's public entry point and no other module needs
    to change — even though the underlying provider is now Groq.

    The two optional trailing arguments are backward-compatible: Phase 3
    added `editorial_reviews` (AI gatekeeper evidence) and Phase 5 added
    `historical_context` (prior same-topic coverage from working memory) so
    the writing prompt can reference genuine continuity instead of treating
    every article as if it appeared out of nowhere. Both are optional and
    default to None so older callers still work unmodified.

    NOTE: previously this function was defined twice in this module (once
    here, once earlier accepting `historical_context` but using the older
    SYSTEM_INSTRUCTION). The second definition silently shadowed the first,
    so every call from main.py — which always passes 5 positional args —
    raised TypeError and fell back to a degraded 3-arg call that dropped
    both editorial_reviews and historical_context. This merged version
    restores both without changing the public signature main.py already
    expects.
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
        prior = (historical_context or {}).get(r.url, [])
        if prior:
            lines.append("Relevant historical context (use only if factually connected):")
            for item in prior[:2]:
                lines.append(
                    f"- {item.get('reported_at','')}: {item.get('title','')} | "
                    f"topic={item.get('topic','')} | evidence={(item.get('content') or '')[:500]}"
                )
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
