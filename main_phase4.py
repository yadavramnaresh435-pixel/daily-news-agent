#!/usr/bin/env python3
"""
Hindu Research Daily Intelligence & Content Agent
===================================================

Phase 2 keeps the existing orchestration and public service interfaces, but
adds deterministic article quality scoring, freshness, source credibility,
relevance, editorial review, duplicate detection, and topic diversity before
Groq summarization.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from urllib.parse import urlparse

from tavily import TavilyClient

from config.constants import CATEGORIES, MEMORY_FILE, MEMORY_RETENTION_DAYS
from config.settings import load_settings
from services.ai_service import (
    CategoryDigest,
    GroqClient,
    format_full_message,
    EditorialReview,
    evaluate_article_with_groq,
    summarize_with_gemini,
)
from services.tavily_service import SearchResult, run_tavily_search
from services.telegram_service import deliver_digest
from utils.logger import get_logger

log = get_logger()

_EMPTY_CONTENT_MARKERS = (
    "no significant information",
    "no information available",
    "no information is present",
    "no content available",
    "no relevant information",
)
_AI_FAILURE_MARKER = "⚠️ AI summary unavailable"
_MIN_CONTENT_LENGTH = 20

# Final digest guardrails. These are internal ranking limits, not public APIs.
_FINAL_MAX_ARTICLES = 8
_MAX_PER_CATEGORY = 3
_EDITORIAL_SHORTLIST_PER_CATEGORY = 4

# Strong relevance signals. Matching is intentionally broad enough to preserve
# legitimate small research organisations and less famous institutions.
_RELEVANCE_GROUPS: dict[str, tuple[str, ...]] = {
    "archaeology": (
        "archaeolog", "excavation", "inscription", "epigraphy", "epigraphic",
        "artifact", "artefact", "burial", "stratigraphy", "heritage site",
        "archaeological survey", "ancient site",
    ),
    "temples": (
        "temple", "mandir", "shrine", "sanctum", "garbhagriha",
        "temple architecture", "conservation", "restoration", "heritage",
    ),
    "manuscripts": (
        "manuscript", "palm-leaf", "palm leaf", "codex", "sanskrit text",
        "critical edition", "textual study", "inscription",
    ),
    "ancient_science": (
        "ancient science", "mathematics", "astronomy", "jyotisha", "medicine",
        "ayurveda", "metallurgy", "water management", "engineering",
    ),
    "philosophy": (
        "indian philosophy", "vedanta", "upanishad", "bhagavad gita",
        "sanskrit", "nyaya", "samkhya", "mimamsa", "yoga philosophy",
    ),
    "museums": (
        "museum", "archive", "national archives", "collection", "exhibition",
        "digital archive",
    ),
    "knowledge": (
        "indian knowledge systems", "vedic", "shastra", "ancient education",
        "gurukula", "gurukul", "indology", "vedic studies",
    ),
    "culture": (
        "hindu culture", "diaspora", "cultural heritage", "cultural milestone",
        "temple inauguration", "heritage project",
    ),
}

_REJECT_TERMS = (
    "instagram.com",
    "facebook.com",
    "tiktok.com",
    "pinterest.com",
    "festival wishes",
    "happy diwali",
    "happy holi",
    "buy now",
    "book now",
    "sponsored",
    "affiliate",
    "advertisement",
    "casino",
)

_GENERIC_TRAVEL_TERMS = (
    "travel guide",
    "tourist guide",
    "things to do",
    "best places to visit",
    "how to reach",
    "travel itinerary",
    "tour package",
)

_COMMERCIAL_HINTS = (
    "shop", "store", "hotel", "resort", "booking", "tickets",
    "real estate", "product", "course", "consulting",
)

_MAJOR_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "thehindu.com", "indianexpress.com",
    "hindustantimes.com", "timesofindia.indiatimes.com", "deccanherald.com",
    "telegraphindia.com", "theprint.in", "scroll.in",
}

_HIGH_CREDIBILITY_DOMAINS = {
    "asi.nic.in", "ignca.gov.in", "ichr.ac.in", "isro.gov.in",
    "nationalarchives.nic.in", "culture.gov.in", "education.gov.in",
    "pib.gov.in", "sahitya-akademi.gov.in",
}



# ---------------------------------------------------------------------------
# Lightweight historical memory
# ---------------------------------------------------------------------------
# Memory is deliberately file-based so the existing architecture, environment
# variables, and service interfaces remain unchanged. Any memory failure is
# non-fatal: the agent simply behaves like the previous daily workflow.
_MEMORY_VERSION = 1
_MEMORY_MAX_ENTRIES = 500


def _memory_path() -> str:
    return MEMORY_FILE


def _load_memory() -> dict:
    try:
        with open(_memory_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {"version": _MEMORY_VERSION, "entries": []}
        entries = data.get("entries")
        if not isinstance(entries, list):
            entries = []
        return {"version": data.get("version", _MEMORY_VERSION), "entries": entries}
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("Historical memory unavailable; continuing without it: %s", exc)
        return {"version": _MEMORY_VERSION, "entries": []}
    except Exception as exc:  # noqa: BLE001 - memory must never break a run
        log.warning("Unexpected historical memory load failure: %s", exc)
        return {"version": _MEMORY_VERSION, "entries": []}


def _save_memory(memory: dict) -> None:
    try:
        directory = os.path.dirname(_memory_path())
        if directory:
            os.makedirs(directory, exist_ok=True)
        temp_path = f"{_memory_path()}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(memory, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, _memory_path())
    except Exception as exc:  # noqa: BLE001 - memory persistence is best-effort
        log.warning("Could not save historical memory; run will still complete: %s", exc)
        try:
            if os.path.exists(f"{_memory_path()}.tmp"):
                os.remove(f"{_memory_path()}.tmp")
        except OSError:
            pass


def _memory_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _content_similarity(a: str, b: str) -> float:
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _memory_match(result: SearchResult, entry: dict) -> bool:
    """Return True only for a near-duplicate, not merely a related topic."""
    old_url = str(entry.get("url") or "").strip().lower()
    new_url = (result.url or "").strip().lower()
    if old_url and new_url and old_url == new_url:
        old_content = str(entry.get("content") or "")
        # A changed page is allowed through when its supplied material is
        # materially different; this is the significant-update escape hatch.
        if old_content and _content_similarity(result.content, old_content) < 0.78:
            return False
        return True

    old_title = str(entry.get("title") or "")
    old_content = str(entry.get("content") or "")
    if not old_title:
        return False
    title_similarity = SequenceMatcher(
        None, _normalize_text(result.title), _normalize_text(old_title)
    ).ratio()
    token_a, token_b = _title_tokens(result.title), _title_tokens(old_title)
    overlap = len(token_a & token_b) / max(1, min(len(token_a), len(token_b)))
    content_similarity = _content_similarity(result.content, old_content)

    # Require strong title/story evidence before suppressing a result.
    return (
        (title_similarity >= 0.90 or overlap >= 0.88)
        and content_similarity >= 0.70
    )


def _prune_memory(memory: dict) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(days=MEMORY_RETENTION_DAYS)
    kept = []
    for entry in memory.get("entries", []):
        try:
            when = _memory_datetime(entry.get("reported_at"))
            if when is None or when >= cutoff:
                kept.append(entry)
        except Exception:
            continue
    memory["entries"] = kept[-_MEMORY_MAX_ENTRIES:]
    memory["version"] = _MEMORY_VERSION
    return memory


def _historical_context(result: SearchResult, memory: dict) -> list[dict]:
    """Find related prior reports for context/continuity without suppressing them."""
    related = []
    current_topic = _topic_for_result(result)
    for entry in memory.get("entries", []):
        try:
            when = _memory_datetime(entry.get("reported_at"))
            if when is None:
                continue
            title = str(entry.get("title") or "")
            content = str(entry.get("content") or "")
            title_similarity = SequenceMatcher(
                None, _normalize_text(result.title), _normalize_text(title)
            ).ratio() if title else 0.0
            token_a, token_b = _title_tokens(result.title), _title_tokens(title)
            overlap = len(token_a & token_b) / max(1, min(len(token_a), len(token_b)))
            same_topic = str(entry.get("topic") or "") == current_topic
            content_similarity = _content_similarity(result.content, content)
            if title_similarity >= 0.55 or overlap >= 0.50 or (same_topic and content_similarity >= 0.35):
                related.append({
                    "reported_at": when.isoformat(),
                    "title": title,
                    "url": str(entry.get("url") or ""),
                    "topic": str(entry.get("topic") or current_topic),
                    "content": content[:900],
                })
        except Exception:
            continue

    related.sort(key=lambda item: item["reported_at"], reverse=True)
    return related[:3]


def _apply_historical_memory(
    results_by_category: dict[str, list[SearchResult]], memory: dict
) -> tuple[dict[str, list[SearchResult]], dict[str, list[dict]], int]:
    """Suppress only near-duplicates and collect context for continuing stories."""
    filtered: dict[str, list[SearchResult]] = {}
    context: dict[str, list[dict]] = {}
    suppressed = 0
    entries = memory.get("entries", [])

    for category_name, results in results_by_category.items():
        kept = []
        for result in results:
            try:
                matches = [entry for entry in entries if _memory_match(result, entry)]
                if matches:
                    # A near-duplicate is suppressed. The most recent matching
                    # entry is still available as historical memory for future runs.
                    suppressed += 1
                    continue
                related = _historical_context(result, memory)
                if related:
                    context[result.url] = related
                kept.append(result)
            except Exception as exc:
                log.warning("Memory comparison failed for '%s': %s", result.title, exc)
                kept.append(result)
        filtered[category_name] = kept

    return filtered, context, suppressed


def _record_memory(
    memory: dict,
    selected_by_category: dict[str, list[SearchResult]],
    historical_context: dict[str, list[dict]],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    entries = memory.setdefault("entries", [])
    existing_urls = {str(item.get("url") or "").strip().lower() for item in entries}

    for category_name, results in selected_by_category.items():
        for result in results:
            url_key = (result.url or "").strip().lower()
            # Do not duplicate an entry when multiple categories surface the same URL.
            if url_key and url_key in existing_urls:
                continue
            topic = _topic_for_result(result)
            related = historical_context.get(result.url, [])
            entries.append({
                "reported_at": now,
                "category": category_name,
                "topic": topic,
                "title": result.title,
                "url": result.url,
                "content": (result.content or "")[:1800],
                "historical_context_urls": [item["url"] for item in related if item.get("url")],
            })
            if url_key:
                existing_urls.add(url_key)

    _prune_memory(memory)
    _save_memory(memory)


def _is_meaningful_result(result: SearchResult) -> bool:
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
    meaningful: list[SearchResult] = []
    skipped = 0
    for result in results:
        try:
            if _is_meaningful_result(result):
                meaningful.append(result)
            else:
                skipped += 1
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            log.warning("Skipping malformed article in '%s': %s", category_name, exc)
    return meaningful, skipped


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _title_tokens(title: str) -> set[str]:
    return {t for t in _normalize_text(title).split() if len(t) > 2}


def _same_story(a: SearchResult, b: SearchResult) -> bool:
    a_title = _normalize_text(a.title)
    b_title = _normalize_text(b.title)
    if not a_title or not b_title:
        return False
    similarity = SequenceMatcher(None, a_title, b_title).ratio()
    ta, tb = _title_tokens(a.title), _title_tokens(b.title)
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    return similarity >= 0.86 or overlap >= 0.80


def _dedupe_results(results: list[SearchResult]) -> tuple[list[SearchResult], int]:
    """Collapse exact URLs and near-identical stories, retaining the richest source."""
    kept: list[SearchResult] = []
    seen_urls: set[str] = set()
    removed = 0

    for result in results:
        url_key = (result.url or "").strip().lower()
        if url_key in seen_urls:
            removed += 1
            continue

        duplicate_index = next(
            (i for i, existing in enumerate(kept) if _same_story(existing, result)),
            None,
        )
        if duplicate_index is None:
            kept.append(result)
            seen_urls.add(url_key)
            continue

        existing = kept[duplicate_index]
        # Prefer the more information-rich source. If lengths are close, keep
        # the first result to avoid changing Tavily's natural ordering.
        if len(result.content or "") > len(existing.content or "") * 1.10:
            kept[duplicate_index] = result
            seen_urls.add(url_key)
        removed += 1

    return kept, removed


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _source_credibility_score(result: SearchResult) -> float:
    """Score source type without equating small organisations with low quality."""
    domain = _domain(result.url)
    text = f"{result.title} {result.content}".lower()

    if not domain:
        return 10.0
    if domain in _HIGH_CREDIBILITY_DOMAINS or domain.endswith(".gov.in") or domain.endswith(".gov"):
        return 100.0
    if domain.endswith(".edu") or domain.endswith(".ac.in") or domain.endswith(".ac.uk"):
        return 95.0

    research_signals = (
        "university", "research institute", "research centre", "research center",
        "museum", "archive", "journal", "peer reviewed", "peer-reviewed",
        "archaeological survey", "institute of", "academy",
    )
    if any(signal in domain or signal in text for signal in research_signals):
        return 82.0

    if domain in _MAJOR_NEWS_DOMAINS:
        return 78.0

    # Genuine independent sources get a middle score rather than rejection.
    if any(word in text for word in ("historian", "archaeologist", "scholar", "researcher")):
        return 62.0

    if any(hint in domain or hint in text for hint in _COMMERCIAL_HINTS):
        return 25.0

    return 52.0


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _derive_date_from_text(result: SearchResult) -> datetime | None:
    """Best-effort date extraction when Tavily has no publication date."""
    text = f"{result.title} {result.content}"
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if match:
        try:
            return datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            return None
    return None


def _freshness_score(result: SearchResult) -> float:
    published = _parse_datetime(getattr(result, "published_date", None))
    published = published or _derive_date_from_text(result)
    if published is None:
        # Tavily is already queried with days=1, so missing metadata should not
        # be treated as ancient; give it a cautious middle score.
        return 62.0

    age_hours = max(0.0, (datetime.now(timezone.utc) - published).total_seconds() / 3600)
    if age_hours <= 12:
        return 100.0
    if age_hours <= 24:
        return 88.0
    if age_hours <= 48:
        return 68.0
    if age_hours <= 72:
        return 45.0
    if age_hours <= 168:
        return 20.0
    return 5.0


def _relevance_score(result: SearchResult, category_name: str) -> float:
    text = f"{result.title} {result.content}".lower()
    score = 0.0

    matched_groups = 0
    for terms in _RELEVANCE_GROUPS.values():
        if any(term in text for term in terms):
            matched_groups += 1
            score += 15.0

    # Category alignment gets a small bonus, while global research signals
    # remain dominant so an excellent article can surface across categories.
    category_terms = [t for t in _normalize_text(category_name).split() if len(t) > 3]
    score += min(10.0, sum(3.0 for t in category_terms if t in text))

    if any(term in text for term in ("study", "research", "excavation", "discovery",
                                     "paper", "journal", "conservation", "survey")):
        score += 10.0

    if any(term in text for term in _REJECT_TERMS):
        score -= 45.0
    if any(term in text for term in _GENERIC_TRAVEL_TERMS):
        score -= 35.0

    # More than three strong research/topic groups is a useful signal, but
    # cap the deterministic score to keep one verbose snippet from dominating.
    score += min(10.0, matched_groups * 2.0)
    return max(0.0, min(100.0, score))


def _quality_gate(result: SearchResult, category_name: str) -> bool:
    text = f"{result.title} {result.url} {result.content}".lower()
    if any(term in text for term in _REJECT_TERMS):
        return False
    if any(term in text for term in _GENERIC_TRAVEL_TERMS):
        return False
    if _relevance_score(result, category_name) < 20:
        return False
    if _source_credibility_score(result) < 22:
        return False
    return True


def _topic_for_result(result: SearchResult) -> str:
    text = f"{result.title} {result.content}".lower()
    priority = (
        ("Archaeology", _RELEVANCE_GROUPS["archaeology"]),
        ("Temple Conservation", _RELEVANCE_GROUPS["temples"]),
        ("Manuscripts", _RELEVANCE_GROUPS["manuscripts"]),
        ("Ancient Science", _RELEVANCE_GROUPS["ancient_science"]),
        ("Ayurveda", ("ayurveda",)),
        ("Astronomy", ("astronomy", "jyotisha")),
        ("Museums", _RELEVANCE_GROUPS["museums"]),
        ("Indian Philosophy", _RELEVANCE_GROUPS["philosophy"]),
        ("Ancient Education", ("ancient education", "gurukul", "gurukula")),
        ("Culture", _RELEVANCE_GROUPS["culture"]),
    )
    for topic, terms in priority:
        if any(term in text for term in terms):
            return topic
    return "Other Research"



def _merge_related_sources(results: list[SearchResult]) -> list[SearchResult]:
    """Merge same-event coverage so the AI sees complementary facts before dedupe."""
    merged: list[SearchResult] = []
    for result in results:
        match_index = next(
            (i for i, existing in enumerate(merged) if _same_story(existing, result)), None
        )
        if match_index is None:
            merged.append(result)
            continue

        existing = merged[match_index]
        existing_credibility = _source_credibility_score(existing)
        new_credibility = _source_credibility_score(result)
        if new_credibility > existing_credibility:
            primary, secondary = result, existing
        else:
            primary, secondary = existing, result

        primary_content = (primary.content or '').strip()
        secondary_content = (secondary.content or '').strip()
        combined = primary_content
        if secondary_content and secondary_content not in primary_content:
            combined = f"{primary_content}\nAdditional source coverage:\n{secondary_content}"

        merged[match_index] = SearchResult(
            title=primary.title,
            url=primary.url,
            content=combined[:3500],
            published_date=getattr(primary, "published_date", None),
        )
    return merged


def _research_value_heuristic(result: SearchResult) -> float:
    """Deterministic fallback research-value estimate used before AI review."""
    text = f"{result.title} {result.content}".lower()
    high_value = (
        "archaeological discovery", "archaeological discoveries", "excavation",
        "newly discovered", "newly digitized", "digitized manuscript", "digitised manuscript",
        "newly translated", "translated sanskrit", "sanskrit text", "inscription", "epigraphy",
        "temple restoration", "temple conservation", "asi", "ancient astronomy",
        "ancient mathematics", "ayurveda research", "museum acquisition", "archive discovery",
        "heritage initiative", "heritage project", "government heritage", "research paper",
        "peer-reviewed", "journal article", "scholarly publication", "archaeological survey",
        "manuscript", "museum collection",
    )
    low_value = (
        "festival wishes", "happy diwali", "happy holi", "motivational", "travel guide",
        "tourist guide", "tour package", "promotion", "discount", "opinion",
        "spirituality tips", "religious marketing", "sponsored", "affiliate", "commercial",
    )
    score = 35.0
    score += min(45.0, sum(8.0 for term in high_value if term in text))
    score -= min(35.0, sum(12.0 for term in low_value if term in text))
    return max(0.0, min(100.0, score))


def _deterministic_score(result: SearchResult, category_name: str) -> dict[str, float | str]:
    freshness = _freshness_score(result)
    relevance = _relevance_score(result, category_name)
    credibility = _source_credibility_score(result)
    research_value = _research_value_heuristic(result)
    base = (
        freshness * 0.22
        + relevance * 0.30
        + credibility * 0.23
        + research_value * 0.25
    )
    return {
        "freshness": freshness,
        "relevance": relevance,
        "credibility": credibility,
        "research_value": research_value,
        "base": base,
        "topic": _topic_for_result(result),
    }


def _apply_editorial_reviews(
    candidates: list[tuple[SearchResult, object]],
    ai_client: GroqClient,
    category,
) -> list[tuple[SearchResult, object, EditorialReview]]:
    reviewed = []
    for result, score_data in candidates[:_EDITORIAL_SHORTLIST_PER_CATEGORY]:
        try:
            review = evaluate_article_with_groq(ai_client, category, result)
        except Exception as exc:  # noqa: BLE001
            log.warning("Editorial evaluation exception for '%s': %s", result.title, exc)
            review = EditorialReview(None, 0.0, 0.0, "Editorial evaluation unavailable.")
        reviewed.append((result, score_data, review))
    return reviewed


def _finalize_ranked_results(
    reviewed_by_category: list[tuple[object, list[tuple[SearchResult, object, EditorialReview]]]],
) -> tuple[dict[str, list[SearchResult]], dict[str, EditorialReview], int]:
    """Global ranking with AI confidence, research value and topic diversity."""
    pool = []
    for category, reviewed in reviewed_by_category:
        for result, scores, review in reviewed:
            # A successful REJECT is a hard editorial veto. Review failures are
            # neutral so a temporary Groq problem cannot empty the digest.
            if review.decision is False:
                continue
            # A KEEP with very low confidence is normally rejected because the model
            # is explicitly signalling that the evidence is too weak.
            if review.decision is True and review.confidence < 35.0:
                continue
            scores = dict(scores)
            scores["ai_confidence"] = review.confidence if review.decision is not None else 50.0
            scores["ai_research_value"] = review.research_value if review.decision is not None else scores["research_value"]
            scores["editorial_reason"] = review.reason
            pool.append((category, result, scores, review))

    pool.sort(
        key=lambda item: (
            float(item[2]["base"]) * 0.55
            + float(item[2]["ai_confidence"]) * 0.15
            + float(item[2]["ai_research_value"]) * 0.30,
            float(item[2]["freshness"]),
            float(item[2]["credibility"]),
        ),
        reverse=True,
    )

    selected = []
    category_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}

    for diversity_pass in (True, False):
        for category, result, scores, review in pool:
            if len(selected) >= _FINAL_MAX_ARTICLES:
                break
            cat_name = category.name
            topic = str(scores["topic"])
            if category_counts.get(cat_name, 0) >= _MAX_PER_CATEGORY:
                continue
            if any(existing_result.url == result.url for _, existing_result, _, _ in selected):
                continue
            if diversity_pass and topic_counts.get(topic, 0) > 0:
                continue

            diversity_score = max(0.0, 100.0 - topic_counts.get(topic, 0) * 35.0)
            final_score = (
                float(scores["freshness"]) * 0.14
                + float(scores["relevance"]) * 0.18
                + float(scores["credibility"]) * 0.16
                + float(scores["research_value"]) * 0.16
                + float(scores["ai_confidence"]) * 0.16
                + float(scores["ai_research_value"]) * 0.15
                + diversity_score * 0.05
            )
            scores["final"] = final_score
            selected.append((category, result, scores, review))
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    selected.sort(key=lambda item: float(item[2]["final"]), reverse=True)

    by_category = {category.name: [] for category in CATEGORIES}
    reviews_by_url: dict[str, EditorialReview] = {}
    for category, result, _, review in selected:
        by_category.setdefault(category.name, []).append(result)
        reviews_by_url[result.url] = review

    return by_category, reviews_by_url, len(selected)


def _build_snippet_fallback(results: list[SearchResult]) -> str:
    if not results:
        return "⚠️ AI summary unavailable and no raw sources to fall back on for this category."

    blocks = []
    for result in results:
        snippet = (result.content or "").strip()
        if len(snippet) > 200:
            snippet = snippet[:200].rsplit(" ", 1)[0] + "…"
        blocks.append(
            f"<b>• {result.title}</b>\n{snippet}\n🔗 <a href=\"{result.url}\">Source</a>"
        )

    return "⚠️ AI summary unavailable — showing raw snippets:\n\n" + "\n\n".join(blocks)


def generate_digest(
    tavily_client: TavilyClient, ai_client: GroqClient
) -> tuple[list[CategoryDigest], dict[str, int]]:
    """Search, score, review, diversify, then summarize using existing interfaces."""
    digests: list[CategoryDigest] = []
    stats = {
        "found": 0,
        "skipped_empty": 0,
        "duplicates_removed": 0,
        "quality_rejected": 0,
        "after_filtering": 0,
        "editorial_rejected": 0,
        "editorial_failures": 0,
        "low_confidence_rejected": 0,
        "ai_research_value": 0,
        "final_selected": 0,
        "ai_summaries_generated": 0,
        "fallback_summaries_used": 0,
        "memory_suppressed": 0,
    }

    reviewed_by_category = []

    for category in CATEGORIES:
        digest = CategoryDigest(category=category)
        try:
            raw_results = run_tavily_search(tavily_client, category)
            stats["found"] += len(raw_results)

            meaningful, skipped_empty = _filter_meaningful_results(raw_results, category.name)
            stats["skipped_empty"] += skipped_empty

            merged_results = _merge_related_sources(meaningful)
            deduped, duplicates_removed = _dedupe_results(merged_results)
            stats["duplicates_removed"] += duplicates_removed

            quality_results = []
            for result in deduped:
                try:
                    if _quality_gate(result, category.name):
                        quality_results.append(result)
                    else:
                        stats["quality_rejected"] += 1
                except Exception as exc:  # noqa: BLE001
                    stats["quality_rejected"] += 1
                    log.warning("Scoring failed for '%s': %s", result.title, exc)

            stats["after_filtering"] += len(quality_results)

            scored = []
            for result in quality_results:
                try:
                    scored.append((result, _deterministic_score(result, category.name)))
                except Exception as exc:  # noqa: BLE001
                    log.warning("Ranking failed for '%s': %s", result.title, exc)

            scored.sort(
                key=lambda item: (
                    float(item[1]["base"]),
                    float(item[1]["freshness"]),
                    float(item[1]["credibility"]),
                ),
                reverse=True,
            )

            reviewed = _apply_editorial_reviews(scored, ai_client, category)
            stats["editorial_rejected"] += sum(1 for _, _, review in reviewed if review.decision is False)
            stats["editorial_failures"] += sum(1 for _, _, review in reviewed if review.decision is None)
            stats["low_confidence_rejected"] += sum(
                1 for _, _, review in reviewed
                if review.decision is True and review.confidence < 35.0
            )
            stats["ai_research_value"] += sum(
                int(round(review.research_value)) for _, _, review in reviewed
            )
            reviewed_by_category.append((category, reviewed))
        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected failure processing '%s': %s", category.name, exc)
            reviewed_by_category.append((category, []))

    selected_by_category, selected_reviews, selected_count = _finalize_ranked_results(reviewed_by_category)

    # Historical memory is applied after editorial ranking so only genuinely
    # publishable candidates are compared against prior published reports.
    memory = _prune_memory(_load_memory())
    selected_by_category, historical_context, memory_suppressed = _apply_historical_memory(
        selected_by_category, memory
    )
    stats["memory_suppressed"] = memory_suppressed
    stats["final_selected"] = sum(len(items) for items in selected_by_category.values())

    for category in CATEGORIES:
        digest = CategoryDigest(category=category)
        digest.results = selected_by_category.get(category.name, [])

        try:
            summary = summarize_with_gemini(
                ai_client,
                category,
                digest.results,
                selected_reviews,
                historical_context,
            )
            if summary.startswith(_AI_FAILURE_MARKER):
                stats["fallback_summaries_used"] += 1
                summary = _build_snippet_fallback(digest.results)
            else:
                stats["ai_summaries_generated"] += 1
            digest.summary_html = summary
        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected summarization failure for '%s': %s", category.name, exc)
            digest.error = str(exc)
            digest.summary_html = _build_snippet_fallback(digest.results)

        digests.append(digest)

    _record_memory(memory, selected_by_category, historical_context)
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
    log.info("After quality filtering: %d", stats["after_filtering"])
    log.info("Quality rejected: %d", stats["quality_rejected"])
    log.info("Duplicates removed: %d", stats["duplicates_removed"])
    log.info("Editorial rejected: %d", stats["editorial_rejected"])
    log.info("Editorial review failures: %d", stats["editorial_failures"])
    log.info("Low-confidence editorial rejects: %d", stats["low_confidence_rejected"])
    log.info("Final selected: %d", stats["final_selected"])
    log.info("Memory-suppressed near-duplicates: %d", stats["memory_suppressed"])
    log.info("Skipped empty articles: %d", stats["skipped_empty"])
    log.info("AI summaries generated: %d", stats["ai_summaries_generated"])
    log.info("Fallback summaries used: %d", stats["fallback_summaries_used"])

    full_message = format_full_message(digests)
    log.info("Digest assembled (%d characters). Sending to Telegram...", len(full_message))

    try:
        deliver_digest(settings.telegram_bot_token, settings.telegram_chat_id, full_message)
        log.info("Telegram message sent successfully")
    except Exception as exc:  # noqa: BLE001
        log.error("Telegram delivery failed: %s", exc)

    log.info("Run complete.")


if __name__ == "__main__":
    main()
