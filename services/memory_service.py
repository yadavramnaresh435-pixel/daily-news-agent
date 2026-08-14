"""Lightweight, failure-safe persistent research memory for the daily agent."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from services.tavily_service import SearchResult
from utils.logger import get_logger

log = get_logger()

_MEMORY_PATH = Path(__file__).resolve().parent.parent / "memory" / "research_memory.json"
_MAX_STORIES = 500
_RECENT_DAYS = 30
_DUPLICATE_SIMILARITY = 0.88


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _tokens(value: str) -> set[str]:
    return {x for x in _normalize(value).split() if len(x) > 2}


def _fingerprint(result: SearchResult) -> str:
    text = _normalize(f"{result.title} {result.content[:500]}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_memory() -> dict[str, Any]:
    """Load memory; an unavailable/corrupt memory never stops a run."""
    try:
        _MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not _MEMORY_PATH.exists():
            return {"stories": [], "topics": []}
        data = json.loads(_MEMORY_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Memory root must be an object.")
        data.setdefault("stories", [])
        data.setdefault("topics", [])
        return data
    except Exception as exc:
        log.warning("Research memory unavailable: %s", exc)
        return {"stories": [], "topics": []}


def _same_historical_story(result: SearchResult, old: dict[str, Any]) -> bool:
    url = (result.url or "").strip().lower()
    old_url = str(old.get("url", "")).strip().lower()
    if url and old_url and url == old_url:
        return True

    title = _normalize(result.title)
    old_title = _normalize(str(old.get("title", "")))
    if title and old_title:
        similarity = SequenceMatcher(None, title, old_title).ratio()
        overlap = len(_tokens(title) & _tokens(old_title)) / max(1, min(len(_tokens(title)), len(_tokens(old_title))))
        if similarity >= _DUPLICATE_SIMILARITY or overlap >= 0.86:
            return True
    return False


def filter_historical_duplicates(
    results: list[SearchResult], memory: dict[str, Any]
) -> tuple[list[SearchResult], int]:
    """Deterministically remove only clear near-duplicates from recent memory."""
    kept: list[SearchResult] = []
    removed = 0
    stories = memory.get("stories", [])
    cutoff = datetime.now(timezone.utc).timestamp() - (_RECENT_DAYS * 86400)

    recent = []
    for item in stories if isinstance(stories, list) else []:
        try:
            ts = datetime.fromisoformat(str(item.get("reported_at", "")).replace("Z", "+00:00")).timestamp()
            if ts >= cutoff:
                recent.append(item)
        except Exception:
            recent.append(item)

    for result in results:
        duplicate = next((old for old in recent if _same_historical_story(result, old)), None)
        if duplicate is not None:
            removed += 1
            continue
        kept.append(result)
    return kept, removed


def update_memory(memory: dict[str, Any], digests: list[Any]) -> None:
    """Persist selected stories and lightweight topic continuity after a successful digest."""
    stories = memory.setdefault("stories", [])
    topics = memory.setdefault("topics", [])
    now = datetime.now(timezone.utc).isoformat()

    for digest in digests:
        for result in getattr(digest, "results", []) or []:
            topic = _derive_topic(result)
            entry = {
                "fingerprint": _fingerprint(result),
                "title": result.title,
                "url": result.url,
                "reported_at": now,
                "published_date": getattr(result, "published_date", None),
                "topic": topic,
            }
            if not any(str(x.get("url", "")).strip().lower() == result.url.strip().lower() for x in stories if isinstance(x, dict)):
                stories.append(entry)

            topic_entry = next((x for x in topics if isinstance(x, dict) and x.get("name") == topic), None)
            if topic_entry is None:
                topics.append(
                    {
                        "name": topic,
                        "last_reported_at": now,
                        "timeline": [
                            {
                                "date": now,
                                "title": result.title,
                                "url": result.url,
                                "content": (result.content or "")[:300],
                            }
                        ],
                    }
                )
            else:
                topic_entry["last_reported_at"] = now
                topic_entry.setdefault("timeline", [])
                timeline = topic_entry["timeline"]
                timeline.append(
                    {
                        "date": now,
                        "title": result.title,
                        "url": result.url,
                        "content": (result.content or "")[:300],
                    }
                )
                topic_entry["timeline"] = timeline[-20:]

    memory["stories"] = stories[-_MAX_STORIES:]
    memory["topics"] = topics[-100:]
    _MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _MEMORY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _MEMORY_PATH)


def _derive_topic(result: SearchResult) -> str:
    text = _normalize(f"{result.title} {result.content}")
    groups = {
        "Archaeology": ("archaeolog", "excavation", "inscription", "epigraphy"),
        "Temple Conservation": ("temple", "restoration", "conservation", "heritage"),
        "Manuscripts": ("manuscript", "palm leaf", "sanskrit text", "codex"),
        "Ancient Science": ("mathematics", "astronomy", "ayurveda", "metallurgy"),
        "Museums": ("museum", "archive", "collection", "exhibition"),
        "Indian Philosophy": ("vedanta", "upanishad", "nyaya", "mimamsa"),
    }
    for topic, terms in groups.items():
        if any(term in text for term in terms):
            return topic
    return "Other Research"


def build_historical_context(
    results: list[SearchResult], memory: dict[str, Any], limit: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """For each result, surface prior stories memory already has on the same
    topic, so the AI can (optionally) reference genuine continuity instead of
    treating every article as if it appeared out of nowhere.

    Returns a dict keyed by result URL -> a short list of prior story dicts
    (most recent first), each with reported_at/title/topic/content. A lookup
    failure for one result is logged and skipped; it never blocks the rest.
    """
    topics = memory.get("topics", [])
    context: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        try:
            topic = _derive_topic(result)
            topic_entry = next(
                (t for t in topics if isinstance(t, dict) and t.get("name") == topic), None
            )
            if not topic_entry:
                continue
            timeline = topic_entry.get("timeline", [])
            own_url = (result.url or "").strip().lower()
            prior = [
                item
                for item in reversed(timeline)
                if isinstance(item, dict) and str(item.get("url", "")).strip().lower() != own_url
            ][:limit]
            if prior:
                context[result.url] = [
                    {
                        "reported_at": item.get("date", ""),
                        "title": item.get("title", ""),
                        "topic": topic,
                        "content": item.get("content", ""),
                    }
                    for item in prior
                ]
        except Exception as exc:  # noqa: BLE001 - one bad lookup must never block the rest
            log.warning("Historical context lookup failed for '%s': %s", result.title, exc)
    return context


# ==========================================================================
# Permanent published-article audit log (distinct from the 30-day working
# memory above). Best-effort: a failure here must never break an otherwise
# successful run, since it's called unguarded at the end of generate_digest.
# ==========================================================================

_ARCHIVE_PATH = Path(__file__).resolve().parent.parent / "memory" / "published_archive.json"
_MAX_ARCHIVE_ENTRIES = 2000


def archive_selected(digests: list[Any], selected_reviews: dict[str, Any]) -> None:
    """Append today's final, published selection (with its editorial review)
    to a permanent local audit log. Never raises."""
    try:
        _ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict[str, Any]] = []
        if _ARCHIVE_PATH.exists():
            try:
                loaded = json.loads(_ARCHIVE_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing = loaded
            except Exception as exc:  # noqa: BLE001
                log.warning("Published archive unreadable, starting fresh: %s", exc)

        now = datetime.now(timezone.utc).isoformat()
        for digest in digests:
            category = getattr(digest, "category", None)
            category_name = getattr(category, "name", "") if category else ""
            for result in getattr(digest, "results", []) or []:
                review = selected_reviews.get(result.url)
                existing.append(
                    {
                        "title": result.title,
                        "url": result.url,
                        "category": category_name,
                        "archived_at": now,
                        "published_date": getattr(result, "published_date", None),
                        "editorial_confidence": getattr(review, "confidence", None),
                        "editorial_research_value": getattr(review, "research_value", None),
                        "editorial_reason": getattr(review, "reason", None),
                    }
                )

        existing = existing[-_MAX_ARCHIVE_ENTRIES:]
        tmp = _ARCHIVE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, _ARCHIVE_PATH)
    except Exception as exc:  # noqa: BLE001 - archive must never fail the run
        log.warning("Published archive update failed: %s", exc)
