"""
Tavily search service.

Wraps all interaction with the Tavily API: querying a category and
normalizing the raw response into a clean list of `SearchResult` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from tavily import TavilyClient

from config.constants import Category, TAVILY_MAX_RESULTS, TAVILY_SEARCH_DAYS
from utils.logger import get_logger

log = get_logger()


@dataclass
class SearchResult:
    title: str
    url: str
    content: str


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
