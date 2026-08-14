"""
Website publisher for hinduresearch.com — Phase 6.0 (Cross Repository Integration).

daily-news-agent (this repo) = Content Producer
hinduresearch.com            = Content Consumer

The two repositories stay fully independent. This module only ever:
1. Builds a clean, extensible JSON record per selected article.
2. Clones a fresh copy of the separate hinduresearch.com repository.
3. Merges today's articles into its permanent JSON archive (newest first,
   no duplicates — a matched story is updated in place instead).
4. Commits and pushes, only if there is something to commit.

Nothing here is required for the existing Telegram pipeline. If the website
secrets below aren't configured, or anything fails at any stage, this module
logs the failure and returns — `publish_to_website` never raises, so it can
never break an otherwise-successful run (see main.py, which only calls this
after Telegram delivery has already succeeded).

Required GitHub Secrets / environment variables (new in Phase 6.0, all
optional — publishing is simply skipped if they're absent):
    WEBSITE_REPO         "owner/hinduresearch.com" — target repository
    WEBSITE_REPO_TOKEN   A PAT with write access to that repository
                          (the default GITHUB_TOKEN in Actions is scoped to
                          the triggering repo only, so it cannot push here)

Optional, with sensible defaults (see config/constants.py):
    WEBSITE_REPO_BRANCH  Target branch (default: "main")
    WEBSITE_DATA_PATH    JSON file path inside that repo (default: "data/articles.json")
    WEBSITE_COMMIT_NAME  Commit author name
    WEBSITE_COMMIT_EMAIL Commit author email
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from config.constants import (
    Category,
    WEBSITE_DEFAULT_BRANCH,
    WEBSITE_DEFAULT_COMMIT_EMAIL,
    WEBSITE_DEFAULT_COMMIT_NAME,
    WEBSITE_DEFAULT_DATA_PATH,
    WEBSITE_SAME_STORY_SIMILARITY,
)
from services.ai_service import EditorialReview, WebsiteArticle
from services.tavily_service import SearchResult
from utils.logger import get_logger

log = get_logger()


# ==========================================================================
# Record building
# ==========================================================================

def _article_id(url: str) -> str:
    return hashlib.sha256((url or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def _source_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:  # noqa: BLE001
        return ""


def build_website_record(
    category: Category,
    result: SearchResult,
    review: Optional[EditorialReview],
    content: Optional[WebsiteArticle],
    run_date: str,
) -> dict[str, Any]:
    """Assemble one clean, extensible website article record."""
    if content is not None:
        title = content.title or result.title
        summary = content.summary
        key_takeaways = content.key_takeaways
        why_this_matters = content.why_this_matters
        research_hook = content.research_hook
    else:
        title = result.title
        summary = (result.content or "")[:400]
        key_takeaways = []
        why_this_matters = ""
        research_hook = ""

    return {
        "id": _article_id(result.url),
        "title": title,
        "summary": summary,
        "category": category.name,
        "date": run_date,
        "source": _source_domain(result.url),
        "source_url": result.url,
        "key_takeaways": key_takeaways,
        "why_this_matters": why_this_matters,
        "research_hook": research_hook,
        "editorial_confidence": int(round(review.confidence)) if review is not None else None,
    }


# ==========================================================================
# Merge: latest-first, permanent archive, intelligent update (no duplicates)
# ==========================================================================

def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _same_story(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("id") and a.get("id") == b.get("id"):
        return True
    url_a = str(a.get("source_url", "")).strip().lower()
    url_b = str(b.get("source_url", "")).strip().lower()
    if url_a and url_b and url_a == url_b:
        return True
    title_a, title_b = _normalize(str(a.get("title", ""))), _normalize(str(b.get("title", "")))
    if not title_a or not title_b:
        return False
    return SequenceMatcher(None, title_a, title_b).ratio() >= WEBSITE_SAME_STORY_SIMILARITY


def merge_articles(
    existing: list[dict[str, Any]], new_articles: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int, int]:
    """Merge today's articles into the permanent archive.

    New stories are prepended (newest first, in the order supplied). A story
    that already exists (same id, URL, or near-identical title) is updated
    in place and moved to the top rather than duplicated; its original
    `first_published` date is preserved. Nothing is ever removed — the
    working-memory (30 day) store is entirely separate from this archive.
    """
    merged = list(existing)
    added = 0
    updated = 0

    # Insert in reverse so the final order matches new_articles' own order
    # (each insert(0, ...) pushes previously-inserted items further down).
    for new in reversed(new_articles):
        match_index = next((i for i, old in enumerate(merged) if _same_story(old, new)), None)
        entry = dict(new)
        if match_index is not None:
            entry["first_published"] = merged[match_index].get("first_published") or merged[match_index].get("date")
            entry["updated_at"] = _now_iso()
            merged.pop(match_index)
            updated += 1
        else:
            entry["first_published"] = entry.get("date")
            added += 1
        merged.insert(0, entry)

    return merged, added, updated


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ==========================================================================
# Git-backed publish to the separate hinduresearch.com repository
# ==========================================================================

def _run_git(args: list[str], cwd: str, redact: Optional[list[str]] = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        stderr = result.stderr.strip()[:500]
        for secret in redact or []:
            if secret:
                stderr = stderr.replace(secret, "***")
        raise RuntimeError(f"git {args[0]} failed: {stderr}")
    return result.stdout.strip()


def publish_to_website(articles: list[dict[str, Any]]) -> None:
    """Publish today's selected articles to the hinduresearch.com repository.

    Best-effort and non-fatal by design: missing configuration or a failure
    at any stage is logged and this function simply returns. It never raises,
    so it can never affect the Telegram delivery that has already succeeded
    by the time this is called, and never crashes the workflow (Phase 6.0
    requirement: "If the push fails, Telegram should still succeed and the
    workflow should finish gracefully").
    """
    if not articles:
        log.info("Website publish skipped: no new articles selected today.")
        return

    repo = os.environ.get("WEBSITE_REPO", "").strip()
    token = os.environ.get("WEBSITE_REPO_TOKEN", "").strip()
    if not repo or not token:
        log.info("Website publish skipped: WEBSITE_REPO / WEBSITE_REPO_TOKEN not configured.")
        return

    branch = os.environ.get("WEBSITE_REPO_BRANCH", WEBSITE_DEFAULT_BRANCH).strip() or WEBSITE_DEFAULT_BRANCH
    data_path = os.environ.get("WEBSITE_DATA_PATH", WEBSITE_DEFAULT_DATA_PATH).strip() or WEBSITE_DEFAULT_DATA_PATH
    commit_name = os.environ.get("WEBSITE_COMMIT_NAME", WEBSITE_DEFAULT_COMMIT_NAME)
    commit_email = os.environ.get("WEBSITE_COMMIT_EMAIL", WEBSITE_DEFAULT_COMMIT_EMAIL)

    work_dir = tempfile.mkdtemp(prefix="hinduresearch-site-")
    clone_url = f"https://x-access-token:{token}@github.com/{repo}.git"

    try:
        try:
            _run_git(
                ["clone", "--depth", "1", "--branch", branch, clone_url, work_dir],
                cwd=tempfile.gettempdir(),
                redact=[token, clone_url],
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Website publish failed: could not clone %s (%s)", repo, exc)
            return

        _run_git(["config", "user.name", commit_name], cwd=work_dir)
        _run_git(["config", "user.email", commit_email], cwd=work_dir)

        target_path = Path(work_dir) / data_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        existing_payload: dict[str, Any] = {"articles": []}
        if target_path.exists():
            try:
                loaded = json.loads(target_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("Website data root must be an object.")
                loaded.setdefault("articles", [])
                existing_payload = loaded
            except Exception as exc:  # noqa: BLE001
                log.warning("Existing website data file unreadable, starting fresh: %s", exc)

        merged_articles, added, updated = merge_articles(existing_payload.get("articles", []), articles)

        payload = {
            "updated_at": _now_iso(),
            "count": len(merged_articles),
            "articles": merged_articles,
        }
        target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        status = _run_git(["status", "--porcelain"], cwd=work_dir)
        if not status:
            log.info("Website publish: no changes to commit.")
            return

        _run_git(["add", data_path], cwd=work_dir)
        commit_message = f"chore: auto-publish research digest ({added} new, {updated} updated)"
        try:
            _run_git(["commit", "-m", commit_message], cwd=work_dir)
        except Exception as exc:  # noqa: BLE001
            log.error("Website publish failed: commit failed (%s)", exc)
            return

        try:
            _run_git(["push", "origin", f"HEAD:{branch}"], cwd=work_dir, redact=[token, clone_url])
        except Exception as exc:  # noqa: BLE001
            log.error("Website publish failed: push to %s/%s failed (%s)", repo, branch, exc)
            return

        log.info(
            "Website publish succeeded: %d new, %d updated (%d total) pushed to %s/%s.",
            added, updated, len(merged_articles), repo, branch,
        )
    except Exception as exc:  # noqa: BLE001 - website publishing must never fail the run
        log.error("Website publish failed unexpectedly: %s", exc)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
