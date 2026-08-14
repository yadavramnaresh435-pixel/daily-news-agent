"""Production static publisher for hinduresearch.com — Phase 6.1B.

The daily-news-agent remains the content producer and hinduresearch.com remains
the static consumer. This module extends the existing Phase 6.0 Git publisher;
it does not create a second publishing mechanism.

Publishing flow:
    selected structured records
        -> permanent data/articles.json
        -> data/latest.json
        -> data/manifest.json
        -> data/archive/YYYY.json
        -> news/articles/YYYY/MM/slug.html
        -> news/index.html
        -> feed/rss.xml
        -> sitemap-news.xml
        -> one Git commit + push

AI never renders HTML. HTML is rendered from templates stored in the website
repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from xml.sax.saxutils import escape as xml_escape

from config.constants import (
    Category,
    WEBSITE_DEFAULT_ARCHIVE_DIR,
    WEBSITE_DEFAULT_ARTICLE_DIR,
    WEBSITE_DEFAULT_BASE_URL,
    WEBSITE_DEFAULT_BRANCH,
    WEBSITE_DEFAULT_COMMIT_EMAIL,
    WEBSITE_DEFAULT_COMMIT_NAME,
    WEBSITE_DEFAULT_DATA_PATH,
    WEBSITE_DEFAULT_FEED_PATH,
    WEBSITE_DEFAULT_LATEST_PATH,
    WEBSITE_DEFAULT_MANIFEST_PATH,
    WEBSITE_DEFAULT_NEWS_INDEX_PATH,
    WEBSITE_DEFAULT_NEWS_SITEMAP_PATH,
    WEBSITE_DEFAULT_RSS_LIMIT,
    WEBSITE_DEFAULT_SITEMAP_LIMIT,
    WEBSITE_DEFAULT_LATEST_LIMIT,
    WEBSITE_SAME_STORY_SIMILARITY,
)
from services.ai_service import EditorialReview, WebsiteArticle
from services.tavily_service import SearchResult
from services.website_renderer import (
    article_date,
    article_path,
    ensure_slug,
    render_article,
    render_index,
)
from utils.logger import get_logger

log = get_logger()
GENERATED_MARKER = "data-generated-by=\"hinduresearch-news-agent\""
SCHEMA_VERSION = "6.1B"


def _article_id(url: str) -> str:
    return hashlib.sha256((url or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def _source_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:  # noqa: BLE001
        return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 10000) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
        return max(minimum, min(maximum, value))
    except (TypeError, ValueError):
        return default


def _slugify(value: str, article_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)[:96].strip("-") or "article"
    suffix = re.sub(r"[^a-z0-9]", "", article_id.lower())[:8]
    return f"{slug}-{suffix}" if suffix else slug


def build_website_record(
    category: Category,
    result: SearchResult,
    review: Optional[EditorialReview],
    content: Optional[WebsiteArticle],
    run_date: str,
) -> dict[str, Any]:
    """Assemble one clean, extensible website article record.

    The public Phase 6.0 fields remain intact. Additional deterministic fields
    support stable URLs, source dates, SEO rendering and future migrations.
    """
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

    article_id = _article_id(result.url)
    source_published_at = getattr(result, "published_date", None)
    english_payload = {
        "title": title,
        "summary": summary,
        "key_takeaways": list(key_takeaways),
        "why_this_matters": why_this_matters,
        "research_hook": research_hook,
    }
    translations = dict(getattr(content, "translations", {}) or {}) if content is not None else {}
    translations.setdefault("en", english_payload)
    return {
        "id": article_id,
        "slug": _slugify(title, article_id),
        "title": title,
        "summary": summary,
        "category": category.name,
        "date": run_date,
        "published_at": run_date,
        "source_published_at": source_published_at,
        "source": _source_domain(result.url),
        "source_url": result.url,
        "key_takeaways": key_takeaways,
        "why_this_matters": why_this_matters,
        "research_hook": research_hook,
        "editorial_confidence": int(round(review.confidence)) if review is not None else None,
        "schema_version": SCHEMA_VERSION,
        "language": "en",
        "image_url": None,
        # Additive multilingual extension; all original Phase 6.0 fields remain intact.
        "translations": translations,
    }


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
    """Merge selected articles without deleting permanent history.

    Existing first_published/slug values are preserved on updates so article
    URLs remain permanent even if the AI later rewrites a headline.
    """
    merged = list(existing)
    added = 0
    updated = 0

    for new in reversed(new_articles):
        match_index = next((i for i, old in enumerate(merged) if _same_story(old, new)), None)
        entry = dict(new)
        if match_index is not None:
            old = merged[match_index]
            entry["first_published"] = old.get("first_published") or old.get("published_at") or old.get("date")
            entry["slug"] = old.get("slug") or entry.get("slug") or _slugify(str(entry.get("title")), str(entry.get("id")))
            entry["updated_at"] = _now_iso()
            updated += 1
            merged.pop(match_index)
        else:
            entry["first_published"] = entry.get("first_published") or entry.get("published_at") or entry.get("date")
            entry["slug"] = entry.get("slug") or _slugify(str(entry.get("title")), str(entry.get("id")))
            entry["updated_at"] = entry.get("updated_at") or entry.get("first_published")
            added += 1
        merged.insert(0, entry)

    return merged, added, updated


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read %s: %s", path, exc)
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _read_template(work_dir: str, template_name: str) -> str:
    path = Path(work_dir) / "templates" / template_name
    if not path.exists():
        raise FileNotFoundError(f"Required website template missing: {path}")
    return path.read_text(encoding="utf-8")


def _is_generated_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:12000]
        return GENERATED_MARKER in sample
    except OSError:
        return False


def _write_generated_text(path: Path, content: str, *, allow_existing_generated: bool = True) -> bool:
    """Write only absent or previously-agent-generated files.

    Returns False rather than overwriting an unmanaged existing website page.
    """
    if path.exists() and not (allow_existing_generated and _is_generated_file(path)):
        log.warning("Refusing to overwrite unmanaged website file: %s", path)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)
    return True


def _render_article_pages(work_dir: str, articles: list[dict[str, Any]], base_url: str) -> int:
    template = _read_template(work_dir, "article-template.html")
    rendered = 0
    for article in articles:
        path = Path(work_dir) / article_path(article).lstrip("/")
        content = render_article(template, article, base_url)
        if _write_generated_text(path, content):
            rendered += 1
    return rendered


def _render_news_index(work_dir: str, articles: list[dict[str, Any]], base_url: str, generated_at: str, latest_limit: int) -> bool:
    template = _read_template(work_dir, "news-index-template.html")
    years: dict[int, int] = {}
    for article in articles:
        years[article_date(article).year] = years.get(article_date(article).year, 0) + 1
    html = render_index(template, articles[:latest_limit], sorted(years.items()), base_url, generated_at)
    return _write_generated_text(Path(work_dir) / "news/index.html", html)


def _build_rss(articles: list[dict[str, Any]], base_url: str, limit: int) -> str:
    items = []
    for article in articles[:limit]:
        url = base_url.rstrip("/") + article_path(article)
        published = article_date(article).astimezone(timezone.utc)
        items.append(
            "<item>"
            f"<title>{xml_escape(str(article.get('title', '')))}</title>"
            f"<link>{xml_escape(url)}</link>"
            f"<guid isPermaLink=\"true\">{xml_escape(url)}</guid>"
            f"<description>{xml_escape(str(article.get('summary', '')))}</description>"
            f"<category>{xml_escape(str(article.get('category', 'Research')))}</category>"
            f"<pubDate>{published.strftime('%a, %d %b %Y %H:%M:%S GMT')}</pubDate>"
            "</item>"
        )
    now = datetime.now(timezone.utc)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0">\n<channel>\n'
        f"<title>Hindu Research Portal — Research News</title>\n"
        f"<link>{xml_escape(base_url.rstrip('/') + '/news/')}</link>\n"
        f"<description>Latest research, archaeology, manuscripts, heritage and cultural news from Hindu Research Portal.</description>\n"
        f"<lastBuildDate>{now.strftime('%a, %d %b %Y %H:%M:%S GMT')}</lastBuildDate>\n"
        + "\n".join(items)
        + "\n</channel>\n</rss>\n"
    )


def _build_news_sitemap(articles: list[dict[str, Any]], base_url: str, limit: int) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)
    candidates = [a for a in articles if article_date(a).astimezone(timezone.utc) >= cutoff]
    candidates = candidates[:limit]
    urls = []
    for article in candidates:
        url = base_url.rstrip("/") + article_path(article)
        published = article_date(article).astimezone(timezone.utc)
        urls.append(
            "<url>"
            f"<loc>{xml_escape(url)}</loc>"
            "<news:news>"
            "<news:publication>"
            "<news:name>Hindu Research Portal</news:name>"
            "<news:language>en</news:language>"
            "</news:publication>"
            f"<news:publication_date>{published.isoformat()}</news:publication_date>"
            f"<news:title>{xml_escape(str(article.get('title', '')))}</news:title>"
            "</news:news>"
            "</url>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
        'xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">\n'
        + "\n".join(urls)
        + "\n</urlset>\n"
    )


def _build_manifest(articles: list[dict[str, Any]], base_url: str, generated_at: str, latest_limit: int) -> dict[str, Any]:
    year_counts: dict[str, int] = {}
    for article in articles:
        year = str(article_date(article).year)
        year_counts[year] = year_counts.get(year, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "site": base_url.rstrip("/"),
        "total_articles": len(articles),
        "latest_count": min(len(articles), latest_limit),
        "years": dict(sorted(year_counts.items(), reverse=True)),
        "paths": {
            "permanent_archive": "/data/articles.json",
            "latest": "/data/latest.json",
            "manifest": "/data/manifest.json",
            "year_archive_pattern": "/data/archive/YYYY.json",
            "article_pattern": "/news/articles/YYYY/MM/slug.html",
            "news_index": "/news/",
            "rss": "/feed/rss.xml",
            "news_sitemap": "/sitemap-news.xml",
        },
    }


def _publish_data_artifacts(
    work_dir: str,
    articles: list[dict[str, Any]],
    base_url: str,
    generated_at: str,
    latest_limit: int,
) -> None:
    # Phase 6.0 permanent archive stays the canonical all-article JSON store.
    _write_json(
        Path(work_dir) / "data/articles.json",
        {"schema_version": SCHEMA_VERSION, "updated_at": generated_at, "count": len(articles), "articles": articles},
    )
    _write_json(
        Path(work_dir) / "data/latest.json",
        {"schema_version": SCHEMA_VERSION, "updated_at": generated_at, "count": min(len(articles), latest_limit), "articles": articles[:latest_limit]},
    )

    by_year: dict[str, list[dict[str, Any]]] = {}
    for article in articles:
        year = str(article_date(article).year)
        by_year.setdefault(year, []).append(article)
    archive_dir = Path(work_dir) / WEBSITE_DEFAULT_ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    for year, year_articles in by_year.items():
        _write_json(
            archive_dir / f"{year}.json",
            {"schema_version": SCHEMA_VERSION, "year": int(year), "updated_at": generated_at, "count": len(year_articles), "articles": year_articles},
        )

    _write_json(Path(work_dir) / "data/manifest.json", _build_manifest(articles, base_url, generated_at, latest_limit))


def _run_git(args: list[str], cwd: str, redact: Optional[list[str]] = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        stderr = result.stderr.strip()[:1000]
        for secret in redact or []:
            if secret:
                stderr = stderr.replace(secret, "***")
        raise RuntimeError(f"git {args[0]} failed: {stderr}")
    return result.stdout.strip()


def publish_to_website(articles: list[dict[str, Any]]) -> None:
    """Extend Phase 6.0 publishing into the full Phase 6.1B static pipeline.

    Best-effort/non-fatal behavior is retained exactly: website problems are
    logged and returned so Telegram remains the successful primary workflow.
    """
    if not articles:
        log.info("Website publish skipped: no selected articles today.")
        return

    repo = os.environ.get("WEBSITE_REPO", "").strip()
    token = os.environ.get("WEBSITE_REPO_TOKEN", "").strip()
    if not repo or not token:
        log.info("Website publish skipped: WEBSITE_REPO / WEBSITE_REPO_TOKEN not configured.")
        return

    branch = _safe_env("WEBSITE_REPO_BRANCH", WEBSITE_DEFAULT_BRANCH)
    data_path = _safe_env("WEBSITE_DATA_PATH", WEBSITE_DEFAULT_DATA_PATH)
    base_url = _safe_env("WEBSITE_BASE_URL", WEBSITE_DEFAULT_BASE_URL)
    commit_name = _safe_env("WEBSITE_COMMIT_NAME", WEBSITE_DEFAULT_COMMIT_NAME)
    commit_email = _safe_env("WEBSITE_COMMIT_EMAIL", WEBSITE_DEFAULT_COMMIT_EMAIL)
    latest_limit = _int_env("WEBSITE_LATEST_LIMIT", WEBSITE_DEFAULT_LATEST_LIMIT, 10, 500)
    rss_limit = _int_env("WEBSITE_RSS_LIMIT", WEBSITE_DEFAULT_RSS_LIMIT, 10, 100)
    sitemap_limit = _int_env("WEBSITE_SITEMAP_LIMIT", WEBSITE_DEFAULT_SITEMAP_LIMIT, 10, 1000)

    work_dir = tempfile.mkdtemp(prefix="hinduresearch-site-")
    clone_url = f"https://x-access-token:{token}@github.com/{repo}.git"
    generated_at = _now_iso()

    try:
        _run_git(
            ["clone", "--depth", "1", "--branch", branch, clone_url, work_dir],
            cwd=tempfile.gettempdir(),
            redact=[token, clone_url],
        )
        _run_git(["config", "user.name", commit_name], cwd=work_dir)
        _run_git(["config", "user.email", commit_email], cwd=work_dir)

        # Keep the configured Phase 6.0 data path compatible. All new generated
        # artifacts use their fixed public paths under data/ and news/.
        target_path = Path(work_dir) / data_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        existing_payload = _read_json(target_path, {"articles": []})
        if not isinstance(existing_payload, dict):
            existing_payload = {"articles": []}
        existing_articles = existing_payload.get("articles", [])
        if not isinstance(existing_articles, list):
            existing_articles = []

        merged_articles, added, updated = merge_articles(existing_articles, articles)

        # Ensure all records have stable slugs before rendering. Existing records
        # without a slug are upgraded once and then preserved permanently.
        for article in merged_articles:
            article["slug"] = article.get("slug") or ensure_slug(article)
            article.setdefault("schema_version", SCHEMA_VERSION)

        _publish_data_artifacts(work_dir, merged_articles, base_url, generated_at, latest_limit)
        # If Phase 6.0 was configured to a non-default data path, preserve that
        # compatibility path too. The canonical generated archive remains data/articles.json.
        if data_path != WEBSITE_DEFAULT_DATA_PATH:
            _write_json(target_path, {"schema_version": SCHEMA_VERSION, "updated_at": generated_at, "count": len(merged_articles), "articles": merged_articles})

        rendered_articles = _render_article_pages(work_dir, merged_articles, base_url)
        rendered_index = _render_news_index(work_dir, merged_articles, base_url, generated_at, latest_limit)

        rss_path = Path(work_dir) / WEBSITE_DEFAULT_FEED_PATH
        if rss_path.exists() and not _is_generated_file(rss_path):
            log.warning("Refusing to overwrite unmanaged RSS file: %s", rss_path)
        else:
            rss_path.parent.mkdir(parents=True, exist_ok=True)
            rss_content = _build_rss(merged_articles, base_url, rss_limit)
            rss_path.write_text(rss_content.replace('<rss version="2.0">', '<rss version="2.0" data-generated-by="hinduresearch-news-agent">', 1), encoding="utf-8")

        sitemap_path = Path(work_dir) / WEBSITE_DEFAULT_NEWS_SITEMAP_PATH
        if sitemap_path.exists() and not _is_generated_file(sitemap_path):
            log.warning("Refusing to overwrite unmanaged sitemap: %s", sitemap_path)
        else:
            sitemap_path.parent.mkdir(parents=True, exist_ok=True)
            sitemap_content = _build_news_sitemap(merged_articles, base_url, sitemap_limit)
            sitemap_path.write_text(sitemap_content.replace('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:news="http://www.google.com/schemas/sitemap-news/0.9" data-generated-by="hinduresearch-news-agent">', 1), encoding="utf-8")

        # Stage only files owned by this publisher. Existing unrelated site work
        # is never included in the commit.
        stage_paths = [
            data_path,
            "data/articles.json",
            "data/latest.json",
            "data/manifest.json",
            "data/archive",
            "news/index.html",
            "news/articles",
            "feed/rss.xml",
            "sitemap-news.xml",
        ]
        _run_git(["add", "--", *stage_paths], cwd=work_dir)
        status = _run_git(["status", "--porcelain"], cwd=work_dir)
        if not status:
            log.info("Website publish: no changes to commit.")
            return

        commit_message = f"chore: auto-publish research news ({added} new, {updated} updated)"
        _run_git(["commit", "-m", commit_message], cwd=work_dir)
        _run_git(["push", "origin", f"HEAD:{branch}"], cwd=work_dir, redact=[token, clone_url])
        log.info(
            "Website publish succeeded: %d new, %d updated, %d article pages rendered, index=%s, total=%d pushed to %s/%s.",
            added,
            updated,
            rendered_articles,
            rendered_index,
            len(merged_articles),
            repo,
            branch,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Website publish failed unexpectedly: %s", exc)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
