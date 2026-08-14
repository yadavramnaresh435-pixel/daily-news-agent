"""Static HTML/feed renderer for the Hindu Research Portal news system.

This module deliberately contains no AI or network logic. It reads reusable
HTML templates from the target website repository and fills them with already
validated structured article data.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _text(value: Any) -> str:
    return str(value or "").strip()


def _escape(value: Any) -> str:
    return html.escape(_text(value), quote=True)


def _slugify(value: str, article_id: str = "") -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    value = re.sub(r"-+", "-", value)
    value = value[:96].strip("-") or "article"
    suffix = re.sub(r"[^a-z0-9]", "", article_id.lower())[:8]
    return f"{value}-{suffix}" if suffix else value


def ensure_slug(article: dict[str, Any]) -> str:
    slug = _text(article.get("slug"))
    if slug:
        return slug
    return _slugify(_text(article.get("title")), _text(article.get("id")))


def _iso_date(value: Any) -> str:
    raw = _text(value)
    if not raw:
        return datetime.now(timezone.utc).date().isoformat()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        # Existing Phase 6.0 records commonly use YYYY-MM-DD.
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return raw + "T00:00:00+00:00"
        return datetime.now(timezone.utc).isoformat()


def article_date(article: dict[str, Any]) -> datetime:
    raw = _text(article.get("first_published") or article.get("date"))
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            dt = datetime.now(timezone.utc)
    return dt.replace(tzinfo=dt.tzinfo or timezone.utc)


def article_path(article: dict[str, Any]) -> str:
    dt = article_date(article)
    return f"/news/articles/{dt.year:04d}/{dt.month:02d}/{ensure_slug(article)}.html"


def canonical_url(base_url: str, article: dict[str, Any]) -> str:
    return base_url.rstrip("/") + article_path(article)


def _description(article: dict[str, Any]) -> str:
    summary = _text(article.get("summary"))
    if len(summary) <= 160:
        return summary
    return summary[:157].rsplit(" ", 1)[0] + "..."


def _source_name(article: dict[str, Any]) -> str:
    return _text(article.get("source")) or "Source"


def _takeaways_html(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    clean = [_text(item) for item in items if _text(item)]
    if not clean:
        return ""
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in clean[:3]) + "</ul>"


def _article_json_ld(base_url: str, article: dict[str, Any], image_url: str) -> str:
    canonical = canonical_url(base_url, article)
    published = _iso_date(article.get("first_published") or article.get("date"))
    modified = _iso_date(article.get("updated_at") or article.get("first_published") or article.get("date"))
    data = {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": _text(article.get("title")),
        "description": _description(article),
        "datePublished": published,
        "dateModified": modified,
        "mainEntityOfPage": {"@type": "WebPage", "@id": canonical},
        "author": {"@type": "Organization", "name": "Hindu Research Portal", "url": base_url.rstrip("/")},
        "publisher": {"@type": "Organization", "name": "Hindu Research Portal", "url": base_url.rstrip("/")},
        "image": [image_url],
        "articleSection": _text(article.get("category")),
        "isAccessibleForFree": True,
    }
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _breadcrumb_json_ld(base_url: str, article: dict[str, Any]) -> str:
    canonical = canonical_url(base_url, article)
    data = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "Home", "item": base_url.rstrip("/") + "/"},
            {"@type": "ListItem", "position": 2, "name": "News", "item": base_url.rstrip("/") + "/news/"},
            {"@type": "ListItem", "position": 3, "name": _text(article.get("category")) or "Research", "item": base_url.rstrip("/") + "/news/"},
            {"@type": "ListItem", "position": 4, "name": _text(article.get("title")), "item": canonical},
        ],
    }
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def render_article(template: str, article: dict[str, Any], base_url: str) -> str:
    canonical = canonical_url(base_url, article)
    source_url = _text(article.get("source_url"))
    source_domain = _source_name(article)
    image_url = _text(article.get("image_url")) or (base_url.rstrip("/") + "/images/logo.png")
    published_dt = article_date(article)
    modified_raw = _text(article.get("updated_at"))
    modified_iso = _iso_date(modified_raw or article.get("first_published") or article.get("date"))
    published_iso = _iso_date(article.get("first_published") or article.get("date"))
    published_display = published_dt.strftime("%d %B %Y")
    try:
        modified_dt = datetime.fromisoformat(modified_iso.replace("Z", "+00:00"))
    except ValueError:
        modified_dt = published_dt
    modified_display = modified_dt.strftime("%d %B %Y")

    values = {
        "SEO_TITLE": _escape(_text(article.get("title")) + " | Hindu Research Portal"),
        "META_DESCRIPTION": _escape(_description(article)),
        "CANONICAL_URL": _escape(canonical),
        "OG_IMAGE_URL": _escape(image_url),
        "PUBLISHED_ISO": _escape(published_iso),
        "MODIFIED_ISO": _escape(modified_iso),
        "ARTICLE_JSON_LD": _article_json_ld(base_url, article, image_url),
        "BREADCRUMB_JSON_LD": _breadcrumb_json_ld(base_url, article),
        "TITLE": _escape(article.get("title")),
        "SUMMARY": _escape(article.get("summary")),
        "CATEGORY": _escape(article.get("category")),
        "SOURCE_DOMAIN": _escape(source_domain),
        "SOURCE_URL": _escape(source_url),
        "KEY_TAKEAWAYS_HTML": _takeaways_html(article.get("key_takeaways")),
        "WHY_THIS_MATTERS": _escape(article.get("why_this_matters")),
        "RESEARCH_HOOK": _escape(article.get("research_hook")),
        "PUBLISHED_DISPLAY": _escape(published_display),
        "MODIFIED_DISPLAY": _escape(modified_display),
        "ARTICLE_PATH": _escape(article_path(article)),
    }
    output = template
    for key, value in values.items():
        output = output.replace("{{" + key + "}}", value)
    return output


def render_index(template: str, articles: list[dict[str, Any]], years: list[tuple[int, int]], base_url: str, generated_at: str) -> str:
    cards: list[str] = []
    for article in articles:
        url = canonical_url(base_url, article)
        date_display = article_date(article).strftime("%d %b %Y")
        cards.append(
            '<article class="news-card">'
            f'<div class="news-card-meta"><span>{_escape(article.get("category"))}</span><time datetime="{_escape(_iso_date(article.get("first_published") or article.get("date")))}">{_escape(date_display)}</time></div>'
            f'<h2><a href="{_escape(url)}">{_escape(article.get("title"))}</a></h2>'
            f'<p>{_escape(_description(article))}</p>'
            f'<a class="read-more" href="{_escape(url)}">Read research brief →</a>'
            '</article>'
        )

    archive_links = []
    for year, count in sorted(years, reverse=True):
        archive_links.append(
            f'<a href="{base_url.rstrip("/")}/data/archive/{year}.json">{year} <span>{count}</span></a>'
        )

    values = {
        "GENERATED_AT": _escape(generated_at),
        "TOTAL_ARTICLES": str(sum(count for _, count in years)),
        "LATEST_COUNT": str(len(articles)),
        "ARTICLE_CARDS_HTML": "\n".join(cards),
        "ARCHIVE_LINKS_HTML": "\n".join(archive_links),
    }
    output = template
    for key, value in values.items():
        output = output.replace("{{" + key + "}}", value)
    return output
