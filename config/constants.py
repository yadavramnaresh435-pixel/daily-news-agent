"""
Static configuration for the Hindu Research Daily Intelligence Agent.

Nothing in this module reads the environment or performs I/O — it only
defines the fixed values (categories, model name, limits, prompt text)
that shape the agent's behavior. Runtime/secret configuration (API keys,
tokens) lives in `config.settings` instead.
"""

from __future__ import annotations

from dataclasses import dataclass


# ==========================================================================
# Research categories
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


# ==========================================================================
# Groq (AI summarization) settings
#
# These are default values. AI_PROVIDER, MODEL_NAME, and GROQ_BASE_URL
# can each be overridden at runtime via environment variables of the same
# name (see config/settings.py) — these constants are only the fallbacks
# used when no override is set.
# ==========================================================================

DEFAULT_AI_PROVIDER = "GROQ"
DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GROQ_TIMEOUT_SECONDS = 30
GROQ_MAX_RETRIES = 3
GROQ_RETRY_BACKOFF_SECONDS = 2  # multiplied by attempt number between retries

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
# Tavily (search) settings
# ==========================================================================

TAVILY_MAX_RESULTS = 4
TAVILY_SEARCH_DAYS = 1  # freshness window in days


# ==========================================================================
# Phase 5 — broad discovery queries
#
# main.py's generate_digest() runs a wide discovery pass across several
# focused queries (rather than just the 4 CATEGORIES.query strings), split
# between "news" and "general"/scholarly topics, before any deterministic
# filtering, ranking, or AI review happens. Each tuple is:
#   (category_name, query, topic)
# - category_name must match a CATEGORIES[i].name exactly.
# - topic is passed straight to Tavily's search `topic` parameter — Tavily
#   only accepts "news" or "general".
# The category_name here only affects logging / the initial bucket; the
# real per-article category is re-derived from content afterwards via
# _category_for_result(), so it is not load-bearing.
# ==========================================================================

DISCOVERY_QUERIES: list[tuple[str, str, str]] = [
    # Archaeology & Ancient Discoveries
    ("Archaeology & Ancient Discoveries", "archaeological excavation discovery India ancient site", "news"),
    ("Archaeology & Ancient Discoveries", "ASI Archaeological Survey of India excavation report", "news"),
    ("Archaeology & Ancient Discoveries", "ancient inscription epigraphy discovery India", "general"),
    ("Archaeology & Ancient Discoveries", "temple restoration conservation heritage site India", "general"),
    # Vedic & Manuscript Research
    ("Vedic & Manuscript Research", "Sanskrit manuscript digitization translation research", "news"),
    ("Vedic & Manuscript Research", "Indology Vedic studies research paper published", "news"),
    ("Vedic & Manuscript Research", "palm-leaf manuscript critical edition Sanskrit text", "general"),
    ("Vedic & Manuscript Research", "ancient Indian mathematics astronomy Ayurveda research study", "general"),
    # Global Cultural & Temple Events
    ("Global Cultural & Temple Events", "Hindu temple inauguration diaspora cultural event", "news"),
    ("Global Cultural & Temple Events", "museum exhibition Indian heritage archive acquisition", "news"),
    ("Global Cultural & Temple Events", "Hindu cultural heritage milestone global community", "general"),
    ("Global Cultural & Temple Events", "national archives digital archive Indian history collection", "general"),
    # Daily Research Hook / Topic Idea
    ("Daily Research Hook / Topic Idea", "Indian philosophy Vedanta Upanishad scholarly analysis", "news"),
    ("Daily Research Hook / Topic Idea", "Bhagavad Gita research interpretation study", "news"),
    ("Daily Research Hook / Topic Idea", "Indian knowledge systems Gurukul Shastra research", "general"),
    ("Daily Research Hook / Topic Idea", "Nyaya Samkhya Mimamsa Indian philosophy research paper", "general"),
]

# Informational counts only (not read elsewhere in the codebase) — how many
# of the queries above are "news" vs "general"/research topic, kept for
# maintainers tuning DISCOVERY_QUERIES.
DISCOVERY_TARGET_NEWS = sum(1 for _, _, topic in DISCOVERY_QUERIES if topic == "news")
DISCOVERY_TARGET_RESEARCH = sum(1 for _, _, topic in DISCOVERY_QUERIES if topic != "news")

# Caps how many top deterministically-ranked candidates get an expensive
# Groq editorial-review call, after discovery + quality filtering + memory
# suppression. Tune down to cut API usage, up for a more thorough review pass.
DISCOVERY_MAX_CANDIDATES_FOR_AI = 24




# ==========================================================================
# Historical research memory
# ==========================================================================

# Informational only: services/memory_service.py resolves its own path
# (<repo_root>/memory/research_memory.json, via `Path(__file__)`) rather than
# reading this constant, so this value is not load-bearing. It is kept in
# sync with the real location for documentation purposes only.
MEMORY_FILE = "memory/research_memory.json"
MEMORY_RETENTION_DAYS = 90

# ==========================================================================
# Telegram (delivery) settings
# ==========================================================================

TELEGRAM_CHUNK_LIMIT = 4000  # keep a safety margin under Telegram's 4096 cap
TELEGRAM_API_ROOT = "https://api.telegram.org"


# ==========================================================================
# Required environment variables (secrets)
# ==========================================================================

REQUIRED_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TAVILY_API_KEY",
    "GROQ_API_KEY",
]


# ==========================================================================
# Website publishing (hinduresearch.com) settings — Phase 6.0
#
# These are only defaults/fallbacks. The actual repository, token, and
# branch are read from environment variables / GitHub Secrets at runtime
# (see services/website_service.py) and are intentionally NOT defined here,
# so nothing sensitive or environment-specific is hardcoded. Publishing is
# entirely optional: if those secrets aren't set, website publishing is
# skipped and the rest of the run (Telegram, memory) is unaffected.
# ==========================================================================

WEBSITE_DEFAULT_BRANCH = "main"
WEBSITE_DEFAULT_DATA_PATH = "data/articles.json"
WEBSITE_DEFAULT_COMMIT_NAME = "Hindu Research Agent"
WEBSITE_DEFAULT_COMMIT_EMAIL = "actions@users.noreply.github.com"
WEBSITE_SAME_STORY_SIMILARITY = 0.88  # same threshold style as memory_service dedupe
