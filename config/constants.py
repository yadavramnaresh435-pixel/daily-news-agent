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
