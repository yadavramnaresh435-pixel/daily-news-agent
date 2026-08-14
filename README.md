# daily-news-agent

An agent that researches the internet for high-quality Sanatan Dharma, Indian
Knowledge Systems, archaeology, manuscripts, Ayurveda, ancient mathematics,
astronomy and heritage news, filters it down to the strongest, most credible
stories, and delivers a formatted daily digest to a **Telegram** chat. It also
publishes structured per-article JSON to the separate `hinduresearch.com`
website repository. It's designed to run as a scheduled GitHub Actions job,
but also runs locally via a `.env` file.

## Two repositories

This repository (`daily-news-agent`) is the **content producer**: search,
ranking, AI editorial review, memory, Telegram delivery, and generating +
pushing the website's JSON data file.

A separate repository, `hinduresearch.com`, is the **content consumer**: it
only reads the JSON this agent pushes and renders the site (news list,
categories, search, archive). The two repositories are intentionally
independent — this repo never contains website UI code, and the website
repo never calls any API in this repo directly.

## Architecture

```
daily-news-agent/
├── config/
│   ├── constants.py      # Static config: categories, discovery queries, limits, AI prompts
│   └── settings.py       # Runtime config: reads & validates env vars / secrets
├── services/
│   ├── tavily_service.py    # Talks to Tavily: runs searches, returns SearchResult objects
│   ├── ai_service.py        # Talks to Groq: editorial review, summarization, website content
│   ├── memory_service.py    # 30-day working memory + permanent published-article archive
│   ├── telegram_service.py  # Talks to Telegram: splits + sends the digest
│   └── website_service.py   # Builds article JSON and pushes it to the hinduresearch.com repo
├── utils/
│   ├── logger.py          # Shared logging setup (one place, consistent format)
│   └── helpers.py         # Small generic helpers (e.g. date formatting)
├── memory/                 # research_memory.json (working memory) + published_archive.json
├── .github/workflows/
│   └── daily_news.yml      # Scheduled GitHub Actions run
├── main.py                 # Orchestrator: discovery -> filtering -> ranking -> AI review -> delivery -> publish
├── requirements.txt
└── README.md
```

### How a run flows

1. `main.py` loads settings (`config.settings.load_settings`) and constructs
   the `TavilyClient` and `GroqClient`.
2. **Discovery:** for each `(category, query, topic)` in
   `config.constants.DISCOVERY_QUERIES` (a broad set of focused queries, not
   just the four `CATEGORIES.query` strings), it calls
   `services.tavily_service.run_tavily_search`.
3. **Filtering & ranking:** results are checked for meaningful content,
   merged/deduplicated (same-story detection), filtered against the last 30
   days of `memory_service` history, scored deterministically (freshness,
   relevance, source credibility, research-value heuristic), and sorted.
4. **AI editorial review:** only the top-ranked candidates (capped by
   `DISCOVERY_MAX_CANDIDATES_FOR_AI`) get an AI review call
   (`services.ai_service.evaluate_article_with_groq`) so API cost stays
   bounded regardless of how many raw results were discovered.
5. **Final selection:** `main._finalize_ranked_results` combines the
   deterministic score with the AI's confidence/research-value and enforces
   per-category and per-topic diversity limits.
6. **Telegram digest:** `services.ai_service.summarize_with_gemini` (name
   kept for interface stability; the provider is Groq) writes the final
   HTML digest per category, given the selected sources, their editorial
   review, and any relevant historical context from memory.
   `services.ai_service.format_full_message` assembles all category digests
   into one message, and `services.telegram_service.deliver_digest` splits
   and sends it.
7. **Website publishing (Phase 6.0):** for the same selected articles,
   `services.ai_service.generate_website_content` produces structured
   per-article fields (title, summary, key takeaways, why it matters, a
   research hook), `services.website_service.build_website_record` turns
   that into a clean JSON record, and — only after Telegram delivery has
   actually succeeded — `services.website_service.publish_to_website` clones
   the `hinduresearch.com` repo, merges today's articles into its permanent
   archive (newest first, no duplicates), and pushes.
8. `memory_service.update_memory` and `memory_service.archive_selected`
   persist the run's results for future duplicate/continuity checks. Both
   are best-effort and never fail the run.

Website publishing is entirely optional and non-fatal: if it isn't
configured, or fails for any reason, the run still completes successfully —
Telegram delivery is never affected by a website publish failure.

### AI provider: Groq

The agent talks to Groq's OpenAI-compatible `/chat/completions` REST API via
a small reusable `GroqClient` in `services/ai_service.py`. It handles auth, a
configurable request timeout, and up to 3 retries with backoff on transient
failures. If every retry fails, the agent falls back to a plain list of
source links for that category (never a hard crash) so the digest still
contains the sources.

> **Note on naming:** the digest-writing function is still called
> `summarize_with_gemini` even though the provider is Groq. This is
> intentional — it keeps the AI service's public interface stable so no
> other module needs to change if the provider is swapped again.

### Configuration

All fixed/static values (research categories, discovery queries, Groq
defaults, Tavily's result/day limits, Telegram's chunk limit, ranking
weights, the AI system prompts) live in `config/constants.py`.

All runtime secrets are read from environment variables (locally via a
`.env` file, or injected directly by GitHub Actions) and validated up front
in `config/settings.py`:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TAVILY_API_KEY`
- `GROQ_API_KEY`

Optional, with fallbacks in `config/constants.py` if unset:

- `AI_PROVIDER` (default: `GROQ`)
- `MODEL_NAME` (default: `llama-3.3-70b-versatile`)
- `GROQ_BASE_URL` (default: `https://api.groq.com/openai/v1`)

Optional, for Phase 6.0 website publishing (publishing is skipped entirely
if these aren't set — everything else keeps working):

- `WEBSITE_REPO` — target repo as `owner/hinduresearch.com`
- `WEBSITE_REPO_TOKEN` — a PAT with write access to that repo (the default
  `GITHUB_TOKEN` in Actions is scoped to this repo only, so it can't push
  to a separate one)
- `WEBSITE_REPO_BRANCH` (default: `main`)
- `WEBSITE_DATA_PATH` (default: `data/articles.json`)
- `WEBSITE_COMMIT_NAME` / `WEBSITE_COMMIT_EMAIL` (default: a generic bot
  identity)

### Logging

`utils/logger.py` configures logging once and hands every module the same
logger via `get_logger()`, so log output stays consistent across the whole
project.

## Running locally

```bash
pip install -r requirements.txt
# create a .env file with at least the four required variables listed above
python main.py
```

## Running on GitHub Actions

The workflow in `.github/workflows/daily_news.yml` runs on a daily schedule
(and can be triggered manually via `workflow_dispatch`). It installs
dependencies and runs `python main.py`, pulling `TELEGRAM_BOT_TOKEN`,
`TELEGRAM_CHAT_ID`, `TAVILY_API_KEY`, `GROQ_API_KEY`, and the optional
`WEBSITE_*` secrets from the repository's GitHub secrets.
