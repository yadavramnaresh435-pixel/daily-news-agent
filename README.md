# daily-news-agent

An agent that gives top daily news on your Telegram after researching the internet and filtering the top quality content with sources and categories before sending it to you 😀 makes you feel like a boss.

It fetches fresh news across four research niches using **Tavily**, summarizes each result into a factual bullet + content-writing hook using an AI model via the **OpenRouter** REST API, and delivers a formatted daily digest to a **Telegram** chat. It's designed to run as a scheduled GitHub Actions job, but also runs locally via a `.env` file.

## Architecture

The project is organized as a small modular pipeline: `config` supplies settings, `services` do the actual work, `utils` provide shared low-level helpers, and `main.py` wires them together in order.

```
daily-news-agent/
├── config/
│   ├── constants.py      # Static config: categories, model defaults, limits, AI prompt
│   └── settings.py       # Runtime config: reads & validates env vars / secrets
├── services/
│   ├── tavily_service.py    # Talks to Tavily: runs the search, returns SearchResult objects
│   ├── ai_service.py        # Talks to OpenRouter: summarizes results, assembles the final message
│   └── telegram_service.py  # Talks to Telegram: splits + sends the digest
├── utils/
│   ├── logger.py          # Shared logging setup (one place, consistent format)
│   └── helpers.py         # Small generic helpers (e.g. date formatting)
├── output/                 # Reserved for future use (currently empty)
├── .github/workflows/
│   └── daily_news.yml      # Scheduled GitHub Actions run
├── main.py                 # Orchestrator only — no business logic
├── requirements.txt
└── README.md
```

### How a run flows

1. `main.py` loads settings (`config.settings.load_settings`) and constructs the Tavily client and the OpenRouter client (`services.ai_service.OpenRouterClient`).
2. For each category in `config.constants.CATEGORIES`, it calls `services.tavily_service.run_tavily_search`, then `services.ai_service.summarize_with_gemini`.
3. `services.ai_service.format_full_message` assembles all category digests into one HTML-formatted message.
4. `services.telegram_service.deliver_digest` splits the message (if needed) and sends it to Telegram.

Each service module owns exactly one external integration, so, for example, swapping the search provider or the AI model only touches one file.

> **Note on naming:** the AI summarization function is still called `summarize_with_gemini` even though the provider is now OpenRouter. This is intentional — it keeps the AI service's public interface stable so no other module needed to change when the provider was swapped.

### AI provider: OpenRouter

The agent talks to OpenRouter's REST API (`/chat/completions`) via a small reusable `OpenRouterClient` in `services/ai_service.py`. It handles auth, a configurable request timeout, and up to 3 retries with backoff on transient failures. If every retry fails, the agent falls back to a plain list of source links for that category (never Gemini, never a hard crash) so the digest still contains the sources.

### Configuration

All fixed/static values (research categories, OpenRouter defaults, Tavily's result/day limits, Telegram's chunk limit, the AI system prompt) live in `config/constants.py` — no magic numbers are scattered through the code.

All runtime secrets are read from environment variables (locally via a `.env` file, or injected directly by GitHub Actions) and validated up front in `config/settings.py`:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TAVILY_API_KEY`
- `OPENROUTER_API_KEY`

Three additional values are optional and can be overridden via environment variables; if unset, they fall back to the defaults in `config/constants.py`:

- `AI_PROVIDER` (default: `OPENROUTER`)
- `MODEL_NAME` (default: `google/gemini-2.5-flash`, requested via OpenRouter)
- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)

### Logging

`utils/logger.py` configures logging once and hands every module the same logger via `get_logger()`, so log output stays consistent across the whole project.

## Running locally

```bash
pip install -r requirements.txt
# create a .env file with at least the four required variables listed above
python main.py
```

## Running on GitHub Actions

The workflow in `.github/workflows/daily_news.yml` runs on a daily schedule (and can be triggered manually via `workflow_dispatch`). It installs dependencies and runs `python main.py`, pulling required secrets from the repository's GitHub secrets.

**Action needed:** the workflow file itself was intentionally left unmodified as part of this change. Its `env:` block still references `GEMINI_API_KEY`, which the code no longer reads. Add an `OPENROUTER_API_KEY` repository secret and update that `env:` block (`GEMINI_API_KEY` → `OPENROUTER_API_KEY`) before the next scheduled run, or the agent will exit early with a missing-environment-variable error.

## Behavior

Search behavior, the AI prompt, the summary output format, and the Telegram message format are unchanged. Only the AI provider backing the summarization step changed, from the Gemini SDK to OpenRouter's REST API.
