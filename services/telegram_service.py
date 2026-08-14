"""
Telegram delivery service.

Handles splitting an oversized message into chunks that respect
Telegram's character limit, and sending those chunks via the Bot API.
"""

from __future__ import annotations

import time

import requests

from config.constants import TELEGRAM_API_ROOT, TELEGRAM_CHUNK_LIMIT
from utils.logger import get_logger

log = get_logger()


def split_message(text: str, limit: int = TELEGRAM_CHUNK_LIMIT) -> list[str]:
    """
    Split a long message into chunks under Telegram's character limit.
    Splits on paragraph/line boundaries where possible to avoid breaking
    HTML tags (e.g. an unclosed <a href="..."> tag) mid-way.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""

    for line in text.split("\n"):
        # +1 accounts for the newline that will rejoin lines
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current.rstrip())
            current = line
            # Edge case: a single line longer than the limit on its own.
            while len(current) > limit:
                chunks.append(current[:limit])
                current = current[limit:]
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current.rstrip())

    return chunks


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> bool:
    """Send a single message chunk to Telegram. Returns True on success."""
    url = f"{TELEGRAM_API_ROOT}/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, data=payload, timeout=15)
        if response.status_code == 200:
            return True
        log.error(
            "Telegram API error (status %s): %s",
            response.status_code,
            response.text[:500],
        )
        return False
    except requests.RequestException as exc:
        log.error("Telegram request failed: %s", exc)
        return False


def deliver_digest(bot_token: str, chat_id: str, full_text: str) -> bool:
    """Split (if needed) and deliver the digest, logging per-chunk outcomes.

    Returns True only if every chunk was sent successfully. Previously this
    returned None unconditionally and never raised on a failed send, so
    main.py's `try/except` around this call always treated the call as a
    success (it only logs and swallows failures per-chunk) — meaning
    website publishing could fire even after Telegram delivery had actually
    failed. Callers should check the return value rather than relying on
    the absence of an exception.
    """
    chunks = split_message(full_text)
    log.info("Delivering digest in %d chunk(s).", len(chunks))

    all_succeeded = True
    for i, chunk in enumerate(chunks, start=1):
        success = send_telegram_message(bot_token, chat_id, chunk)
        all_succeeded = all_succeeded and success
        status = "sent" if success else "FAILED"
        log.info("Chunk %d/%d %s (%d chars).", i, len(chunks), status, len(chunk))
        if len(chunks) > 1:
            time.sleep(0.5)  # gentle pacing to avoid Telegram rate limits

    return all_succeeded
