"""Новини з RSS спортивних медіа (без ключів).

Новини фільтруються за назвами команд і передаються Claude як контекст.
Додатково Claude може сам шукати свіжу інформацію через web search.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser

from ..utils import http_get, norm_team

log = logging.getLogger("betai.news")


def fetch_news(feeds: list[str], max_age_hours: int = 72) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    items: list[dict[str, Any]] = []
    for url in feeds:
        try:
            parsed = feedparser.parse(http_get(url, retries=2, timeout=20).content)
        except Exception as exc:  # noqa: BLE001
            log.warning("RSS %s недоступний: %s", url, exc)
            continue
        for e in parsed.entries:
            ts = e.get("published_parsed") or e.get("updated_parsed")
            published = datetime(*ts[:6], tzinfo=timezone.utc) if ts else None
            if published and published < cutoff:
                continue
            summary = re.sub(r"<[^>]+>", " ", e.get("summary", ""))
            items.append({
                "title": e.get("title", "").strip(),
                "summary": re.sub(r"\s+", " ", summary).strip()[:400],
                "link": e.get("link"),
                "published": published.isoformat() if published else None,
                "source": parsed.feed.get("title", url),
            })
    seen: set[str] = set()
    unique = []
    for it in items:  # прибираємо дублікати між джерелами
        key = it["title"].lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(it)
    log.info("Новини: зібрано %d статей з %d джерел", len(unique), len(feeds))
    return unique


def _keywords(team: str) -> list[str]:
    """Ключові слова для пошуку команди в тексті."""
    words = {team.lower(), norm_team(team)}
    # останнє значуще слово ("Manchester United" → "united" надто загальне, тож беремо перше)
    parts = [p for p in re.split(r"\W+", team.lower()) if len(p) > 3 and p not in {"united", "city", "club"}]
    words.update(parts[:1])
    return [w for w in words if len(w) > 2]


def news_for_match(news: list[dict[str, Any]], home: str, away: str, limit: int = 8) -> list[dict[str, Any]]:
    kh, ka = _keywords(home), _keywords(away)
    scored = []
    for item in news:
        text = f"{item['title']} {item['summary']}".lower()
        hit_h = any(k in text for k in kh)
        hit_a = any(k in text for k in ka)
        if hit_h or hit_a:
            scored.append((2 if hit_h and hit_a else 1, item))
    scored.sort(key=lambda x: (x[0], x[1].get("published") or ""), reverse=True)
    return [i for _, i in scored[:limit]]
