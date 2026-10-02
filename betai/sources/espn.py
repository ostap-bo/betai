"""ESPN (публічний API без ключа) — розклад і результати матчів.

Використовується для:
- розкладу ліг, яких немає в The Odds API (УПЛ, товариські матчі збірних);
- розрахунку результатів ставок без витрати кредитів The Odds API.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from ..utils import http_get, match_team, parse_dt, utcnow

log = logging.getLogger("betai.espn")
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard"


def fetch_scoreboard(code: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Матчі ліги за період: [{id, commence_time, home, away, completed, home_score, away_score}]."""
    dates = f"{start:%Y%m%d}-{end:%Y%m%d}"
    data = http_get(BASE.format(code=code), params={"dates": dates, "limit": 200}, retries=2).json()
    out = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in teams or "away" not in teams:
            continue
        status = (ev.get("status") or comp.get("status") or {}).get("type", {})

        def score(side: str) -> int | None:
            try:
                return int(teams[side].get("score"))
            except (TypeError, ValueError):
                return None

        out.append({
            "id": f"espn-{ev['id']}",
            "commence_time": parse_dt(ev["date"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home": teams["home"]["team"].get("displayName") or teams["home"]["team"].get("name"),
            "away": teams["away"]["team"].get("displayName") or teams["away"]["team"].get("name"),
            "completed": bool(status.get("completed")),
            "state": status.get("state"),
            "home_score": score("home"),
            "away_score": score("away"),
        })
    return out


def upcoming(code: str, league: dict[str, Any], lookahead_hours: int) -> list[dict[str, Any]]:
    """Майбутні матчі як події без коефіцієнтів (для AI-прогнозу)."""
    now = utcnow()
    rows = fetch_scoreboard(code, now, now + timedelta(hours=lookahead_hours))
    events = []
    for r in rows:
        kick = parse_dt(r["commence_time"])
        if r["state"] != "pre" or not (now < kick <= now + timedelta(hours=lookahead_hours)):
            continue
        events.append({
            "id": r["id"], "league": league["id"], "league_name": league["name"],
            "commence_time": r["commence_time"], "home": r["home"], "away": r["away"],
            "odds": {}, "bookmakers": 0, "espn": code,
        })
    log.info("ESPN [%s]: %d майбутніх матчів", code, len(events))
    return events


def results_for(picks: list[dict[str, Any]], code: str) -> dict[str, dict[str, Any]]:
    """Шукає завершені матчі для ставок: {event_id: {home_score, away_score}}."""
    if not picks:
        return {}
    times = [parse_dt(p["commence_time"]) for p in picks]
    rows = fetch_scoreboard(code, min(times) - timedelta(days=1), max(times) + timedelta(days=1))
    done = [r for r in rows if r["completed"] and r["home_score"] is not None]
    out: dict[str, dict[str, Any]] = {}
    for p in picks:
        kick = parse_dt(p["commence_time"])
        for r in done:
            if abs((parse_dt(r["commence_time"]) - kick).total_seconds()) > 36 * 3600:
                continue
            if match_team(p["home"], [r["home"]]) and match_team(p["away"], [r["away"]]):
                out[p["event_id"]] = {"home_score": r["home_score"], "away_score": r["away_score"]}
                break
    return out
