"""The Odds API (https://the-odds-api.com) — актуальні коефіцієнти та результати.

Безкоштовний план: 500 кредитів/місяць.
Вартість запиту /odds = кількість ринків × кількість регіонів.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from statistics import mean
from typing import Any

from ..utils import http_get, parse_dt, utcnow

log = logging.getLogger("betai.odds")
BASE = "https://api.the-odds-api.com/v4"


def _summarise(prices: list[tuple[float, str]]) -> dict[str, Any] | None:
    """Найкращий коефіцієнт, де він є, і середній по ринку."""
    if not prices:
        return None
    best_price, best_book = max(prices, key=lambda p: p[0])
    return {"best": round(best_price, 3), "book": best_book,
            "avg": round(mean(p for p, _ in prices), 3), "n": len(prices)}


def normalise_event(ev: dict[str, Any], league: dict[str, Any], totals_line: float = 2.5) -> dict[str, Any]:
    """Перетворює подію The Odds API у внутрішній формат."""
    home, away = ev["home_team"], ev["away_team"]
    h2h: dict[str, list] = {"home": [], "draw": [], "away": []}
    tot: dict[str, list] = {"over": [], "under": []}
    for bm in ev.get("bookmakers", []):
        book = bm.get("title") or bm.get("key")
        for mkt in bm.get("markets", []):
            if mkt["key"] == "h2h":
                for o in mkt["outcomes"]:
                    side = "home" if o["name"] == home else "away" if o["name"] == away else "draw"
                    h2h[side].append((float(o["price"]), book))
            elif mkt["key"] == "totals":
                for o in mkt["outcomes"]:
                    if abs(float(o.get("point", 0)) - totals_line) < 1e-6:
                        tot[o["name"].lower()].append((float(o["price"]), book))

    odds: dict[str, Any] = {}
    if all(h2h.values()):
        odds["h2h"] = {k: _summarise(v) for k, v in h2h.items()}
    if all(tot.values()):
        odds["totals"] = {"line": totals_line, **{k: _summarise(v) for k, v in tot.items()}}

    return {
        "id": ev["id"],
        "league": league.get("id", league["odds_key"]),
        "league_name": league["name"],
        "commence_time": ev["commence_time"],
        "home": home,
        "away": away,
        "odds": odds,
        "bookmakers": len(ev.get("bookmakers", [])),
    }


def fetch_odds(api_key: str, league: dict[str, Any], *, regions: str = "eu",
               markets: list[str] | None = None, lookahead_hours: int = 72,
               totals_line: float = 2.5) -> tuple[list[dict[str, Any]], int | None]:
    """Повертає (події, залишок кредитів). Порожня відповідь кредитів не витрачає."""
    markets = markets or ["h2h", "totals"]
    now = utcnow()
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": (now + timedelta(hours=lookahead_hours)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    resp = http_get(f"{BASE}/sports/{league['odds_key']}/odds", params=params)
    remaining = _int(resp.headers.get("x-requests-remaining"))
    log.info("The Odds API [%s]: %d подій, залишок кредитів: %s",
             league["odds_key"], len(resp.json()), remaining)
    events = [normalise_event(e, league, totals_line) for e in resp.json()]
    return [e for e in events if "h2h" in e["odds"] and parse_dt(e["commence_time"]) > now], remaining


def _int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def fetch_scores(api_key: str, sport_key: str, days_from: int = 3) -> dict[str, dict[str, Any]]:
    """Завершені матчі за останні N днів: {event_id: {home_score, away_score}}."""
    resp = http_get(f"{BASE}/sports/{sport_key}/scores",
                    params={"apiKey": api_key, "daysFrom": days_from, "dateFormat": "iso"})
    out: dict[str, dict[str, Any]] = {}
    for ev in resp.json():
        if not ev.get("completed") or not ev.get("scores"):
            continue
        scores = {s["name"]: int(s["score"]) for s in ev["scores"]}
        out[ev["id"]] = {
            "home": ev["home_team"], "away": ev["away_team"],
            "home_score": scores.get(ev["home_team"]), "away_score": scores.get(ev["away_team"]),
        }
    return out
