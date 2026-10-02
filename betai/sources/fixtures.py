"""API-Football (https://www.api-football.com) — травми, H2H, склади.

Необов'язкове джерело. Безкоштовний план: 100 запитів/день.
Якщо ключа немає — платформа працює без нього (H2H береться з історії,
травми шукає Claude через web search).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from ..utils import http_get, match_team

log = logging.getLogger("betai.fixtures")
BASE = "https://v3.football.api-sports.io"


class ApiFootball:
    def __init__(self, api_key: str):
        self.headers = {"x-apisports-key": api_key}
        self.calls = 0

    def _get(self, path: str, **params: Any) -> list[dict[str, Any]]:
        self.calls += 1
        data = http_get(f"{BASE}/{path}", params=params, headers=self.headers).json()
        errors = data.get("errors")
        if errors:
            log.warning("API-Football /%s: %s", path, errors)
            return []
        return data.get("response", [])

    def fixtures(self, league_id: int, days: int = 3) -> list[dict[str, Any]]:
        today = date.today()
        season = today.year if today.month >= 7 else today.year - 1
        return self._get("fixtures", league=league_id, season=season,
                         **{"from": today.isoformat(), "to": (today + timedelta(days=days)).isoformat()})

    def injuries(self, fixture_id: int) -> list[dict[str, Any]]:
        rows = self._get("injuries", fixture=fixture_id)
        return [{
            "team": r["team"]["name"],
            "player": r["player"]["name"],
            "type": r["player"].get("type"),
            "reason": r["player"].get("reason"),
        } for r in rows]

    def h2h(self, home_id: int, away_id: int, last: int = 10) -> list[dict[str, Any]]:
        rows = self._get("fixtures/headtohead", h2h=f"{home_id}-{away_id}", last=last)
        out = []
        for r in rows:
            g = r.get("goals", {})
            if g.get("home") is None:
                continue
            out.append({
                "date": r["fixture"]["date"][:10],
                "home": r["teams"]["home"]["name"],
                "away": r["teams"]["away"]["name"],
                "score": f"{g['home']}-{g['away']}",
                "competition": r["league"]["name"],
            })
        return out


def find_fixture(fixtures: list[dict[str, Any]], home: str, away: str) -> dict[str, Any] | None:
    """Шукає фікстуру API-Football, що відповідає події з The Odds API."""
    homes = {f["teams"]["home"]["name"]: f for f in fixtures}
    hit = match_team(home, homes)
    if hit:
        fx = homes[hit]
        if match_team(away, [fx["teams"]["away"]["name"]]):
            return fx
    return None


def enrich_with_api_football(client: ApiFootball, league_id: int,
                             events: list[dict[str, Any]], fixtures_cache: dict[int, list]) -> None:
    """Додає до подій травми та H2H (in-place)."""
    if league_id not in fixtures_cache:
        try:
            fixtures_cache[league_id] = client.fixtures(league_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Фікстури API-Football недоступні: %s", exc)
            fixtures_cache[league_id] = []
    for ev in events:
        fx = find_fixture(fixtures_cache[league_id], ev["home"], ev["away"])
        if not fx:
            continue
        try:
            ev["injuries"] = client.injuries(fx["fixture"]["id"])
            ev["h2h_all_comps"] = client.h2h(fx["teams"]["home"]["id"], fx["teams"]["away"]["id"])
            ev["venue"] = (fx["fixture"].get("venue") or {}).get("name")
            ev["referee"] = fx["fixture"].get("referee")
        except Exception as exc:  # noqa: BLE001
            log.warning("API-Football для %s — %s: %s", ev["home"], ev["away"], exc)
