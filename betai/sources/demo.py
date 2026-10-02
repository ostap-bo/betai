"""Демо-дані: дозволяють запустити всю платформу без жодного API-ключа.

Генерує синтетичну історію матчів і майбутні події з коефіцієнтами.
Назви навмисно в різних форматах (як у реальних джерелах), щоб перевірити зіставлення.
"""
from __future__ import annotations

import random
import uuid
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..utils import utcnow

# (назва football-data, назва The Odds API, атака, захист)
TEAMS = [
    ("Man City", "Manchester City", 1.55, 0.70), ("Arsenal", "Arsenal", 1.45, 0.65),
    ("Liverpool", "Liverpool", 1.50, 0.75), ("Chelsea", "Chelsea", 1.25, 0.90),
    ("Tottenham", "Tottenham Hotspur", 1.25, 1.05), ("Newcastle", "Newcastle United", 1.20, 0.90),
    ("Aston Villa", "Aston Villa", 1.15, 0.95), ("Man United", "Manchester United", 1.05, 1.00),
    ("Brighton", "Brighton and Hove Albion", 1.10, 1.00), ("West Ham", "West Ham United", 0.95, 1.10),
    ("Crystal Palace", "Crystal Palace", 0.90, 1.00), ("Fulham", "Fulham", 0.95, 1.05),
    ("Brentford", "Brentford", 1.00, 1.10), ("Bournemouth", "AFC Bournemouth", 0.95, 1.10),
    ("Everton", "Everton", 0.80, 1.00), ("Wolves", "Wolverhampton Wanderers", 0.85, 1.15),
    ("Nott'm Forest", "Nottingham Forest", 0.90, 1.00), ("Leeds", "Leeds United", 0.85, 1.20),
    ("Burnley", "Burnley", 0.70, 1.25), ("Sunderland", "Sunderland", 0.75, 1.20),
]
HOME_ADV = 1.25
BASE_GOALS = 1.30


def demo_history(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = utcnow().replace(tzinfo=None) - timedelta(days=700)
    for season in range(2):
        for i, (h, _, ah, dh) in enumerate(TEAMS):
            for j, (a, _, aa, da) in enumerate(TEAMS):
                if i == j:
                    continue
                lam_h = BASE_GOALS * ah * da * HOME_ADV / 1.12
                lam_a = BASE_GOALS * aa * dh / 1.12
                day = season * 350 + rng.integers(0, 280)
                rows.append({"date": start + timedelta(days=int(day)), "home": h, "away": a,
                             "hg": int(rng.poisson(lam_h)), "ag": int(rng.poisson(lam_a))})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _price(p: float, margin: float, rnd: random.Random) -> float:
    return round(max(1.01, 1 / (p * (1 + margin)) * rnd.uniform(0.96, 1.05)), 2)


def demo_events(league: dict[str, Any], n: int = 8, seed: int = 11) -> list[dict[str, Any]]:
    from ..model.poisson import outcome_probs, score_matrix  # локальний імпорт (уникаємо циклу)

    rnd = random.Random(seed)
    teams = TEAMS[:]
    rnd.shuffle(teams)
    events = []
    now = utcnow()
    books = ["Pinnacle", "Bet365", "Unibet", "William Hill", "1xBet", "Betfair"]
    for k in range(n):
        h, a = teams[2 * k], teams[2 * k + 1]
        lam_h = BASE_GOALS * h[2] * a[3] * HOME_ADV / 1.12
        lam_a = BASE_GOALS * a[2] * h[3] / 1.12
        p = outcome_probs(score_matrix(lam_h * rnd.uniform(0.9, 1.1), lam_a * rnd.uniform(0.9, 1.1)))

        def side(prob: float) -> dict[str, Any]:
            prices = sorted(((_price(prob, 0.05, rnd), b) for b in books), reverse=True)
            return {"best": prices[0][0], "book": prices[0][1],
                    "avg": round(sum(x for x, _ in prices) / len(prices), 3), "n": len(prices)}

        kickoff = (now + timedelta(hours=6 + 7 * k)).replace(minute=0, second=0, microsecond=0)
        events.append({
            "id": uuid.UUID(int=rnd.getrandbits(128)).hex,
            "league": league["odds_key"], "league_name": league["name"],
            "commence_time": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home": h[1], "away": a[1],
            "odds": {
                "h2h": {"home": side(p["home"]), "draw": side(p["draw"]), "away": side(p["away"])},
                "totals": {"line": 2.5, "over": side(p["over"]), "under": side(p["under"])},
            },
            "bookmakers": len(books),
            "injuries": [{"team": h[1], "player": "Demo Player", "type": "Missing Fixture",
                          "reason": "Hamstring"}] if k % 3 == 0 else [],
        })
    return events


def demo_news(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ev in events[:4]:
        out.append({
            "title": f"{ev['home']} boss provides team news ahead of {ev['away']} clash",
            "summary": f"{ev['home']} could be without a key midfielder, while {ev['away']} "
                       f"report a clean bill of health. (демо-новина)",
            "link": None, "published": utcnow().isoformat(), "source": "Demo Feed",
        })
    return out
