"""Трекінг рекомендацій: розрахунок результатів, ROI, CLV."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .utils import parse_dt, utcnow


def pick_id(event_id: str, market: str, side: str) -> str:
    return f"{event_id}:{market}:{side}"


def outcome(pick: dict[str, Any], hs: int, as_: int) -> str:
    """won / lost / void для ставки за фінальним рахунком."""
    side, mkt = pick["side"], pick["market"]
    if mkt == "h2h":
        actual = "home" if hs > as_ else "away" if as_ > hs else "draw"
        return "won" if actual == side else "lost"
    if mkt == "totals":
        total, line = hs + as_, float(pick.get("line", 2.5))
        if total == line:
            return "void"
        return "won" if (total > line) == (side == "over") else "lost"
    return "void"


def settle(picks: list[dict[str, Any]], scores: dict[str, dict[str, Any]]) -> int:
    """Розраховує відкриті ставки за результатами. Повертає кількість розрахованих."""
    n = 0
    for p in picks:
        if p["status"] != "pending" or p["event_id"] not in scores:
            continue
        s = scores[p["event_id"]]
        if s.get("home_score") is None or s.get("away_score") is None:
            continue
        p["status"] = outcome(p, s["home_score"], s["away_score"])
        p["score"] = f"{s['home_score']}-{s['away_score']}"
        p["settled_at"] = utcnow().isoformat()
        p["profit"] = round(p["stake"] * (p["price"] - 1), 3) if p["status"] == "won" else \
            -p["stake"] if p["status"] == "lost" else 0.0
        n += 1
    return n


def update_clv(picks: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    """Оновлює останній відомий коефіцієнт для відкритих ставок (≈ closing line)."""
    by_id = {e["id"]: e for e in events}
    for p in picks:
        ev = by_id.get(p["event_id"])
        if p["status"] != "pending" or not ev:
            continue
        side = ev["odds"].get(p["market"], {}).get(p["side"])
        if side:
            p["latest_price"] = side["avg"]
            p["clv"] = round(p["price"] / side["avg"] - 1, 4)


def stale_pending(picks: list[dict[str, Any]], days: int = 7) -> None:
    """Ставки без результату понад N днів після старту → void (напр., матч перенесено)."""
    now = utcnow()
    for p in picks:
        if p["status"] == "pending" and (now - parse_dt(p["commence_time"])).days > days:
            p["status"], p["profit"] = "void", 0.0


def stats(picks: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [p for p in picks if p["status"] in ("won", "lost")]
    staked = sum(p["stake"] for p in settled)
    profit = sum(p.get("profit", 0) for p in settled)
    won = sum(p["status"] == "won" for p in settled)
    clvs = [p["clv"] for p in picks if p.get("clv") is not None]

    curve, running = [], 0.0
    for p in sorted(settled, key=lambda x: x["commence_time"]):
        running += p["profit"]
        curve.append({"date": p["commence_time"][:10], "profit": round(running, 3)})

    def breakdown(key: str) -> list[dict[str, Any]]:
        agg: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "staked": 0.0, "profit": 0.0, "won": 0})
        for p in settled:
            a = agg[p.get(key) or "—"]
            a["n"] += 1
            a["staked"] += p["stake"]
            a["profit"] += p["profit"]
            a["won"] += p["status"] == "won"
        return [{"name": k, "n": int(v["n"]), "won": int(v["won"]), "profit": round(v["profit"], 2),
                 "roi": round(v["profit"] / v["staked"], 4) if v["staked"] else 0.0}
                for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["n"])]

    return {
        "total": len(picks),
        "pending": sum(p["status"] == "pending" for p in picks),
        "settled": len(settled),
        "won": won,
        "lost": len(settled) - won,
        "hit_rate": round(won / len(settled), 4) if settled else None,
        "staked": round(staked, 2),
        "profit": round(profit, 2),
        "roi": round(profit / staked, 4) if staked else None,
        "avg_odds": round(sum(p["price"] for p in settled) / len(settled), 2) if settled else None,
        "avg_clv": round(sum(clvs) / len(clvs), 4) if clvs else None,
        "curve": curve,
        "by_market": breakdown("label_market"),
        "by_league": breakdown("league_name"),
    }
