"""Пошук value: порівняння наших ймовірностей з коефіцієнтами букмекерів."""
from __future__ import annotations

from typing import Any

# Опис ринків: (ринок, сторона) → людська назва
SELECTIONS = {
    ("h2h", "home"): "П1",
    ("h2h", "draw"): "Нічия",
    ("h2h", "away"): "П2",
    ("totals", "over"): "ТБ {line}",
    ("totals", "under"): "ТМ {line}",
}


def fair_probs(prices: dict[str, float]) -> dict[str, float]:
    """Прибирає маржу букмекера (пропорційний метод): {side: price} → {side: prob}."""
    inv = {k: 1.0 / v for k, v in prices.items() if v and v > 1}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()} if total else {}


def market_probs(odds: dict[str, Any]) -> dict[str, float]:
    """Справедливі ймовірності ринку (за середніми коефіцієнтами)."""
    out: dict[str, float] = {}
    if "h2h" in odds:
        out.update(fair_probs({k: odds["h2h"][k]["avg"] for k in ("home", "draw", "away")}))
    if "totals" in odds:
        out.update(fair_probs({k: odds["totals"][k]["avg"] for k in ("over", "under")}))
    return out


def margin(odds_side: dict[str, Any], keys: tuple[str, ...]) -> float:
    return sum(1 / odds_side[k]["avg"] for k in keys) - 1


def blend(model: dict[str, float], market: dict[str, float], market_weight: float) -> dict[str, float]:
    """Зважене поєднання моделі та ринку (ринок — сильний орієнтир)."""
    if not model:
        return dict(market)
    out = {}
    for k, pm in market.items():
        out[k] = market_weight * pm + (1 - market_weight) * model.get(k, pm)
    # нормалізуємо групи
    for group in (("home", "draw", "away"), ("over", "under")):
        s = sum(out.get(g, 0) for g in group)
        if s:
            for g in group:
                if g in out:
                    out[g] /= s
    return out


def edge(prob: float, price: float) -> float:
    """Очікуваний прибуток на 1 юніт: p·k − 1."""
    return prob * price - 1.0


def kelly_stake(prob: float, price: float, fraction: float = 0.25, cap: float = 0.03) -> float:
    """Частковий Келлі як частка банку, обмежена зверху."""
    b = price - 1.0
    if b <= 0:
        return 0.0
    f = (b * prob - (1 - prob)) / b
    return max(0.0, min(cap, f * fraction))


def candidates(event: dict[str, Any], probs: dict[str, float]) -> list[dict[str, Any]]:
    """Усі варіанти ставок на подію з edge за найкращими коефіцієнтами."""
    out = []
    odds = event["odds"]
    for (mkt, side), label in SELECTIONS.items():
        if mkt not in odds or side not in probs:
            continue
        o = odds[mkt][side]
        p = probs[side]
        out.append({
            "market": mkt, "side": side,
            "label": label.format(line=odds[mkt].get("line", "")),
            "price": o["best"], "book": o["book"], "avg_price": o["avg"],
            "prob": round(p, 4), "edge": round(edge(p, o["best"]), 4),
        })
    return sorted(out, key=lambda c: c["edge"], reverse=True)
