"""Головний конвеєр: дані → модель → AI → рекомендації → дашборд."""
from __future__ import annotations

import logging
import random
import zlib
from datetime import timedelta
from typing import Any

from .ai.analyst import ClaudeAnalyst
from .config import DOCS_DATA_DIR, Settings
from .model import poisson, value
from .sources import demo, history, news, odds
from .sources.fixtures import ApiFootball, enrich_with_api_football
from .tracker import pick_id, settle, stale_pending, stats, update_clv
from .utils import parse_dt, read_json, utcnow, write_json

log = logging.getLogger("betai.pipeline")

PICKS_PATH = DOCS_DATA_DIR / "picks.json"
LATEST_PATH = DOCS_DATA_DIR / "latest.json"


# ─── 1. Розрахунок минулих ставок ───────────────────────────────────────────
def settle_picks(cfg: Settings, picks: list[dict[str, Any]]) -> None:
    if cfg.demo or not cfg.odds_api_key:
        return
    now = utcnow()
    leagues = {p["league"] for p in picks
               if p["status"] == "pending" and parse_dt(p["commence_time"]) + timedelta(hours=2) < now}
    for lg in leagues:
        try:
            n = settle(picks, odds.fetch_scores(cfg.odds_api_key, lg, days_from=3))
            log.info("Розраховано %d ставок у %s", n, lg)
        except Exception as exc:  # noqa: BLE001
            log.warning("Не вдалося отримати результати %s: %s", lg, exc)
    stale_pending(picks)


# ─── 2. Збір даних та модель ────────────────────────────────────────────────
def collect(cfg: Settings) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Повертає (події з контекстом, новини)."""
    ocfg, mcfg = cfg.section("odds"), cfg.section("model")
    line = float(ocfg.get("totals_line", 2.5))
    all_events: list[dict[str, Any]] = []

    for league in cfg.leagues:
        if cfg.demo:
            events, hist = demo.demo_events(league, seed=zlib.crc32(league["odds_key"].encode()) % 1000), demo.demo_history()
        else:
            if not cfg.odds_api_key:
                raise SystemExit("ODDS_API_KEY не задано. Додайте ключ або запустіть з --demo.")
            try:
                events = odds.fetch_odds(cfg.odds_api_key, league, regions=ocfg.get("regions", "eu"),
                                         markets=ocfg.get("markets"), totals_line=line,
                                         lookahead_hours=int(cfg.raw.get("lookahead_hours", 72)))
            except Exception as exc:  # noqa: BLE001
                log.error("Коефіцієнти для %s недоступні: %s", league["name"], exc)
                continue
            hist = history.load_history(league["history_code"], int(mcfg.get("history_seasons", 3)))

        ratings = poisson.fit_ratings(hist, half_life_days=float(mcfg.get("decay_half_life_days", 240)))
        for ev in events:
            build_context(ev, hist, ratings, mcfg, line)
        all_events.extend(events)
        if cfg.demo:
            break  # у демо достатньо однієї ліги

    if cfg.demo:
        news_items = demo.demo_news(all_events)
    else:
        ncfg = cfg.section("news")
        news_items = news.fetch_news(ncfg.get("feeds", []), int(ncfg.get("max_age_hours", 72)))
    for ev in all_events:
        ev["news"] = news.news_for_match(news_items, ev["home"], ev["away"])
    return all_events, news_items


def build_context(ev: dict[str, Any], hist, ratings: poisson.TeamRatings,
                  mcfg: dict[str, Any], line: float) -> None:
    """Модельні ймовірності, ринкові ймовірності, форма та H2H для події (in-place)."""
    market = value.market_probs(ev["odds"])
    model_probs: dict[str, float] = {}
    xg = ratings.expected_goals(ev["home"], ev["away"])
    if xg:
        m = poisson.score_matrix(*xg, max_goals=int(mcfg.get("max_goals", 10)))
        model_probs = poisson.outcome_probs(m, line)
        ev["expected_goals"] = {"home": round(xg[0], 2), "away": round(xg[1], 2)}
        ev["likely_scores"] = poisson.top_scores(m)
    hname, aname = ratings.resolve(ev["home"]), ratings.resolve(ev["away"])
    if hname and aname:
        ev["form"] = {"home": history.team_form(hist, hname), "away": history.team_form(hist, aname)}
        ev["h2h"] = history.head_to_head(hist, hname, aname)

    ev["model_probs"] = {k: round(v, 4) for k, v in model_probs.items()}
    ev["market_probs"] = {k: round(v, 4) for k, v in market.items()}
    ev["base_probs"] = {k: round(v, 4) for k, v in
                        value.blend(model_probs, market, float(mcfg.get("market_weight", 0.6))).items()}
    ev["candidates"] = value.candidates(ev, ev["base_probs"])
    ev["totals_line"] = line


# ─── 3. AI-аналіз ───────────────────────────────────────────────────────────
def ai_context(ev: dict[str, Any]) -> dict[str, Any]:
    keep = ("league_name", "commence_time", "home", "away", "odds", "expected_goals", "likely_scores",
            "form", "h2h", "h2h_all_comps", "injuries", "venue", "referee", "news", "totals_line")
    ctx = {k: ev[k] for k in keep if ev.get(k)}
    ctx["model_probabilities"] = ev["model_probs"]
    ctx["market_fair_probabilities"] = ev["market_probs"]
    ctx["base_probabilities"] = ev["base_probs"]
    ctx["value_candidates_by_base"] = ev["candidates"][:3]
    return ctx


def mock_ai(ev: dict[str, Any]) -> dict[str, Any]:
    """Демо-заміна Claude: детермінований «аналіз» без API."""
    rnd = random.Random(ev["id"])
    probs = {k: v * rnd.uniform(0.97, 1.06) for k, v in ev["base_probs"].items()}
    from .ai.analyst import sanitize_probs
    probs = sanitize_probs(probs, ev["base_probs"])
    best = max(value.candidates(ev, probs), key=lambda c: c["edge"])
    return {
        "summary": f"ДЕМО-аналіз: {ev['home']} проти {ev['away']}. Очікувані голи "
                   f"{ev.get('expected_goals', {}).get('home', '?')}–{ev.get('expected_goals', {}).get('away', '?')}.",
        "key_factors": ["Форма команд за останні 6 матчів", "Сила атаки/захисту з моделі Пуассона",
                        "Демо-новина про склад"],
        "absences": [{"team": i["team"], "player": i["player"], "status": "out", "impact": "medium"}
                     for i in ev.get("injuries", [])],
        "probabilities": probs,
        "recommendation": {"market": best["market"], "side": best["side"],
                           "confidence": rnd.randint(4, 8),
                           "reasoning": "Демо-режим: рекомендація згенерована без Claude API."},
        "risks": ["Це демо-дані — не використовуйте для реальних ставок"],
        "sources": [],
    }


def run_ai(cfg: Settings, events: list[dict[str, Any]]) -> dict[str, Any]:
    acfg = cfg.section("ai")
    usage: dict[str, Any] = {"mode": "off"}
    now = utcnow()
    horizon = now + timedelta(hours=int(cfg.raw.get("lookahead_hours", 72)))
    pool = [e for e in events if now < parse_dt(e["commence_time"]) <= horizon]
    # пріоритет — матчі з найбільшим базовим edge
    pool.sort(key=lambda e: e["candidates"][0]["edge"] if e["candidates"] else -1, reverse=True)
    selected = pool[: int(acfg.get("max_matches", 12))]

    # API-Football: травми та H2H тільки для відібраних матчів (економія ліміту)
    if cfg.api_football_key and not cfg.demo:
        client, cache = ApiFootball(cfg.api_football_key), {}
        ids = {lg["odds_key"]: lg.get("api_football_id") for lg in cfg.leagues}
        for lg_key, lg_id in ids.items():
            if lg_id:
                enrich_with_api_football(client, lg_id, [e for e in selected if e["league"] == lg_key], cache)
        log.info("API-Football: %d запитів", client.calls)

    analyst = None
    if cfg.demo:
        usage["mode"] = "demo"
    elif cfg.ai_enabled:
        analyst = ClaudeAnalyst(cfg.anthropic_api_key, acfg)
        usage["mode"] = "claude"
        usage["model"] = analyst.model

    for ev in selected:
        try:
            if cfg.demo:
                ev["ai"] = mock_ai(ev)
            elif analyst:
                log.info("Claude аналізує: %s — %s", ev["home"], ev["away"])
                ev["ai"] = analyst.analyse(ai_context(ev))
        except Exception as exc:  # noqa: BLE001
            log.error("AI-аналіз %s — %s не вдався: %s", ev["home"], ev["away"], exc)
            ev["ai_error"] = str(exc)[:300]
    if analyst:
        usage.update(analyst.usage)
    return usage


# ─── 4. Відбір ставок ───────────────────────────────────────────────────────
def select_picks(cfg: Settings, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pc = cfg.section("picks")
    min_edge, min_conf = float(pc.get("min_edge", 0.03)), int(pc.get("min_confidence", 5))
    lo, hi = float(pc.get("min_odds", 1.4)), float(pc.get("max_odds", 5.0))
    units = float(cfg.section("bankroll").get("units", 100))
    out = []

    for ev in events:
        ai = ev.get("ai")
        if ai:
            ev["final_probs"] = ai["probabilities"]
            rec = ai.get("recommendation") or {}
            conf = int(rec.get("confidence") or 0)
            cands = value.candidates(ev, ev["final_probs"])
            ev["final_candidates"] = cands
            choice = next((c for c in cands if c["market"] == rec.get("market")
                           and c["side"] == rec.get("side")), None)
            source = "ai"
        else:
            ev["final_probs"] = ev["base_probs"]
            ev["final_candidates"] = ev["candidates"]
            # без AI — тільки модель; вимагаємо подвійний поріг edge
            choice = next((c for c in ev["candidates"] if c["edge"] >= 2 * min_edge), None)
            conf, source = None, "model"
            if cfg.ai_enabled or cfg.demo:
                choice = None  # AI був доступний, але матч не аналізувався — не ставимо

        ok = bool(choice) and choice["edge"] >= min_edge and lo <= choice["price"] <= hi \
            and (conf is None or conf >= min_conf)
        ev["verdict"] = "bet" if ok else "skip"
        if not ok:
            continue
        stake_pct = value.kelly_stake(choice["prob"], choice["price"],
                                      float(pc.get("kelly_fraction", 0.25)), float(pc.get("max_stake_pct", 0.03)))
        stake = round(max(0.5, stake_pct * units), 1)
        ev["stake"] = stake
        out.append({
            "id": pick_id(ev["id"], choice["market"], choice["side"]),
            "event_id": ev["id"], "league": ev["league"], "league_name": ev["league_name"],
            "commence_time": ev["commence_time"], "home": ev["home"], "away": ev["away"],
            "market": choice["market"], "side": choice["side"], "label": choice["label"],
            "label_market": "1X2" if choice["market"] == "h2h" else "Тотал",
            "line": ev["odds"].get("totals", {}).get("line") if choice["market"] == "totals" else None,
            "price": choice["price"], "book": choice["book"], "prob": choice["prob"],
            "edge": choice["edge"], "stake": stake, "confidence": conf, "source": source,
            "reasoning": (ai or {}).get("recommendation", {}).get("reasoning"),
            "created_at": utcnow().isoformat(), "status": "pending",
            "demo": cfg.demo,
        })

    out.sort(key=lambda p: (p["confidence"] or 0, p["edge"]), reverse=True)
    return out[: int(pc.get("max_picks_per_run", 8))]


# ─── 5. Оркестрація ─────────────────────────────────────────────────────────
def run(cfg: Settings) -> dict[str, Any]:
    started = utcnow()
    store = read_json(PICKS_PATH, {"picks": []})
    picks: list[dict[str, Any]] = store.get("picks", [])
    if cfg.demo:
        picks = [p for p in picks if p.get("demo")] or demo_settled_history()
    else:
        picks = [p for p in picks if not p.get("demo")]  # демо-дані не змішуємо з реальними

    settle_picks(cfg, picks)
    events, news_items = collect(cfg)
    log.info("Зібрано %d подій", len(events))
    usage = run_ai(cfg, events)
    new_picks = select_picks(cfg, events)
    update_clv(picks, events)

    existing = {p["id"] for p in picks}
    added = [p for p in new_picks if p["id"] not in existing]
    picks.extend(added)
    log.info("Нових рекомендацій: %d", len(added))

    write_json(PICKS_PATH, {"updated_at": utcnow().isoformat(), "stats": stats(picks), "picks": picks})
    events.sort(key=lambda e: e["commence_time"])
    latest = {
        "updated_at": utcnow().isoformat(),
        "duration_sec": round((utcnow() - started).total_seconds(), 1),
        "demo": cfg.demo,
        "ai": usage,
        "settings": {k: cfg.section("picks").get(k) for k in ("min_edge", "min_odds", "max_odds", "min_confidence")},
        "news_count": len(news_items),
        "events": [slim(e) for e in events],
    }
    write_json(LATEST_PATH, latest)
    return {"events": len(events), "new_picks": len(added), "ai": usage}


def slim(ev: dict[str, Any]) -> dict[str, Any]:
    keep = ("id", "league_name", "commence_time", "home", "away", "odds", "expected_goals", "likely_scores",
            "form", "h2h", "injuries", "news", "model_probs", "market_probs", "base_probs", "final_probs",
            "final_candidates", "ai", "ai_error", "verdict", "stake", "venue")
    return {k: ev[k] for k in keep if k in ev}


def demo_settled_history(n: int = 40, seed: int = 5) -> list[dict[str, Any]]:
    """Синтетична історія ставок для демонстрації дашборду."""
    rnd = random.Random(seed)
    teams = [t[1] for t in demo.TEAMS]
    out = []
    for i in range(n):
        h, a = rnd.sample(teams, 2)
        market = rnd.choice(["h2h", "totals"])
        side = rnd.choice(["home", "draw", "away"]) if market == "h2h" else rnd.choice(["over", "under"])
        price = round(rnd.uniform(1.6, 3.6), 2)
        prob = min(0.9, 1 / price * rnd.uniform(1.03, 1.12))
        won = rnd.random() < prob * 0.97
        stake = round(rnd.uniform(0.8, 3.0), 1)
        when = utcnow() - timedelta(days=n - i + 1, hours=rnd.randint(0, 10))
        label = value.SELECTIONS[(market, side)].format(line=2.5)
        out.append({
            "id": f"demo-{i}:{market}:{side}", "event_id": f"demo-{i}", "league": "soccer_epl",
            "league_name": "Англія — Прем'єр-ліга", "commence_time": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home": h, "away": a, "market": market, "side": side, "label": label,
            "label_market": "1X2" if market == "h2h" else "Тотал", "line": 2.5 if market == "totals" else None,
            "price": price, "book": rnd.choice(["Pinnacle", "Bet365", "Unibet"]), "prob": round(prob, 4),
            "edge": round(prob * price - 1, 4), "stake": stake, "confidence": rnd.randint(5, 8),
            "source": "ai", "reasoning": "Демо-ставка.", "created_at": when.isoformat(),
            "status": "won" if won else "lost", "score": f"{rnd.randint(0, 3)}-{rnd.randint(0, 3)}",
            "profit": round(stake * (price - 1), 3) if won else -stake,
            "clv": round(rnd.uniform(-0.03, 0.06), 4), "demo": True,
        })
    return out
