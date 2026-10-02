import json

import numpy as np
import pytest

from betai import pipeline
from betai.ai import analyst
from betai.config import load_settings
from betai.model import poisson, value
from betai.sources import demo, odds
from betai.tracker import outcome, settle, stats
from betai.utils import match_team, norm_team


# ── нормалізація назв ─────────────────────────────────────
@pytest.mark.parametrize("a,b", [
    ("Manchester United", "Man United"), ("Wolverhampton Wanderers", "Wolves"),
    ("Nottingham Forest", "Nott'm Forest"), ("Brighton and Hove Albion", "Brighton"),
    ("Bayer Leverkusen", "Leverkusen"), ("Atlético Madrid", "Ath Madrid"), ("Inter Milan", "Inter"),
])
def test_team_aliases(a, b):
    assert norm_team(a) == norm_team(b)


def test_match_team_does_not_confuse_city_and_united():
    assert match_team("Manchester City", ["Man United", "Man City"]) == "Man City"
    assert match_team("Manchester United", ["Man United", "Man City"]) == "Man United"
    assert match_team("Real Madrid", ["Barcelona", "Sevilla"]) is None


# ── модель ────────────────────────────────────────────────
def test_score_matrix_probs_sum_to_one():
    m = poisson.score_matrix(1.6, 1.1)
    p = poisson.outcome_probs(m)
    assert m.sum() == pytest.approx(1.0)
    assert p["home"] + p["draw"] + p["away"] == pytest.approx(1.0)
    assert p["over"] + p["under"] == pytest.approx(1.0)
    assert p["home"] > p["away"]


def test_ratings_rank_strong_team_higher():
    r = poisson.fit_ratings(demo.demo_history())
    assert r.attack["Man City"] > r.attack["Burnley"]
    assert r.defence["Arsenal"] < r.defence["Leeds"]  # менше = кращий захист
    xg = r.expected_goals("Manchester City", "Burnley")
    assert xg and xg[0] > xg[1]


# ── value ─────────────────────────────────────────────────
def test_fair_probs_remove_margin():
    p = value.fair_probs({"a": 1.9, "b": 1.9})
    assert p["a"] == pytest.approx(0.5)


def test_kelly():
    assert value.kelly_stake(0.5, 1.9) == 0.0          # негативне очікування
    assert 0 < value.kelly_stake(0.55, 2.0) <= 0.03    # обмеження зверху
    assert value.edge(0.5, 2.2) == pytest.approx(0.1)


# ── трекінг ───────────────────────────────────────────────
def test_outcomes():
    assert outcome({"market": "h2h", "side": "home"}, 2, 1) == "won"
    assert outcome({"market": "h2h", "side": "draw"}, 2, 1) == "lost"
    assert outcome({"market": "totals", "side": "over", "line": 2.5}, 2, 1) == "won"
    assert outcome({"market": "totals", "side": "under", "line": 2.5}, 2, 1) == "lost"


def test_settle_and_stats():
    picks = [{"event_id": "e1", "market": "h2h", "side": "away", "status": "pending", "stake": 2.0,
              "price": 3.0, "commence_time": "2026-01-01T15:00:00Z"}]
    assert settle(picks, {"e1": {"home_score": 0, "away_score": 1}}) == 1
    assert picks[0]["profit"] == pytest.approx(4.0)
    s = stats(picks)
    assert s["roi"] == pytest.approx(2.0) and s["won"] == 1


# ── The Odds API ──────────────────────────────────────────
def test_normalise_event():
    ev = {"id": "x", "commence_time": "2030-01-01T15:00:00Z", "home_team": "A", "away_team": "B",
          "bookmakers": [
              {"title": "B1", "markets": [
                  {"key": "h2h", "outcomes": [{"name": "A", "price": 2.0}, {"name": "B", "price": 4.0},
                                              {"name": "Draw", "price": 3.4}]},
                  {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9, "point": 2.5},
                                                 {"name": "Under", "price": 1.95, "point": 2.5}]}]},
              {"title": "B2", "markets": [
                  {"key": "h2h", "outcomes": [{"name": "A", "price": 2.1}, {"name": "B", "price": 3.8},
                                              {"name": "Draw", "price": 3.3}]}]}]}
    out = odds.normalise_event(ev, {"odds_key": "soccer_epl", "name": "EPL"})
    assert out["odds"]["h2h"]["home"]["best"] == 2.1 and out["odds"]["h2h"]["home"]["book"] == "B2"
    assert out["odds"]["totals"]["over"]["best"] == 1.9


# ── AI ────────────────────────────────────────────────────
def test_parse_json_from_fenced_block():
    text = 'Аналіз...\n```json\n{"summary": "ok", "probabilities": {"home": 0.5}}\n```'
    assert analyst.parse_json(text)["summary"] == "ok"


def test_sanitize_limits_shift():
    base = {"home": 0.5, "draw": 0.25, "away": 0.25, "over": 0.5, "under": 0.5}
    out = analyst.sanitize_probs({"home": 95, "draw": 0.02, "away": 0.03, "over": 0.5, "under": 0.5}, base)
    assert out["home"] < 0.75
    assert out["home"] + out["draw"] + out["away"] == pytest.approx(1.0, abs=1e-3)


def test_claude_analyst_handles_pause_turn(monkeypatch):
    calls = []
    reply = {"summary": "s", "key_factors": [], "probabilities":
             {"home": 0.5, "draw": 0.25, "away": 0.25, "over": 0.55, "under": 0.45},
             "recommendation": {"market": "h2h", "side": "home", "confidence": 7, "reasoning": "r"}}

    class Resp:
        def __init__(self, data):
            self.status_code, self._d, self.text = 200, data, ""

        def json(self):
            return self._d

    def fake_post(url, headers, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:
            return Resp({"stop_reason": "pause_turn", "content": [{"type": "text", "text": "шукаю..."}],
                         "usage": {"input_tokens": 10, "output_tokens": 5}})
        return Resp({"stop_reason": "end_turn", "usage": {"input_tokens": 20, "output_tokens": 30,
                     "server_tool_use": {"web_search_requests": 2}},
                     "content": [{"type": "text", "text": "```json\n" + __import__("json").dumps(reply) + "\n```"}]})

    monkeypatch.setattr(analyst.requests, "post", fake_post)
    a = analyst.ClaudeAnalyst("key", {"model": "claude-sonnet-5-5", "web_search": True})
    base = {"home": 0.48, "draw": 0.26, "away": 0.26, "over": 0.55, "under": 0.45}
    res = a.analyse({"base_probabilities": base, "home": "A", "away": "B"})
    assert res["recommendation"]["side"] == "home"
    assert calls[0]["tools"][0]["name"] == "web_search"
    assert len(calls[1]["messages"]) == 2  # продовження після pause_turn
    assert a.usage["web_searches"] == 2


# ── повний конвеєр у демо-режимі ───────────────────────────
def test_demo_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PICKS_PATH", tmp_path / "picks.json")
    monkeypatch.setattr(pipeline, "LATEST_PATH", tmp_path / "latest.json")
    res = pipeline.run(load_settings(demo=True))
    assert res["events"] > 0
    latest = json.loads((tmp_path / "latest.json").read_text())
    store = json.loads((tmp_path / "picks.json").read_text())
    assert latest["demo"] is True and latest["events"]
    assert all(np.isfinite(p["price"]) for p in store["picks"])
    for p in store["picks"]:
        if p["status"] == "pending":
            assert p["edge"] >= 0.03


def test_gemini_analyst(monkeypatch):
    sent = []
    reply = {"summary": "s", "probabilities": {"home": 0.5, "draw": 0.25, "away": 0.25, "over": 0.5, "under": 0.5},
             "recommendation": {"market": "totals", "side": "over", "confidence": 6, "reasoning": "r"}}

    class Resp:
        status_code, text = 200, ""

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "```json\n" + json.dumps(reply) + "\n```"}]},
                                    "groundingMetadata": {"webSearchQueries": ["q1"],
                                                          "groundingChunks": [{"web": {"uri": "https://bbc.com/x"}}]}}],
                    "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50}}

    def fake_post(url, params, json, timeout):  # noqa: A002
        sent.append((url, params, json))
        return Resp()

    monkeypatch.setattr(analyst.requests, "post", fake_post)
    a = analyst.make_analyst("gemini", {"gemini": "g"}, {"gemini_pause_sec": 0})
    base = {"home": 0.5, "draw": 0.25, "away": 0.25, "over": 0.5, "under": 0.5}
    res = a.analyse({"base_probabilities": base})
    assert "gemini-2.5-flash:generateContent" in sent[0][0]
    assert sent[0][2]["tools"] == [{"google_search": {}}]
    assert res["sources"] == ["https://bbc.com/x"] and a.usage["web_searches"] == 1


def test_provider_selection(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert load_settings().ai_provider == "claude"
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg = load_settings()
    assert cfg.ai_provider == "gemini" and cfg.ai_enabled


# ── ESPN і прогнози без коефіцієнтів ───────────────────────
def test_espn_scoreboard_and_results(monkeypatch):
    from datetime import timedelta
    from betai.sources import espn
    from betai.utils import utcnow
    kick = utcnow() - timedelta(hours=4)
    data = {"events": [{"id": "1", "date": kick.strftime("%Y-%m-%dT%H:%MZ"),
                        "status": {"type": {"state": "post", "completed": True}},
                        "competitions": [{"competitors": [
                            {"homeAway": "home", "team": {"displayName": "Shakhtar Donetsk"}, "score": "2"},
                            {"homeAway": "away", "team": {"displayName": "Dynamo Kyiv"}, "score": "2"}]}]}]}
    monkeypatch.setattr(espn, "http_get", lambda *a, **k: type("R", (), {"json": lambda s: data})())
    picks = [{"event_id": "e", "home": "Shakhtar Donetsk", "away": "Dynamo Kyiv",
              "commence_time": kick.strftime("%Y-%m-%dT%H:%M:%SZ")}]
    assert espn.results_for(picks, "ukr.1") == {"e": {"home_score": 2, "away_score": 2}}


def test_prediction_without_odds():
    ai = {"probabilities": {"home": 0.5, "draw": 0.3, "away": 0.2, "over": 0.5, "under": 0.5},
          "recommendation": {"market": "h2h", "side": "home", "confidence": 6}}
    pr = pipeline.make_prediction(ai, 0.05)
    assert pr["label"] == "П1" and pr["fair_odds"] == 2.0 and pr["min_odds"] == 2.1


def test_sanitize_without_base():
    out = analyst.sanitize_probs({"home": 0.6, "draw": 0.25, "away": 0.25, "over": 0.4, "under": 0.6}, {})
    assert out["home"] + out["draw"] + out["away"] == pytest.approx(1.0, abs=1e-3)
