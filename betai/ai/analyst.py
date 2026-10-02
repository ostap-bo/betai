"""AI-аналітик на базі Claude API.

Claude отримує: коефіцієнти, ймовірності моделі та ринку, форму, H2H, травми, новини.
За потреби сам шукає свіжу інформацію (web search) і повертає структурований JSON:
скориговані ймовірності, ключові фактори, рекомендацію та впевненість.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests

log = logging.getLogger("betai.ai")
API_URL = "https://api.anthropic.com/v1/messages"
MAX_SHIFT = 0.12  # AI не може зсунути ймовірність більш ніж на 12 п.п. від бази

SYSTEM_PROMPT = """Ти — досвідчений футбольний аналітик і спеціаліст зі ставок на value.
Твоє завдання — оцінити матч і чесно відповісти, чи є в ньому ставка з позитивним очікуванням.

Принципи:
- Букмекерський ринок дуже точний. Базова ймовірність (blend моделі та ринку) — твоя відправна точка.
- Зсувай ймовірності лише через КОНКРЕТНІ, свіжі факти: травми/дискваліфікації ключових гравців,
  ротація перед єврокубками, тренерські зміни, мотивація, погода, стиль суперників. Типовий зсув — 1–6 п.п.
- Не вигадуй факти. Якщо інформації немає — так і скажи, і не зсувай ймовірності.
- Якщо value немає — рекомендація "none". Краще пропустити матч, ніж зробити погану ставку.
- Будь каліброваним: впевненість 8–10 — лише за сильних і перевірених аргументів.
- Пиши українською, коротко і по суті.
"""

OUTPUT_SPEC = """Поверни ЛИШЕ JSON у блоці ```json з такою структурою:
{
  "summary": "2–3 речення: головне про матч",
  "key_factors": ["фактор 1", "фактор 2", "..."],
  "absences": [{"team": "...", "player": "...", "status": "out|doubtful", "impact": "high|medium|low"}],
  "probabilities": {"home": 0.0, "draw": 0.0, "away": 0.0, "over": 0.0, "under": 0.0},
  "recommendation": {
    "market": "h2h|totals|none",
    "side": "home|draw|away|over|under|none",
    "confidence": 1,
    "reasoning": "чому саме ця ставка (або чому пропустити)"
  },
  "risks": ["ризик 1", "ризик 2"],
  "sources": ["url або назва джерела"]
}
Ймовірності — числа від 0 до 1; home+draw+away = 1, over+under = 1 (тотал {line})."""


class ClaudeAnalyst:
    def __init__(self, api_key: str, cfg: dict[str, Any]):
        self.api_key = api_key
        self.model = cfg.get("model", "claude-sonnet-5-5")
        self.max_tokens = int(cfg.get("max_tokens", 4000))
        self.web_search = bool(cfg.get("web_search", True))
        self.tool_type = cfg.get("web_search_tool", "web_search_20250305")
        self.max_uses = int(cfg.get("web_search_max_uses", 3))
        self.usage = {"input_tokens": 0, "output_tokens": 0, "web_searches": 0, "calls": 0}

    # ── HTTP ────────────────────────────────────────────────────────────────
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        for attempt in range(3):
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code in (429, 500, 502, 503, 529) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Claude API {resp.status_code}: {resp.text[:500]}")
            return resp.json()
        raise RuntimeError("Claude API: вичерпано спроби")

    def _call(self, user_prompt: str) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        payload: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT, "messages": messages,
        }
        if self.web_search:
            payload["tools"] = [{"type": self.tool_type, "name": "web_search", "max_uses": self.max_uses}]

        texts: list[str] = []
        for _ in range(4):  # pause_turn → продовжуємо довгий хід із пошуком
            data = self._post(payload)
            self.usage["calls"] += 1
            u = data.get("usage", {})
            self.usage["input_tokens"] += u.get("input_tokens", 0)
            self.usage["output_tokens"] += u.get("output_tokens", 0)
            self.usage["web_searches"] += (u.get("server_tool_use") or {}).get("web_search_requests", 0)
            texts += [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            if data.get("stop_reason") != "pause_turn":
                break
            messages.append({"role": "assistant", "content": data["content"]})
        return "\n".join(texts)

    # ── Аналіз матчу ────────────────────────────────────────────────────────
    def analyse(self, ctx: dict[str, Any]) -> dict[str, Any]:
        line = ctx.get("totals_line", 2.5)
        prompt = (
            "Проаналізуй футбольний матч і знайди value-ставку (або поясни, чому її немає).\n"
            + ("Спершу знайди в інтернеті свіжі новини: травми, дискваліфікації, ймовірні склади, "
               "заяви тренерів, ротацію.\n" if self.web_search else "")
            + "\nДані:\n```json\n" + json.dumps(ctx, ensure_ascii=False, indent=1, default=str) + "\n```\n\n"
            + OUTPUT_SPEC.replace("{line}", str(line))
        )
        text = self._call(prompt)
        result = parse_json(text)
        if result is None:
            raise ValueError("Claude повернув відповідь без коректного JSON")
        result["probabilities"] = sanitize_probs(result.get("probabilities", {}), ctx["base_probabilities"])
        return result


def parse_json(text: str) -> dict[str, Any] | None:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = blocks[::-1] or []
    if not candidates:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidates = [text[start:end + 1]]
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def sanitize_probs(ai: dict[str, Any], base: dict[str, float]) -> dict[str, float]:
    """Обмежує зсув від бази та нормалізує групи — захист від галюцинацій."""
    out: dict[str, float] = {}
    for k, b in base.items():
        try:
            v = float(ai.get(k, b))
        except (TypeError, ValueError):
            v = b
        if v > 1:  # раптом повернув у відсотках
            v /= 100
        out[k] = min(max(v, b - MAX_SHIFT, 0.01), b + MAX_SHIFT, 0.99)
    for group in (("home", "draw", "away"), ("over", "under")):
        s = sum(out[g] for g in group if g in out)
        if s:
            for g in group:
                if g in out:
                    out[g] = round(out[g] / s, 4)
    return out
