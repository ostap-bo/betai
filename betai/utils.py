"""Спільні утиліти: HTTP з повторами, JSON, нормалізація назв команд."""
from __future__ import annotations

import difflib
import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

log = logging.getLogger("betai")

USER_AGENT = "BetAI/1.0 (+https://github.com)"


def http_get(url: str, *, params: dict | None = None, headers: dict | None = None,
             retries: int = 3, timeout: int = 30) -> requests.Response:
    """GET із повторами та експоненційною паузою. Кидає виняток після останньої спроби."""
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # noqa: PERF203
            last_exc = exc
            wait = 2 ** attempt
            log.warning("GET %s не вдався (%s), повтор через %ss", url, exc, wait)
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: str) -> datetime:
    """ISO-8601 → aware datetime (UTC)."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.error("Пошкоджений JSON: %s — використовую значення за замовчуванням", path)
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ─── Нормалізація назв команд ────────────────────────────────────────────────
# Різні джерела пишуть назви по-різному ("Manchester United" / "Man United").
# Тут — канонічні псевдоніми; решту добирає нечіткий пошук.
ALIASES: dict[str, str] = {
    "manchester united": "man united", "manchester utd": "man united", "man utd": "man united",
    "manchester city": "man city",
    "tottenham hotspur": "tottenham", "spurs": "tottenham",
    "newcastle united": "newcastle",
    "west ham united": "west ham",
    "wolverhampton wanderers": "wolves", "wolverhampton": "wolves",
    "brighton and hove albion": "brighton", "brighton & hove albion": "brighton",
    "nottingham forest": "nott'm forest", "nottm forest": "nott'm forest",
    "leicester city": "leicester", "leeds united": "leeds", "ipswich town": "ipswich",
    "afc bournemouth": "bournemouth", "sheffield united": "sheffield united",
    "luton town": "luton", "burnley fc": "burnley", "sunderland afc": "sunderland",
    "atletico madrid": "ath madrid", "atlético madrid": "ath madrid", "atletico de madrid": "ath madrid",
    "athletic bilbao": "ath bilbao", "athletic club": "ath bilbao",
    "real betis": "betis", "real sociedad": "sociedad", "celta vigo": "celta",
    "rayo vallecano": "vallecano", "deportivo alaves": "alaves", "alavés": "alaves",
    "espanyol": "espanol", "rcd espanyol": "espanol", "rcd mallorca": "mallorca",
    "ca osasuna": "osasuna", "real valladolid": "valladolid", "cadiz cf": "cadiz",
    "girona fc": "girona", "ud las palmas": "las palmas", "real oviedo": "oviedo",
    "inter milan": "inter", "internazionale": "inter", "fc internazionale": "inter",
    "ac milan": "milan", "as roma": "roma", "ss lazio": "lazio", "ssc napoli": "napoli",
    "hellas verona": "verona", "atalanta bc": "atalanta",
    "bayern munich": "bayern munich", "fc bayern munchen": "bayern munich", "bayern münchen": "bayern munich",
    "borussia dortmund": "dortmund", "bayer leverkusen": "leverkusen", "bayer 04 leverkusen": "leverkusen",
    "borussia monchengladbach": "m'gladbach", "borussia mönchengladbach": "m'gladbach",
    "eintracht frankfurt": "ein frankfurt", "vfb stuttgart": "stuttgart", "vfl wolfsburg": "wolfsburg",
    "sc freiburg": "freiburg", "1. fc union berlin": "union berlin", "fc union berlin": "union berlin",
    "1. fsv mainz 05": "mainz", "fsv mainz 05": "mainz", "mainz 05": "mainz",
    "tsg hoffenheim": "hoffenheim", "tsg 1899 hoffenheim": "hoffenheim",
    "fc augsburg": "augsburg", "werder bremen": "werder bremen", "sv werder bremen": "werder bremen",
    "1. fc koln": "fc koln", "1. fc köln": "fc koln", "fc köln": "fc koln",
    "1. fc heidenheim": "heidenheim", "fc st. pauli": "st pauli", "st. pauli": "st pauli",
    "hamburger sv": "hamburg", "rb leipzig": "rb leipzig", "vfl bochum": "bochum",
    "paris saint germain": "paris sg", "paris saint-germain": "paris sg", "psg": "paris sg",
    "olympique marseille": "marseille", "olympique lyonnais": "lyon", "as monaco": "monaco",
}

_STRIP_TOKENS = re.compile(r"\b(fc|cf|afc|sc|ac|ssc|calcio|club)\b")


def norm_team(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    s = s.replace("&", "and")
    if s in ALIASES:
        return ALIASES[s]
    raw = name.lower().strip()
    if raw in ALIASES:
        return ALIASES[raw]
    stripped = re.sub(r"\s+", " ", _STRIP_TOKENS.sub("", s)).strip()
    return ALIASES.get(stripped, stripped or s)


def match_team(name: str, candidates: Iterable[str], cutoff: float = 0.72) -> str | None:
    """Знаходить найближчу назву серед кандидатів (за нормалізованою формою)."""
    cands = list(candidates)
    if not cands:
        return None
    target = norm_team(name)
    normed = {norm_team(c): c for c in cands}
    if target in normed:
        return normed[target]
    # входження підрядка ("leverkusen" ⊂ "bayer leverkusen")
    for n, orig in normed.items():
        if len(target) >= 4 and (target in n or n in target):
            return orig
    best = difflib.get_close_matches(target, list(normed), n=1, cutoff=cutoff)
    return normed[best[0]] if best else None
