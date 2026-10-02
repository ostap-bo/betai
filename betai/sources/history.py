"""Історичні результати з football-data.co.uk (безкоштовно, без ключа).

Використовується для: статистичної моделі (сила атаки/захисту), форми команд,
особистих зустрічей (H2H) у межах ліги.
"""
from __future__ import annotations

import io
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR
from ..utils import http_get

log = logging.getLogger("betai.history")
BASE = "https://www.football-data.co.uk/mmz4281"
CACHE_DIR = DATA_DIR / "cache" / "history"
COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def season_codes(n: int, today: date | None = None) -> list[str]:
    """Коди останніх n сезонів, напр. ['2627', '2526', '2425']."""
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{(y % 100):02d}{((y + 1) % 100):02d}" for y in range(start, start - n, -1)]


def _download(code: str, season: str, current: bool) -> pd.DataFrame | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path: Path = CACHE_DIR / f"{code}_{season}.csv"
    ttl = 6 * 3600 if current else 30 * 24 * 3600
    if path.exists() and time.time() - path.stat().st_mtime < ttl:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        try:
            text = http_get(f"{BASE}/{season}/{code}.csv", retries=2).content.decode("latin-1")
            path.write_text(text, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.warning("Історія %s/%s недоступна: %s", code, season, exc)
            if not path.exists():
                return None
            text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        df = pd.read_csv(io.StringIO(text), usecols=lambda c: c in COLUMNS, on_bad_lines="skip")
    except Exception as exc:  # noqa: BLE001
        log.warning("Не вдалося прочитати CSV %s/%s: %s", code, season, exc)
        return None
    return df


def load_history(code: str, seasons: int = 3) -> pd.DataFrame:
    """Повертає DataFrame: date, home, away, hg, ag — відсортований за датою."""
    frames = []
    for i, season in enumerate(season_codes(seasons)):
        df = _download(code, season, current=(i == 0))
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "home", "away", "hg", "ag"])
    return clean_history(pd.concat(frames, ignore_index=True))


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["date"])
    out = pd.DataFrame({
        "date": df["date"],
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
        "hg": df["FTHG"].astype(int),
        "ag": df["FTAG"].astype(int),
    })
    return out.sort_values("date").reset_index(drop=True)


def team_form(df: pd.DataFrame, team: str, n: int = 6) -> list[dict]:
    """Останні n матчів команди: результат W/D/L, рахунок, суперник."""
    games = df[(df["home"] == team) | (df["away"] == team)].tail(n)
    out = []
    for _, g in games.iterrows():
        is_home = g["home"] == team
        gf, ga = (g["hg"], g["ag"]) if is_home else (g["ag"], g["hg"])
        out.append({
            "date": g["date"].strftime("%Y-%m-%d"),
            "opponent": g["away"] if is_home else g["home"],
            "venue": "H" if is_home else "A",
            "score": f"{gf}-{ga}",
            "result": "W" if gf > ga else "D" if gf == ga else "L",
        })
    return out


def head_to_head(df: pd.DataFrame, home: str, away: str, n: int = 8) -> list[dict]:
    games = df[((df["home"] == home) & (df["away"] == away)) |
               ((df["home"] == away) & (df["away"] == home))].tail(n)
    return [{"date": g["date"].strftime("%Y-%m-%d"), "home": g["home"], "away": g["away"],
             "score": f"{g['hg']}-{g['ag']}"} for _, g in games.iterrows()]
