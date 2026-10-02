"""Статистична модель: Пуассон з поправкою Діксона–Коулза.

1. Оцінюємо силу атаки та захисту кожної команди з історії (свіжі матчі важать більше).
2. Розраховуємо очікувані голи (λ) для майбутнього матчу.
3. Будуємо матрицю ймовірностей рахунків → 1X2, тотал, обидві заб'ють.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..utils import match_team

DC_RHO = -0.06  # поправка на кореляцію низьких рахунків (0:0, 1:1 трохи частіші)


@dataclass
class TeamRatings:
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_avg: float = 1.5
    away_avg: float = 1.2
    games: dict[str, int] = field(default_factory=dict)

    def resolve(self, name: str) -> str | None:
        return match_team(name, self.attack.keys())

    def expected_goals(self, home: str, away: str) -> tuple[float, float] | None:
        h, a = self.resolve(home), self.resolve(away)
        if not h or not a:
            return None
        lam_h = self.home_avg * self.attack[h] * self.defence[a]
        lam_a = self.away_avg * self.attack[a] * self.defence[h]
        return float(lam_h), float(lam_a)


def fit_ratings(df: pd.DataFrame, half_life_days: float = 240, as_of: datetime | None = None,
                iterations: int = 25, prior_games: float = 4.0) -> TeamRatings:
    """Ітеративна оцінка рейтингів з експоненційним згасанням та шрінкажем до середнього."""
    if df.empty:
        return TeamRatings()
    as_of = pd.Timestamp(as_of or df["date"].max())
    age = (as_of - df["date"]).dt.days.clip(lower=0).to_numpy()
    w = 0.5 ** (age / half_life_days)

    hg, ag = df["hg"].to_numpy(float), df["ag"].to_numpy(float)
    home_avg = float(np.sum(w * hg) / np.sum(w))
    away_avg = float(np.sum(w * ag) / np.sum(w))

    teams = sorted(set(df["home"]) | set(df["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    hi = df["home"].map(idx).to_numpy()
    ai = df["away"].map(idx).to_numpy()
    att = np.ones(len(teams))
    dfn = np.ones(len(teams))

    for _ in range(iterations):
        # атака: забиті / очікувані з урахуванням захисту суперника
        exp_h = home_avg * dfn[ai]
        exp_a = away_avg * dfn[hi]
        num = np.bincount(hi, w * hg, len(teams)) + np.bincount(ai, w * ag, len(teams))
        den = np.bincount(hi, w * exp_h, len(teams)) + np.bincount(ai, w * exp_a, len(teams))
        att = (num + prior_games * home_avg) / (den + prior_games * home_avg)
        # захист: пропущені / очікувані з урахуванням атаки суперника
        exp_h = home_avg * att[hi]
        exp_a = away_avg * att[ai]
        num = np.bincount(ai, w * hg, len(teams)) + np.bincount(hi, w * ag, len(teams))
        den = np.bincount(ai, w * exp_h, len(teams)) + np.bincount(hi, w * exp_a, len(teams))
        dfn = (num + prior_games * home_avg) / (den + prior_games * home_avg)
        att /= np.exp(np.mean(np.log(att)))  # нормування: середня атака = 1
        dfn /= np.exp(np.mean(np.log(dfn)))

    games = (np.bincount(hi, minlength=len(teams)) + np.bincount(ai, minlength=len(teams)))
    return TeamRatings(
        attack={t: float(att[i]) for t, i in idx.items()},
        defence={t: float(dfn[i]) for t, i in idx.items()},
        home_avg=home_avg, away_avg=away_avg,
        games={t: int(games[i]) for t, i in idx.items()},
    )


def score_matrix(lam_h: float, lam_a: float, max_goals: int = 10, rho: float = DC_RHO) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
    # поправка Діксона–Коулза для рахунків 0:0, 1:0, 0:1, 1:1
    m[0, 0] *= 1 - lam_h * lam_a * rho
    m[0, 1] *= 1 + lam_h * rho
    m[1, 0] *= 1 + lam_a * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def outcome_probs(m: np.ndarray, line: float = 2.5) -> dict[str, float]:
    n = m.shape[0]
    total = np.add.outer(np.arange(n), np.arange(n))
    over = float(m[total > line].sum())
    return {
        "home": float(np.tril(m, -1).sum()),
        "draw": float(np.trace(m)),
        "away": float(np.triu(m, 1).sum()),
        "over": over,
        "under": 1.0 - over,
        "btts": float(m[1:, 1:].sum()),
    }


def top_scores(m: np.ndarray, k: int = 3) -> list[dict[str, float | str]]:
    flat = np.argsort(m, axis=None)[::-1][:k]
    return [{"score": f"{i}-{j}", "p": round(float(m[i, j]), 3)}
            for i, j in (np.unravel_index(f, m.shape) for f in flat)]
