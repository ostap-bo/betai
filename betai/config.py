"""Завантаження конфігурації з config.yaml та змінних середовища."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DATA_DIR = ROOT / "docs" / "data"


def _load_dotenv(path: Path) -> None:
    """Мінімальний парсер .env (без зовнішніх залежностей)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    raw: dict[str, Any]
    odds_api_key: str | None = None
    api_football_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    demo: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # зручні аксесори
    @property
    def leagues(self) -> list[dict[str, Any]]:
        out = []
        for lg in self.raw.get("leagues", []) or []:
            lg = dict(lg)
            lg.setdefault("id", lg.get("odds_key") or f"espn:{lg.get('espn')}")
            out.append(lg)
        return out

    def league_by_id(self, league_id: str) -> dict[str, Any] | None:
        return next((lg for lg in self.leagues if lg["id"] == league_id or lg.get("odds_key") == league_id), None)

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {}) or {}

    @property
    def ai_provider(self) -> str:
        return str(self.section("ai").get("provider", "gemini")).lower()

    @property
    def ai_keys(self) -> dict[str, str | None]:
        return {"claude": self.anthropic_api_key, "gemini": self.gemini_api_key}

    @property
    def ai_enabled(self) -> bool:
        return bool(self.section("ai").get("enabled", True)) and bool(self.ai_keys.get(self.ai_provider))


def load_settings(config_path: Path | None = None, demo: bool = False) -> Settings:
    _load_dotenv(ROOT / ".env")
    path = config_path or ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # Дозволяємо перевизначити модель Claude через змінну середовища
    ai = raw.setdefault("ai", {})
    if os.getenv("ANTHROPIC_MODEL"):
        ai["model"] = os.environ["ANTHROPIC_MODEL"]
    if os.getenv("AI_PROVIDER"):
        ai["provider"] = os.environ["AI_PROVIDER"]
    elif not os.getenv("GEMINI_API_KEY") and os.getenv("ANTHROPIC_API_KEY"):
        ai["provider"] = "claude"  # є лише ключ Claude — використовуємо його

    return Settings(
        raw=raw,
        odds_api_key=os.getenv("ODDS_API_KEY") or None,
        api_football_key=os.getenv("API_FOOTBALL_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        demo=demo or os.getenv("BETAI_DEMO") == "1",
    )
