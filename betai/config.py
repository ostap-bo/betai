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
    demo: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # зручні аксесори
    @property
    def leagues(self) -> list[dict[str, Any]]:
        return self.raw.get("leagues", [])

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {}) or {}

    @property
    def ai_enabled(self) -> bool:
        return bool(self.section("ai").get("enabled", True)) and bool(self.anthropic_api_key)


def load_settings(config_path: Path | None = None, demo: bool = False) -> Settings:
    _load_dotenv(ROOT / ".env")
    path = config_path or ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # Дозволяємо перевизначити модель Claude через змінну середовища
    if os.getenv("ANTHROPIC_MODEL"):
        raw.setdefault("ai", {})["model"] = os.environ["ANTHROPIC_MODEL"]

    return Settings(
        raw=raw,
        odds_api_key=os.getenv("ODDS_API_KEY") or None,
        api_football_key=os.getenv("API_FOOTBALL_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        demo=demo or os.getenv("BETAI_DEMO") == "1",
    )
