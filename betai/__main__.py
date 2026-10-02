"""CLI: python -m betai [run|settle] [--demo]"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_settings
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="betai", description="BetAI — AI-аналітика футбольних ставок")
    parser.add_argument("command", nargs="?", default="run", choices=["run"],
                        help="run — зібрати дані, проаналізувати, оновити дашборд")
    parser.add_argument("--demo", action="store_true", help="запуск на демо-даних без API-ключів")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_settings(demo=args.demo)
    result = run(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
