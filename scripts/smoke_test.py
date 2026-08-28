#!/usr/bin/env python3
"""Run pipeline smoke tests (unit + optional env integration)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_pytest() -> int:
    print("==> Running unit tests")
    return subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=ROOT)


def run_mock_eval(rom_path: str, *, max_turns: int, episodes: int) -> int:
    print("==> Running mock-agent env integration")
    env = os.environ.copy()
    env["MARIO_ROM_PATH"] = rom_path
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_zero_shot_eval.py"),
        "--mock-agent",
        "--episodes",
        str(episodes),
        "--max-turns",
        str(max_turns),
        "--output",
        str(ROOT / "results" / "smoke_mock"),
    ]
    return subprocess.call(cmd, cwd=ROOT, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Odysseus pipeline smoke test")
    parser.add_argument("--with-env", action="store_true", help="Run mock eval if ROM is available")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    code = run_pytest()
    if code != 0:
        return code

    rom_path = os.environ.get("MARIO_ROM_PATH", "")
    if args.with_env:
        if not rom_path or not Path(rom_path).is_file():
            print("WARN: MARIO_ROM_PATH not set or missing; skipping env integration")
            return 0
        return run_mock_eval(rom_path, max_turns=args.max_turns, episodes=args.episodes)

    print("Smoke test complete (unit tests only).")
    print("Set MARIO_ROM_PATH and rerun with --with-env for game integration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
