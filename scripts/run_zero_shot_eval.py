#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from odysseus.agent.qwen_agent import MockAgent, QwenAgent
from odysseus.env.mario_env import MarioEnv
from odysseus.eval.runner import EvalRunner


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_rom_path(config: dict) -> str:
    rom_path = os.environ.get("MARIO_ROM_PATH") or config.get("rom_path") or ""
    if not rom_path:
        raise SystemExit(
            "ROM path is required. Set MARIO_ROM_PATH or rom_path in the config file."
        )
    if not Path(rom_path).is_file():
        raise SystemExit(f"ROM file not found: {rom_path}")
    return rom_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of Qwen3-VL-8B-Instruct on Super Mario Land"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "zero_shot.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override num_episodes")
    parser.add_argument("--max-turns", type=int, default=None, help="Override max_turns")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--mock-agent",
        action="store_true",
        help="Use deterministic mock agent (no GPU inference)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    rom_path = resolve_rom_path(config)

    level = config.get("level", {})
    inference = config.get("inference", {})
    episode_cfg = config.get("episode", {})
    frame_cfg = config.get("frame", {})

    num_episodes = args.episodes or episode_cfg.get("num_episodes", 3)
    max_turns = args.max_turns or episode_cfg.get("max_turns", 80)
    output_dir = args.output or config.get("output_dir", "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    env = MarioEnv(
        rom_path,
        world=int(level.get("world", 1)),
        stage=int(level.get("stage", 1)),
        upscale_factor=int(frame_cfg.get("upscale_factor", 8)),
        headless=True,
    )

    if args.mock_agent:
        agent = MockAgent()
    else:
        agent = QwenAgent(
            model_id=config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct"),
            temperature=float(inference.get("temperature", 1.0)),
            top_p=float(inference.get("top_p", 1.0)),
            max_new_tokens=int(inference.get("max_new_tokens", 1024)),
        )

    runner = EvalRunner(
        env=env,
        agent=agent,
        max_turns=max_turns,
        num_episodes=num_episodes,
        output_dir=output_dir,
    )

    try:
        summary = runner.run()
    finally:
        env.close()

    print("Evaluation complete.")
    print(f"  Mean progress: {summary['mean_progress']:.2f}")
    print(f"  Std progress:  {summary['std_progress']:.2f}")
    print(f"  Mean turns:    {summary['mean_turns']:.2f}")
    print(f"  Results dir:   {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
