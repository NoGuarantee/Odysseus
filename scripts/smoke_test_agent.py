#!/usr/bin/env python3
"""Quick smoke test for Qwen3-VL agent inference (no game ROM required)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from odysseus.agent.parser import parse_action
from odysseus.agent.prompt import ODYSSEUS_PROMPT


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Qwen3-VL agent inference")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    print("Prompt length:", len(ODYSSEUS_PROMPT))
    frame = Image.new("RGB", (1280, 1152), color=(135, 206, 235))

    from odysseus.agent.qwen_agent import QwenAgent

    print(f"Loading {args.model_id} ...")
    agent = QwenAgent(model_id=args.model_id, max_new_tokens=256)
    print("Running one inference step ...")
    response = agent.act(frame)
    print("Buttons:", response.buttons)
    print("Parse error:", response.parse_error)
    print("Raw response preview:", response.raw_text[:400])

    # Validate parser independently
    buttons, err = parse_action(response.raw_text)
    assert buttons == response.buttons
    print("Agent smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
