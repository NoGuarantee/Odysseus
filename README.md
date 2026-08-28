# Odysseus

Zero-shot evaluation pipeline for **Qwen3-VL-8B-Instruct** playing **Super Mario Land**, following the interaction protocol from the paper [Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games via RL](https://arxiv.org/abs/2605.00347).

## Requirements

- Python 3.10+
- `python3-dev` (needed for CUDA/Triton kernels during inference)
- NVIDIA GPU with ~16 GB VRAM (for local Qwen3-VL-8B inference in bf16)
- A legally obtained Super Mario Land (Game Boy) ROM (`.gb`)

## Installation

```bash
# Debian/Ubuntu: install dev headers if inference fails with "Python.h: No such file"
sudo apt install python3-dev

pip install -e .
```

## Configuration

Edit [`config/zero_shot.yaml`](config/zero_shot.yaml) or set environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MARIO_ROM_PATH` | — | Path to `super_mario_land.gb` |
| `model_id` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model |
| `max_turns` | 80 | Turns per episode |
| `num_episodes` | 3 | Evaluation episodes |

## Run Zero-Shot Evaluation

```bash
export MARIO_ROM_PATH=/path/to/super_mario_land.gb

python scripts/run_zero_shot_eval.py \
  --config config/zero_shot.yaml \
  --episodes 3 \
  --output results/
```

### Smoke test without VLM (env + parser only)

```bash
export MARIO_ROM_PATH=/path/to/super_mario_land.gb

python scripts/run_zero_shot_eval.py \
  --mock-agent \
  --episodes 1 \
  --max-turns 10 \
  --output results/

# Or run all unit tests + optional env integration:
python scripts/smoke_test.py --with-env
```

### Agent-only smoke test (no ROM)

```bash
python scripts/smoke_test_agent.py
```

## Protocol (from paper Appendix B)

- **Prompt:** structured CoT with `<perception>`, `<reasoning>`, `<answer>`
- **Frame:** 160×144 native, upscaled ×8 → 1280×1152
- **Frame-skip:** 15 frames if action includes `a` (jump), else 5 frames
- **Actions:** up to 2 buttons from `{a, b, up, down, left, right, noop}`
- **Metric:** level progress = Mario's horizontal position from level start (RAM)

## Output

Results are written to the output directory:

- `zero_shot_1-1_<timestamp>.jsonl` — per-turn trajectories
- `summary_final.json` — aggregated metrics (mean progress, std, parse error rate)

## Expected Baseline

Per Table 2 of the paper, **Qwen3-VL-8B-Instruct (base)** on World 1-1 achieves mean progress ≈ **513.57 ± 21.08** (256 runs). A small smoke test (3–5 episodes) will be noisier but should stay in the same order of magnitude.

## Project Layout

```
odysseus/
  env/          # PyBoy wrapper, actions, RAM progress
  agent/        # Qwen3-VL agent, prompt, parser
  eval/         # Evaluation loop, JSONL logger
  metrics/      # Progress aggregation
scripts/
  run_zero_shot_eval.py
config/
  zero_shot.yaml
```

## License

Code is provided for research. You must supply your own legally obtained game ROM.
