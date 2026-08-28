from __future__ import annotations

from unittest.mock import MagicMock, patch

from PIL import Image

from odysseus.agent.qwen_agent import MockAgent
from odysseus.eval.runner import EvalRunner
from odysseus.metrics.progress import MetricsCollector


def test_eval_runner_with_mocked_env(tmp_path):
    frame = Image.new("RGB", (1280, 1152), color=(0, 0, 0))
    agent = MockAgent()

    env = MagicMock()
    env.world = 1
    env.stage = 1
    env.reset.return_value = frame
    env.get_progress_from_start.side_effect = [0, 12, 28, 45]

    step_info = MagicMock()
    step_info.progress = 100
    step_info.delta_progress = 10
    step_info.died = False
    env.step.return_value = (frame, step_info)

    runner = EvalRunner(
        env=env,
        agent=agent,
        max_turns=3,
        num_episodes=1,
        output_dir=str(tmp_path),
        run_name="test_run",
    )
    summary = runner.run()

    assert summary["num_episodes"] == 1
    assert summary["mean_turns"] == 3.0
    assert (tmp_path / "test_run.jsonl").is_file()
    assert (tmp_path / "summary_final.json").is_file()
    env.reset.assert_called_once()
    assert env.step.call_count == 3


def test_metrics_collector_empty():
    collector = MetricsCollector()
    summary = collector.aggregate()
    assert summary["num_episodes"] == 0
    assert summary["mean_progress"] == 0.0
