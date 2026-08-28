from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

from PIL import Image
from tqdm import tqdm

from odysseus.agent.qwen_agent import AgentResponse
from odysseus.env.mario_env import MarioEnv
from odysseus.eval.logger import EvalLogger, TurnRecord
from odysseus.metrics.progress import EpisodeMetrics, MetricsCollector

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    last_raw_response: str

    def act(self, frame: Image.Image) -> AgentResponse: ...


class EvalRunner:
    def __init__(
        self,
        env: MarioEnv,
        agent: AgentProtocol,
        *,
        max_turns: int = 80,
        num_episodes: int = 3,
        output_dir: str = "results",
        run_name: str | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.max_turns = max_turns
        self.num_episodes = num_episodes
        self.output_dir = output_dir
        self.run_name = run_name or self._default_run_name()

    @staticmethod
    def _default_run_name() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"zero_shot_1-1_{ts}"

    def run(self) -> dict:
        metrics = MetricsCollector()

        with EvalLogger(self.output_dir, self.run_name) as eval_logger:
            for episode_idx in tqdm(range(self.num_episodes), desc="Episodes"):
                episode_metrics = self._run_episode(episode_idx, eval_logger)
                metrics.add(episode_metrics)
                eval_logger.log_episode_summary(
                    {
                        "type": "episode",
                        "episode": episode_idx,
                        "progress": episode_metrics.progress,
                        "turns": episode_metrics.turns,
                        "died": episode_metrics.died,
                        "parse_error_rate": episode_metrics.parse_error_rate,
                    }
                )

            summary = metrics.aggregate()
            summary["run_name"] = self.run_name
            summary["max_turns"] = self.max_turns
            summary["level"] = f"{self.env.world}-{self.env.stage}"
            eval_logger.write_final_summary(summary)
            return summary

    def _run_episode(self, episode_idx: int, eval_logger: EvalLogger) -> EpisodeMetrics:
        frame = self.env.reset()
        parse_errors = 0
        died = False
        turns = 0

        for turn in range(self.max_turns):
            response = self.agent.act(frame)
            if response.parse_error:
                parse_errors += 1

            frame, info = self.env.step(response.buttons)
            turns += 1

            eval_logger.log_turn(
                TurnRecord(
                    episode=episode_idx,
                    turn=turn,
                    buttons=response.buttons,
                    progress=info.progress,
                    delta_progress=info.delta_progress,
                    died=info.died,
                    parse_error=response.parse_error,
                    raw_response=response.raw_text,
                )
            )

            if info.died:
                died = True
                break

        progress = self.env.get_progress_from_start()
        logger.info(
            "Episode %d finished: progress=%d turns=%d died=%s",
            episode_idx,
            progress,
            turns,
            died,
        )
        return EpisodeMetrics(
            episode=episode_idx,
            progress=progress,
            turns=turns,
            died=died,
            parse_errors=parse_errors,
        )
