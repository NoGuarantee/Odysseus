from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev


@dataclass
class EpisodeMetrics:
    episode: int
    progress: int
    turns: int
    died: bool
    parse_errors: int = 0

    @property
    def parse_error_rate(self) -> float:
        if self.turns == 0:
            return 0.0
        return self.parse_errors / self.turns


@dataclass
class MetricsCollector:
    episodes: list[EpisodeMetrics] = field(default_factory=list)

    def add(self, metrics: EpisodeMetrics) -> None:
        self.episodes.append(metrics)

    def aggregate(self) -> dict:
        if not self.episodes:
            return {
                "num_episodes": 0,
                "mean_progress": 0.0,
                "std_progress": 0.0,
                "mean_turns": 0.0,
                "mean_parse_error_rate": 0.0,
                "episodes": [],
            }

        progresses = [ep.progress for ep in self.episodes]
        turns = [ep.turns for ep in self.episodes]
        parse_rates = [ep.parse_error_rate for ep in self.episodes]

        return {
            "num_episodes": len(self.episodes),
            "mean_progress": mean(progresses),
            "std_progress": pstdev(progresses) if len(progresses) > 1 else 0.0,
            "mean_turns": mean(turns),
            "mean_parse_error_rate": mean(parse_rates),
            "episodes": [
                {
                    "episode": ep.episode,
                    "progress": ep.progress,
                    "turns": ep.turns,
                    "died": ep.died,
                    "parse_error_rate": ep.parse_error_rate,
                }
                for ep in self.episodes
            ],
        }


def aggregate_metrics(episodes: list[EpisodeMetrics]) -> dict:
    collector = MetricsCollector()
    for episode in episodes:
        collector.add(episode)
    return collector.aggregate()
