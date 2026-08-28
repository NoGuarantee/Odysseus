from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TurnRecord:
    episode: int
    turn: int
    buttons: list[str]
    progress: int
    delta_progress: int
    died: bool
    parse_error: bool
    raw_response: str


class EvalLogger:
    def __init__(self, output_dir: str | Path, run_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_path = self.output_dir / f"{run_name}.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self._file = self.trajectory_path.open("w", encoding="utf-8")

    def log_turn(self, record: TurnRecord) -> None:
        self._file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._file.flush()

    def log_episode_summary(self, episode_summary: dict[str, Any]) -> None:
        episode_summary["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode_summary, ensure_ascii=False) + "\n")

    def write_final_summary(self, summary: dict[str, Any]) -> None:
        summary["timestamp"] = datetime.now(timezone.utc).isoformat()
        final_path = self.output_dir / "summary_final.json"
        final_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> EvalLogger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
