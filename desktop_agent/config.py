from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AgentConfig:
    task: str
    max_steps: int = 10
    planner_model: str = "qwen2.5-coder:3b"
    vlm_model: str = "moondream"
    use_vlm: str = "auto"
    ollama_host: str = "http://localhost:11434"
    execute: bool = False
    confirm_each_action: bool = False
    hitl: str = "risky"
    runs_dir: Path = Path("runs")
    confidence_threshold: float = 0.35
    stop_on_repeat: int = 2
    max_retries: int = 2
    action_delay: float = 0.4
    langgraph: bool = True

    def validate(self) -> None:
        if not self.task.strip():
            raise ValueError("Task cannot be empty.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.use_vlm not in {"auto", "always", "never"}:
            raise ValueError("use_vlm must be one of: auto, always, never.")
        if self.hitl not in {"off", "risky", "always"}:
            raise ValueError("hitl must be one of: off, risky, always.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        if self.action_delay < 0:
            raise ValueError("action_delay cannot be negative.")
