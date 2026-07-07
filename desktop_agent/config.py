import os
from dataclasses import dataclass
from pathlib import Path


def load_env() -> None:
    env_path = Path(".env")
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")
        except Exception:
            pass


load_env()

# Stronger local defaults: qwen2.5 follows JSON/structured instructions well, and
# qwen2.5vl supports UI grounding. Override with --planner-model / --vlm-model.
DEFAULT_PLANNER_MODEL = "qwen2.5:7b"

# NVIDIA API defaults
NVIDIA_DEFAULT_PLANNER = "nvidia:nvidia/nemotron-3-ultra-550b-a55b,ollama:qwen2.5:7b"


@dataclass(slots=True)
class Config:
    """All runtime settings for a single agent run."""

    task: str
    planner_model: str = DEFAULT_PLANNER_MODEL
    ollama_host: str = "http://localhost:11434"
    runs_dir: Path = Path("runs")
    verbose_logs: bool = False
    nvidia_api_key: str | None = None

    # Populated by validate(); tells agent.py which provider to use.
    planner_provider: str = "ollama"  # "nvidia" | "ollama"

    def validate(self) -> None:
        # Only override model defaults if the user hasn't explicitly set them.
        user_set_planner = self.planner_model != DEFAULT_PLANNER_MODEL

        # Determine planner provider
        if self.nvidia_api_key:
            self.planner_provider = "nvidia"
            if not user_set_planner:
                self.planner_model = NVIDIA_DEFAULT_PLANNER
        else:
            self.planner_provider = "ollama"


        if not self.task.strip():
            raise ValueError("Task cannot be empty.")
