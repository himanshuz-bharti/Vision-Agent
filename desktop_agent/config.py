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
DEFAULT_VLM_MODEL = "qwen2.5vl:7b"

# Groq free-tier defaults (much more generous rate limits than HF).
GROQ_DEFAULT_PLANNER = "llama-3.3-70b-versatile"
GROQ_DEFAULT_VLM = "llama-3.2-90b-vision-preview"

# Hugging Face serverless defaults.
HF_DEFAULT_PLANNER = "Qwen/Qwen2.5-7B-Instruct"
HF_DEFAULT_VLM = "Qwen/Qwen3-VL-30B-A3B-Instruct"


@dataclass(slots=True)
class Config:
    """All runtime settings for a single agent run."""

    task: str
    execute: bool = False              # dry-run unless True
    plan_only: bool = False            # stop after producing the plan
    hitl: str = "subgoal"              # subgoal | action | off
    max_steps_per_subgoal: int = 6
    use_vlm: str = "always"             # auto | always | never
    grounding: str = "uia"             # uia
    planner_model: str = DEFAULT_PLANNER_MODEL
    vlm_model: str = DEFAULT_VLM_MODEL
    ollama_host: str = "http://localhost:11434"
    max_ui_candidates: int = 150
    action_delay: float = 0.4
    confidence_threshold: float = 0.35
    stop_on_repeat: int = 2
    runs_dir: Path = Path("runs")
    verbose_logs: bool = False
    hf_token: str | None = None
    groq_api_key: str | None = None

    # Populated by validate(); tells agent.py which provider to use.
    provider: str = "ollama"  # "groq" | "hf" | "ollama"

    def validate(self) -> None:
        # Provider priority: Groq > HuggingFace > Ollama.
        # Only override model defaults if the user hasn't explicitly set them.
        user_set_planner = self.planner_model != DEFAULT_PLANNER_MODEL
        user_set_vlm = self.vlm_model != DEFAULT_VLM_MODEL

        if self.groq_api_key:
            self.provider = "groq"
            if not user_set_planner:
                self.planner_model = GROQ_DEFAULT_PLANNER
            if not user_set_vlm:
                self.vlm_model = GROQ_DEFAULT_VLM
        elif self.hf_token:
            self.provider = "hf"
            if not user_set_planner:
                self.planner_model = HF_DEFAULT_PLANNER
            if not user_set_vlm:
                self.vlm_model = HF_DEFAULT_VLM
        else:
            self.provider = "ollama"

        if not self.task.strip() and not self.plan_only:
            raise ValueError("Task cannot be empty.")
        if not self.task.strip():
            raise ValueError("Task cannot be empty.")
        if self.hitl not in {"subgoal", "action", "off"}:
            raise ValueError("hitl must be one of: subgoal, action, off.")
        if self.use_vlm not in {"auto", "always", "never"}:
            raise ValueError("use_vlm must be one of: auto, always, never.")
        if self.grounding not in {"uia"}:
            raise ValueError("grounding must be 'uia'.")
        if self.max_steps_per_subgoal < 1:
            raise ValueError("max_steps_per_subgoal must be at least 1.")
        if self.max_ui_candidates < 1:
            raise ValueError("max_ui_candidates must be at least 1.")
        if self.action_delay < 0:
            raise ValueError("action_delay cannot be negative.")
