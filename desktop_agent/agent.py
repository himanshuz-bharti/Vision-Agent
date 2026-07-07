from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config
from .logging_utils import log_event, log_exception, say, setup_run_logging
from .models import NvidiaClient, OllamaClient
from .planner import (
    PlannerError,
    make_plan,
)


class Agent:
    """Plans a task into subgoals, then executes them one at a time.

    Human-in-the-loop gate (default --hitl subgoal): before each subgoal the user
    chooses [E]xecute / [S]kip / [Q]uit. Within an approved subgoal a small loop runs
    perceive -> plan one action -> safety -> execute -> verify until the subgoal is done,
    the step budget is spent, or it blocks.
    """

    def __init__(self, config: Config):
        self.config = config
        self.config.validate()

        # Parse planner cascade list
        self.planner_models = [m.strip() for m in self.config.planner_model.split(",") if m.strip()]
        self.current_planner_idx = 0

        # Initialize primary planner
        primary_model_str = self.planner_models[0]
        if primary_model_str.startswith("nvidia:"):
            self.active_planner_provider = "nvidia"
            self.active_planner_model = primary_model_str[7:]
        elif primary_model_str.startswith("ollama:"):
            self.active_planner_provider = "ollama"
            self.active_planner_model = primary_model_str[7:]
        else:
            self.active_planner_provider = config.planner_provider
            self.active_planner_model = primary_model_str

        if self.active_planner_provider == "nvidia":
            self.planner_client = NvidiaClient(config.nvidia_api_key)
        else:
            self.planner_client = OllamaClient(config.ollama_host)


    # -- top level ---------------------------------------------------------

    def _hotswap_planner(self, reason: str) -> bool:
        self.current_planner_idx += 1
        if self.current_planner_idx >= len(self.planner_models):
            say(f"Fatal: All {len(self.planner_models)} models in the cascade router failed.", level="ERROR", event="cascade_exhausted")
            return False
            
        next_model_str = self.planner_models[self.current_planner_idx]
        say(f"Primary model {self.active_planner_model} failed ({reason}). Hot-swapping to fallback model {next_model_str}...", level="WARNING", event="model_cascade")
        
        if next_model_str.startswith("nvidia:"):
            self.active_planner_provider, self.active_planner_model = "nvidia", next_model_str[7:]
        elif next_model_str.startswith("ollama:"):
            self.active_planner_provider, self.active_planner_model = "ollama", next_model_str[7:]
        else:
            self.active_planner_provider, self.active_planner_model = self.config.planner_provider, next_model_str
        
        if self.active_planner_provider == "nvidia":
            self.planner_client = NvidiaClient(self.config.nvidia_api_key)
        else:
            self.planner_client = OllamaClient(self.config.ollama_host)
        return True

    def run(self) -> dict[str, Any]:
        run_dir = self._make_run_dir()
        paths = setup_run_logging(run_dir, verbose_console=self.config.verbose_logs)
        self._report(run_dir, paths)

        plan = None
        while True:
            try:
                plan = make_plan(self.planner_client, self.active_planner_model, self.config.task)
                break
            except Exception as exc:
                if not self._hotswap_planner(str(exc)):
                    return {"status": "blocked", "reason": f"Plan failed: {exc}", "run_dir": str(run_dir)}
                    
        print_plan(plan)
        log_event("plan_ready", subgoal_count=len(plan.get("subgoals") or []), plan=plan)
        return {"status": "planned", "plan": plan, "run_dir": str(run_dir)}
    def _report(self, run_dir: Path, paths: dict[str, str]) -> None:
        say(f"Task: {self.config.task}", event="run_started", task=self.config.task)
        p_provider = self.config.planner_provider
        if p_provider in ("nvidia", "hf"):
            say(
                "Models: "
                f"planner={p_provider}:{self.config.planner_model}",
                event="model_report",
            )
            say(f"Run logs: {paths['agent_log']}", event="log_paths", paths=paths)
            return

        preflight = self.planner_client.available_models()
        say(
            "Models: "
            f"planner=ollama:{self.config.planner_model}",
            event="model_report",
        )
        if not preflight["reachable"]:
            say(
                f"WARNING: Ollama not reachable at {self.config.ollama_host} ({preflight['error']}). "
                "Start it with `ollama serve`.",
                level="WARNING",
                event="ollama_unreachable",
            )
        else:
            missing = [
                m for m in {self.config.planner_model}
                if not _model_present(m, preflight["models"])
            ]
        say(f"Run logs: {paths['agent_log']}", event="log_paths", paths=paths)



    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def print_plan(plan: dict[str, Any]) -> None:
    subgoals = (plan or {}).get("subgoals") or []
    say(f"\nPlan ({len(subgoals)} subgoals):", event="plan_printed", subgoal_count=len(subgoals))
    for subgoal in subgoals:
        done_when = subgoal.get("done_when")
        suffix = f"   (done when: {done_when})" if done_when else ""
        print(f"  {subgoal.get('id')}. {subgoal.get('goal')}{suffix}")
    if (plan or {}).get("notes"):
        print(f"  notes: {plan['notes']}")


def _prompt(message: str) -> str:
    try:
        return input(message).strip().lower()
    except EOFError:
        return ""

