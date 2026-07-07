import argparse
import os
from pathlib import Path

from .agent import Agent
from .config import Config, DEFAULT_PLANNER_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="desktop-agent",
        description="Local desktop agent: task -> plan -> per-subgoal execution "
        "(screenshot + UIA/VLM) with a human confirmation gate. Ollama only.",
    )
    parser.add_argument("task", nargs="*", help="Natural-language task, e.g. \"Open Notepad and type hello\".")

    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL, help="Ollama model for planning.")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Directory for screenshots/logs.")
    parser.add_argument("--verbose-logs", action="store_true", help="Also stream structured logs to the console.")
    parser.add_argument("--nvidia-key", help="NVIDIA API Key for Planner (e.g. Nemotron).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task = " ".join(args.task).strip()

    config = Config(
        task=task,
        planner_model=args.planner_model,
        ollama_host=args.ollama_host,
        runs_dir=args.runs_dir,
        verbose_logs=args.verbose_logs,
        nvidia_api_key=args.nvidia_key or os.environ.get("NVIDIA_API_KEY"),
    )
    config.validate()

    result = Agent(config).run()
    print(f"\nStatus: {result.get('status', 'unknown')}")
    if result.get("reason"):
        print(f"Reason: {result['reason']}")
    if result.get("run_dir"):
        print(f"Run log: {result['run_dir']}")
    return 0 if result.get("status") in {"complete", "planned", "dry_run", "blocked", "needs_user"} else 1
