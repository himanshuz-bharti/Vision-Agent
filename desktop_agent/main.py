from __future__ import annotations

import argparse
from pathlib import Path

from .config import AgentConfig
from .runner import DesktopAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="desktop-agent",
        description="Local screenshot -> OCR/VLM -> planner -> HITL safety -> PyAutoGUI desktop agent.",
    )
    parser.add_argument("task", nargs="*", help="Task for the agent to attempt.")
    parser.add_argument("--execute", action="store_true", help="Actually move/click/type. Default is dry-run.")
    parser.add_argument(
        "--confirm-each-action",
        action="store_true",
        help="Prompt before every low-risk action when --execute is enabled. Equivalent to --hitl always.",
    )
    parser.add_argument(
        "--hitl",
        choices=["off", "risky", "always"],
        default="risky",
        help="Human-in-the-loop mode for --execute: off, risky approvals only, or every action.",
    )
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum loop count.")
    parser.add_argument("--max-retries", type=int, default=2, help="Consecutive failed/uncertain actions before blocking.")
    parser.add_argument("--action-delay", type=float, default=0.4, help="Seconds to wait after an executed action before verification.")
    parser.add_argument("--planner-model", default="qwen2.5-coder:3b", help="Local Ollama text model for action planning.")
    parser.add_argument("--vlm-model", default="moondream", help="Local Ollama vision model for screenshots.")
    parser.add_argument(
        "--use-vlm",
        choices=["auto", "always", "never"],
        default="auto",
        help="Use the vision model never, always, or only when OCR is weak.",
    )
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Directory for screenshots/logs.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
        help="Require HITL approval or block planner actions below this confidence.",
    )
    parser.add_argument(
        "--no-langgraph",
        action="store_true",
        help="Use the built-in loop even if LangGraph is installed.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task = " ".join(args.task).strip()
    hitl = "always" if args.confirm_each_action else args.hitl
    config = AgentConfig(
        task=task,
        max_steps=args.max_steps,
        planner_model=args.planner_model,
        vlm_model=args.vlm_model,
        use_vlm=args.use_vlm,
        ollama_host=args.ollama_host,
        execute=args.execute,
        confirm_each_action=args.confirm_each_action,
        hitl=hitl,
        runs_dir=args.runs_dir,
        confidence_threshold=args.confidence_threshold,
        max_retries=args.max_retries,
        action_delay=args.action_delay,
        langgraph=not args.no_langgraph,
    )
    config.validate()

    agent = DesktopAgent(config)
    result = agent.run()
    print(f"\nStatus: {result.get('status', 'unknown')}")
    if result.get("reason"):
        print(f"Reason: {result['reason']}")
    if result.get("run_dir"):
        print(f"Run log: {result['run_dir']}")
    return 0 if result.get("status") in {"complete", "dry_run", "blocked", "needs_user"} else 1