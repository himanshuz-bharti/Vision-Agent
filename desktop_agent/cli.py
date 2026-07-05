import argparse
import os
from pathlib import Path

from .agent import Agent
from .config import Config, DEFAULT_PLANNER_MODEL, DEFAULT_VLM_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="desktop-agent",
        description="Local desktop agent: task -> plan -> per-subgoal execution "
        "(screenshot + UIA/VLM) with a human confirmation gate. Ollama only.",
    )
    parser.add_argument("task", nargs="*", help="Natural-language task, e.g. \"Open Notepad and type hello\".")
    parser.add_argument("--plan-only", action="store_true", help="Only produce and print the plan, then stop.")
    parser.add_argument("--execute", action="store_true", help="Actually move/click/type. Default is dry-run.")
    parser.add_argument(
        "--hitl",
        choices=["subgoal", "action", "off"],
        default="subgoal",
        help="Human-in-the-loop: confirm before each subgoal (default), before each action, or off.",
    )
    parser.add_argument("--max-steps-per-subgoal", type=int, default=6, help="Max actions attempted per subgoal.")
    parser.add_argument(
        "--use-vlm",
        choices=["auto", "always", "never"],
        default="always",
        help="Use the vision model never, always (default), or auto.",
    )
    parser.add_argument(
        "--grounding",
        choices=["uia"],
        default="uia",
        help="Clickable-target source: Windows UI Automation (default).",
    )
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL, help="Ollama model for planning.")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL, help="Ollama model for screenshots/grounding.")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--max-ui-candidates", type=int, default=150, help="Max UI/OCR candidates per screenshot.")
    parser.add_argument("--action-delay", type=float, default=0.4, help="Seconds to wait after an action before verifying.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Directory for screenshots/logs.")
    parser.add_argument("--verbose-logs", action="store_true", help="Also stream structured logs to the console.")
    parser.add_argument("--hf-token", help="Hugging Face API Token.")
    parser.add_argument("--groq-key", help="Groq API Key (free tier: ~30 req/min). Get one at https://console.groq.com/keys")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task = " ".join(args.task).strip()

    config = Config(
        task=task,
        execute=args.execute,
        plan_only=args.plan_only,
        hitl=args.hitl,
        max_steps_per_subgoal=args.max_steps_per_subgoal,
        use_vlm=args.use_vlm,
        grounding=args.grounding,
        planner_model=args.planner_model,
        vlm_model=args.vlm_model,
        ollama_host=args.ollama_host,
        max_ui_candidates=args.max_ui_candidates,
        action_delay=args.action_delay,
        runs_dir=args.runs_dir,
        verbose_logs=args.verbose_logs,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        groq_api_key=args.groq_key or os.environ.get("GROQ_API_KEY"),
    )
    config.validate()

    result = Agent(config).run()
    print(f"\nStatus: {result.get('status', 'unknown')}")
    if result.get("reason"):
        print(f"Reason: {result['reason']}")
    if result.get("run_dir"):
        print(f"Run log: {result['run_dir']}")
    return 0 if result.get("status") in {"complete", "planned", "dry_run", "blocked", "needs_user"} else 1
