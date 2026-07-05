from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .actions import execute_action
from .config import Config
from .logging_utils import log_event, log_exception, say, setup_run_logging
from .models import OllamaClient, HuggingFaceClient, GroqClient
from .perception.perceiver import Perceiver
from .perception.screen import CaptureError, capture_screenshot
from .perception.uia import uia_status
from .perception.vlm import ground_target
from .planner import (
    PlannerError,
    current_subgoal,
    make_plan,
    next_action,
    remaining_subgoals,
    subgoal_done_by_text,
)
from .safety import evaluate_safety
from .verifier import verify_result


class Agent:
    """Plans a task into subgoals, then executes them one at a time.

    Human-in-the-loop gate (default --hitl subgoal): before each subgoal the user
    chooses [E]xecute / [S]kip / [Q]uit. Within an approved subgoal a small loop runs
    perceive -> plan one action -> safety -> execute -> verify until the subgoal is done,
    the step budget is spent, or it blocks.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        if config.provider == "groq":
            self.client = GroqClient(config.groq_api_key)
        elif config.provider == "hf":
            self.client = HuggingFaceClient(config.hf_token)
        else:
            self.client = OllamaClient(config.ollama_host)
        self.perceiver = Perceiver(
            self.client,
            vlm_model=config.vlm_model,
            use_vlm=config.use_vlm,
            grounding=config.grounding,
            max_ui_candidates=config.max_ui_candidates,
        )

    # -- top level ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        run_dir = self._make_run_dir()
        paths = setup_run_logging(run_dir, verbose_console=self.config.verbose_logs)
        self._report(run_dir, paths)

        plan = make_plan(self.client, self.config.planner_model, self.config.task)
        print_plan(plan)
        log_event("plan_ready", subgoal_count=len(plan.get("subgoals") or []), plan=plan)

        if self.config.plan_only:
            say("Plan-only mode: stopping before execution.", event="plan_only_stop")
            return {"status": "planned", "plan": plan, "run_dir": str(run_dir)}

        if self.config.hitl != "off":
            approved = _prompt("\nApprove and execute this plan? [Y/n] ") in {"y", "yes", ""}
            if not approved:
                say("Plan rejected by user. Quitting.", event="plan_rejected")
                return {"status": "needs_user", "reason": "Plan rejected by user.", "run_dir": str(run_dir)}
            # Once plan is approved, run all subgoals and actions autonomously
            self.config.hitl = "off"

        return self._run_subgoals(plan, run_dir)

    def _run_subgoals(self, plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        subgoals = plan.get("subgoals") or []
        if not self.config.execute:
            say(
                "\nDRY RUN (no --execute): the agent will show the FIRST action it would take "
                "for each approved subgoal, without touching the mouse/keyboard.",
                event="dry_run_notice",
            )
        results: list[dict[str, Any]] = []
        for index, subgoal in enumerate(subgoals):
            remaining = remaining_subgoals(plan, index)
            choice = self._confirm_subgoal(subgoal, index, len(subgoals))
            if choice == "quit":
                say("Stopped by user.", event="user_quit")
                results.append({"subgoal": subgoal, "status": "quit"})
                return self._summary(results, run_dir, status="needs_user", reason="User quit.")
            if choice == "skip":
                say(f"Skipped subgoal {subgoal.get('id')}.", event="subgoal_skipped", subgoal=subgoal)
                results.append({"subgoal": subgoal, "status": "skipped"})
                continue

            outcome = self._run_one_subgoal(subgoal, remaining, run_dir, index)
            results.append({"subgoal": subgoal, "status": outcome["status"], "reason": outcome.get("reason")})
            if outcome["status"] in {"blocked", "needs_user"} and self.config.execute:
                cont = _prompt("Continue with the next subgoal anyway? [y/N] ")
                if cont not in {"y", "yes"}:
                    return self._summary(results, run_dir, status=outcome["status"], reason=outcome.get("reason"))

        return self._summary(results, run_dir, status="complete", reason="All subgoals processed.")

    # -- per-subgoal inner loop -------------------------------------------

    def _run_one_subgoal(
        self, subgoal: dict[str, Any], remaining: list[dict[str, Any]], run_dir: Path, sg_index: int
    ) -> dict[str, Any]:
        say(f"\n--- Subgoal {subgoal.get('id')}: {subgoal.get('goal')} ---", event="subgoal_started", subgoal=subgoal)
        history: list[dict[str, Any]] = []
        critiques: list[str] = []
        consecutive_failures = 0
        last_fingerprint: str | None = None
        repeat_count = 0

        for step in range(1, self.config.max_steps_per_subgoal + 1):
            # 1. Capture + perceive
            before_path = run_dir / f"sg{sg_index + 1:02d}_step{step:02d}_before.png"
            try:
                meta = capture_screenshot(before_path)
            except CaptureError as exc:
                say(f"Capture failed: {exc}", level="ERROR", event="capture_error")
                return {"status": "blocked", "reason": str(exc)}
            observation = self.perceiver.perceive(before_path, meta, self.config.task)
            self._print_perception(step, observation)

            if subgoal_done_by_text(subgoal, observation):
                say("Subgoal's done_when condition looks satisfied on screen.", event="subgoal_done_by_text")
                return {"status": "done", "reason": "done_when visible before action"}

            rereason_count = 0
            while True:
                # 2. Plan one action
                try:
                    action, raw = next_action(
                        self.client,
                        self.config.planner_model,
                        task=self.config.task,
                        subgoal=subgoal,
                        remaining=remaining,
                        observation=observation,
                        history=history,
                        step=step,
                        max_steps=self.config.max_steps_per_subgoal,
                        critiques=critiques,
                    )
                except PlannerError as exc:
                    say(f"Planner could not produce an action: {exc}", level="WARNING", event="planner_failed")
                    return {"status": "needs_user", "reason": str(exc)}

                say(
                    f"Action: {action['action']} "
                    f"(confidence {action.get('confidence', 0):.2f}) - {action.get('reason', '')}",
                    event="action_decided",
                    action=action,
                )

                # 3. Ground a described click target to a real candidate via the VLM
                action = self._ground_if_needed(action, observation, step)

                # 4. Safety gate
                decision = evaluate_safety(
                    self.config.task, action, observation, confidence_threshold=self.config.confidence_threshold
                )
                if not decision["allowed"]:
                    if decision.get("hard_block") or self.config.hitl == "off":
                        say(f"Safety blocked: {decision['reason']}", level="WARNING", event="safety_blocked", decision=decision)
                        return {"status": "blocked", "reason": decision["reason"]}
                    say(f"Safety warning: {decision['reason']}", level="WARNING")

                # 5. Human feedback gate
                if self.config.hitl != "off":
                    print("\nProposed Action:")
                    print(json.dumps(action, indent=2, ensure_ascii=True))
                    feedback = _prompt("Approve this action? [Y/n] (or type feedback/reason if 'No'): ")
                    if feedback in {"n", "no"} or (feedback and feedback not in {"y", "yes"}):
                        rereason_count += 1
                        if rereason_count >= 3:
                            say("Action declined 3 times. Quitting the task.", level="WARNING", event="rereason_limit_reached")
                            return {"status": "needs_user", "reason": "Action declined 3 times by user."}

                        critique_msg = f"User rejected the proposed action: {action['action']}. "
                        if feedback not in {"n", "no"}:
                            critique_msg += f"User feedback: {feedback}"
                        else:
                            critique_msg += "Please reason and choose a different action."
                        critiques.append(critique_msg)
                        say(f"Action declined. Rereasoning (attempt {rereason_count}/3)...", event="rereasoning")
                        continue

                if not decision["allowed"]:
                    say(f"Safety: human-approved once ({decision['reason']})", event="safety_override")
                elif self.config.hitl != "off":
                    say(f"Safety: allowed ({decision['reason']})", event="safety_allowed")

                break

            if action["action"] == "finish":
                say(f"Subgoal complete: {action.get('text') or subgoal.get('goal')}", event="subgoal_finished")
                return {"status": "done", "reason": action.get("text") or "planner finished subgoal"}
            if action["action"] == "ask_user":
                say(f"Agent needs input: {action.get('text')}", event="ask_user", action=action)
                return {"status": "needs_user", "reason": action.get("text") or "planner asked user"}

            # 6. Dry-run: show the first action and stop this subgoal
            if not self.config.execute:
                result = execute_action(action, observation, dry_run=True)
                say(result["message"], event="dry_run_action")
                return {"status": "dry_run", "reason": "Dry-run shows the first planned action only."}

            # 7. Execute
            try:
                execution = execute_action(action, observation, dry_run=False)
            except Exception as exc:
                execution = {"executed": False, "dry_run": False, "error": type(exc).__name__, "message": str(exc)}
                log_exception("execution_error", exc, action=action)
            say(f"Executed: {execution['message']}", event="executed", execution=execution)

            # 8. Capture after + verify
            after_observation = None
            if not execution.get("error"):
                delay = self.config.action_delay
                if action["action"] in {"open_app", "open_url"}:
                    delay = max(delay, 2.5)
                time.sleep(delay)
                after_path = run_dir / f"sg{sg_index + 1:02d}_step{step:02d}_after.png"
                try:
                    after_meta = capture_screenshot(after_path)
                    after_observation = self.perceiver.perceive(after_path, after_meta, self.config.task, skip_vlm=True)
                except CaptureError as exc:
                    say(f"Post-action capture failed: {exc}", level="WARNING", event="post_capture_error")

            verification = verify_result(action, observation, after_observation, execution)
            say(f"Verify: {verification['status']} - {verification['message']}", event="verified", verification=verification)

            history.append({"step": step, "action": action, "verification": verification})
            self._write_step_log(run_dir, sg_index, step, subgoal, observation, action, decision, execution, verification)

            # 9. Progress / repetition / failure accounting
            if verification.get("success") is True:
                consecutive_failures = 0
                if subgoal_done_by_text(subgoal, after_observation):
                    say("Subgoal's done_when condition looks satisfied on screen.", event="subgoal_done_by_text")
                    return {"status": "done", "reason": "done_when visible after action"}
            elif verification["status"] in {"failed", "uncertain"}:
                consecutive_failures += 1
                critiques.append(
                    f"Step {step} action '{action['action']}' did not clearly progress "
                    f"({verification['message']}). Try a different approach."
                )
                if consecutive_failures >= self.config.stop_on_repeat:
                    say("Too many non-progressing actions; stopping this subgoal.", level="WARNING", event="subgoal_stalled")
                    return {"status": "blocked", "reason": f"No progress after {consecutive_failures} attempts."}

            fingerprint = _fingerprint(action)
            repeat_count = repeat_count + 1 if fingerprint == last_fingerprint else 0
            last_fingerprint = fingerprint
            if repeat_count + 1 >= self.config.stop_on_repeat and verification.get("success") is not True:
                say("Same action repeated without progress; stopping this subgoal.", level="WARNING", event="repeat_guard")
                return {"status": "blocked", "reason": "Repeated the same action without progress."}

        say(f"Reached the {self.config.max_steps_per_subgoal}-step budget for this subgoal.", event="subgoal_budget")
        return {"status": "blocked", "reason": "Step budget exhausted for this subgoal."}

    # -- grounding ---------------------------------------------------------

    def _ground_if_needed(self, action: dict[str, Any], observation: dict[str, Any], step: int) -> dict[str, Any]:
        if action.get("action") not in {"click", "double_click", "type"}:
            return action
        if action.get("target_id") and _candidate_exists(str(action["target_id"]), observation):
            return action
        description = action.get("target_description")
        if not description or self.config.use_vlm == "never":
            return action
        marks_image = observation.get("marks_image")
        mark_map = {int(m): tid for m, tid in (observation.get("mark_map") or {}).items()}
        if not marks_image or not mark_map:
            return action
        say(f"Grounding '{description}' to an on-screen mark via the VLM ...", event="grounding_started")
        resolved = ground_target(self.client, self.config.vlm_model, Path(marks_image), description, mark_map)
        if resolved.get("target_id"):
            action = dict(action)
            action["target_id"] = resolved["target_id"]
            say(f"Grounded to {resolved['target_id']} (mark {resolved.get('mark')}).", event="grounding_resolved", resolved=resolved)
        else:
            say(f"Grounding failed: {resolved.get('error')}", level="WARNING", event="grounding_failed", resolved=resolved)
        return action

    # -- console + reporting ----------------------------------------------

    def _confirm_subgoal(self, subgoal: dict[str, Any], index: int, total: int) -> str:
        if self.config.hitl == "off":
            return "execute"
        print(f"\n>>> Subgoal {index + 1}/{total}: {subgoal.get('goal')}")
        if subgoal.get("done_when"):
            print(f"    done when: {subgoal['done_when']}")
        answer = _prompt("    [E]xecute / [S]kip / [Q]uit? ")
        if answer in {"q", "quit"}:
            return "quit"
        if answer in {"s", "skip"}:
            return "skip"
        return "execute"

    def _print_perception(self, step: int, observation: dict[str, Any]) -> None:
        vlm = observation.get("vlm") or {}
        candidates = observation.get("ui_candidates") or []
        say(
            f"Perceived (step {step}): "
            f"{observation.get('uia', {}).get('item_count', 0)} UIA controls, "
            f"{len(candidates)} click candidates, VLM {'on' if vlm.get('summary') else 'off'}.",
            event="perceived",
        )

    def _report(self, run_dir: Path, paths: dict[str, str]) -> None:
        say(f"Task: {self.config.task}", event="run_started", task=self.config.task)
        provider = self.config.provider
        if provider in ("groq", "hf"):
            say(
                "Models: "
                f"planner={provider}:{self.config.planner_model} | vlm={provider}:{self.config.vlm_model} | "
                f"use_vlm={self.config.use_vlm} | grounding={self.config.grounding} | "
                f"uia={'on' if uia_status().get('available') else 'off'} | "
                f"hitl={self.config.hitl} | execute={self.config.execute}",
                event="model_report",
            )
            say(f"Run logs: {paths['agent_log']}", event="log_paths", paths=paths)
            return

        preflight = self.client.available_models()
        say(
            "Models: "
            f"planner=ollama:{self.config.planner_model} | vlm=ollama:{self.config.vlm_model} | "
            f"use_vlm={self.config.use_vlm} | grounding={self.config.grounding} | "
            f"uia={'on' if uia_status().get('available') else 'off'} | "
            f"hitl={self.config.hitl} | execute={self.config.execute}",
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
                m for m in {self.config.planner_model, self.config.vlm_model}
                if not _model_present(m, preflight["models"])
            ]
            if missing:
                say(
                    f"WARNING: these models are not pulled yet: {', '.join(sorted(missing))}. "
                    "Pull with `ollama pull <model>`.",
                    level="WARNING",
                    event="models_missing",
                    missing=missing,
                    available=preflight["models"],
                )
        say(f"Run logs: {paths['agent_log']}", event="log_paths", paths=paths)

    def _summary(self, results: list[dict[str, Any]], run_dir: Path, *, status: str, reason: str) -> dict[str, Any]:
        say("\n=== Summary ===", event="summary")
        for item in results:
            sg = item["subgoal"]
            print(f"  [{item['status']:>8}] {sg.get('id')}. {sg.get('goal')}")
        return {"status": status, "reason": reason, "results": results, "run_dir": str(run_dir)}

    def _write_step_log(
        self, run_dir: Path, sg_index: int, step: int, subgoal, observation, action, decision, execution, verification
    ) -> None:
        path = run_dir / f"sg{sg_index + 1:02d}_step{step:02d}.json"
        payload = {
            "task": self.config.task,
            "subgoal": subgoal,
            "step": step,
            "observation": {k: v for k, v in observation.items() if k != "ui_candidates"} | {
                "ui_candidate_count": len(observation.get("ui_candidates") or [])
            },
            "action": action,
            "safety": decision,
            "execution": execution,
            "verification": verification,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")

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


def _confirm_action(action: dict[str, Any]) -> bool:
    print("\nAction confirmation:")
    print(json.dumps(action, indent=2, ensure_ascii=True))
    approved = _prompt("Execute this action? [y/N] ") in {"y", "yes"}
    log_event("hitl_action_confirmation", approved=approved, action=action)
    return approved


def _confirm_safety(action: dict[str, Any], decision: dict[str, Any]) -> bool:
    print(f"\nSafety approval required: {decision.get('reason')}")
    print(json.dumps(action, indent=2, ensure_ascii=True))
    approved = _prompt("Approve this one action? [y/N] ") in {"y", "yes"}
    log_event("hitl_safety_override", approved=approved, action=action, decision=decision)
    return approved


def _candidate_exists(target_id: str, observation: dict[str, Any]) -> bool:
    return any(str(c.get("id")) == target_id for c in observation.get("ui_candidates") or [])


def _fingerprint(action: dict[str, Any]) -> str:
    return json.dumps(
        {
            "action": action.get("action"),
            "target_id": action.get("target_id"),
            "text": action.get("text"),
            "keys": action.get("keys"),
            "scroll_amount": action.get("scroll_amount"),
        },
        sort_keys=True,
    )


def _model_present(model: str, available: list[str]) -> bool:
    wanted = model if ":" in model else f"{model}:latest"
    return any(name == model or name == wanted for name in available)
