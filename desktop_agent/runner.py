from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .actions import execute_action
from .config import AgentConfig
from .models.ollama_client import OllamaClient
from .perception.capture import CaptureError, capture_screenshot
from .perception.perceiver import Perceiver
from .planner import Planner, PlannerError
from .safety import evaluate_safety
from .verifier import verify_result


class DesktopAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = OllamaClient(config.ollama_host)
        self.perceiver = Perceiver(self.client, vlm_model=config.vlm_model, use_vlm=config.use_vlm)
        self.planner = Planner(self.client, config.planner_model)

    def run(self) -> dict[str, Any]:
        if self.config.langgraph:
            try:
                app = self._compile_langgraph()
            except ImportError:
                print("LangGraph is not installed; using built-in loop.")
            except Exception as exc:
                print(f"Could not compile LangGraph graph; using built-in loop. Details: {exc}")
            else:
                try:
                    return app.invoke(self._initial_state())
                except CaptureError as exc:
                    return {"status": "blocked", "reason": str(exc), "run_dir": None}
        try:
            return self._run_builtin()
        except CaptureError as exc:
            return {"status": "blocked", "reason": str(exc), "run_dir": None}

    def _initial_state(self) -> dict[str, Any]:
        run_dir = self._make_run_dir()
        return {
            "task": self.config.task,
            "max_steps": self.config.max_steps,
            "step": 0,
            "run_dir": str(run_dir),
            "history": [],
            "action_counts": {},
            "last_action": None,
            "last_result": None,
            "status": "running",
            "reason": None,
        }

    def _compile_langgraph(self):
        from langgraph.graph import END, StateGraph

        graph = StateGraph(dict)
        graph.add_node("capture", self._capture_node)
        graph.add_node("perceive", self._perceive_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("safety", self._safety_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("verify", self._verify_node)

        graph.set_entry_point("capture")
        graph.add_edge("capture", "perceive")
        graph.add_edge("perceive", "plan")
        graph.add_edge("plan", "safety")
        graph.add_edge("safety", "execute")
        graph.add_edge("execute", "verify")
        graph.add_conditional_edges("verify", self._route_after_verify, {"continue": "capture", "finish": END})

        return graph.compile()

    def _run_builtin(self) -> dict[str, Any]:
        state = self._initial_state()
        while state.get("status") == "running" and state["step"] < state["max_steps"]:
            for node in (
                self._capture_node,
                self._perceive_node,
                self._plan_node,
                self._safety_node,
                self._execute_node,
                self._verify_node,
            ):
                state = node(state)
            if self._route_after_verify(state) == "finish":
                break
        return state

    def _capture_node(self, state: dict[str, Any]) -> dict[str, Any]:
        state = dict(state)
        state["step"] = int(state.get("step", 0)) + 1
        screenshot_path = Path(state["run_dir"]) / f"step_{state['step']:02d}_before.png"
        state["screenshot_meta"] = capture_screenshot(screenshot_path)
        print(f"\nStep {state['step']}: captured screenshot")
        return state

    def _perceive_node(self, state: dict[str, Any]) -> dict[str, Any]:
        state = dict(state)
        screenshot_path = Path(state["screenshot_meta"]["path"])
        state["observation"] = self.perceiver.perceive(screenshot_path, state["screenshot_meta"])
        ocr_count = len((state["observation"].get("ocr") or {}).get("items") or [])
        vlm = state["observation"].get("vlm") or {}
        print(f"Perception: {ocr_count} OCR items, VLM {'used' if vlm.get('summary') else 'not used'}")
        observation_action = maybe_answer_observation_task(state["task"], state["observation"])
        if observation_action:
            state["action"] = observation_action
            state["planner_bypassed"] = True
        return state

    def _plan_node(self, state: dict[str, Any]) -> dict[str, Any]:
        state = dict(state)
        if state.get("action"):
            action = state["action"]
            print(
                f"Planner: bypassed action={action['action']} "
                f"confidence={action.get('confidence', 0):.2f} reason={action.get('reason', '')}"
            )
            return state
        try:
            action = self.planner.plan(state)
        except PlannerError as exc:
            action = {
                "action": "ask_user",
                "target_id": None,
                "coordinates": None,
                "text": "The planner model returned an invalid action. Try a stronger planner model or simplify the task.",
                "keys": None,
                "scroll_amount": None,
                "seconds": None,
                "confidence": 0.0,
                "expected_result": "User changes planner model or task wording.",
                "reason": str(exc),
            }
            state["planner_error"] = str(exc)
            state["planner_raw_response"] = exc.raw_response
        except Exception as exc:
            action = {
                "action": "ask_user",
                "target_id": None,
                "coordinates": None,
                "text": "The planner failed before producing an action.",
                "keys": None,
                "scroll_amount": None,
                "seconds": None,
                "confidence": 0.0,
                "expected_result": "User checks the local model and logs.",
                "reason": str(exc),
            }
            state["planner_error"] = str(exc)
            state["planner_raw_response"] = None
        state["action"] = action
        print(f"Planner: {action['action']} confidence={action.get('confidence', 0):.2f} reason={action.get('reason', '')}")
        return state

    def _safety_node(self, state: dict[str, Any]) -> dict[str, Any]:
        state = dict(state)
        decision = evaluate_safety(
            state["task"],
            state["action"],
            state["observation"],
            confidence_threshold=self.config.confidence_threshold,
        )
        state["safety"] = decision
        if decision["allowed"]:
            print(f"Safety: allowed ({decision['reason']})")
        else:
            state["status"] = "needs_user" if decision.get("requires_user") else "blocked"
            state["reason"] = decision["reason"]
            print(f"Safety: blocked ({decision['reason']})")
        return state

    def _execute_node(self, state: dict[str, Any]) -> dict[str, Any]:
        state = dict(state)
        action = state["action"]

        if not state.get("safety", {}).get("allowed"):
            state["execution"] = {"executed": False, "dry_run": False, "message": state.get("reason")}
            return state

        if action["action"] == "finish":
            state["status"] = "complete"
        elif action["action"] == "ask_user":
            state["status"] = "needs_user"
            state["reason"] = action.get("text") or "Planner requested user input."

        dry_run = not self.config.execute and action["action"] not in {"finish", "ask_user"}
        if self.config.execute and self.config.confirm_each_action and action["action"] not in {"finish", "ask_user"}:
            if not confirm_action(action):
                state["status"] = "needs_user"
                state["reason"] = "User declined action confirmation."
                state["execution"] = {"executed": False, "dry_run": False, "message": state["reason"]}
                return state

        state["execution"] = execute_action(action, state["observation"], dry_run=dry_run)
        print(f"Executor: {state['execution']['message']}")
        if dry_run:
            state["status"] = "dry_run"
            state["reason"] = "Dry-run mode stops after the first planned action. Add --execute to run the loop."
        elif action["action"] not in {"finish", "ask_user"}:
            time.sleep(0.4)
            after_path = Path(state["run_dir"]) / f"step_{state['step']:02d}_after.png"
            state["after_screenshot_meta"] = capture_screenshot(after_path)
            state["after_observation"] = self.perceiver.perceive(after_path, state["after_screenshot_meta"])
        return state

    def _verify_node(self, state: dict[str, Any]) -> dict[str, Any]:
        state = dict(state)
        after_observation = state.get("after_observation")
        verification = verify_result(state["action"], state["observation"], after_observation, state["execution"])
        state["verification"] = verification
        state["last_action"] = state["action"]
        state["last_result"] = verification

        if state.get("status") == "running" and verification["status"] == "failed":
            state["reason"] = verification["message"]
        elif verification["status"] in {"complete", "needs_user", "dry_run"}:
            state["status"] = verification["status"]
            state["reason"] = verification["message"]

        self._track_repetition(state)
        if state.get("status") == "running" and int(state.get("step", 0)) >= int(state.get("max_steps", 0)):
            state["status"] = "blocked"
            state["reason"] = f"Reached max_steps={state.get('max_steps')}."
        self._write_step_log(state)
        print(f"Verifier: {verification['status']} ({verification['message']})")
        return state

    def _route_after_verify(self, state: dict[str, Any]) -> str:
        if state.get("status") != "running":
            return "finish"
        if int(state.get("step", 0)) >= int(state.get("max_steps", 0)):
            state["status"] = "blocked"
            state["reason"] = f"Reached max_steps={state.get('max_steps')}."
            return "finish"
        return "continue"

    def _track_repetition(self, state: dict[str, Any]) -> None:
        if state.get("status") != "running":
            return
        action = state.get("action") or {}
        fingerprint = json.dumps(
            {
                "action": action.get("action"),
                "target_id": action.get("target_id"),
                "coordinates": action.get("coordinates"),
                "text": action.get("text"),
                "keys": action.get("keys"),
                "scroll_amount": action.get("scroll_amount"),
            },
            sort_keys=True,
        )
        counts = state.setdefault("action_counts", {})
        counts[fingerprint] = int(counts.get(fingerprint, 0)) + 1
        if counts[fingerprint] >= self.config.stop_on_repeat:
            state["status"] = "blocked"
            state["reason"] = "Same action repeated without completing the task."

    def _write_step_log(self, state: dict[str, Any]) -> None:
        log_path = Path(state["run_dir"]) / f"step_{state['step']:02d}.json"
        payload = {
            "task": state["task"],
            "step": state["step"],
            "observation": state.get("observation"),
            "action": state.get("action"),
            "planner_bypassed": state.get("planner_bypassed"),
            "planner_error": state.get("planner_error"),
            "planner_raw_response": state.get("planner_raw_response"),
            "safety": state.get("safety"),
            "execution": state.get("execution"),
            "verification": state.get("verification"),
            "status": state.get("status"),
            "reason": state.get("reason"),
        }
        log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


OBSERVATION_ACTION_WORDS = {
    "click",
    "type",
    "scroll",
    "open",
    "close",
    "fill",
    "submit",
    "search",
    "copy",
    "paste",
    "move",
    "delete",
    "install",
    "run",
    "execute",
    "press",
}


def maybe_answer_observation_task(task: str, observation: dict[str, Any]) -> dict[str, Any] | None:
    lowered = f" {task.lower().strip()} "
    asks_about_screen = (
        "screen" in lowered
        and any(phrase in lowered for phrase in (" describe", "what is", "what's", "what do", "summarize", "read", "visible"))
    ) or " what do you see " in lowered
    wants_action = any(f" {word} " in lowered for word in OBSERVATION_ACTION_WORDS)
    if not asks_about_screen or wants_action:
        return None

    vlm = observation.get("vlm") or {}
    ocr = observation.get("ocr") or {}
    summary = (vlm.get("summary") or "").strip()
    ocr_text = (ocr.get("text") or "").strip()

    if summary:
        text = summary
        confidence = 0.9
        reason = "The task only asks for a screen description, so no desktop action is needed."
    elif ocr_text:
        text = f"Visible text on screen: {ocr_text}"
        confidence = 0.7
        reason = "The task only asks for screen contents and OCR text is available."
    else:
        return {
            "action": "ask_user",
            "target_id": None,
            "coordinates": None,
            "text": "I captured the screen, but neither OCR nor the vision model produced a usable description.",
            "keys": None,
            "scroll_amount": None,
            "seconds": None,
            "confidence": 0.1,
            "expected_result": "User checks OCR/VLM setup.",
            "reason": "No screen description was available.",
        }

    return {
        "action": "finish",
        "target_id": None,
        "coordinates": None,
        "text": text,
        "keys": None,
        "scroll_amount": None,
        "seconds": None,
        "confidence": confidence,
        "expected_result": "User receives a screen description.",
        "reason": reason,
    }


def confirm_action(action: dict[str, Any]) -> bool:
    print(json.dumps(action, indent=2, ensure_ascii=True))
    answer = input("Execute this action? [y/N] ").strip().lower()
    return answer in {"y", "yes"}

