from __future__ import annotations

import unittest

from desktop_agent.config import Config
from desktop_agent.planner import make_plan, subgoal_done_by_text
from desktop_agent.safety import evaluate_safety
from desktop_agent.verifier import verify_result


class FakeClient:
    """Stand-in for OllamaClient that returns a canned chat response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def chat(self, model, messages, *, temperature=0.0, json_mode=False, timeout=180) -> str:
        self.calls += 1
        return self.response


class MakePlanTests(unittest.TestCase):
    def test_parses_ordered_subgoals(self) -> None:
        client = FakeClient(
            '{"subgoals":[{"id":1,"goal":"Open Notepad","done_when":"Notepad window visible"},'
            '{"id":2,"goal":"Type hello","done_when":"hello visible"}],"notes":""}'
        )
        plan = make_plan(client, "m", "Open Notepad and type hello")
        self.assertEqual(len(plan["subgoals"]), 2)
        self.assertEqual(plan["subgoals"][0]["goal"], "Open Notepad")

    def test_falls_back_to_single_subgoal_on_garbage(self) -> None:
        plan = make_plan(FakeClient("not json at all"), "m", "Do the thing")
        self.assertEqual(len(plan["subgoals"]), 1)
        self.assertIn("fallback", plan["notes"])


class SubgoalDoneTests(unittest.TestCase):
    def test_done_when_terms_visible(self) -> None:
        subgoal = {"id": 1, "goal": "Open Notepad", "done_when": "Notepad window is visible"}
        obs = {"ocr": {"text": ""}, "vlm": {"summary": "Untitled - Notepad window"}}
        self.assertTrue(subgoal_done_by_text(subgoal, obs))

    def test_not_done_when_absent(self) -> None:
        subgoal = {"id": 1, "goal": "Open Notepad", "done_when": "Notepad window is visible"}
        obs = {"ocr": {"text": ""}, "vlm": {"summary": "Desktop icons only"}}
        self.assertFalse(subgoal_done_by_text(subgoal, obs))


class SafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.obs = {"ui_candidates": [{"id": "uia_001"}], "ocr": {"text": ""}, "vlm": {"summary": ""}}

    def test_benign_type_allowed(self) -> None:
        action = {"action": "type", "text": "hello", "confidence": 0.9, "reason": ""}
        self.assertTrue(evaluate_safety("type hello", action, self.obs, confidence_threshold=0.35)["allowed"])

    def test_destructive_hotkey_requires_user(self) -> None:
        action = {"action": "hotkey", "keys": ["shift", "delete"], "confidence": 0.9, "reason": "delete"}
        decision = evaluate_safety("delete files", action, self.obs, confidence_threshold=0.35)
        self.assertFalse(decision["allowed"])
        self.assertTrue(decision["requires_user"])

    def test_low_confidence_requires_user(self) -> None:
        action = {"action": "click", "target_id": "uia_001", "confidence": 0.1, "reason": ""}
        decision = evaluate_safety("click something", action, self.obs, confidence_threshold=0.35)
        self.assertTrue(decision["requires_user"])


class VerifierTests(unittest.TestCase):
    def test_typed_text_becomes_visible(self) -> None:
        before = {"ocr": {"text": ""}, "vlm": {"summary": ""}}
        after = {"ocr": {"text": ""}, "vlm": {"summary": "hello world"}, "ui_candidates": []}
        result = verify_result(
            {"action": "type", "text": "hello", "expected_result": "hello visible"},
            before,
            after,
            {"executed": True, "dry_run": False, "message": "typed"},
        )
        self.assertEqual(result["status"], "success")


class ConfigTests(unittest.TestCase):
    def test_empty_task_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Config(task="  ").validate()

    def test_bad_hitl_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Config(task="x", hitl="sometimes").validate()


class AgentTests(unittest.TestCase):
    def test_rereason_loop_on_user_decline(self) -> None:
        from unittest.mock import patch
        from pathlib import Path
        from desktop_agent.agent import Agent
        from desktop_agent.config import Config

        config = Config(task="Open Notepad", hitl="subgoal", execute=False)
        agent = Agent(config)

        with patch("desktop_agent.agent.capture_screenshot") as mock_capture, \
             patch.object(agent.perceiver, "perceive") as mock_perceive, \
             patch("desktop_agent.agent.next_action") as mock_next_action, \
             patch("desktop_agent.agent.evaluate_safety") as mock_safety, \
             patch("desktop_agent.agent._prompt") as mock_prompt:

            mock_capture.return_value = {}
            mock_perceive.return_value = {"ui_candidates": []}
            mock_next_action.return_value = ({"action": "click", "target_id": None, "target_description": "editor"}, "raw")
            mock_safety.return_value = {"allowed": True, "reason": "allowed"}
            
            # First two prompts return "no", third returns "yes"
            mock_prompt.side_effect = ["no", "no", "yes"]

            subgoal = {"id": 1, "goal": "Open Notepad", "done_when": "Notepad window visible"}
            outcome = agent._run_one_subgoal(subgoal, [], Path("runs"), 0)

            self.assertEqual(mock_next_action.call_count, 3)
            self.assertEqual(outcome["status"], "dry_run")


if __name__ == "__main__":
    unittest.main()
