from __future__ import annotations

import unittest

from desktop_agent.perception.vlm import is_degenerate_vlm_output
from desktop_agent.planner import normalize_action


class PlannerPlaceholderGuardTests(unittest.TestCase):
    def test_rejects_example_url_placeholder(self) -> None:
        action = {
            "action": "open_url",
            "target_id": None,
            "target_description": None,
            "coordinates": None,
            "text": "https://example.com/search?q=topic",
            "keys": None,
            "scroll_amount": None,
            "seconds": None,
            "confidence": 0.8,
            "expected_result": "A browser opens.",
            "reason": "Copied from a format example.",
        }

        with self.assertRaisesRegex(ValueError, "placeholder/example"):
            normalize_action(action)

    def test_allows_task_derived_open_app_text(self) -> None:
        action = {
            "action": "open_app",
            "target_id": None,
            "target_description": None,
            "coordinates": None,
            "text": "notepad",
            "keys": None,
            "scroll_amount": None,
            "seconds": None,
            "confidence": 0.8,
            "expected_result": "Notepad opens.",
            "reason": "The user asked to open Notepad.",
        }

        normalized = normalize_action(action)
        self.assertEqual(normalized["text"], "notepad")


class VlmGarbageFilterTests(unittest.TestCase):
    def test_rejects_empty_and_repeated_output(self) -> None:
        self.assertTrue(is_degenerate_vlm_output(""))
        self.assertTrue(is_degenerate_vlm_output("!!!!!!!!!!!!"))
        self.assertTrue(is_degenerate_vlm_output("aaaaaaaaaaaaaaaa"))

    def test_allows_real_summary_and_mark_json(self) -> None:
        self.assertFalse(is_degenerate_vlm_output("A Notepad window is open with a blank editing area."))
        self.assertFalse(is_degenerate_vlm_output('{"mark": 7}'))


if __name__ == "__main__":
    unittest.main()
