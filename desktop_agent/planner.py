from __future__ import annotations

import json
import re
from typing import Any

from .logging_utils import log_event, log_exception, say
from .models import OllamaClient, OpenAICompatibleClient

ClientType = OllamaClient | OpenAICompatibleClient


class PlannerError(RuntimeError):
    def __init__(self, message: str, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


# ---------------------------------------------------------------------------
# 1. Task decomposition:  task text -> ordered subgoals
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = (
    "You are a desktop automation planner. Break the user's task into an ordered list of "
    "small, verifiable subgoals a computer-control agent can execute one at a time using "
    "only these capabilities: open apps, open URLs, click visible controls, type text, "
    "press keys/hotkeys, scroll, wait, and search inside an app. Keep each subgoal to a "
    "single app interaction (not individual keystrokes). "
    "Decompose complex tasks (like web searches) into explicit, distinct subgoals: e.g. "
    "1. open the browser page, 2. click specifically on the in-page document search input (not the browser's address bar), "
    "3. type the search query, 4. click the in-page search button. Return only valid JSON. "
    "Break down the query into highly detailed instructions for achieving it."
)

PLAN_SCHEMA = {
    "subgoals": [
        {
            "id": "1-based integer",
            "goal": "what to accomplish in this step",
            "done_when": "a short, observable condition that means this subgoal is finished",
        }
    ],
    "notes": "optional short strategy note or empty string",
}


def make_plan(client: ClientType, model: str, task: str) -> dict[str, Any]:
    """Ask the model for an ordered subgoal decomposition of the task.

    Always returns a usable plan: on any model/parse failure it falls back to a single
    subgoal wrapping the whole task so execution can still proceed.
    """
    say(f"Planning: decomposing task with {model} ...", event="plan_started", model=model, task=task)
    messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": task,
                    "required_json_schema": PLAN_SCHEMA,
                    "rules": [
                        "Return exactly one JSON object with a non-empty 'subgoals' array.",
                        "Order subgoals in execution order.",
                        "Each subgoal must have id, goal, and done_when.",
                        "Prefer 2-6 subgoals. Do not include chain-of-thought or markdown.",
                        "Decompose complex actions into granular, explicit subgoals (e.g. separate clicking from typing and clicking search buttons).",
                        "Avoid combined subgoals like 'search and play video' if they require multiple UI interactions.",
                        "done_when should mention text/UI that would be visible when the step is done.",
                    ],
                },
                ensure_ascii=True,
            ),
        },
    ]
    kwargs = {}
    if model == "nvidia/nemotron-3-ultra-550b-a55b":
        kwargs.update({
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 16384,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384}
        })

    raw = client.chat(model, messages, json_mode=True, **kwargs)
    say(f"\n[RAW PLANNER OUTPUT - {model}]\n{raw}\n", event="raw_model_output")
    plan = _parse_plan(raw, task)
    log_event("plan_completed", model=model, task=task, subgoal_count=len(plan["subgoals"]), plan=plan)
    return plan


def _parse_plan(raw: str, task: str) -> dict[str, Any]:
    parsed = parse_json_object(raw)
    subgoals_raw = parsed.get("subgoals")
    if not isinstance(subgoals_raw, list) or not subgoals_raw:
        raise ValueError("plan is missing a non-empty 'subgoals' array")
    subgoals: list[dict[str, Any]] = []
    for index, item in enumerate(subgoals_raw, start=1):
        if not isinstance(item, dict):
            continue
        goal = str(item.get("goal") or "").strip()
        if not goal:
            continue
        subgoals.append(
            {
                "id": index,
                "goal": goal,
                "done_when": str(item.get("done_when") or "").strip(),
            }
        )
    if not subgoals:
        raise ValueError("plan had no usable subgoals after normalization")
    return {"subgoals": subgoals, "notes": str(parsed.get("notes") or "")}


def _observation_text(observation: dict[str, Any]) -> str:
    ocr = observation.get("ocr") or {}
    vlm = observation.get("vlm") or {}
    uia_labels = [c.get("label") for c in observation.get("ui_candidates") or [] if c.get("label")]
    return f"{ocr.get('text') or ''} {vlm.get('summary') or ''} {' '.join(uia_labels)}"


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {text[:100]}...")
    
    return json.loads(text[start:end+1])
