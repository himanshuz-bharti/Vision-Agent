from __future__ import annotations

import time
from typing import Any


class ActionError(RuntimeError):
    pass


def execute_action(action: dict[str, Any], observation: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"executed": False, "dry_run": True, "message": describe_action(action, observation)}

    import pyautogui

    pyautogui.FAILSAFE = True
    name = action["action"]

    if name in {"click", "double_click"}:
        x, y = resolve_point(action, observation)
        if name == "click":
            pyautogui.click(x=x, y=y)
        else:
            pyautogui.doubleClick(x=x, y=y)
        return {"executed": True, "dry_run": False, "message": f"{name} at ({x}, {y})"}

    if name == "type":
        text = action.get("text")
        if not isinstance(text, str) or not text:
            raise ActionError("type action requires non-empty text.")
        pyautogui.write(text, interval=0.01)
        return {"executed": True, "dry_run": False, "message": f"typed {len(text)} characters"}

    if name == "hotkey":
        keys = normalize_keys(action)
        pyautogui.hotkey(*keys)
        return {"executed": True, "dry_run": False, "message": f"hotkey {keys}"}

    if name == "press":
        keys = normalize_keys(action)
        if len(keys) != 1:
            raise ActionError("press action requires exactly one key.")
        pyautogui.press(keys[0])
        return {"executed": True, "dry_run": False, "message": f"pressed {keys[0]}"}

    if name == "scroll":
        amount = int(action.get("scroll_amount") or 0)
        if amount == 0:
            raise ActionError("scroll action requires non-zero scroll_amount.")
        pyautogui.scroll(amount)
        return {"executed": True, "dry_run": False, "message": f"scrolled {amount}"}

    if name == "wait":
        seconds = float(action.get("seconds") or 1)
        time.sleep(max(0.1, min(seconds, 10.0)))
        return {"executed": True, "dry_run": False, "message": f"waited {seconds:.1f}s"}

    if name == "ask_user":
        return {"executed": False, "dry_run": False, "message": action.get("text") or "Agent asked the user."}

    if name == "finish":
        return {"executed": False, "dry_run": False, "message": action.get("text") or "Task finished."}

    raise ActionError(f"Unsupported action: {name}")


def describe_action(action: dict[str, Any], observation: dict[str, Any]) -> str:
    name = action.get("action")
    if name in {"click", "double_click"}:
        x, y = resolve_point(action, observation)
        return f"DRY RUN: would {name} at ({x}, {y})"
    if name == "type":
        text = action.get("text") or ""
        return f"DRY RUN: would type {len(text)} characters"
    if name == "hotkey":
        return f"DRY RUN: would press hotkey {normalize_keys(action)}"
    if name == "press":
        return f"DRY RUN: would press {normalize_keys(action)}"
    if name == "scroll":
        return f"DRY RUN: would scroll {action.get('scroll_amount')}"
    if name == "wait":
        return f"DRY RUN: would wait {action.get('seconds') or 1}s"
    return f"DRY RUN: would {name}"


def normalize_keys(action: dict[str, Any]) -> list[str]:
    keys = action.get("keys")
    if isinstance(keys, str):
        return [keys.lower()]
    if not isinstance(keys, list) or not keys:
        raise ActionError("Keyboard action requires keys.")
    return [str(key).lower() for key in keys]


def resolve_point(action: dict[str, Any], observation: dict[str, Any]) -> tuple[int, int]:
    target_id = action.get("target_id")
    if target_id:
        for candidate in observation.get("ui_candidates") or []:
            if str(candidate.get("id")) == str(target_id):
                x1, y1, x2, y2 = candidate["bbox"]
                return (int(x1 + x2) // 2, int(y1 + y2) // 2)

    coordinates = action.get("coordinates")
    if isinstance(coordinates, list | tuple) and len(coordinates) == 2:
        return int(coordinates[0]), int(coordinates[1])

    raise ActionError("Action requires target_id or coordinates.")
