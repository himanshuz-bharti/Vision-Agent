from __future__ import annotations

import os
import re
import subprocess
import time
import webbrowser
from typing import Any

from .logging_utils import log_event, log_exception


class ActionError(RuntimeError):
    pass


def execute_action(action: dict[str, Any], observation: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    name = action["action"]
    log_event("action_dispatch_started", action=action, dry_run=dry_run)
    if dry_run:
        message = describe_action(action, observation)
        result = {"executed": False, "dry_run": True, "message": message}
        log_event("action_dispatch_completed", action_name=name, result=result)
        return result

    if name == "open_url":
        url = extract_text(action, field="text", label="open_url")
        opened = webbrowser.open(url, new=2)
        result = {"executed": True, "dry_run": False, "message": f"opened URL: {url}", "webbrowser_opened": opened}
        log_event("open_url_executed", url=url, webbrowser_opened=opened, result=result)
        return result

    if name == "open_app":
        app_name = extract_text(action, field="text", label="open_app")
        message = open_app(app_name)
        result = {"executed": True, "dry_run": False, "message": message}
        log_event("open_app_executed", app_name=app_name, result=result)
        return result

    try:
        import pyautogui
    except Exception as exc:
        log_exception("pyautogui_import_error", exc, action=action)
        raise

    pyautogui.FAILSAFE = True
    log_event("pyautogui_ready", failsafe=pyautogui.FAILSAFE, action_name=name)

    if name == "search_app":
        query = extract_text(action, field="text", label="search_app")
        log_event("search_app_started", query=query)
        pyautogui.press("esc")
        pyautogui.hotkey("ctrl", "f")
        time.sleep(0.1)
        pyautogui.write(query, interval=0.01)
        result = {"executed": True, "dry_run": False, "message": f"searched current app for {query!r}"}
        log_event("search_app_completed", query=query, result=result)
        return result

    if name == "click":
        x, y = resolve_point(action, observation)
        log_event("click_started", target_id=action.get("target_id"), resolved=[x, y])
        pyautogui.click(x=x, y=y)
        result = {"executed": True, "dry_run": False, "message": f"click at ({x}, {y})"}
        log_event("click_completed", resolved=[x, y], result=result)
        return result

    if name == "double_click":
        x, y = resolve_point(action, observation)
        log_event("double_click_started", target_id=action.get("target_id"), resolved=[x, y])
        pyautogui.doubleClick(x=x, y=y)
        result = {"executed": True, "dry_run": False, "message": f"double_click at ({x}, {y})"}
        log_event("double_click_completed", resolved=[x, y], result=result)
        return result

    if name == "type":
        text = extract_text(action, field="text", label="type")
        target_id = action.get("target_id")
        clicked_target = False
        if target_id:
            try:
                x, y = resolve_point(action, observation)
                log_event("type_click_focus_started", target_id=target_id, resolved=[x, y])
                pyautogui.click(x=x, y=y)
                time.sleep(0.15)
                clicked_target = True
            except Exception as exc:
                log_event("type_click_focus_failed", level="WARNING", target_id=target_id, error=str(exc))
        
        log_event("type_started", text_chars=len(text), text_preview=text[:500], clicked_target=clicked_target)
        pyautogui.write(text, interval=0.01)
        msg = f"typed {len(text)} characters"
        if clicked_target:
            msg = f"clicked target {target_id} and " + msg
        result = {"executed": True, "dry_run": False, "message": msg}
        log_event("type_completed", text_chars=len(text), result=result)
        return result

    if name == "hotkey":
        keys = normalize_keys(action)
        log_event("hotkey_started", keys=keys)
        pyautogui.hotkey(*keys)
        result = {"executed": True, "dry_run": False, "message": f"hotkey {keys}"}
        log_event("hotkey_completed", keys=keys, result=result)
        return result

    if name == "press":
        keys = normalize_keys(action)
        if len(keys) != 1:
            raise ActionError("press action requires exactly one key.")
        log_event("press_started", key=keys[0])
        pyautogui.press(keys[0])
        result = {"executed": True, "dry_run": False, "message": f"pressed {keys[0]}"}
        log_event("press_completed", key=keys[0], result=result)
        return result

    if name == "scroll":
        amount = int(action.get("scroll_amount") or 0)
        if amount == 0:
            raise ActionError("scroll action requires non-zero scroll_amount.")
        log_event("scroll_started", amount=amount)
        pyautogui.scroll(amount)
        result = {"executed": True, "dry_run": False, "message": f"scrolled {amount}"}
        log_event("scroll_completed", amount=amount, result=result)
        return result

    if name == "wait":
        seconds = float(action.get("seconds") or 1)
        bounded_seconds = max(0.1, min(seconds, 10.0))
        log_event("wait_started", requested_seconds=seconds, bounded_seconds=bounded_seconds)
        time.sleep(bounded_seconds)
        result = {"executed": True, "dry_run": False, "message": f"waited {seconds:.1f}s"}
        log_event("wait_completed", requested_seconds=seconds, bounded_seconds=bounded_seconds, result=result)
        return result

    if name == "ask_user":
        result = {"executed": False, "dry_run": False, "message": action.get("text") or "Agent asked the user."}
        log_event("ask_user_action", result=result, action=action)
        return result

    if name == "finish":
        result = {"executed": False, "dry_run": False, "message": action.get("text") or "Task finished."}
        log_event("finish_action", result=result, action=action)
        return result

    raise ActionError(f"Unsupported action: {name}")


def open_app(app_name: str) -> str:
    command = normalize_app_command(app_name)
    log_event("open_app_command_resolved", app_name=app_name, command=command)
    try:
        if looks_like_uri_scheme(command):
            os.startfile(command)  # type: ignore[attr-defined]
            message = f"opened app via URI/protocol: {command}"
            log_event("open_app_protocol_started", app_name=app_name, protocol=command, message=message)
            return message
        process = subprocess.Popen([command], shell=False)
        message = f"opened app via process: {command}"
        log_event("open_app_process_started", app_name=app_name, executable=command, pid=process.pid, message=message)
        return message
    except Exception as exc:
        log_exception("open_app_primary_error", exc, app_name=app_name, command=command)
        try:
            fallback_message = open_app_via_os_search(app_name)
        except Exception as fallback_exc:
            log_exception("open_app_fallback_error", fallback_exc, app_name=app_name, command=command)
            raise ActionError(
                f"Could not open app {app_name!r} using command {command!r} or generic OS search: {fallback_exc}"
            ) from fallback_exc
        return f"{fallback_message} (direct command failed: {exc})"


def open_app_via_os_search(app_name: str) -> str:
    query = app_name.strip()
    if not query:
        raise ActionError("open_app fallback requires a non-empty search query.")
    import pyautogui

    pyautogui.FAILSAFE = True
    log_event("open_app_os_search_started", query=query)
    pyautogui.press("win")
    time.sleep(0.25)
    pyautogui.write(query, interval=0.01)
    time.sleep(0.15)
    pyautogui.press("enter")
    message = f"opened app via generic OS search: {query}"
    log_event("open_app_os_search_completed", query=query, message=message)
    return message


def normalize_app_command(app_name: str) -> str:
    value = app_name.strip()
    if not value:
        raise ActionError("open_app action requires a non-empty app name or executable command.")
    if looks_like_uri_scheme(value) or has_executable_suffix(value) or contains_path_separator(value):
        return value
    return value


def looks_like_uri_scheme(value: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:$", value.strip()))


def has_executable_suffix(value: str) -> bool:
    return value.lower().endswith((".exe", ".bat", ".cmd", ".com"))


def contains_path_separator(value: str) -> bool:
    return "\\" in value or "/" in value


def describe_action(action: dict[str, Any], observation: dict[str, Any]) -> str:
    name = action.get("action")
    if name == "open_url":
        return f"DRY RUN: would open URL {action.get('text')!r}"
    if name == "open_app":
        return f"DRY RUN: would open app/command {action.get('text')!r}"
    if name == "search_app":
        return f"DRY RUN: would search current app for {action.get('text')!r}"
    if name in {"click", "double_click"}:
        if not action.get("target_id") and action.get("target_description"):
            return f"DRY RUN: would {name} the control described as {action.get('target_description')!r} (needs VLM grounding to resolve)"
        x, y = resolve_point(action, observation)
        return f"DRY RUN: would {name} target_id={action.get('target_id')!r} at resolved center ({x}, {y})"
    if name == "type":
        text = action.get("text") or ""
        target_id = action.get("target_id")
        if target_id:
            return f"DRY RUN: would click target_id={target_id!r} and type {len(text)} characters"
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


def extract_text(action: dict[str, Any], *, field: str, label: str) -> str:
    text = action.get(field)
    if not isinstance(text, str) or not text.strip():
        log_event("action_validation_error", level="ERROR", label=label, field=field, value=text, action=action)
        raise ActionError(f"{label} action requires non-empty {field}.")
    return text.strip()


def normalize_keys(action: dict[str, Any]) -> list[str]:
    keys = action.get("keys")
    if isinstance(keys, str):
        normalized = [keys.lower()]
        log_event("keys_normalized", original=keys, normalized=normalized)
        return normalized
    if not isinstance(keys, list) or not keys:
        log_event("action_validation_error", level="ERROR", label="keyboard", field="keys", value=keys, action=action)
        raise ActionError("Keyboard action requires keys.")
    normalized = [str(key).lower() for key in keys]
    log_event("keys_normalized", original=keys, normalized=normalized)
    return normalized


def resolve_point(action: dict[str, Any], observation: dict[str, Any]) -> tuple[int, int]:
    target_id = action.get("target_id")
    if not target_id:
        log_event("point_resolution_failed", level="ERROR", action=action, reason="missing_target_id")
        raise ActionError("Click actions require target_id from current ui_candidates; raw coordinates are not allowed.")

    for candidate in observation.get("ui_candidates") or []:
        if str(candidate.get("id")) == str(target_id):
            center = candidate.get("center")
            if isinstance(center, (list, tuple)) and len(center) == 2:
                point = (int(center[0]), int(center[1]))
            else:
                x1, y1, x2, y2 = candidate["bbox"]
                point = ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)
            log_event("point_resolved_from_target", target_id=target_id, candidate=candidate, point=point)
            return point

    log_event("target_id_not_found", level="ERROR", target_id=target_id, available_targets=[c.get("id") for c in observation.get("ui_candidates") or []])
    raise ActionError(f"Unknown target_id {target_id!r}; planner must choose a visible ui_candidate or ask_user.")


