from __future__ import annotations

from typing import Any


HARD_BLOCK_KEYWORDS = {
    "password",
    "passcode",
    "otp",
    "2fa",
    "credential",
    "secret key",
    "api key",
    "private key",
    "seed phrase",
    "credit card",
    "debit card",
}

APPROVAL_KEYWORDS = {
    "bank",
    "payment",
    "purchase",
    "buy",
    "checkout",
    "send email",
    "send message",
    "delete",
    "remove file",
    "format",
    "install",
    "uninstall",
    "disable security",
    "registry",
    "admin",
}

DESTRUCTIVE_HOTKEYS = {
    ("alt", "f4"),
    ("ctrl", "w"),
    ("ctrl", "q"),
    ("shift", "delete"),
    ("ctrl", "shift", "esc"),
}


def allowed_decision(reason: str = "Allowed by safety policy.") -> dict[str, Any]:
    return {
        "allowed": True,
        "reason": reason,
        "requires_user": False,
        "hard_block": False,
        "severity": "allowed",
    }


def approval_decision(reason: str) -> dict[str, Any]:
    return {
        "allowed": False,
        "reason": reason,
        "requires_user": True,
        "hard_block": False,
        "severity": "approval_required",
    }


def blocked_decision(reason: str) -> dict[str, Any]:
    return {
        "allowed": False,
        "reason": reason,
        "requires_user": True,
        "hard_block": True,
        "severity": "blocked",
    }


def evaluate_safety(
    task: str,
    action: dict[str, Any],
    observation: dict[str, Any],
    *,
    confidence_threshold: float,
) -> dict[str, Any]:
    action_name = action.get("action")
    if action_name in {"finish", "ask_user", "wait"}:
        return allowed_decision("Non-mutating action.")

    target_id = action.get("target_id")
    if target_id and not target_exists(str(target_id), observation):
        return blocked_decision(f"Unknown target_id: {target_id}.")

    coordinates = action.get("coordinates")
    if coordinates is not None and not valid_coordinates(coordinates, observation):
        return blocked_decision("Coordinates are outside the current screen.")

    combined = f"{task} {action.get('text') or ''} {action.get('reason') or ''}".lower()
    for keyword in HARD_BLOCK_KEYWORDS:
        if keyword in combined:
            return blocked_decision(f"Sensitive credential/payment data detected: '{keyword}'.")

    confidence = float(action.get("confidence") or 0)
    if confidence < confidence_threshold:
        return approval_decision(
            f"Planner confidence {confidence:.2f} is below threshold {confidence_threshold:.2f}."
        )

    for keyword in APPROVAL_KEYWORDS:
        if keyword in combined:
            return approval_decision(f"Risky intent detected: '{keyword}'.")

    keys = tuple(str(key).lower() for key in (action.get("keys") or []))
    if action_name == "hotkey":
        if "win" in keys or "cmd" in keys:
            return approval_decision("Windows/Cmd hotkey combos require human approval.")
        if keys in DESTRUCTIVE_HOTKEYS:
            return approval_decision(f"Potentially destructive hotkey requires approval: {keys}.")

    if action_name == "press":
        blocked_keys = {"delete", "del"}
        if any(key in blocked_keys for key in keys):
            return approval_decision("Delete key requires human approval.")

    if action_name in {"click", "double_click"} and coordinates is not None and not target_id:
        return approval_decision("Raw coordinate click requires human approval.")

    return allowed_decision("Allowed by safety policy.")


def valid_coordinates(coordinates: Any, observation: dict[str, Any]) -> bool:
    if not isinstance(coordinates, list | tuple) or len(coordinates) != 2:
        return False
    try:
        x = int(coordinates[0])
        y = int(coordinates[1])
    except (TypeError, ValueError):
        return False
    width, height = observation.get("screen_size") or [0, 0]
    return 0 <= x <= int(width) and 0 <= y <= int(height)


def target_exists(target_id: str, observation: dict[str, Any]) -> bool:
    candidates = observation.get("ui_candidates") or []
    return any(str(candidate.get("id")) == target_id for candidate in candidates)