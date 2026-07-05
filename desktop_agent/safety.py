from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from .logging_utils import log_event


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
    "message",
    "send",
    "post",
    "publish",
    "email",
    "bank",
    "payment",
    "purchase",
    "buy",
    "checkout",
    "delete",
    "remove file",
    "format",
    "install",
    "uninstall",
    "disable security",
    "registry",
    "admin",
}

SUBMIT_INTENT_WORDS = {
    "send",
    "sent",
    "message",
    "post",
    "publish",
    "email",
    "submit",
    "checkout",
    "purchase",
    "pay",
    "delete",
}

DESTRUCTIVE_HOTKEYS = {
    ("alt", "f4"),
    ("ctrl", "w"),
    ("ctrl", "q"),
    ("shift", "delete"),
    ("ctrl", "shift", "esc"),
}

CAPTCHA_WORDS = {
    "captcha",
    "recaptcha",
    "hcaptcha",
    "verify you are human",
    "i'm not a robot",
    "i am not a robot",
    "select all images",
    "select all squares",
    "image matching",
    "security check",
    "unusual traffic",
}

LOGIN_WORDS = {
    "login",
    "log in",
    "sign in",
    "signin",
    "password",
    "passcode",
    "otp",
    "one-time code",
    "two-factor",
    "2fa",
    "verification code",
    "authenticator",
    "credential",
}

BLOCKER_ALLOWED_ACTIONS = {"ask_user", "finish", "wait", "open_app", "open_url"}


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
    combined = f"{task} {action.get('text') or ''} {action.get('reason') or ''}".lower()
    screen_text = visible_observation_text(observation)
    blocker = detect_human_blocker(screen_text)
    log_event(
        "safety_evaluation_started",
        task=task,
        action=action,
        confidence_threshold=confidence_threshold,
        human_blocker=blocker,
    )
    if blocker and action_name not in BLOCKER_ALLOWED_ACTIONS:
        return log_decision(approval_decision(blocker["reason"]), action=action)
    if action_name in {"finish", "ask_user", "wait"}:
        return log_decision(allowed_decision("Non-mutating action."), action=action)

    if action_name == "open_url":
        url = str(action.get("text") or "")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return log_decision(blocked_decision("open_url only allows full http/https URLs."), action=action)
        for keyword in HARD_BLOCK_KEYWORDS:
            if keyword in combined:
                return log_decision(blocked_decision(f"Sensitive credential/payment data detected: '{keyword}'."), action=action)
        return log_decision(allowed_decision("Opening a browser URL is allowed."), action=action)

    if action_name == "open_app":
        for keyword in HARD_BLOCK_KEYWORDS:
            if keyword in combined:
                return log_decision(blocked_decision(f"Sensitive credential/payment data detected: '{keyword}'."), action=action)
        return log_decision(allowed_decision("Opening an application or executable is allowed."), action=action)
    if action_name == "search_app":
        return log_decision(allowed_decision("Searching inside the current app is allowed."), action=action)

    target_id = action.get("target_id")
    if target_id and not target_exists(str(target_id), observation):
        return log_decision(blocked_decision(f"Unknown target_id: {target_id}."), action=action)

    coordinates = action.get("coordinates")
    if coordinates is not None:
        return log_decision(blocked_decision("Raw coordinates are not allowed. Use target_id from current ui_candidates or ask_user."), action=action)

    for keyword in HARD_BLOCK_KEYWORDS:
        if keyword in combined:
            return log_decision(blocked_decision(f"Sensitive credential/payment data detected: '{keyword}'."), action=action)

    confidence = float(action.get("confidence") or 0)
    if confidence < confidence_threshold:
        return log_decision(
            approval_decision(f"Planner confidence {confidence:.2f} is below threshold {confidence_threshold:.2f}."),
            action=action,
        )

    if action_name == "type" and any(word in combined for word in SUBMIT_INTENT_WORDS):
        return log_decision(allowed_decision("Drafting text is allowed; submitting or sending requires a separate approved action."), action=action)

    for keyword in APPROVAL_KEYWORDS:
        if keyword in combined:
            return log_decision(approval_decision(f"Risky intent detected: '{keyword}'."), action=action)

    keys = tuple(str(key).lower() for key in (action.get("keys") or []))
    if action_name == "hotkey":
        if "win" in keys or "cmd" in keys:
            return log_decision(approval_decision("Windows/Cmd hotkey combos require human approval."), action=action)
        if keys in DESTRUCTIVE_HOTKEYS:
            return log_decision(approval_decision(f"Potentially destructive hotkey requires approval: {keys}."), action=action)

    if action_name == "press":
        blocked_keys = {"delete", "del"}
        if any(key in blocked_keys for key in keys):
            return log_decision(approval_decision("Delete key requires human approval."), action=action)
        if any(key in {"enter", "return"} for key in keys) and any(word in combined for word in SUBMIT_INTENT_WORDS):
            return log_decision(approval_decision("Pressing Enter may submit/send content and requires approval."), action=action)

    if action_name in {"click", "double_click"} and not target_id:
        return log_decision(blocked_decision("Click actions require target_id from current ui_candidates."), action=action)

    return log_decision(allowed_decision("Allowed by safety policy."), action=action)


def log_decision(decision: dict[str, Any], *, action: dict[str, Any]) -> dict[str, Any]:
    log_event("safety_decision", decision=decision, action=action)
    return decision


def target_exists(target_id: str, observation: dict[str, Any]) -> bool:
    candidates = observation.get("ui_candidates") or []
    return any(str(candidate.get("id")) == target_id for candidate in candidates)

def visible_observation_text(observation: dict[str, Any]) -> str:
    vlm = observation.get("vlm") or {}
    return f"{vlm.get('summary') or ''}".lower()


def detect_human_blocker(screen_text: str) -> dict[str, str] | None:
    for phrase in CAPTCHA_WORDS:
        if phrase in screen_text:
            blocker = {
                "type": "captcha",
                "matched": phrase,
                "reason": f"Human input required: CAPTCHA or anti-bot challenge detected ({phrase!r}).",
            }
            log_event("human_blocker_detected", blocker=blocker)
            return blocker
    for phrase in LOGIN_WORDS:
        if phrase in screen_text:
            blocker = {
                "type": "login_or_credentials",
                "matched": phrase,
                "reason": f"Human input required: login, credential, or verification step detected ({phrase!r}).",
            }
            log_event("human_blocker_detected", blocker=blocker)
            return blocker
    return None



