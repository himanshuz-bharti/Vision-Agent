from __future__ import annotations

import re
from typing import Any

from .logging_utils import log_event


# A change of at least this fraction of tokens (added or removed) counts as a real
# structural change, rather than incidental noise (clock, cursor, one blinking char).
_MIN_STRUCTURAL_CHANGE = 0.08
_MIN_CHANGED_TOKENS = 3


def verify_result(
    action: dict[str, Any],
    before_observation: dict[str, Any],
    after_observation: dict[str, Any] | None,
    execution: dict[str, Any],
) -> dict[str, Any]:
    name = action.get("action")
    log_event("verifier_started", action=action, execution=execution, has_after_observation=after_observation is not None)
    if execution.get("error"):
        return verified("failed", False, execution["message"], reason="execution_error")
    if execution.get("dry_run"):
        return verified("dry_run", None, execution["message"], reason="dry_run")
    if name == "finish":
        return verified("complete", True, action.get("text") or "Planner marked task complete.", reason="planner_finish")
    if name == "ask_user":
        return verified("needs_user", None, action.get("text") or "Planner needs user input.", reason="planner_ask_user")
    if after_observation is None:
        return verified("uncertain", None, "No after-observation was captured.", reason="missing_after_observation")

    before_vlm = ((before_observation.get("vlm") or {}).get("summary")) or ""
    before_visible = before_vlm.lower()
    after_vlm = ((after_observation.get("vlm") or {}).get("summary")) or ""
    after_visible = after_vlm.lower()

    # 1. Action-specific positive signals (strongest evidence).
    if name in {"open_app", "open_url"}:
        requested = (action.get("text") or "").lower()
        visible_hint = most_specific_visible_hint(requested)
        if visible_hint and visible_hint in after_visible:
            return verified("success", True, f"Visible screen contains requested hint: {visible_hint!r}.", reason="requested_hint_visible")

    if name == "type":
        typed = action.get("text") or ""
        if typed and typed[:30].lower() in after_visible:
            return verified("success", True, "Typed text appears in the screen text.", reason="typed_text_visible")

    # 2. expected_result keywords becoming visible after the action.
    expected = action.get("expected_result") or ""
    newly_visible = expected_keywords_newly_visible(expected, before_visible, after_visible)
    if newly_visible:
        return verified("success", True, f"Expected result appeared: {', '.join(newly_visible)}.", reason="expected_result_visible")

    # 3. Structural change in the on-screen text / candidate labels.
    change = structural_change(before_observation, after_observation)
    if change["changed"]:
        return verified(
            "success",
            True,
            f"Screen changed structurally ({change['added']} added / {change['removed']} removed tokens).",
            reason="structural_change",
        )

    # 4. No meaningful change. Waits are inherently inconclusive.
    if name == "wait":
        return verified("uncertain", None, "Wait completed; no visible change detected.", reason="wait_no_change")

    # A raw pixel change with no textual/structural change is not trustworthy evidence.
    return verified("uncertain", None, "No meaningful change detected after the action.", reason="no_structural_change")


def structural_change(before_observation: dict[str, Any], after_observation: dict[str, Any]) -> dict[str, Any]:
    """Compare token/label sets before and after; return whether the change is meaningful."""
    before_tokens = observation_tokens(before_observation)
    after_tokens = observation_tokens(after_observation)
    added = after_tokens - before_tokens
    removed = before_tokens - after_tokens
    union = before_tokens | after_tokens
    change_ratio = (len(added) + len(removed)) / len(union) if union else 0.0
    changed = (len(added) + len(removed)) >= _MIN_CHANGED_TOKENS and change_ratio >= _MIN_STRUCTURAL_CHANGE
    result = {
        "changed": changed,
        "added": len(added),
        "removed": len(removed),
        "change_ratio": round(change_ratio, 3),
    }
    log_event("verifier_structural_change", result=result)
    return result


def observation_tokens(observation: dict[str, Any]) -> set[str]:
    labels = " ".join(str(c.get("label") or "") for c in observation.get("ui_candidates") or [])
    return tokenize(labels)


def expected_keywords_newly_visible(expected: str, before_visible: str, after_visible: str) -> list[str]:
    keywords = [kw for kw in tokenize(expected) if len(kw) >= 4]
    newly = [kw for kw in keywords if kw in after_visible and kw not in before_visible]
    return newly[:5]


def tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (value or "").lower()) if len(token) >= 2}


def most_specific_visible_hint(value: str) -> str:
    cleaned = value.replace("https://", "").replace("http://", "")
    cleaned = cleaned.split("/", 1)[0] if "." in cleaned else cleaned
    cleaned = cleaned.split(":", 1)[0] if cleaned.endswith(":") else cleaned
    parts = [part for part in cleaned.replace("-", " ").replace("_", " ").split() if len(part) >= 3]
    if parts:
        return max(parts, key=len)
    return cleaned.strip()


def verified(status: str, success: bool | None, message: str, *, reason: str) -> dict[str, Any]:
    result = {"status": status, "success": success, "message": message}
    log_event("verifier_completed", result=result, reason=reason)
    return result
