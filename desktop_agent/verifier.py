from __future__ import annotations

from typing import Any


def verify_result(
    action: dict[str, Any],
    before_observation: dict[str, Any],
    after_observation: dict[str, Any] | None,
    execution: dict[str, Any],
) -> dict[str, Any]:
    name = action.get("action")
    if execution.get("error"):
        return {"status": "failed", "success": False, "message": execution["message"]}
    if execution.get("dry_run"):
        return {"status": "dry_run", "success": None, "message": execution["message"]}
    if name == "finish":
        return {"status": "complete", "success": True, "message": action.get("text") or "Planner marked task complete."}
    if name == "ask_user":
        return {"status": "needs_user", "success": None, "message": action.get("text") or "Planner needs user input."}
    if after_observation is None:
        return {"status": "uncertain", "success": None, "message": "No after-observation was captured."}

    before_hash = ((before_observation.get("screenshot") or {}).get("sha256")) or ""
    after_hash = ((after_observation.get("screenshot") or {}).get("sha256")) or ""
    before_text = ((before_observation.get("ocr") or {}).get("text")) or ""
    after_text = ((after_observation.get("ocr") or {}).get("text")) or ""

    if name == "type":
        typed = action.get("text") or ""
        if typed and typed[:30] in after_text:
            return {"status": "success", "success": True, "message": "Typed text appears in OCR output."}

    if before_hash and after_hash and before_hash != after_hash:
        return {"status": "success", "success": True, "message": "Screenshot changed after action."}

    if before_text != after_text:
        return {"status": "success", "success": True, "message": "OCR text changed after action."}

    if name == "wait":
        return {"status": "uncertain", "success": None, "message": "Wait completed; no visible change detected."}

    return {"status": "failed", "success": False, "message": "No visible change detected."}