from __future__ import annotations

import time
from typing import Any

from desktop_agent.logging_utils import log_event, log_exception


# Interactive UI Automation control types worth clicking. Names match
# uiautomation.ControlType.*Control attribute names (without the "Control" suffix).
INTERACTIVE_CONTROLS = {
    "Button",
    "SplitButton",
    "Edit",
    "Document",
    "CheckBox",
    "RadioButton",
    "ComboBox",
    "List",
    "ListItem",
    "MenuItem",
    "Menu",
    "Hyperlink",
    "Tab",
    "TabItem",
    "TreeItem",
    "Slider",
    "Spinner",
    "Text",
}

# Control types that are only useful when they carry a visible name.
NAME_REQUIRED_CONTROLS = {"Text", "Document", "List", "Menu", "Tab"}


def _import_uiautomation():
    """Import uiautomation lazily; return the module or None if unavailable."""
    try:
        import uiautomation as auto  # type: ignore
    except Exception as exc:  # pragma: no cover - platform/dependency dependent
        log_event("uia_import_unavailable", level="WARNING", error=str(exc))
        return None, str(exc)
    return auto, None


def uia_status() -> dict[str, Any]:
    """Report whether UI Automation is usable in this process (for the model report)."""
    auto, error = _import_uiautomation()
    if auto is None:
        return {"available": False, "error": error}
    return {"available": True, "error": None}


def capture_uia_candidates(
    *,
    max_items: int = 120,
    time_budget: float = 4.0,
    max_depth: int = 28,
) -> dict[str, Any]:
    """Walk the foreground window's UI Automation tree for interactive controls.

    Returns {"available": bool, "error": str|None, "items": list[candidate]} where each
    candidate is {"id", "source": "uia", "kind", "label", "bbox", "center", "enabled"}.
    The walk is bounded by max_items, time_budget (seconds), and max_depth so it can
    never hang the perception step on a huge or slow tree.
    """
    started = time.perf_counter()
    auto, error = _import_uiautomation()
    if auto is None:
        return {"available": False, "error": error, "items": []}

    try:
        # GetForegroundWindow() returns an HWND (int); resolve it to a control.
        handle = auto.GetForegroundWindow()
        window = auto.ControlFromHandle(handle) if handle else None
    except Exception as exc:
        log_exception("uia_foreground_error", exc)
        return {"available": False, "error": f"foreground window lookup failed: {exc}", "items": []}

    if window is None:
        log_event("uia_no_foreground_window", level="WARNING")
        return {"available": True, "error": "no foreground window", "items": []}

    items: list[dict[str, Any]] = []
    truncated = False

    def deadline_reached() -> bool:
        return (time.perf_counter() - started) >= time_budget

    def walk(control: Any, depth: int) -> None:
        nonlocal truncated
        if len(items) >= max_items:
            truncated = True
            return
        if depth > max_depth or deadline_reached():
            truncated = True
            return
        try:
            children = control.GetChildren()
        except Exception:
            children = []
        for child in children:
            if len(items) >= max_items or deadline_reached():
                truncated = True
                return
            candidate = _to_candidate(child, auto)
            if candidate is not None:
                candidate["id"] = f"uia_{len(items) + 1:03d}"
                items.append(candidate)
            walk(child, depth + 1)

    try:
        walk(window, 0)
    except Exception as exc:
        log_exception("uia_walk_error", exc)
        return {"available": True, "error": f"tree walk failed: {exc}", "items": items}

    result = {
        "available": True,
        "error": None,
        "items": items,
        "truncated": truncated,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
    log_event(
        "uia_capture_completed",
        item_count=len(items),
        truncated=truncated,
        elapsed_ms=result["elapsed_ms"],
        sample_items=items[:15],
    )
    return result


def _to_candidate(control: Any, auto: Any) -> dict[str, Any] | None:
    """Convert a uiautomation control to a candidate dict, or None to skip it."""
    try:
        kind = _control_kind(control, auto)
        if kind not in INTERACTIVE_CONTROLS:
            return None
        name = (control.Name or "").strip()
        if kind in NAME_REQUIRED_CONTROLS and not name:
            return None

        rect = control.BoundingRectangle
        left, top, right, bottom = int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
        if right <= left or bottom <= top:
            return None
        # Reject absurdly large rects (usually the root/window itself, not a control).
        if (right - left) <= 0 or (bottom - top) <= 0:
            return None

        try:
            offscreen = bool(control.IsOffscreen)
        except Exception:
            offscreen = False
        if offscreen:
            return None

        try:
            enabled = bool(control.IsEnabled)
        except Exception:
            enabled = True

        return {
            "source": "uia",
            "kind": kind,
            "label": name,
            "bbox": [left, top, right, bottom],
            "center": [(left + right) // 2, (top + bottom) // 2],
            "enabled": enabled,
        }
    except Exception:
        # Any COM/access error on a single node: skip it, keep walking.
        return None


def _control_kind(control: Any, auto: Any) -> str:
    """Map a control's ControlType id to a short name like 'Button'."""
    try:
        control_type = control.ControlType
    except Exception:
        return "Unknown"
    name = auto.ControlTypeNames.get(control_type)
    if not name:
        return "Unknown"
    # ControlTypeNames values look like "ButtonControl"; strip the suffix.
    if name.endswith("Control"):
        name = name[: -len("Control")]
    return name
