"""Perception: screenshot -> OCR + UIA + merged click candidates + set-of-marks + optional VLM.

Consolidates the old perceiver / ui_candidates / set_of_marks modules.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from desktop_agent.logging_utils import log_event, log_exception
from desktop_agent.models import OllamaClient

from .screen import image_size
from .uia import capture_uia_candidates
from .vlm import summarize_screen


class Perceiver:
    def __init__(
        self,
        client: OllamaClient,
        *,
        vlm_model: str,
        use_vlm: str,
        grounding: str = "uia",
        max_ui_candidates: int = 150,
    ) -> None:
        self.client = client
        self.vlm_model = vlm_model
        self.use_vlm = use_vlm
        self.grounding = grounding
        self.max_ui_candidates = max_ui_candidates

    def perceive(self, image_path: Path, screenshot_meta: dict | None = None, task: str = "", skip_vlm: bool = False) -> dict:
        width, height = image_size(image_path)
        ocr = {"available": False, "error": "OCR disabled", "items": [], "text": ""}

        uia = self._capture_uia()
        candidates = self._build_candidates(uia, (width, height))
        mark_map = assign_marks(candidates)

        marks_path = image_path.with_name(image_path.stem + "_marks.png")
        marks = render_marks(image_path, candidates, marks_path)
        vlm_image = Path(marks["path"]) if marks.get("available") and marks.get("path") else image_path

        visual_task = task_needs_visual_context(task)
        should_use_vlm = self.use_vlm != "never" and not skip_vlm
        log_event(
            "vlm_decision",
            mode=self.use_vlm,
            should_use_vlm=should_use_vlm,
            skip_vlm=skip_vlm,
            visual_task=visual_task,
            ocr_available=False,
            ocr_item_count=0,
            uia_available=uia.get("available"),
            candidate_count=len(candidates),
        )
        vlm = {"available": False, "error": None, "summary": ""}
        if should_use_vlm:
            vlm = summarize_screen(self.client, self.vlm_model, vlm_image)

        observation = {
            "screenshot": screenshot_meta or {"path": str(image_path), "width": width, "height": height},
            "screen_size": [width, height],
            "ocr": ocr,
            "uia": {"available": uia.get("available"), "error": uia.get("error"), "item_count": len(uia.get("items") or [])},
            "vlm": vlm,
            "ui_candidates": candidates,
            "mark_map": {str(mark): target_id for mark, target_id in mark_map.items()},
            "marks_image": marks.get("path"),
        }
        log_event(
            "perceiver_completed",
            screen_size=[width, height],
            ocr_item_count=0,
            uia_item_count=len(uia.get("items") or []),
            vlm_available=vlm.get("available"),
            ui_candidate_count=len(candidates),
        )
        return observation

    def _capture_uia(self) -> dict:
        if self.grounding != "uia":
            return {"available": False, "error": "uia grounding disabled", "items": []}
        return capture_uia_candidates(max_items=self.max_ui_candidates)

    def _build_candidates(self, uia: dict, screen_size: tuple[int, int]) -> list[dict]:
        uia_items = uia.get("items") or []
        merged: list[dict[str, Any]] = []
        for candidate in uia_items:
            entry = dict(candidate)
            entry.setdefault("center", _bbox_center(entry.get("bbox") or []))
            merged.append(entry)
            if len(merged) >= self.max_ui_candidates:
                break
        return merged


# ---------------------------------------------------------------------------
# Candidate building / merging  (from ui_candidates.py)
# ---------------------------------------------------------------------------

def _bbox_center(bbox: list[int]) -> list[int]:
    if len(bbox) != 4:
        return [0, 0]
    x1, y1, x2, y2 = bbox
    return [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2]


# ---------------------------------------------------------------------------
# Set-of-marks annotation  (from set_of_marks.py)
# ---------------------------------------------------------------------------

_SOURCE_COLORS = {"uia": (0, 180, 255), "text": (255, 120, 0), "grounded": (0, 220, 0)}
_DEFAULT_COLOR = (220, 0, 220)


def assign_marks(candidates: list[dict[str, Any]]) -> dict[int, str]:
    """Attach an integer `mark` to each candidate in place; return {mark: candidate_id}."""
    mark_map: dict[int, str] = {}
    for index, candidate in enumerate(candidates, start=1):
        candidate["mark"] = index
        mark_map[index] = str(candidate.get("id"))
    return mark_map


def render_marks(image_path: Path, candidates: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Draw numbered boxes onto a copy of the screenshot. Never raises."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        log_exception("set_of_marks_import_error", exc, image_path=image_path)
        return {"available": False, "path": None, "error": f"Pillow missing: {exc}"}
    try:
        with Image.open(image_path) as base:
            image = base.convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        for candidate in candidates:
            mark = candidate.get("mark")
            bbox = candidate.get("bbox") or []
            if mark is None or len(bbox) != 4:
                continue
            left, top, right, bottom = (int(v) for v in bbox)
            color = _SOURCE_COLORS.get(candidate.get("source"), _DEFAULT_COLOR)
            draw.rectangle((left, top, right, bottom), outline=color, width=2)
            _draw_label(draw, str(mark), left, top, color, font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        log_event("set_of_marks_rendered", path=out_path, mark_count=len(candidates))
        return {"available": True, "path": str(out_path), "error": None}
    except Exception as exc:
        log_exception("set_of_marks_render_error", exc, image_path=image_path, out_path=out_path)
        return {"available": False, "path": None, "error": str(exc)}


def _draw_label(draw: Any, text: str, left: int, top: int, color: tuple[int, int, int], font: Any) -> None:
    try:
        text_box = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
    except Exception:
        text_w, text_h = 8 * len(text), 12
    pad = 2
    tag_top = max(0, top - text_h - 2 * pad)
    draw.rectangle((left, tag_top, left + text_w + 2 * pad, tag_top + text_h + 2 * pad), fill=color)
    draw.text((left + pad, tag_top + pad), text, fill=(0, 0, 0), font=font)


# ---------------------------------------------------------------------------
# Task heuristic  (from task_analysis.py)
# ---------------------------------------------------------------------------

_VISUAL_CONTEXT_PATTERNS = (
    r"\b(?:click|double click|press|scroll|drag|select|choose)\b",
    r"\b(?:read|describe|summarize|look at|watch)\b",
    r"\b(?:visible|screen|image|picture|button|icon|menu|window|tab|current)\b",
)


def task_needs_visual_context(task: str) -> bool:
    normalized = re.sub(r"\s+", " ", (task or "").lower()).strip()
    needs = any(re.search(pattern, normalized) for pattern in _VISUAL_CONTEXT_PATTERNS)
    log_event("task_visual_context_analyzed", task=task, needs_visual_context=needs)
    return needs
