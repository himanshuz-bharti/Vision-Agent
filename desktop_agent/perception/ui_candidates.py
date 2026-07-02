from __future__ import annotations

from typing import Any


BUTTON_WORDS = {
    "ok",
    "yes",
    "no",
    "cancel",
    "next",
    "back",
    "finish",
    "submit",
    "save",
    "open",
    "close",
    "apply",
    "search",
    "send",
    "continue",
    "done",
}


def build_ui_candidates(ocr_items: list[dict[str, Any]], screen_size: tuple[int, int]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in ocr_items:
        text = item["text"]
        lower = text.lower().strip(" :")
        kind = "button_or_text" if lower in BUTTON_WORDS else "text"
        candidates.append(
            {
                "id": item["id"],
                "kind": kind,
                "label": text,
                "bbox": item["bbox"],
            }
        )

    width, height = screen_size
    candidates.append(
        {
            "id": "screen_center",
            "kind": "region",
            "label": "center of the screen",
            "bbox": [width // 2 - 30, height // 2 - 30, width // 2 + 30, height // 2 + 30],
        }
    )
    return candidates[:150]
