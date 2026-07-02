from __future__ import annotations

from pathlib import Path

from desktop_agent.models.ollama_client import OllamaClient

from .capture import image_size
from .ocr import run_ocr
from .ui_candidates import build_ui_candidates
from .vlm import summarize_screen


class Perceiver:
    def __init__(self, client: OllamaClient, *, vlm_model: str, use_vlm: str) -> None:
        self.client = client
        self.vlm_model = vlm_model
        self.use_vlm = use_vlm

    def perceive(self, image_path: Path, screenshot_meta: dict | None = None) -> dict:
        width, height = image_size(image_path)
        ocr = run_ocr(image_path)
        should_use_vlm = self.use_vlm == "always" or (
            self.use_vlm == "auto" and (not ocr["available"] or len(ocr["items"]) < 5)
        )
        vlm = {"available": False, "error": None, "summary": ""}
        if should_use_vlm:
            vlm = summarize_screen(self.client, self.vlm_model, image_path)

        candidates = build_ui_candidates(ocr["items"], (width, height))
        return {
            "screenshot": screenshot_meta or {"path": str(image_path), "width": width, "height": height},
            "screen_size": [width, height],
            "ocr": ocr,
            "vlm": vlm,
            "ui_candidates": candidates,
        }
