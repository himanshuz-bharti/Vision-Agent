from __future__ import annotations

from pathlib import Path
from typing import Any


def run_ocr(image_path: Path, *, min_confidence: float = 35.0, max_items: int = 120) -> dict[str, Any]:
    try:
        import pytesseract
        from PIL import Image
        from pytesseract import Output
    except ImportError as exc:
        return {"available": False, "error": f"pytesseract/Pillow missing: {exc}", "items": [], "text": ""}

    try:
        with Image.open(image_path) as image:
            data = pytesseract.image_to_data(image, output_type=Output.DICT)
    except Exception as exc:
        return {
            "available": False,
            "error": f"OCR failed. Install Tesseract OCR or use --use-vlm always. Details: {exc}",
            "items": [],
            "text": "",
        }

    items: list[dict[str, Any]] = []
    for i, raw_text in enumerate(data.get("text", [])):
        text = (raw_text or "").strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1.0
        if confidence < min_confidence:
            continue

        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        items.append(
            {
                "id": f"text_{len(items) + 1:03d}",
                "text": text,
                "confidence": round(confidence, 1),
                "bbox": [left, top, left + width, top + height],
            }
        )
        if len(items) >= max_items:
            break

    joined = " ".join(item["text"] for item in items)
    return {"available": True, "error": None, "items": items, "text": joined}
