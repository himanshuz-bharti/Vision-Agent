from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


class CaptureError(RuntimeError):
    pass


def capture_screenshot(path: Path) -> dict:
    import pyautogui

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image = pyautogui.screenshot()
    except Exception as exc:
        raise CaptureError(
            "Screen capture failed. Run this from your normal interactive Windows desktop session, "
            "not from a restricted sandbox, background service, SSH session, or elevated/non-interactive terminal."
        ) from exc
    image.save(path)
    return {"path": str(path), "width": image.width, "height": image.height, "sha256": image_hash(path)}


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height
