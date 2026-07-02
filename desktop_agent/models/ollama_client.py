from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout: int = 180,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout)
        except requests.RequestException as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. Start it with `ollama serve` "
                "and pull the configured models."
            ) from exc

        if response.status_code >= 400:
            raise OllamaError(f"Ollama returned HTTP {response.status_code}: {response.text[:500]}")

        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError(f"Unexpected Ollama response: {data}")
        return content.strip()

    def chat_with_image(
        self,
        model: str,
        prompt: str,
        image_path: Path,
        *,
        temperature: float = 0.0,
        timeout: int = 240,
    ) -> str:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return self.chat(
            model,
            [{"role": "user", "content": prompt, "images": [image_b64]}],
            temperature=temperature,
            timeout=timeout,
        )
