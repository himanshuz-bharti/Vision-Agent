from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import requests

from .logging_utils import log_event, log_exception


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Minimal local Ollama chat client (text + image), no other providers."""

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

        started = time.perf_counter()
        log_event(
            "ollama_chat_started",
            base_url=self.base_url,
            model=model,
            json_mode=json_mode,
            timeout=timeout,
            temperature=temperature,
            messages=summarize_messages(messages),
        )
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout)
        except requests.RequestException as exc:
            log_exception("ollama_chat_request_error", exc, base_url=self.base_url, model=model)
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. Start it with `ollama serve` "
                "and pull the configured models."
            ) from exc

        if response.status_code >= 400:
            log_event(
                "ollama_chat_http_error",
                level="ERROR",
                base_url=self.base_url,
                model=model,
                status_code=response.status_code,
                response_text=response.text[:1000],
                elapsed_ms=_elapsed_ms(started),
            )
            raise OllamaError(f"Ollama returned HTTP {response.status_code}: {response.text[:500]}")

        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            log_event(
                "ollama_chat_unexpected_response",
                level="ERROR",
                base_url=self.base_url,
                model=model,
                response=data,
                elapsed_ms=_elapsed_ms(started),
            )
            raise OllamaError(f"Unexpected Ollama response: {data}")
        log_event(
            "ollama_chat_completed",
            base_url=self.base_url,
            model=model,
            response_chars=len(content),
            response_preview=content[:1200],
            elapsed_ms=_elapsed_ms(started),
        )
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
        started = time.perf_counter()
        log_event(
            "ollama_chat_with_image_started",
            model=model,
            image_path=image_path,
            image_bytes=image_path.stat().st_size if image_path.exists() else None,
            timeout=timeout,
            temperature=temperature,
        )
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        try:
            content = self.chat(
                model,
                [{"role": "user", "content": prompt, "images": [image_b64]}],
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as exc:
            log_exception("ollama_chat_with_image_error", exc, model=model, image_path=image_path)
            raise
        log_event(
            "ollama_chat_with_image_completed",
            model=model,
            image_path=image_path,
            response_chars=len(content),
            response_preview=content[:1200],
            elapsed_ms=_elapsed_ms(started),
        )
        return content

    def available_models(self, *, timeout: int = 10) -> dict[str, Any]:
        """Return {"reachable": bool, "models": [names], "error": str|None} for preflight."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            response.raise_for_status()
            names = [m.get("name", "") for m in (response.json().get("models") or [])]
            return {"reachable": True, "models": names, "error": None}
        except requests.RequestException as exc:
            return {"reachable": False, "models": [], "error": str(exc)}


def summarize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        content_text = content if isinstance(content, str) else str(content)
        images = message.get("images") or []
        summary.append(
            {
                "role": message.get("role"),
                "content_chars": len(content_text),
                "content_preview": content_text[:800],
                "image_count": len(images) if isinstance(images, list) else 1,
            }
        )
    return summary


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


class OpenAICompatibleClient:
    """Base client for any OpenAI-compatible inference API (HF, Groq, etc.).

    Subclasses only need to set ``provider_name`` and ``base_url``
    (via ``__init__``).  Retry logic for HTTP 402/429 rate-limit responses
    is baked in.
    """

    provider_name: str = "openai"

    # Retry settings for rate-limited (402/429) responses on free tiers.
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 10  # seconds; doubles each retry

    def __init__(self, token: str, base_url: str) -> None:
        self.token = token
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
        headers = {"Authorization": f"Bearer {self.token}"}
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        log_event(
            f"{self.provider_name}_chat_started",
            model=model,
            json_mode=json_mode,
            timeout=timeout,
            temperature=temperature,
            messages=summarize_messages(messages),
        )

        last_exc: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, headers=headers, timeout=timeout,
                )
                response.raise_for_status()
                break  # success
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                last_exc = exc
                if status in (402, 429) and attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    log_event(
                        f"{self.provider_name}_rate_limit_retry",
                        level="WARNING",
                        model=model,
                        status_code=status,
                        attempt=attempt,
                        delay_seconds=delay,
                    )
                    print(
                        f"  [rate-limit] {self.provider_name} returned HTTP {status}. "
                        f"Retrying in {delay}s (attempt {attempt}/{self.MAX_RETRIES})..."
                    )
                    time.sleep(delay)
                    continue
                # Non-retryable HTTP error or last attempt
                log_exception(f"{self.provider_name}_chat_request_error", exc, model=model)
                if status in (402, 429):
                    raise RuntimeError(
                        f"{self.provider_name} free-tier rate limit reached (HTTP {status}). "
                        f"The agent retried {self.MAX_RETRIES} times but the quota is still exhausted. "
                        "Options: (1) wait a few minutes and retry, (2) upgrade your plan, "
                        "or (3) switch to a different provider or local Ollama."
                    ) from exc
                raise RuntimeError(
                    f"{self.provider_name} API returned HTTP {status}. "
                    "Please check your internet connection and API token."
                ) from exc
            except requests.RequestException as exc:
                last_exc = exc
                log_exception(f"{self.provider_name}_chat_request_error", exc, model=model)
                raise RuntimeError(
                    f"{self.provider_name} API request failed (network error). "
                    "Please check your internet connection."
                ) from exc

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Unexpected {self.provider_name} response: {data}")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected {self.provider_name} response content: {data}")

        log_event(
            f"{self.provider_name}_chat_completed",
            model=model,
            response_chars=len(content),
            response_preview=content[:1200],
            elapsed_ms=_elapsed_ms(started),
        )
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
        started = time.perf_counter()
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ]

        log_event(
            f"{self.provider_name}_chat_with_image_started",
            model=model,
            image_path=image_path,
            timeout=timeout,
            temperature=temperature,
        )
        try:
            content = self.chat(model, messages, temperature=temperature, timeout=timeout)
        except Exception as exc:
            log_exception(f"{self.provider_name}_chat_with_image_error", exc, model=model, image_path=image_path)
            raise
        log_event(
            f"{self.provider_name}_chat_with_image_completed",
            model=model,
            image_path=image_path,
            response_chars=len(content),
            response_preview=content[:1200],
            elapsed_ms=_elapsed_ms(started),
        )
        return content

    def available_models(self, *, timeout: int = 10) -> dict[str, Any]:
        return {"reachable": True, "models": [], "error": None}


class HuggingFaceClient(OpenAICompatibleClient):
    """Client for Hugging Face Serverless Inference API."""

    provider_name = "hf"

    def __init__(self, token: str) -> None:
        super().__init__(token, "https://router.huggingface.co/v1")


class GroqClient(OpenAICompatibleClient):
    """Client for Groq Cloud Inference API (free tier: ~30 req/min)."""

    provider_name = "groq"

    def __init__(self, token: str) -> None:
        super().__init__(token, "https://api.groq.com/openai/v1")

