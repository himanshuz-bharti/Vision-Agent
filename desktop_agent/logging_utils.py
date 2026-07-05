from __future__ import annotations

import inspect
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_RUN_DIR: Path | None = None
_EVENTS_PATH: Path | None = None
_CONFIGURED = False
_MAX_STRING = 12000
_MAX_LIST = 80
_MAX_DICT = 120


def setup_run_logging(run_dir: Path, *, verbose_console: bool = False) -> dict[str, str]:
    global _RUN_DIR, _EVENTS_PATH, _CONFIGURED
    _RUN_DIR = run_dir
    _EVENTS_PATH = run_dir / "events.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(run_dir / "agent.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if verbose_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    _CONFIGURED = True
    paths = {"agent_log": str(run_dir / "agent.log"), "events_jsonl": str(_EVENTS_PATH)}
    log_event("logging_configured", paths=paths)
    return paths


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(event: str, *, level: str = "INFO", **fields: Any) -> None:
    caller = _caller_info()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "event": event,
        "module": caller["module"],
        "file": caller["file"],
        "function": caller["function"],
        "line": caller["line"],
        "fields": sanitize(fields),
    }

    logger = logging.getLogger(caller["module"])
    log_method = getattr(logger, level.lower(), logger.info)
    log_method("%s %s", event, json.dumps(record["fields"], ensure_ascii=True, default=str))

    if _EVENTS_PATH is not None:
        try:
            with _EVENTS_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
        except OSError:
            logger.exception("failed_to_write_structured_event")


def say(message: str, *, event: str = "console", level: str = "INFO", **fields: Any) -> None:
    """Print a human-facing progress line to the console AND record it as a structured event.

    Use this for anything the user should see live (planning steps, chosen action,
    execution result). Use log_event for detailed disk-only telemetry.
    """
    print(message)
    log_event(event, level=level, message=message, **fields)


def log_exception(event: str, exc: BaseException, **fields: Any) -> None:
    log_event(
        event,
        level="ERROR",
        error_type=type(exc).__name__,
        error=str(exc),
        **fields,
    )
    logging.getLogger(_caller_info()["module"]).exception(event)


def sanitize(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "<max_depth>"
    if isinstance(value, str):
        return _sanitize_string(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, dict):
        items = list(value.items())
        result: dict[str, Any] = {}
        for key, item in items[:_MAX_DICT]:
            key_str = str(key)
            if key_str.lower() in {"images", "image", "image_b64", "base64"}:
                result[key_str] = _summarize_images(item)
            else:
                result[key_str] = sanitize(item, depth=depth + 1)
        if len(items) > _MAX_DICT:
            result["<truncated_keys>"] = len(items) - _MAX_DICT
        return result
    if isinstance(value, list | tuple | set):
        values = list(value)
        output = [sanitize(item, depth=depth + 1) for item in values[:_MAX_LIST]]
        if len(values) > _MAX_LIST:
            output.append(f"<truncated_items={len(values) - _MAX_LIST}>")
        return output
    return _sanitize_string(str(value))


def _sanitize_string(value: str) -> str:
    if len(value) > _MAX_STRING:
        return value[:_MAX_STRING] + f"<truncated_chars={len(value) - _MAX_STRING}>"
    return value


def _summarize_images(value: Any) -> Any:
    if isinstance(value, list):
        return [f"<image len={len(str(item))}>" for item in value]
    return f"<image len={len(str(value))}>"


def _caller_info() -> dict[str, Any]:
    frame = inspect.currentframe()
    if frame is None:
        return {"module": "unknown", "file": "unknown", "function": "unknown", "line": 0}
    current = frame
    while current:
        module = current.f_globals.get("__name__", "unknown")
        if module != __name__:
            code = current.f_code
            return {
                "module": module,
                "file": Path(code.co_filename).name,
                "function": code.co_name,
                "line": current.f_lineno,
            }
        current = current.f_back
    return {"module": __name__, "file": Path(__file__).name, "function": "unknown", "line": 0}