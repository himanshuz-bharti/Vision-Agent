import json
import dataclasses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

@dataclass(slots=True)
class ScreenshotData:
    path: str
    width: int
    height: int
    timestamp: float

@dataclass(slots=True)
class WindowInfo:
    title: str
    process_name: str
    pid: int
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    is_maximized: bool

@dataclass(slots=True)
class ProcessInfo:
    name: str
    pid: int
    cpu: float
    memory: float

@dataclass(slots=True)
class MousePosition:
    x: int
    y: int

@dataclass(slots=True)
class DesktopState:
    timestamp: str
    screen_width: int
    screen_height: int
    active_window: Optional[WindowInfo]
    running_processes: Optional[List[ProcessInfo]]
    screenshot: Optional[ScreenshotData]
    accessibility_tree: Optional[Dict[str, Any]]
    mouse_position: Optional[MousePosition]
    clipboard: Optional[str]

    def to_json(self) -> str:
        """Serialize the entire state to a JSON string, handling nested dataclasses."""
        def default_encoder(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            if isinstance(obj, (datetime, Path)):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Convert self to dict first, then serialize
        data = dataclasses.asdict(self)
        return json.dumps(data, indent=2, ensure_ascii=False, default=default_encoder)

    def save(self, path: str | Path) -> None:
        """Save the JSON state to a file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_json(), encoding="utf-8")
