from dataclasses import dataclass

@dataclass(slots=True)
class ObserverConfig:
    """Configuration for toggling individual desktop collectors."""
    capture_screenshot: bool = True
    capture_processes: bool = True
    capture_clipboard: bool = True
    capture_mouse: bool = True
    capture_active_window: bool = True
    capture_accessibility: bool = True
    
    # Path where screenshots will be saved
    # The Observer will dynamically inject run-specific directories if needed.
    screenshots_dir: str = "runs/screenshots"
