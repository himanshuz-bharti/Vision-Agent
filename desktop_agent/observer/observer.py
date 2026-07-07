import logging
from datetime import datetime
import pyautogui

from .config import ObserverConfig
from .state import DesktopState
from .utils import time_it
from .screenshot import capture_screenshot
from .active_window import get_active_window
from .running_apps import get_running_processes
from .mouse_clip import get_mouse_position, get_clipboard
from .accessibility import AccessibilityCollector, DummyAccessibilityCollector

logger = logging.getLogger(__name__)

class DesktopObserver:
    """
    Master Orchestrator for capturing the desktop state.
    Executes multiple independent collectors securely without executing actions.
    """
    def __init__(
        self, 
        config: ObserverConfig = None,
        accessibility_collector: AccessibilityCollector = None
    ):
        self.config = config or ObserverConfig()
        self.accessibility = accessibility_collector or DummyAccessibilityCollector()

    def observe(self) -> DesktopState:
        """Takes a complete snapshot of the desktop state."""
        timestamp = datetime.now().isoformat()
        
        # Primary Screen dimensions
        screen_width, screen_height = pyautogui.size()
        
        screenshot_data = None
        active_window_data = None
        processes_data = None
        mouse_data = None
        clipboard_data = None
        accessibility_data = None

        with time_it("DesktopObservation"):
            if self.config.capture_screenshot:
                logger.debug("Capturing screenshot...")
                try:
                    with time_it("capture_screenshot"):
                        screenshot_data = capture_screenshot(self.config.screenshots_dir)
                except Exception as e:
                    logger.error(f"Failed to capture screenshot: {e}")

            if self.config.capture_active_window:
                logger.debug("Collecting active window...")
                try:
                    with time_it("get_active_window"):
                        active_window_data = get_active_window()
                except Exception as e:
                    logger.error(f"Failed to collect active window: {e}")

            if self.config.capture_processes:
                logger.debug("Collecting running apps...")
                try:
                    with time_it("get_running_processes"):
                        processes_data = get_running_processes()
                except Exception as e:
                    logger.error(f"Failed to collect running apps: {e}")

            if self.config.capture_mouse:
                logger.debug("Collecting mouse position...")
                try:
                    with time_it("get_mouse_position"):
                        mouse_data = get_mouse_position()
                except Exception as e:
                    logger.error(f"Failed to collect mouse position: {e}")

            if self.config.capture_clipboard:
                logger.debug("Collecting clipboard...")
                try:
                    with time_it("get_clipboard"):
                        clipboard_data = get_clipboard()
                except Exception as e:
                    logger.error(f"Failed to collect clipboard: {e}")

            if self.config.capture_accessibility:
                logger.debug("Collecting accessibility tree...")
                try:
                    with time_it("accessibility.collect"):
                        accessibility_data = self.accessibility.collect()
                except Exception as e:
                    logger.error(f"Failed to collect accessibility tree: {e}")

        logger.debug("Done.")

        return DesktopState(
            timestamp=timestamp,
            screen_width=screen_width,
            screen_height=screen_height,
            active_window=active_window_data,
            running_processes=processes_data,
            screenshot=screenshot_data,
            accessibility_tree=accessibility_data,
            mouse_position=mouse_data,
            clipboard=clipboard_data
        )
