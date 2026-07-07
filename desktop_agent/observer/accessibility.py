from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class AccessibilityCollector(ABC):
    """
    Abstract base class for capturing the accessibility tree.
    Allows easy swapping between Windows UI Automation, AT-SPI, or macOS Accessibility.
    """
    @abstractmethod
    def collect(self) -> Optional[Dict[str, Any]]:
        pass

class DummyAccessibilityCollector(AccessibilityCollector):
    """A placeholder collector that safely returns None until fully implemented."""
    def collect(self) -> Optional[Dict[str, Any]]:
        return None
