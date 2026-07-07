from typing import Optional, Tuple
from .state import MousePosition

def get_mouse_position() -> Optional[MousePosition]:
    """Captures the current (x, y) coordinates of the mouse."""
    try:
        import pyautogui
        x, y = pyautogui.position()
        return MousePosition(x=x, y=y)
    except Exception:
        return None

def get_clipboard() -> Optional[str]:
    """Retrieves the current text from the clipboard."""
    try:
        import pyperclip
        text = pyperclip.paste()
        return text if text else None
    except Exception:
        return None
