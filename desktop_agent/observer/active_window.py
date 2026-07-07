import sys
from typing import Optional
from .state import WindowInfo

def get_active_window() -> Optional[WindowInfo]:
    """
    Safely captures the currently active (foreground) window.
    Returns None if the system is headless or extraction fails.
    """
    if sys.platform != "win32":
        return None
        
    try:
        import win32gui
        import win32process
        
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None
            
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        
        rect = win32gui.GetWindowRect(hwnd)
        x, y, right, bottom = rect
        
        import psutil
        try:
            process_name = psutil.Process(pid).name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_name = "unknown"
            
        return WindowInfo(
            title=title,
            process_name=process_name,
            pid=pid,
            bounds=(x, y, right - x, bottom - y),
            is_maximized=bool(win32gui.IsZoomed(hwnd))
        )
    except Exception:
        return None
