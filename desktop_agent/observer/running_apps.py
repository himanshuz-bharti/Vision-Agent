import sys
from typing import List
from .state import ProcessInfo

def get_running_processes() -> List[ProcessInfo]:
    """
    Retrieves currently running GUI processes.
    Filters out background services by only capturing processes attached to visible windows.
    """
    apps = []
    if sys.platform != "win32":
        return apps
        
    try:
        import win32gui
        import win32process
        import psutil
        
        gui_pids = set()
        
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                gui_pids.add(pid)
                
        win32gui.EnumWindows(callback, None)
        
        for pid in gui_pids:
            try:
                p = psutil.Process(pid)
                apps.append(ProcessInfo(
                    name=p.name(),
                    pid=pid,
                    cpu=p.cpu_percent(),
                    memory=p.memory_percent()
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
    except Exception:
        pass
        
    return apps
