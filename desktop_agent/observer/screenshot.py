import time
from pathlib import Path
from .state import ScreenshotData

def capture_screenshot(save_dir: str | Path) -> ScreenshotData:
    """
    Captures the entire desktop using MSS for maximum performance.
    Saves the output to the specified directory.
    """
    import mss
    import mss.tools

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.time()
    filename = f"screenshot_{int(timestamp * 1000)}.png"
    filepath = out_dir / filename
    
    with mss.mss() as sct:
        # monitor 0 is the "All monitors" encompassing virtual desktop
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        
        # Save raw bytes to PNG
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(filepath))
        
        width = monitor["width"]
        height = monitor["height"]
        
    return ScreenshotData(
        path=str(filepath),
        width=width,
        height=height,
        timestamp=timestamp
    )
