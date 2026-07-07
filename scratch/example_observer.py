import logging
from pathlib import Path

# Setup logging to see the performance timing
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

from desktop_agent.observer import DesktopObserver, ObserverConfig

def main():
    print("\n--- Starting Observer Test ---")
    config = ObserverConfig(
        screenshots_dir="runs/example/screenshots"
    )
    
    observer = DesktopObserver(config)
    
    state = observer.observe()
    
    print("\n--- Desktop State Summary ---")
    if state.active_window:
        print(f"Active Window: '{state.active_window.title}' (Process: {state.active_window.process_name})")
        
    if state.mouse_position:
        print(f"Mouse Position: ({state.mouse_position.x}, {state.mouse_position.y})")
        
    if state.screenshot:
        print(f"Screenshot saved to: {state.screenshot.path}")
        
    print(f"Number of running GUI apps: {len(state.running_processes) if state.running_processes else 0}")
    
    out_path = Path("runs/example/state.json")
    state.save(out_path)
    print(f"\nSaved JSON state to {out_path}")

if __name__ == "__main__":
    main()
