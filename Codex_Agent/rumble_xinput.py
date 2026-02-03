"""Send rumble commands to Xbox controllers using XInput (Windows-specific)."""
import argparse
import ctypes
import time
from ctypes import wintypes

# Load XInput DLL
try:
    xinput = ctypes.windll.xinput1_4
except:
    try:
        xinput = ctypes.windll.xinput1_3
    except:
        try:
            xinput = ctypes.windll.xinput9_1_0
        except:
            print("ERROR: Could not load any XInput DLL")
            exit(1)

# Define XInput structures
class XINPUT_VIBRATION(ctypes.Structure):
    _fields_ = [
        ("wLeftMotorSpeed", wintypes.WORD),
        ("wRightMotorSpeed", wintypes.WORD),
    ]

class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", wintypes.DWORD),
        ("Gamepad", ctypes.c_byte * 12),  # Simplified
    ]


def normalize_strength(value: float) -> int:
    """Convert 0.0-1.0 float to 0-65535 integer for XInput."""
    clamped = max(0.0, min(1.0, value))
    return int(clamped * 65535)


def set_vibration(controller_id: int, left_motor: int, right_motor: int) -> None:
    """Set vibration motors on Xbox controller."""
    vibration = XINPUT_VIBRATION(left_motor, right_motor)
    xinput.XInputSetState(controller_id, ctypes.byref(vibration))


def get_connected(controller_id: int) -> bool:
    """Check if a controller is connected."""
    state = XINPUT_STATE()
    result = xinput.XInputGetState(controller_id, ctypes.byref(state))
    return result == 0  # ERROR_SUCCESS


def rumble(controller_id: int, low: float, high: float, duration: float) -> None:
    """Trigger controller rumble using XInput."""
    low_motor = normalize_strength(low)
    high_motor = normalize_strength(high)
    
    # Set vibration
    set_vibration(controller_id, low_motor, high_motor)
    
    # Wait for duration
    time.sleep(duration)
    
    # Stop vibration
    set_vibration(controller_id, 0, 0)


def find_connected_controller() -> int:
    """Find the first connected Xbox controller."""
    for i in range(4):  # XInput supports up to 4 controllers
        if get_connected(i):
            return i
    raise RuntimeError("No Xbox controller found.")


def run_pattern(controller_id: int, low: float, high: float, duration: float,
                repeats: int, interval: float) -> None:
    """Send a sequence of rumble pulses."""
    for i in range(repeats):
        print(f"Pulse {i+1}/{repeats}...")
        rumble(controller_id, low, high, duration)
        if i < repeats - 1:  # Don't sleep after last pulse
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send rumble to Xbox controller via XInput")
    parser.add_argument("--index", type=int, default=None, help="Controller index 0-3 (auto-detect if not specified)")
    parser.add_argument("--low", type=float, default=0.5, help="Low-frequency motor strength (default: 0.5)")
    parser.add_argument("--high", type=float, default=0.5, help="High-frequency motor strength (default: 0.5)")
    parser.add_argument("--duration", type=float, default=1.0, help="Pulse duration in seconds (default: 1.0)")
    parser.add_argument("--repeats", type=int, default=1, help="Number of pulses")
    parser.add_argument("--interval", type=float, default=0.2, help="Delay between pulses")
    args = parser.parse_args()

    try:
        # Find controller
        if args.index is not None:
            if not get_connected(args.index):
                raise RuntimeError(f"No controller found at index {args.index}")
            controller_id = args.index
        else:
            controller_id = find_connected_controller()
        
        print(f"Connected to Xbox controller at index {controller_id}")
        
        # Run rumble pattern
        run_pattern(controller_id, args.low, args.high, args.duration, args.repeats, args.interval)
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
