"""Send rumble commands to a connected game controller via pygame."""
import argparse
import time

import pygame


def connect_joystick(index: int = 0) -> pygame.joystick.Joystick:
    """Initialize pygame and return the joystick at the given index."""
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick found.")
    js = pygame.joystick.Joystick(index)
    js.init()
    return js


def normalize_strength(value: float) -> float:
    """Clamp rumble strength to the [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def rumble(js: pygame.joystick.Joystick, low: float, high: float, duration: float) -> bool:
    """Try to trigger controller rumble; returns True if pygame reports success."""
    low = normalize_strength(low)
    high = normalize_strength(high)
    duration_ms = int(max(0.0, duration) * 1000)
    if not hasattr(js, "rumble"):
        return False
    return bool(js.rumble(low, high, duration_ms))


def stop_rumble(js: pygame.joystick.Joystick) -> None:
    """Attempt to stop rumble on the controller."""
    if hasattr(js, "stop_rumble"):
        js.stop_rumble()


def run_pattern(js: pygame.joystick.Joystick, low: float, high: float, duration: float,
                repeats: int, interval: float) -> None:
    """Send a sequence of rumble pulses."""
    for _ in range(repeats):
        success = rumble(js, low, high, duration)
        if not success:
            print("Rumble call was not supported or failed.")
            break
        time.sleep(duration)
        stop_rumble(js)
        time.sleep(max(0.0, interval))


def main() -> None:
    parser = argparse.ArgumentParser(description="Send rumble commands to a controller")
    parser.add_argument("--index", type=int, default=0, help="Joystick index (default: 0)")
    parser.add_argument("--low", type=float, default=0.0, help="Low-frequency motor strength")
    parser.add_argument("--high", type=float, default=1.0, help="High-frequency motor strength")
    parser.add_argument("--duration", type=float, default=0.5, help="Pulse duration in seconds")
    parser.add_argument("--repeats", type=int, default=1, help="Number of pulses")
    parser.add_argument("--interval", type=float, default=0.2, help="Delay between pulses")
    args = parser.parse_args()

    js = None
    try:
        js = connect_joystick(args.index)
        print(f"Connected to: {js.get_name()}")
        run_pattern(js, args.low, args.high, args.duration, args.repeats, args.interval)
    finally:
        if js is not None:
            stop_rumble(js)
        pygame.quit()


if __name__ == "__main__":
    main()
