#!/usr/bin/env python3
"""
dmx_spin_dynamixel_only.py
- Uses ONLY your dynamixel_* helpers (same style as your big teleop file).
- Spins Dynamixel Motor 12 by stepping goal position each loop.
- Ctrl+C to stop (clean shutdown).
"""

import time
import threading
import numpy as np

from control_table import (
    MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME,
    MOTOR100_OPEN, MOTOR101_OPEN, MOTOR102_OPEN
)
from dynamixel_driver import (
    dynamixel_connect, dynamixel_drive, dynamixel_disconnect, radians_to_ticks
)

# ---------------- User settings ----------------
STEP_TICKS   = 64       # how many ticks to add per loop (increase if too slow)
LOOP_DT      = 0.01     # loop period (s) → 100 Hz
PRINT_EVERY  = 0.5      # seconds between console prints
# ------------------------------------------------

running = True
running_lock = threading.Lock()

def motor_control():
    """Continuously nudge Motor 12 goal forward by STEP_TICKS each loop."""
    global running

    # Start near home positions for all 6 Dynamixels
    ticks = [
        int(MOTOR11_HOME),  # Joint 11
        int(MOTOR12_HOME),  # Joint 12 (we'll spin this one)
        int(MOTOR13_HOME),  # Joint 13
        int(MOTOR100_OPEN), # Gripper 100
        int(MOTOR101_OPEN), # Gripper 101
        int(MOTOR102_OPEN), # Gripper 102
    ]

    ctrl, gsw = dynamixel_connect()
    print("\033[93mDMX: Connected (GroupSyncWrite ready)\033[0m")

    t_last_print = time.perf_counter()

    try:
        while True:
            with running_lock:
                if not running:
                    break

            # Increment Motor 12 goal (index 1) in extended position ticks
            ticks[1] += STEP_TICKS

            # Write all 6 motors (your driver expects a 6-length tick list)
            ok = dynamixel_drive(ctrl, gsw, ticks)
            if not ok:
                print("\033[91m[WARN] SyncWrite failed\033[0m")

            # Throttle loop
            time.sleep(LOOP_DT)

            # Lightweight status print
            now = time.perf_counter()
            if (now - t_last_print) >= PRINT_EVERY:
                t_last_print = now
                print(f"Motor12 goal ticks: {ticks[1]}  (step={STEP_TICKS})")

    finally:
        dynamixel_disconnect(ctrl)
        print("\033[93mDMX: Disconnected (torque off, port closed)\033[0m")

if __name__ == "__main__":
    try:
        th = threading.Thread(target=motor_control, daemon=True)
        th.start()
        print("\033[96mRUN: Motor 12 stepping — Ctrl+C to stop\033[0m")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        with running_lock:
            running = False
        print("\n\033[93mRUN: stopping...\033[0m")
        th.join(timeout=2.0)
        print("\033[92mRUN: stopped cleanly\033[0m")
