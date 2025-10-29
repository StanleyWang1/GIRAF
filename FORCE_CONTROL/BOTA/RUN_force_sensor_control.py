#!/usr/bin/env python3
"""
force_spin_threaded.py
- Thread 1: Force sensor -> updates shared Fz (blocking read)
- Thread 2: Motor control -> if Fz < -5 N, increment ID12 goal; else hold
- Uses your helpers:
    * force_sensor_driver.ForceSensorDriver
    * dynamixel_driver.{dynamixel_connect, dynamixel_drive, dynamixel_disconnect}
"""

import time
import sys
import signal
import threading

from force_sensor_driver import ForceSensorDriver
from dynamixel_driver import (
    dynamixel_connect, dynamixel_drive, dynamixel_disconnect,
)
from control_table import MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR12_MIN, MOTOR12_MAX, TORQUE_ENABLE, PRESENT_POSITION

# ===================== User settings =====================
FZ_THRESHOLD_N = -1.0   # spin when Fz < -5 N
STEP_TICKS     = 10    # how many ticks to add per motor update while spinning
SENSOR_TIMEOUT = 0.25   # seconds to wait for a new sensor sample in sensor thread
CTRL_HZ        = 100.0  # motor update rate
PRINT_HZ       = 2.0    # status print rate
TARE_ON_START  = True
# ========================================================

# Shared state
running = True
running_lock = threading.Lock()

shared = {"Fz": None}
shared_lock = threading.Lock()

def handle_sigint(_s, _f):
    global running
    with running_lock:
        running = False
signal.signal(signal.SIGINT, handle_sigint) # Ctrl+C handler

# ------------------------- Sensor thread -------------------------
def sensor_thread():
    global running
    fs = ForceSensorDriver(tare_on_start=TARE_ON_START)
    fs.start()
    print("\033[92mSENSOR: started\033[0m")
    try:
        while True:
            with running_lock:
                if not running:
                    break
            fr = fs.read_blocking(timeout_s=SENSOR_TIMEOUT)
            if fr is None:
                continue
            Fz = float(fr.force[2])
            with shared_lock:
                shared["Fz"] = Fz
    except Exception as e:
        print(f"\033[91m[FATAL][SENSOR] {e}\033[0m", file=sys.stderr)
    finally:
        try:
            fs.stop()
        except Exception:
            pass
        print("\033[93mSENSOR: stopped\033[0m")

# ------------------------- Motor thread -------------------------
def motor_thread():
    global running
    
    PITCH_MOTOR = 12 

    dmx_controller, dmx_GSW = dynamixel_connect()
    print("\033[93mTELEOP: Motors Connected!\033[0m")
    time.sleep(0.5)

    dynamixel_drive(dmx_controller, dmx_GSW, [(MOTOR12_MIN + MOTOR12_MAX)/2])

    # Performance benchmarking (loop Hz)
    loop_count = 0
    last_time = time.perf_counter()
    print_interval = 1.0 / PRINT_HZ if PRINT_HZ > 0 else 1.0

    try:
        while running:
            with shared_lock:
                Fz = shared["Fz"]
            
            MOTOR_12_ticks = (MOTOR12_MIN + MOTOR12_MAX)/2 + 100*Fz
            MOTOR_12_ticks = max(MOTOR12_MIN, min(MOTOR12_MAX, MOTOR_12_ticks))
            dynamixel_drive(dmx_controller, dmx_GSW, [int(MOTOR_12_ticks)])
            time.sleep(0.01)

            # perf counting
            loop_count += 1
            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed >= print_interval:
                hz = loop_count / elapsed if elapsed > 0 else float('inf')
                print(f"\033[94m[MOTOR] loop Hz: {hz:.1f}\033[0m")
                loop_count = 0
                last_time = now

    finally:
        dynamixel_disconnect(dmx_controller)
        print("\033[93mTELEOP: Motors Disconnected!\033[0m")

# --------------------------- Main ---------------------------
if __name__ == "__main__":
    print("\033[96mRUN: threaded force→spin — Ctrl+C to stop\033[0m")
    ts = threading.Thread(target=sensor_thread, daemon=True)
    tm = threading.Thread(target=motor_thread,  daemon=True)
    ts.start()
    tm.start()
    try:
        while True:
            with running_lock:
                if not running:
                    break
            time.sleep(0.5)
    except KeyboardInterrupt:
        with running_lock:
            running = False
    finally:
        ts.join(timeout=2.0)
        tm.join(timeout=2.0)
        print("\033[92mDONE\033[0m")
