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
    JOINT1, JOINT2, JOINT3, GRIPPER1, GRIPPER2, GRIPPER3
)
from control_table import PRESENT_POSITION  # used to seed ticks on startup

# ===================== User settings =====================
FZ_THRESHOLD_N = -1.0   # spin when Fz < -5 N
STEP_TICKS     = 10    # how many ticks to add per motor update while spinning
SENSOR_TIMEOUT = 0.25   # seconds to wait for a new sensor sample in sensor thread
CTRL_HZ        = 100.0  # motor update rate
PRINT_HZ       = 5.0    # status print rate
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
signal.signal(signal.SIGINT, handle_sigint)

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
    ctrl, gsw = dynamixel_connect()  # configures EP mode + torque on (per your driver)
    print("\033[92mDMX: connected\033[0m")

    # Seed ticks from present positions so motion is relative to current pose
    ids = [JOINT1, JOINT2, JOINT3, GRIPPER1, GRIPPER2, GRIPPER3]
    ticks = [0, 0, 0, 0, 0, 0]
    for i, mid in enumerate(ids):
        pv = ctrl.read(mid, PRESENT_POSITION)
        ticks[i] = int(pv) if pv is not False else 0

    period = 1.0 / CTRL_HZ
    print_period = 1.0 / PRINT_HZ
    t_last_print = 0.0

    try:
        while True:
            with running_lock:
                if not running:
                    break

            # get latest Fz (may be None until first sample)
            with shared_lock:
                Fz = shared["Fz"]

            # decide and update ticks
            if Fz is not None and Fz < FZ_THRESHOLD_N:
                ticks[1] += STEP_TICKS  # JOINT2 (ID12) spin step

            # send goal for all six (API expects 6-length list)
            if not dynamixel_drive(ctrl, gsw, ticks):
                print("\033[91m[WARN] SyncWrite failed\033[0m")

            # periodic status
            now = time.perf_counter()
            if (now - t_last_print) >= print_period:
                t_last_print = now
                print(f"Fz={Fz if Fz is not None else '---':>7}  action={'SPIN' if (Fz is not None and Fz < FZ_THRESHOLD_N) else 'HOLD'}  id12_goal={ticks[1]}")

            # pace motor loop
            time.sleep(period)

    except Exception as e:
        print(f"\033[91m[FATAL][MOTOR] {e}\033[0m", file=sys.stderr)
    finally:
        try:
            dynamixel_disconnect(ctrl)
        except Exception:
            pass
        print("\033[93mDMX: disconnected\033[0m")

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
