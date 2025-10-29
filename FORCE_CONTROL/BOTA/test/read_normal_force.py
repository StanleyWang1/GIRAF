#!/usr/bin/env python3
import os, sys, json, time, signal
import bota_driver

DURATION_SEC = 10.0
DO_TARE = True

# Repo root = two levels up from this file (matches your working scripts)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH  = os.path.join(project_root, "bota_driver_config", "ethercat_gen0.json")

stop_flag = False
def _sigint(_s, _f):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, _sigint)

def fail(msg):
    print(msg)
    sys.exit(1)

def preflight():
    if not os.path.isfile(CONFIG_PATH):
        fail(f"[FATAL] Config not found at {CONFIG_PATH}")
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        ci = cfg["driver_config"]["communication_interface_name"]
        if ci.lower() != "canopen_over_ethercat_gen0":
            fail(f"[FATAL] communication_interface_name='{ci}' (expected 'CANopen_over_EtherCAT_gen0')")
    except Exception as e:
        fail(f"[FATAL] Config JSON issue: {e}")

def main():
    preflight()

    drv = bota_driver.BotaDriver(CONFIG_PATH)
    if not drv.configure(): fail("[FATAL] Failed to configure driver")
    if DO_TARE and not drv.tare(): fail("[FATAL] Failed to tare sensor")
    if not drv.activate(): fail("[FATAL] Failed to activate driver")

    unique_samples = 0
    last_ts = None

    t0 = time.perf_counter()
    end = t0 + DURATION_SEC

    try:
        while not stop_flag and time.perf_counter() < end:
            frame = drv.read_frame()
            ts = frame.timestamp
            if last_ts is None:
                last_ts = ts
                # Print the first value
                print(frame.force[2])
                unique_samples += 1
                continue

            # Only count/print when a NEW device sample arrives
            if ts != last_ts:
                last_ts = ts
                unique_samples += 1
                print(frame.force[2])  # live Fz (N)
    finally:
        try:
            drv.deactivate()
            drv.shutdown()
        except Exception:
            pass

    elapsed = max(1e-9, time.perf_counter() - t0)  # wall time (s)
    rate_hz = unique_samples / elapsed

    # Print ONLY the sampling rate line (no extra text)
    print(f"{rate_hz:.3f} Hz")

if __name__ == "__main__":
    main()
