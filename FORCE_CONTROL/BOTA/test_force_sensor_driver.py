#!/usr/bin/env python3
"""
test_rs_cli.py
- r : read once from buffer (non-blocking latest frame)
- s : stream only NEW samples until you press Enter (or Ctrl+C)
- q : quit

It auto-finds ../bota_driver_config/ethercat_gen0.json by walking up from this file.
For EtherCAT, ensure the python in THIS venv has caps:
  sudo setcap cap_net_raw,cap_net_admin+eip $(readlink -f $(which python3))
"""

import os, sys, json, time, threading
from pathlib import Path

import bota_driver

DO_TARE = True

def find_config(filename="ethercat_gen0.json") -> str:
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "bota_driver_config" / filename
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(f"Could not find {filename} by walking up from {here}")

def preflight(cfg_path: str):
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    cfg = json.loads(Path(cfg_path).read_text())
    iface = cfg["driver_config"]["communication_interface_name"]
    if iface.lower() != "canopen_over_ethercat_gen0":
        raise RuntimeError(f"Config interface is '{iface}', expected 'CANopen_over_EtherCAT_gen0'")

def read_once(drv):
    fr = drv.read_frame()  # latest frame, non-blocking
    F = fr.force; T = fr.torque
    print(f"READ (buffer): F=[{F[0]:.3f},{F[1]:.3f},{F[2]:.3f}] N | "
          f"T=[{T[0]:.4f},{T[1]:.4f},{T[2]:.4f}] N·m")

def stream_new(drv):
    """
    Stream only NEW device samples (timestamp changes).
    Stops when user presses Enter or on Ctrl+C.
    """
    stop_flag = {"stop": False}

    def waiter():
        try:
            input("(streaming) Press Enter to stop...\n")
        except EOFError:
            pass
        stop_flag["stop"] = True

    threading.Thread(target=waiter, daemon=True).start()

    last_ts = None
    try:
        while not stop_flag["stop"]:
            fr = drv.read_frame()
            ts = fr.timestamp
            if last_ts is None or ts != last_ts:
                last_ts = ts
                F = fr.force; T = fr.torque
                print(f"NEW: F=[{F[0]:.3f},{F[1]:.3f},{F[2]:.3f}] N | "
                      f"T=[{T[0]:.4f},{T[1]:.4f},{T[2]:.4f}] N·m")
            # tiny sleep avoids pegging a core if sample rate is low
            time.sleep(0.0005)
    except KeyboardInterrupt:
        pass
    print("[stream] stopped.")

def main():
    cfg_path = find_config()
    preflight(cfg_path)
    print(f"[INFO] Using config: {cfg_path}")

    drv = bota_driver.BotaDriver(cfg_path)
    if not drv.configure():
        print("[FATAL] configure() failed"); sys.exit(1)
    if DO_TARE and not drv.tare():
        print("[FATAL] tare() failed"); sys.exit(1)
    if not drv.activate():
        print("[FATAL] activate() failed"); sys.exit(1)
    print("[OK] Sensor ready. Commands: r (read once), s (stream), q (quit)")

    try:
        while True:
            cmd = input("[r/s/q] > ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "r":
                read_once(drv)
            elif cmd == "s":
                stream_new(drv)
            else:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if not drv.deactivate(): print("[WARN] deactivate() returned False")
            if not drv.shutdown():   print("[WARN] shutdown() returned False")
        except Exception as e:
            print(f"[WARN] close error: {e}")
        print("[INFO] Closed cleanly.")

if __name__ == "__main__":
    main()
