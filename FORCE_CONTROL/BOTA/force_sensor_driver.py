#!/usr/bin/env python3
"""
force_sensor_driver.py
Minimal wrapper around bota_driver with:
  - start()/stop() lifecycle
  - read_from_buffer()  (non-blocking, latest frame)
  - stream()            (yields only NEW samples as they arrive)
  - read_blocking()     (waits until NEW sample or timeout)

Auto-finds: ../bota_driver_config/ethercat_gen0.json
NOTE: For EtherCAT, the python in THIS venv needs caps:
  sudo setcap cap_net_raw,cap_net_admin+eip $(readlink -f $(which python3))
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import bota_driver  # must be installed in this venv

class ForceSensorDriver:
    def __init__(self, config_path: Optional[str] = None, tare_on_start: bool = True):
        self.config_path = config_path or self._auto_find_config("ethercat_gen0.json")
        self.tare_on_start = tare_on_start
        
        self._drv: Optional[bota_driver.BotaDriver] = None
        self._last_ts = None
        self._t0_wall = None

    # ---------- lifecycle ----------
    def start(self) -> None:
        drv = bota_driver.BotaDriver(self.config_path)
        if not drv.configure():
            raise RuntimeError("configure() failed")
        if self.tare_on_start and not drv.tare():
            raise RuntimeError("tare() failed")
        if not drv.activate():
            raise RuntimeError("activate() failed")
        self._drv = drv
        self._last_ts = None
        self._t0_wall = time.perf_counter()

    def stop(self) -> None:
        if self._drv is None:
            return
        try:
            self._drv.deactivate()
            self._drv.shutdown()
        finally:
            self._drv = None

    # ---------- data access ----------
    def read_from_buffer(self):
        """Return the latest frame immediately (may be same sample if no new data)."""
        self._ensure_started()
        return self._drv.read_frame()

    def stream(self) -> Generator[Tuple[float, list, list, float], None, None]:
        """
        Yield (t_wall_s, force[3], torque[3], dev_ts) **only when a NEW device sample arrives**.
        Ideal for streaming.
        """
        self._ensure_started()
        while True:
            fr = self._drv.read_frame()
            ts = fr.timestamp
            if self._last_ts is None or ts != self._last_ts:
                self._last_ts = ts
                t_wall = time.perf_counter() - self._t0_wall
                yield (t_wall, fr.force, fr.torque, ts)
            # else: same device sample; spin again

    def read_blocking(self, timeout_s: Optional[float] = None):
        """
        Block until a NEW device sample arrives (or timeout). Returns frame or None on timeout.
        """
        self._ensure_started()
        t_start = time.perf_counter()
        while True:
            fr = self._drv.read_frame()
            ts = fr.timestamp
            if self._last_ts is None or ts != self._last_ts:
                self._last_ts = ts
                return fr
            if timeout_s is not None and (time.perf_counter() - t_start) >= timeout_s:
                return None
            # tiny yield to avoid 100% CPU if sensor is slow
            time.sleep(0.0005)

    # ---------- utils ----------
    @staticmethod
    def _auto_find_config(filename: str) -> str:
        # Walk up from this file to find ../bota_driver_config/<filename>
        here = Path(__file__).resolve()
        for d in [here.parent] + list(here.parents):
            cand = d / "bota_driver_config" / filename
            if cand.is_file():
                return str(cand)
        raise FileNotFoundError(f"Could not find {filename} by walking up from {here}")

    def _ensure_started(self):
        if self._drv is None:
            raise RuntimeError("ForceSensorDriver not started. Call start() first.")

# ---------------- DEMO ----------------
if __name__ == "__main__":
    """
    Demo:
      - start the driver
      - print 20 streamed samples (F and T)
      - show an example of read_from_buffer() and read_blocking()
    """
    fs = ForceSensorDriver(tare_on_start=True)
    try:
        fs.start()
        print("[OK] Sensor started. Streaming 20 samples (only NEW ones):")
        n = 0
        for t_wall, F, T, dev_ts in fs.stream():
            print(f"t={t_wall:8.3f}s  F=[{F[0]:.3f},{F[1]:.3f},{F[2]:.3f}] N  "
                  f"T=[{T[0]:.4f},{T[1]:.4f},{T[2]:.4f}] N·m")
            n += 1
            if n >= 20:
                break

        # Non-blocking: latest frame (may duplicate)
        fr = fs.read_from_buffer()
        print(f"[read_from_buffer] Fz={fr.force[2]:.3f} N, Tz={fr.torque[2]:.4f} N·m")

        # Blocking: wait for next new sample (max 0.25 s)
        fr2 = fs.read_blocking(timeout_s=0.25)
        if fr2 is not None:
            print(f"[read_blocking] NEW: Fz={fr2.force[2]:.3f} N")
        else:
            print("[read_blocking] Timeout (no new sample)")

    except KeyboardInterrupt:
        pass
    finally:
        fs.stop()
        print("[INFO] Stopped cleanly.")
