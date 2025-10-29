#!/usr/bin/env python3
import os, sys, json, time, signal, threading
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import bota_driver

# =======================
# USER SETTINGS
# =======================
DURATION_SEC = 30.0       # total runtime
READ_HZ      = 100.0      # sensor read rate
PLOT_HZ      = 20.0       # plot refresh rate
DO_TARE      = True       # tare on startup

# ==== CONFIG PATH (same pattern as your working script) ====
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH  = os.path.join(project_root, "bota_driver_config", "ethercat_gen0.json")

# =======================
# INTERNALS
# =======================
stop_flag = False
def _signal_handler(signum, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, _signal_handler)

WINDOW_SEC   = max(DURATION_SEC, 30.0)              # show at least 30s window
MAX_SAMPLES  = int(WINDOW_SEC * READ_HZ) + 10

# Buffers (use wall-clock time to ensure strictly increasing x-axis)
t_wall_buf = deque(maxlen=MAX_SAMPLES)
fx_buf = deque(maxlen=MAX_SAMPLES)
fy_buf = deque(maxlen=MAX_SAMPLES)
fz_buf = deque(maxlen=MAX_SAMPLES)
tx_buf = deque(maxlen=MAX_SAMPLES)
ty_buf = deque(maxlen=MAX_SAMPLES)
tz_buf = deque(maxlen=MAX_SAMPLES)

def fail(msg: str):
    print(f"[FATAL] {msg}")
    sys.exit(1)

def preflight():
    # Check config file exists and is Gen0
    if not os.path.isfile(CONFIG_PATH):
        fail(f"Config not found at {CONFIG_PATH}\nTip: ls -l {os.path.dirname(CONFIG_PATH)}")
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        ci = cfg["driver_config"]["communication_interface_name"]
        if ci.lower() != "canopen_over_ethercat_gen0":
            fail(f"communication_interface_name='{ci}' (expected 'CANopen_over_EtherCAT_gen0')")
    except Exception as e:
        fail(f"Config JSON issue: {e}")
    print(f"[INFO] Using config: {CONFIG_PATH}")

def _autoscale(ax, data_arrays, pad_ratio=0.15, min_span=1e-2):
    vals_list = [np.asarray(a) for a in data_arrays if len(a) > 0]
    if not vals_list: return
    vals = np.hstack(vals_list)
    span = max(min_span, np.max(np.abs(vals)))
    ymax = span * (1.0 + pad_ratio)
    ax.set_ylim(-ymax, ymax)

def main():
    global stop_flag

    preflight()

    print("[INFO] Starting BotaDriver…")
    drv = bota_driver.BotaDriver(CONFIG_PATH)
    if not drv.configure(): fail("Failed to configure driver")
    if DO_TARE and not drv.tare(): fail("Failed to tare sensor")
    if not drv.activate(): fail("Failed to activate driver")

    # ----- Reader thread -----
    read_period = 1.0 / READ_HZ
    t0 = time.perf_counter()

    def reader():
        nonlocal t0
        next_t = time.perf_counter()
        first = True
        while not stop_flag and (time.perf_counter() - t0) < DURATION_SEC:
            bf = drv.read_frame()
            now_rel = time.perf_counter() - t0

            # Append data
            t_wall_buf.append(now_rel)
            fx_buf.append(bf.force[0]); fy_buf.append(bf.force[1]); fz_buf.append(bf.force[2])
            tx_buf.append(bf.torque[0]); ty_buf.append(bf.torque[1]); tz_buf.append(bf.torque[2])

            if first:
                print(f"[DEBUG] first sample | t_wall={now_rel:.3f}s "
                      f"F={[round(x,3) for x in bf.force]} T={[round(x,3) for x in bf.torque]}")
                first = False

            next_t += read_period
            st = next_t - time.perf_counter()
            if st > 0: time.sleep(st)

    thr = threading.Thread(target=reader, daemon=True)
    thr.start()

    # ----- Live plot -----
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25)
    ax_f = fig.add_subplot(gs[0, 0]); ax_t = fig.add_subplot(gs[1, 0])

    ax_f.set_title("Forces vs Time"); ax_f.set_xlabel("time (s)"); ax_f.set_ylabel("F (N)")
    ln_fx, = ax_f.plot([], [], label="Fx", linewidth=1.5)
    ln_fy, = ax_f.plot([], [], label="Fy", linewidth=1.5)
    ln_fz, = ax_f.plot([], [], label="Fz", linewidth=1.5)
    ax_f.legend(loc="upper right"); ax_f.grid(True, alpha=0.3)

    ax_t.set_title("Torques vs Time"); ax_t.set_xlabel("time (s)"); ax_t.set_ylabel("T (N·m)")
    ln_tx, = ax_t.plot([], [], label="Tx", linewidth=1.5)
    ln_ty, = ax_t.plot([], [], label="Ty", linewidth=1.5)
    ln_tz, = ax_t.plot([], [], label="Tz", linewidth=1.5)
    ax_t.legend(loc="upper right"); ax_t.grid(True, alpha=0.3)

    def _trim_window(t_arr, *ys):
        if len(t_arr) == 0: return np.array([]), [np.array([]) for _ in ys]
        t = np.asarray(t_arr)
        t0_plot = t[-1] - WINDOW_SEC
        idx = np.searchsorted(t, t0_plot, side="left")
        ys_out = [np.asarray(y)[idx:] for y in ys]
        return t[idx:], ys_out

    def update(_frame):
        t, [fx, fy, fz] = _trim_window(t_wall_buf, fx_buf, fy_buf, fz_buf)
        _, [tx, ty, tz] = _trim_window(t_wall_buf, tx_buf, ty_buf, tz_buf)

        ln_fx.set_data(t, fx); ln_fy.set_data(t, fy); ln_fz.set_data(t, fz)
        ln_tx.set_data(t, tx); ln_ty.set_data(t, ty); ln_tz.set_data(t, tz)

        if len(t) > 1:
            ax_f.set_xlim(t[0], t[-1]); ax_t.set_xlim(t[0], t[-1])
        _autoscale(ax_f, [fx, fy, fz]); _autoscale(ax_t, [tx, ty, tz])

        # stop after duration
        if stop_flag or (time.perf_counter() - t0) >= DURATION_SEC:
            raise StopIteration
        return ln_fx, ln_fy, ln_fz, ln_tx, ln_ty, ln_tz

    ani = animation.FuncAnimation(fig, update, interval=int(1000.0 / PLOT_HZ), blit=False)

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        pass

    # ----- Shutdown -----
    stop_flag = True
    thr.join(timeout=2.0)
    try:
        if not drv.deactivate(): print("[WARN] Failed to deactivate driver")
        if not drv.shutdown():   print("[WARN] Failed to shutdown driver")
    except Exception as e:
        print(f"[WARN] Driver close error: {e}")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
