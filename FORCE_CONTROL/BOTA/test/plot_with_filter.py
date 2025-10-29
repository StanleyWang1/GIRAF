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
DURATION_SEC = 120.0       # total runtime
READ_HZ      = 100.0      # sensor read rate
PLOT_HZ      = 20.0       # plot refresh rate
DO_TARE      = True       # tare on startup

# ---- Simple high-frequency noise filter (single-pole low-pass) ----
FILTER_ENABLE = False      # set False to disable
F_CUT_HZ      = 3.0      # cutoff frequency (Hz) ~8–20 typical for contact forces
SHOW_RAW      = False     # if True, overlay raw traces (faint)

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
fx_buf = deque(maxlen=MAX_SAMPLES); fy_buf = deque(maxlen=MAX_SAMPLES); fz_buf = deque(maxlen=MAX_SAMPLES)
tx_buf = deque(maxlen=MAX_SAMPLES); ty_buf = deque(maxlen=MAX_SAMPLES); tz_buf = deque(maxlen=MAX_SAMPLES)

# Optional raw buffers for overlay
fx_raw = deque(maxlen=MAX_SAMPLES); fy_raw = deque(maxlen=MAX_SAMPLES); fz_raw = deque(maxlen=MAX_SAMPLES)
tx_raw = deque(maxlen=MAX_SAMPLES); ty_raw = deque(maxlen=MAX_SAMPLES); tz_raw = deque(maxlen=MAX_SAMPLES)

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

    # EMA (single-pole LPF) state and coeff
    # alpha = dt / (tau + dt), tau = 1/(2*pi*f_c)
    if FILTER_ENABLE and F_CUT_HZ > 0.0:
        dt = 1.0 / READ_HZ
        tau = 1.0 / (2.0 * np.pi * F_CUT_HZ)
        alpha = dt / (tau + dt)
    else:
        alpha = None

    ema_f = [None, None, None]  # Fx,Fy,Fz
    ema_t = [None, None, None]  # Tx,Ty,Tz

    def lp_step(prev, x):
        # Exponential moving average step
        if alpha is None:
            return x
        return x if prev is None else (prev + alpha * (x - prev))

    def reader():
        nonlocal t0
        next_t = time.perf_counter()
        first = True
        while not stop_flag and (time.perf_counter() - t0) < DURATION_SEC:
            bf = drv.read_frame()
            now_rel = time.perf_counter() - t0

            # raw values
            Fx, Fy, Fz = float(bf.force[0]), float(bf.force[1]), float(bf.force[2])
            Tx, Ty, Tz = float(bf.torque[0]), float(bf.torque[1]), float(bf.torque[2])

            # filter step (if enabled)
            Fx_f = lp_step(ema_f[0], Fx); ema_f[0] = Fx_f
            Fy_f = lp_step(ema_f[1], Fy); ema_f[1] = Fy_f
            Fz_f = lp_step(ema_f[2], Fz); ema_f[2] = Fz_f
            Tx_f = lp_step(ema_t[0], Tx); ema_t[0] = Tx_f
            Ty_f = lp_step(ema_t[1], Ty); ema_t[1] = Ty_f
            Tz_f = lp_step(ema_t[2], Tz); ema_t[2] = Tz_f

            # Append data (filtered)
            t_wall_buf.append(now_rel)
            fx_buf.append(Fx_f); fy_buf.append(Fy_f); fz_buf.append(Fz_f)
            tx_buf.append(Tx_f); ty_buf.append(Ty_f); tz_buf.append(Tz_f)

            # Optionally store raw for overlay
            if SHOW_RAW:
                fx_raw.append(Fx); fy_raw.append(Fy); fz_raw.append(Fz)
                tx_raw.append(Tx); ty_raw.append(Ty); tz_raw.append(Tz)

            if first:
                print(f"[DEBUG] first sample | t_wall={now_rel:.3f}s "
                      f"RAW F={[round(x,3) for x in (Fx,Fy,Fz)]} RAW T={[round(x,3) for x in (Tx,Ty,Tz)]}")
                if FILTER_ENABLE and alpha is not None:
                    fcut = F_CUT_HZ
                    print(f"[INFO] LPF enabled: single-pole, f_c={fcut:.2f} Hz, alpha={alpha:.4f}")
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
    ln_fx, = ax_f.plot([], [], label="Fx (filt)", linewidth=1.5)
    ln_fy, = ax_f.plot([], [], label="Fy (filt)", linewidth=1.5)
    ln_fz, = ax_f.plot([], [], label="Fz (filt)", linewidth=1.5)

    # Optional raw overlays
    if SHOW_RAW:
        ln_fx_raw, = ax_f.plot([], [], label="Fx raw", linewidth=0.8, alpha=0.35)
        ln_fy_raw, = ax_f.plot([], [], label="Fy raw", linewidth=0.8, alpha=0.35)
        ln_fz_raw, = ax_f.plot([], [], label="Fz raw", linewidth=0.8, alpha=0.35)

    ax_f.legend(loc="upper right"); ax_f.grid(True, alpha=0.3)

    ax_t.set_title("Torques vs Time"); ax_t.set_xlabel("time (s)"); ax_t.set_ylabel("T (N·m)")
    ln_tx, = ax_t.plot([], [], label="Tx (filt)", linewidth=1.5)
    ln_ty, = ax_t.plot([], [], label="Ty (filt)", linewidth=1.5)
    ln_tz, = ax_t.plot([], [], label="Tz (filt)", linewidth=1.5)

    if SHOW_RAW:
        ln_tx_raw, = ax_t.plot([], [], label="Tx raw", linewidth=0.8, alpha=0.35)
        ln_ty_raw, = ax_t.plot([], [], label="Ty raw", linewidth=0.8, alpha=0.35)
        ln_tz_raw, = ax_t.plot([], [], label="Tz raw", linewidth=0.8, alpha=0.35)

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

        if SHOW_RAW:
            _, [fxr, fyr, fzr] = _trim_window(t_wall_buf, fx_raw, fy_raw, fz_raw)
            _, [txr, tyr, tzr] = _trim_window(t_wall_buf, tx_raw, ty_raw, tz_raw)
            ln_fx_raw.set_data(t, fxr); ln_fy_raw.set_data(t, fyr); ln_fz_raw.set_data(t, fzr)
            ln_tx_raw.set_data(t, txr); ln_ty_raw.set_data(t, tyr); ln_tz_raw.set_data(t, tzr)

        if len(t) > 1:
            ax_f.set_xlim(t[0], t[-1]); ax_t.set_xlim(t[0], t[-1])
        _autoscale(ax_f, [fx, fy, fz]); _autoscale(ax_t, [tx, ty, tz])

        if stop_flag or (time.perf_counter() - t0) >= DURATION_SEC:
            raise StopIteration
        # Return only the artists we created (raw ones only if enabled)
        artists = [ln_fx, ln_fy, ln_fz, ln_tx, ln_ty, ln_tz]
        if SHOW_RAW:
            artists += [ln_fx_raw, ln_fy_raw, ln_fz_raw, ln_tx_raw, ln_ty_raw, ln_tz_raw]
        return artists

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
