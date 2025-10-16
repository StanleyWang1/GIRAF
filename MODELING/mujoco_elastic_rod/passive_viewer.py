# Simple passive viewer + slider panel to drive actuators interactively.
# - Opens the same model as drive_rrr.py in a passive Mujoco viewer
# - Provides a small Tk slider window to set data.ctrl values (one slider per actuator)
# Usage: python passive_viewer.py

import os
import time
import threading

import mujoco
import mujoco.viewer as mjv
import numpy as np
import tkinter as tk
from tkinter import ttk

XML_PATH = "./MODELING/mujoco_elastic_rod/rrr_arm.xml"

# Slider visual range (user can type values beyond this if needed)
SLIDER_MIN = -200.0
SLIDER_MAX = 200.0
SLIDER_RES = 0.1

def make_gui(data, n_ctrl):
    """Create a small TK window with one slider per control to edit data.ctrl in real time."""
    root = tk.Tk()
    root.title("Passive viewer — actuator controls")

    frm = ttk.Frame(root, padding=8)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    sliders = []
    vars_ = []

    def on_close():
        # stop the tkinter mainloop without killing the entire process
        try:
            root.quit()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", on_close)

    for i in range(n_ctrl):
        lbl = ttk.Label(frm, text=f"ctrl[{i}]")
        lbl.grid(row=i, column=0, sticky="w", padx=(0, 6))

        v = tk.DoubleVar(value=float(data.ctrl[i]) if i < len(data.ctrl) else 0.0)
        s = ttk.Scale(frm, orient="horizontal", from_=SLIDER_MIN, to=SLIDER_MAX,
                      command=(lambda val, idx=i, var=v: _on_slide(val, idx, var, data)),
                      variable=v)
        s.grid(row=i, column=1, sticky="ew")
        frm.columnconfigure(1, weight=1)

        # small entry to show/set precise value
        ent = ttk.Entry(frm, textvariable=v, width=8)
        ent.grid(row=i, column=2, padx=(6,0))

        sliders.append(s)
        vars_.append(v)

    # Add a quit button
    btn = ttk.Button(frm, text="Close sliders", command=on_close)
    btn.grid(row=n_ctrl+1, column=0, columnspan=3, pady=(8,0))

    # start TK mainloop (blocking) in this thread
    root.mainloop()

def _on_slide(val, idx, var, data):
    """Callback when a slider moves — update data.ctrl safely."""
    try:
        f = float(val)
    except Exception:
        try:
            f = float(var.get())
        except Exception:
            return
    # clamp / write to data.ctrl if index exists
    if 0 <= idx < data.ctrl.size:
        data.ctrl[idx] = f

def main():
    if not os.path.exists(XML_PATH):
        print(f"[error] XML not found: {XML_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("[info] Model loaded. DOFs:", model.nv, "Actuators:", model.nu)

    # Zero controls initially
    data.ctrl[:] = 0.0

    # Launch a small GUI in a background thread
    n_ctrl = int(model.nu) if hasattr(model, "nu") else int(model.nu if hasattr(model, "nu") else len(data.ctrl))
    gui_thread = threading.Thread(target=make_gui, args=(data, n_ctrl), daemon=True)
    gui_thread.start()

    # Open passive viewer and step the simulation while GUI is open.
    try:
        with mjv.launch_passive(model, data) as viewer:
            # optional: disable contact points for clarity
            try:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            except Exception:
                pass

            print("[info] Viewer open. Use sliders window to change data.ctrl in real time.")
            # run until viewer window closes or an exception occurs
            while True:
                try:
                    # advance simulation one step
                    mujoco.mj_step(model, data)
                except Exception as e:
                    print("[error] mj_step failed:", e)
                    break

                # sync viewer to updated model/data
                try:
                    viewer.sync()
                except Exception:
                    # viewer closed or sync failed -> exit loop
                    break

                # If both viewer and GUI have been closed, exit
                if not gui_thread.is_alive():
                    # allow a few more frames to show final state, then exit
                    time.sleep(0.05)
                    break

                # small sleep to reduce CPU usage (viewer.sync does internal timing, this is extra)
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("[info] Interrupted by user.")
    except Exception as e:
        print("[error] Viewer failed:", e)

    print("[info] Exiting passive_viewer.")

if __name__ == "__main__":
    main()