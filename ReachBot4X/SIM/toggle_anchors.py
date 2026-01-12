import sys
import time
import threading
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_REL_PATH = "./ReachBot4X/SIM/RB4X/env_flat_w_anchors.xml"
TARGET_RTF = 1.0  # Real-time factor: 1.0 = real-time, 2.0 = 2x speed, 0.5 = half-speed


def load_model(path: Path):
    """Load a MuJoCo model and data, exiting with a clear error if the file is missing."""
    model_path = path.resolve()
    if not model_path.exists():
        print(f"ERROR: Model file not found:\n{model_path}")
        sys.exit(1)

    print(f"Loading MuJoCo model:\n{model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    return model, data


def get_anchor_equalities(model):
    """Resolve equality constraint IDs for the 4 foot anchors by name."""
    eq_ids = {
        "1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot1_anchor"),
        "2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot2_anchor"),
        "3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot3_anchor"),
        "4": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot4_anchor"),
    }

    # Optional sanity check
    for key, eq_id in eq_ids.items():
        if eq_id < 0:
            raise RuntimeError(f"Could not find equality constraint for foot{key}_anchor")

    return eq_ids


def start_input_thread(data, eq_ids):
    """
    Start a background thread that listens for user input and toggles anchor constraints.

    Type "1", "2", "3", or "4" + Enter in the terminal to toggle the corresponding anchor.
    """
    def input_loop():
        print("Input thread ready. Type 1/2/3/4 + Enter to toggle anchors.")
        while True:
            try:
                cmd = input().strip()
            except EOFError:
                # e.g. terminal closed; just exit the thread
                break

            if cmd in eq_ids:
                eq_id = eq_ids[cmd]
                # Toggle active flag: 1 -> 0, 0 -> 1
                data.eq_active[eq_id] = 1 - data.eq_active[eq_id]
                state = "ATTACHED" if data.eq_active[eq_id] else "DETACHED"
                print(f"[anchor {cmd}] → {state}")
            elif cmd.lower() in {"q", "quit", "exit"}:
                print("Input thread exiting on user request.")
                break
            else:
                print('Unrecognized command. Use 1, 2, 3, 4, or "q" to quit input thread.')

    thread = threading.Thread(target=input_loop, daemon=True)
    thread.start()
    return thread


def run_simulation(model, data, eq_ids, target_rtf: float = 1.0):
    """Main simulation loop with real-time factor control and terminal anchor toggling."""
    # Initialize all anchors ON
    for eq in eq_ids.values():
        data.eq_active[eq] = 1

    # Start input thread (non-blocking)
    start_input_thread(data, eq_ids)

    print("Opening MuJoCo viewer.")
    print("Anchors 1–4 start ATTACHED. Type 1/2/3/4 + Enter in the terminal to toggle.")

    sim_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Advance physics by one MuJoCo timestep
            mujoco.mj_step(model, data)

            # Render the current state
            viewer.sync()

            # --- Real-time factor control ---
            sim_time = data.time
            wall_time = time.time() - sim_start
            if wall_time <= 0.0:
                continue

            rtf = sim_time / wall_time

            # If simulation is running faster than desired, sleep a bit to let wall time catch up
            if rtf > target_rtf:
                desired_wall = sim_time / target_rtf
                sleep_time = desired_wall - wall_time
                if sleep_time > 0.0:
                    # Clamp sleep to avoid very long pauses if things get out of sync
                    time.sleep(min(sleep_time, 0.01))


def main():
    model, data = load_model(Path(MODEL_REL_PATH))
    eq_ids = get_anchor_equalities(model)
    run_simulation(model, data, eq_ids, target_rtf=TARGET_RTF)


if __name__ == "__main__":
    main()
