"""
Simple anchoring controller (no RMRC).

- Auto-attaches each foot to its corresponding region when the foot tip enters the region sphere.
- Prints analytical FK and actual FK (from MuJoCo site) for all 4 arms in each arm's base frame.
"""

import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from attachment_controller import SimulationConfig, RobotAttachmentController


MODEL_REL_PATH = "./ReachBot4X/SIM/RB4X/env_flat_w_dynamic_anchors.xml"

# Print rate (Hz)
FK_PRINT_HZ = 10.0


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


def get_equalities(model):
    """Resolve equality constraint IDs for each foot."""
    eq_ids = {
        "1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot1_anchor"),
        "2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot2_anchor"),
        "3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot3_anchor"),
        "4": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot4_anchor"),
    }
    for key, eq_id in eq_ids.items():
        if eq_id < 0:
            raise RuntimeError(f"Could not find equality constraint 'foot{key}_anchor'")
    return eq_ids


def get_sites_and_geoms_and_mocap(model):
    """
    Get IDs for:
      - foot tip sites
      - region sites & geoms
      - per-foot anchor mocap bodies (mocap indices)
    """
    foot_site_ids = {
        "1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "boomEndSite1"),
        "2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "boomEndSite2"),
        "3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "boomEndSite3"),
        "4": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "boomEndSite4"),
    }

    region_site_ids = {
        "1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "region_site1"),
        "2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "region_site2"),
        "3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "region_site3"),
        "4": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "region_site4"),
    }
    region_geom_ids = {
        "1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "region1_geom"),
        "2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "region2_geom"),
        "3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "region3_geom"),
        "4": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "region4_geom"),
    }

    anchor_mocap_ids = {}
    for key in ("1", "2", "3", "4"):
        body_name = f"foot{key}_anchor_body"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"Missing body '{body_name}'")
        mocap_id = model.body_mocapid[body_id]
        if mocap_id < 0:
            raise RuntimeError(f"Body '{body_name}' is not marked as mocap='true'")
        anchor_mocap_ids[key] = mocap_id

    # Sanity checks
    for label, sid in foot_site_ids.items():
        if sid < 0:
            raise RuntimeError(f"Missing foot site boomEndSite{label}")
    for label, sid in region_site_ids.items():
        if sid < 0:
            raise RuntimeError(f"Missing region site region_site{label}")
    for label, gid in region_geom_ids.items():
        if gid < 0:
            raise RuntimeError(f"Missing region geom region{label}_geom")

    return foot_site_ids, region_site_ids, region_geom_ids, anchor_mocap_ids


def set_region_color(model, geom_id, rgba):
    """Update the RGBA color of a region geom."""
    model.geom_rgba[geom_id] = rgba


def get_arm_base_body_id(model, arm_key: str) -> int:
    """Get the body ID for an arm's base (e.g., 'arm1', 'arm2', etc.)."""
    body_name = f"arm{arm_key}"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Could not find body '{body_name}'")
    return body_id


def get_arm_joint_qpos_indices(model, arm_key: str) -> dict:
    """
    Return qpos indices (NOT dof indices!) for the 3 joints of an arm:
      revolver{key}1, revolver{key}2, prismatic{key}
    """
    revolver1_name = f"revolver{arm_key}1"
    revolver2_name = f"revolver{arm_key}2"
    prismatic_name = f"prismatic{arm_key}"

    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, revolver1_name)
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, revolver2_name)
    j3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prismatic_name)

    if j1 < 0:
        raise RuntimeError(f"Missing joint '{revolver1_name}'")
    if j2 < 0:
        raise RuntimeError(f"Missing joint '{revolver2_name}'")
    if j3 < 0:
        raise RuntimeError(f"Missing joint '{prismatic_name}'")

    q1 = model.jnt_qposadr[j1]
    q2 = model.jnt_qposadr[j2]
    q3 = model.jnt_qposadr[j3]
    return {"qpos_indices": [q1, q2, q3]}


def compute_fk_analytical(theta1_rad, theta2_rad, d3_m):
    """
    Analytical FK position for your chain, expressed in the arm base frame.

    Chain (as discussed):
      Rz(theta1) -> Tz(h) -> Rx(theta2) -> Tz(z0) -> Ty(y0 + d3)

    Notes:
    - In your XML you had a fixed boom offset like: pos="0 0.059837 -0.0525"
      so we include y0=0.059837 and z0=-0.0525 here.
    - MuJoCo hinge qpos is already radians; prismatic is meters.
    """
    h = 0.105487
    y0 = 0.059837
    z0 = -0.0525

    r = y0 + d3_m  # total +y translation after the Rx(theta2) frame

    c1 = np.cos(theta1_rad)
    s1 = np.sin(theta1_rad)
    c2 = np.cos(theta2_rad)
    s2 = np.sin(theta2_rad)

    # Derived from: p = [0,0,h] + Rz(theta1)*Rx(theta2)*[0, r, z0]
    x = -s1 * (c2 * r - s2 * z0)
    y =  c1 * (c2 * r - s2 * z0)
    z =  h + s2 * r + c2 * z0

    return np.array([x, y, z])


def world_to_body_frame(data, body_id: int, p_world: np.ndarray) -> np.ndarray:
    """
    Convert a world-frame point into a body's local frame:
      p_body = R^T (p_world - p_body_world)
    """
    R = data.xmat[body_id].reshape(3, 3)
    p0 = data.xpos[body_id]
    return R.T @ (p_world - p0)

def _ansi_clear_line():
    return "\x1b[2K"          # clear entire line

def _ansi_cursor_up(n):
    return f"\x1b[{n}A"       # move cursor up n lines

def init_fk_display():
    # Print 4 placeholder lines once
    for k in ("1", "2", "3", "4"):
        print(f"[arm{k}] FK_analytical: x= ----  y= ----  z= ---- | FK_actual: x= ----  y= ----  z= ----")
    sys.stdout.flush()

def update_fk_display(lines):
    """
    lines: list[str] length 4. Each string is a full line to render.
    """
    # Move cursor up 4 lines, overwrite them
    sys.stdout.write(_ansi_cursor_up(4))
    for line in lines:
        sys.stdout.write(_ansi_clear_line() + "\r" + line + "\n")
    sys.stdout.flush()

def get_fk_lines_for_all_arms(model, data, foot_site_ids):
    lines = []
    for k in ("1", "2", "3", "4"):
        arm_base_body_id = get_arm_base_body_id(model, k)
        qpos_info = get_arm_joint_qpos_indices(model, k)
        q1_idx, q2_idx, q3_idx = qpos_info["qpos_indices"]

        theta1 = data.qpos[q1_idx]
        theta2 = data.qpos[q2_idx]
        d3     = data.qpos[q3_idx]

        fk_analytical = compute_fk_analytical(theta1, theta2, d3)

        site_id = foot_site_ids[k]
        foot_site_world = data.site_xpos[site_id].copy()
        fk_actual = world_to_body_frame(data, arm_base_body_id, foot_site_world)

        lines.append(
            f"[arm{k}] "
            f"FK_analytical: x={fk_analytical[0]: .5f} y={fk_analytical[1]: .5f} z={fk_analytical[2]: .5f} | "
            f"FK_actual: x={fk_actual[0]: .5f} y={fk_actual[1]: .5f} z={fk_actual[2]: .5f}"
        )
    return lines

def run_controller(model, data, eq_ids, foot_site_ids, region_site_ids, region_geom_ids, anchor_mocap_ids, config=None):
    if config is None:
        config = SimulationConfig()

    controller = RobotAttachmentController(config)
    controller.reset()

    # Start with all equalities OFF, and regions in inactive color
    for eq_id in eq_ids.values():
        data.eq_active[eq_id] = 0
    for gid in region_geom_ids.values():
        set_region_color(model, gid, config.region_inactive_rgba)

    print("Opening MuJoCo viewer.")
    print("Auto-attach is active for all 4 feet.")
    print(f"Printing FK at ~{FK_PRINT_HZ:.1f} Hz.")

    sim_start = time.time()
    last_fk_print = 0.0
    fk_period = 1.0 / max(FK_PRINT_HZ, 1e-6)
    
    init_fk_display()   # <-- prints the 4 lines once

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            # Throttled FK printing
            now = time.time()
            if (now - last_fk_print) >= fk_period:
                last_fk_print = now
                lines = get_fk_lines_for_all_arms(model, data, foot_site_ids)
                update_fk_display(lines)

            # Auto-attach logic for each foot -> corresponding region
            for k in ("1", "2", "3", "4"):
                foot_site_id = foot_site_ids[k]
                region_site_id = region_site_ids[k]

                foot_pos = data.site_xpos[foot_site_id]
                region_pos = data.site_xpos[region_site_id]

                d = foot_pos - region_pos
                dist_sq = float(d @ d)

                if dist_sq < config.region_radius * config.region_radius:
                    if not controller.is_foot_attached(k):
                        controller.attach_foot_to_region(k, k)

                        # Teleport anchor mocap and activate equality
                        mocap_id = anchor_mocap_ids[k]
                        data.mocap_pos[mocap_id] = region_pos

                        eq_id = eq_ids[k]
                        data.eq_active[eq_id] = 1

                        # Visual: region solid/active
                        set_region_color(model, region_geom_ids[k], config.region_active_rgba)

                        print(f"[SUCCESS] Foot {k} anchored to region {k}!")

            viewer.sync()

            # Real-time factor control (optional)
            sim_time = data.time
            wall_time = time.time() - sim_start
            if wall_time > 0.0:
                rtf = sim_time / wall_time
                if rtf > config.target_rtf:
                    desired_wall = sim_time / config.target_rtf
                    sleep_time = desired_wall - wall_time
                    if sleep_time > 0.0:
                        time.sleep(min(sleep_time, 0.01))


def main():
    config = SimulationConfig(
        region_radius=0.1,
        target_rtf=1.0,
    )

    model, data = load_model(Path(MODEL_REL_PATH))
    eq_ids = get_equalities(model)
    foot_site_ids, region_site_ids, region_geom_ids, anchor_mocap_ids = get_sites_and_geoms_and_mocap(model)

    run_controller(model, data, eq_ids, foot_site_ids, region_site_ids, region_geom_ids, anchor_mocap_ids, config=config)


if __name__ == "__main__":
    main()
