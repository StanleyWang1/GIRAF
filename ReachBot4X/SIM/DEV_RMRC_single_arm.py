"""
Resolved Motion Rate Control (RMRC) anchoring demo.

- When you press Enter in the terminal, Arm1 RMRC activates and steers foot1 toward region1.
- When the foot tip enters the region sphere, we "anchor" by teleporting the corresponding mocap anchor
  and enabling the equality constraint.

Controller (Arm1 only):
  v = Kp * (x_region - x_foot)   (all in arm1 base frame)
  qdot = J^+ v                  (damped least-squares)
  q_cmd <- q_cmd + qdot * dt    (Euler)

Notes:
- Uses the *analytical* position Jacobian derived from the validated FK.
- Uses *measured* foot and region positions from the sim (sites), expressed in arm1 base frame.
"""

import sys
import time
import threading
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from attachment_controller import SimulationConfig, RobotAttachmentController


MODEL_REL_PATH = "./ReachBot4X/SIM/RB4X/env_flat_w_dynamic_anchors.xml"

# Controller gains
K_P = 5.0              # task-space proportional gain (1/s)
V_MAX = 0.01            # max task-space speed (m/step)
DAMPING = 1e-2         # damping for pseudoinverse
POS_TOL = 0.01         # stop commanding if within this distance (m)


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
    """Return qpos indices for [revolver{key}1, revolver{key}2, prismatic{key}]."""
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


def get_arm1_actuator_ids(model) -> list[int]:
    """Actuator IDs in the same order as q = [theta1, theta2, d3] for Arm1."""
    names = ["motor11", "motor12", "boomMotor1"]
    act_ids = []
    for n in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if aid < 0:
            raise RuntimeError(f"Missing actuator '{n}'")
        act_ids.append(aid)
    return act_ids


def compute_fk_analytical(theta1_rad, theta2_rad, d3_m):
    """Analytical FK position for your chain, expressed in the arm base frame."""
    h = 0.105487
    y0 = 0.059837
    z0 = -0.0525

    r = y0 + d3_m

    c1, s1 = np.cos(theta1_rad), np.sin(theta1_rad)
    c2, s2 = np.cos(theta2_rad), np.sin(theta2_rad)

    x = -s1 * (c2 * r - s2 * z0)
    y =  c1 * (c2 * r - s2 * z0)
    z =  h + s2 * r + c2 * z0

    return np.array([x, y, z])


def compute_pos_jacobian_analytical(theta1_rad, theta2_rad, d3_m):
    """
    Position Jacobian of the foot (x,y,z) w.r.t. q = [theta1, theta2, d3].

    Returns:
        J (3x3): columns correspond to partials w.r.t. [theta1, theta2, d3]
                in the arm base frame.
    """
    y0 = 0.059837
    z0 = -0.0525

    r = y0 + d3_m

    c1, s1 = np.cos(theta1_rad), np.sin(theta1_rad)
    c2, s2 = np.cos(theta2_rad), np.sin(theta2_rad)

    A = c2 * r - s2 * z0
    dA_dtheta2 = -s2 * r - c2 * z0
    dA_dd3 = c2

    # x = -s1*A
    dx_dtheta1 = -c1 * A
    dx_dtheta2 = -s1 * dA_dtheta2
    dx_dd3 = -s1 * dA_dd3

    # y = c1*A
    dy_dtheta1 = -s1 * A
    dy_dtheta2 = c1 * dA_dtheta2
    dy_dd3 = c1 * dA_dd3

    # z = h + s2*r + c2*z0
    dz_dtheta1 = 0.0
    dz_dtheta2 = c2 * r - s2 * z0
    dz_dd3 = s2

    return np.array(
        [
            [dx_dtheta1, dx_dtheta2, dx_dd3],
            [dy_dtheta1, dy_dtheta2, dy_dd3],
            [dz_dtheta1, dz_dtheta2, dz_dd3],
        ],
        dtype=float,
    )


def world_to_body_frame(data, body_id: int, p_world: np.ndarray) -> np.ndarray:
    """Convert a world-frame point into a body's local frame: p_body = R^T (p_world - p_body_world)."""
    R = data.xmat[body_id].reshape(3, 3)
    p0 = data.xpos[body_id]
    return R.T @ (p_world - p0)


def start_input_thread(state: dict):
    """Background thread: press Enter to toggle RMRC; type 'q' to quit."""

    def loop():
        print("\n" + "=" * 60)
        print("RMRC READY")
        print("=" * 60)
        print("Press Enter to toggle Arm1 RMRC (foot1 -> region1)")
        print("Type 'q' + Enter to quit")
        print("=" * 60 + "\n")

        while True:
            try:
                cmd = input().strip()
            except EOFError:
                break

            if cmd == "":
                state["rmrc_active"] = not state["rmrc_active"]
                print(f"[RMRC] {'ON' if state['rmrc_active'] else 'OFF'}")
            elif cmd.lower() in {"q", "quit", "exit"}:
                state["quit"] = True
                print("[RMRC] Quit requested.")
                break
            else:
                print("Unrecognized. Press Enter to toggle or 'q' to quit.")

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th


def rmrc_step_arm1(
    model,
    data,
    arm1_base_body_id: int,
    foot1_site_id: int,
    region1_site_id: int,
    qpos_indices_arm1: list[int],
    act_ids_arm1: list[int],
    q_cmd: np.ndarray,
    dt: float,
    kp: float = K_P,
    v_max: float = V_MAX,
    damping: float = DAMPING,
):
    """One RMRC step for Arm1."""
    q1_idx, q2_idx, q3_idx = qpos_indices_arm1

    theta1 = float(data.qpos[q1_idx])
    theta2 = float(data.qpos[q2_idx])
    d3 = float(data.qpos[q3_idx])

    # Measured positions in arm1 base frame
    x_foot_world = data.site_xpos[foot1_site_id].copy()
    x_reg_world = data.site_xpos[region1_site_id].copy()

    x_foot = world_to_body_frame(data, arm1_base_body_id, x_foot_world)
    x_reg = world_to_body_frame(data, arm1_base_body_id, x_reg_world)

    e = x_reg - x_foot

    # Task-space velocity
    v = kp * e
    v_norm = float(np.linalg.norm(v))
    if v_norm > v_max and v_norm > 1e-12:
        v *= (v_max / v_norm)

    # Stop commanding if close
    if float(np.linalg.norm(e)) < POS_TOL:
        return q_cmd

    # Jacobian (analytical)
    J = compute_pos_jacobian_analytical(theta1, theta2, d3)

    # Damped least-squares pseudoinverse
    JJt = J @ J.T
    A = JJt + (damping ** 2) * np.eye(3)
    try:
        qdot = J.T @ np.linalg.solve(A, v)
    except np.linalg.LinAlgError:
        qdot, *_ = np.linalg.lstsq(J, v, rcond=None)

    # Integrate
    q_cmd = q_cmd + qdot * dt

    # Send position commands
    for i, aid in enumerate(act_ids_arm1):
        data.ctrl[aid] = q_cmd[i]

    return q_cmd


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

    # Arm1-specific IDs
    arm1_base_body_id = get_arm_base_body_id(model, "1")
    qpos_indices_arm1 = get_arm_joint_qpos_indices(model, "1")["qpos_indices"]
    act_ids_arm1 = get_arm1_actuator_ids(model)

    foot1_site_id = foot_site_ids["1"]
    region1_site_id = region_site_ids["1"]

    # Initialize commanded position to current qpos
    q_cmd = np.array(
        [
            data.qpos[qpos_indices_arm1[0]],
            data.qpos[qpos_indices_arm1[1]],
            data.qpos[qpos_indices_arm1[2]],
        ],
        dtype=float,
    )

    # Input state
    state = {"rmrc_active": False, "quit": False}
    start_input_thread(state)

    print("Opening MuJoCo viewer.")
    print("RMRC controls Arm1 only: foot1 -> region1.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if state.get("quit", False):
                break

            mujoco.mj_step(model, data)

            # RMRC if active and not already attached
            if state["rmrc_active"] and not controller.is_foot_attached("1"):
                dt = float(model.opt.timestep)
                q_cmd = rmrc_step_arm1(
                    model=model,
                    data=data,
                    arm1_base_body_id=arm1_base_body_id,
                    foot1_site_id=foot1_site_id,
                    region1_site_id=region1_site_id,
                    qpos_indices_arm1=qpos_indices_arm1,
                    act_ids_arm1=act_ids_arm1,
                    q_cmd=q_cmd,
                    dt=dt,
                )

            # Auto-attach: foot1 -> region1
            foot_pos = data.site_xpos[foot1_site_id]
            region_pos = data.site_xpos[region1_site_id]
            d = foot_pos - region_pos
            dist_sq = float(d @ d)

            if dist_sq < config.region_radius * config.region_radius:
                if not controller.is_foot_attached("1"):
                    controller.attach_foot_to_region("1", "1")

                    # Teleport anchor mocap and activate equality
                    data.mocap_pos[anchor_mocap_ids["1"]] = region_pos
                    data.eq_active[eq_ids["1"]] = 1

                    # Visual
                    set_region_color(model, region_geom_ids["1"], config.region_active_rgba)

                    # Turn off RMRC automatically
                    state["rmrc_active"] = False
                    print("[SUCCESS] Foot 1 anchored to region 1! RMRC OFF.")

            viewer.sync()


def main():
    config = SimulationConfig(
        region_radius=0.1,
        target_rtf=1.0,
    )

    model, data = load_model(Path(MODEL_REL_PATH))
    eq_ids = get_equalities(model)
    foot_site_ids, region_site_ids, region_geom_ids, anchor_mocap_ids = get_sites_and_geoms_and_mocap(model)

    run_controller(
        model,
        data,
        eq_ids,
        foot_site_ids,
        region_site_ids,
        region_geom_ids,
        anchor_mocap_ids,
        config=config,
    )


if __name__ == "__main__":
    main()


def main():
    config = SimulationConfig(region_radius=0.1, target_rtf=1.0)
    model, data = load_model(Path(MODEL_REL_PATH))
    eq_ids = get_equalities(model)
    foot_site_ids, region_site_ids, region_geom_ids, anchor_mocap_ids = get_sites_and_geoms_and_mocap(model)

    controller = run_controller(model, data, eq_ids, foot_site_ids, region_site_ids, region_geom_ids, anchor_mocap_ids, config)

    region_inactive_rgba = (0.5, 0.5, 0.5, 0.25)

    print("Opening MuJoCo viewer.")
    print(f"Printing FK at ~{FK_PRINT_HZ:.1f} Hz.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            for k in ("1", "2", "3", "4"):
                foot_pos = data.site_xpos[foot_site_ids[k]]
                region_pos = data.site_xpos[region_site_ids[k]]
                if np.dot(foot_pos - region_pos, foot_pos - region_pos) < config.region_radius**2:
                    if not controller.is_foot_attached(k):
                        controller.attach_foot_to_region(k, k)
                        data.mocap_pos[anchor_mocap_ids[k]] = region_pos
                        data.eq_active[eq_ids[k]] = 1
                        set_region_color(model, region_geom_ids[k], config.region_active_rgba)

            viewer.sync()

if __name__ == "__main__":
    main()
