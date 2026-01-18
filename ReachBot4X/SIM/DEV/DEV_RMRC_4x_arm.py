"""
Resolved Motion Rate Control (RMRC) anchoring demo (SEQUENTIAL + POST-ANCHOR MOVE).

Behavior:
- Press Enter in the terminal to start/pause the anchoring sequence.
- The controller drives ONE arm at a time until it anchors, then moves to the next.

Sequence (arm -> region):
  1 -> 1
  3 -> 3
  2 -> 2
  4 -> 4

After ALL 4 are anchored:
- Smoothly (slowly) drive ALL arms simultaneously to a target configuration:
    motor{k}1 -> 0
    motor{k}2 -> pi/2
    prismatic{k} -> 0.5
  using per-step joint increment clamps.

Control law (for current arm k):
  e = x_region_k - x_foot_k                  (all in arm-k base frame)
  v = Kp * e                                 (task-space velocity)
  qdot = J^+ v                               (damped least-squares)
  q_cmd <- q_cmd + qdot * dt                 (Euler)

Notes:
- Uses *measured* foot/region positions from MuJoCo sites.
- Uses *analytical* position Jacobian derived from your validated FK.
- Assumes per-arm position actuators are named: motor{k}1, motor{k}2, boomMotor{k}.
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

# -------------------- RMRC gains --------------------
K_P = 5.0                # task-space proportional gain (1/s)
V_MAX = 0.01             # max task-space speed (m/s)
DAMPING = 1e-2           # damping for pseudoinverse
POS_TOL = 0.01           # stop commanding if within this distance (m)

# Optional: per-step joint increment clamps DURING RMRC (set None to disable)
MAX_DTHETA_STEP = None   # e.g. 0.02  (rad/step)
MAX_DD3_STEP = None      # e.g. 0.002 (m/step)

# -------------------- Post-anchor targets --------------------
TARGET_THETA1 = 0.0
TARGET_THETA2 = 0.5 * np.pi
TARGET_D3 = 0.0

# Smoothness: max joint increment per step during post-anchor motion
POST_MAX_DTHETA_STEP = 0.0001   # rad/step
POST_MAX_DD3_STEP = 0.00001     # m/step

# Tolerance for “done” in post-anchor mode
POST_DONE_TOL = 1e-4


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
    """Get IDs for foot tip sites, region sites & geoms, per-foot anchor mocap bodies."""
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
        anchor_mocap_ids[key] = int(mocap_id)

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
    model.geom_rgba[geom_id] = rgba


def get_arm_base_body_id(model, arm_key: str) -> int:
    body_name = f"arm{arm_key}"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Could not find body '{body_name}'")
    return int(body_id)


def get_arm_joint_qpos_indices(model, arm_key: str) -> list[int]:
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

    return [
        int(model.jnt_qposadr[j1]),
        int(model.jnt_qposadr[j2]),
        int(model.jnt_qposadr[j3]),
    ]


def get_arm_actuator_ids(model, arm_key: str) -> list[int]:
    """Actuator IDs in order [motor{k}1, motor{k}2, boomMotor{k}]."""
    names = [f"motor{arm_key}1", f"motor{arm_key}2", f"boomMotor{arm_key}"]
    act_ids = []
    for n in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if aid < 0:
            raise RuntimeError(f"Missing actuator '{n}'")
        act_ids.append(int(aid))
    return act_ids


def compute_pos_jacobian_analytical(theta1_rad, theta2_rad, d3_m):
    """J = d(x,y,z)/d(theta1,theta2,d3) in the arm base frame."""
    y0 = 0.059837
    z0 = -0.0525

    r = y0 + d3_m

    c1, s1 = np.cos(theta1_rad), np.sin(theta1_rad)
    c2, s2 = np.cos(theta2_rad), np.sin(theta2_rad)

    A = c2 * r - s2 * z0
    dA_dtheta2 = -s2 * r - c2 * z0
    dA_dd3 = c2

    dx_dtheta1 = -c1 * A
    dx_dtheta2 = -s1 * dA_dtheta2
    dx_dd3 = -s1 * dA_dd3

    dy_dtheta1 = -s1 * A
    dy_dtheta2 = c1 * dA_dtheta2
    dy_dd3 = c1 * dA_dd3

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
    R = data.xmat[body_id].reshape(3, 3)
    p0 = data.xpos[body_id]
    return R.T @ (p_world - p0)


def start_input_thread(state: dict):
    """Background thread: Enter toggles the anchoring sequence; 'q' quits."""

    def loop():
        print("\n" + "=" * 60)
        print("SEQUENTIAL RMRC READY")
        print("=" * 60)
        print("Press Enter to START/PAUSE the anchoring sequence")
        print("Type 'q' + Enter to quit")
        print("=" * 60 + "\n")

        while True:
            try:
                cmd = input().strip()
            except EOFError:
                break

            if cmd == "":
                state["sequence_active"] = not state["sequence_active"]
                print(f"[RMRC] {'RUNNING' if state['sequence_active'] else 'PAUSED'}")
            elif cmd.lower() in {"q", "quit", "exit"}:
                state["quit"] = True
                print("[RMRC] Quit requested.")
                break
            else:
                print("Unrecognized. Press Enter to toggle or 'q' to quit.")

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th


def rmrc_step_arm(
    model,
    data,
    arm_base_body_id: int,
    foot_site_id: int,
    region_site_id: int,
    qpos_indices: list[int],
    act_ids: list[int],
    q_cmd: np.ndarray,
    dt: float,
    kp: float = K_P,
    v_max: float = V_MAX,
    damping: float = DAMPING,
):
    """One RMRC step for a single arm (RRP)."""
    q1_idx, q2_idx, q3_idx = qpos_indices

    theta1 = float(data.qpos[q1_idx])
    theta2 = float(data.qpos[q2_idx])
    d3 = float(data.qpos[q3_idx])

    # Measured positions in *this arm's base frame*
    x_foot_world = data.site_xpos[foot_site_id].copy()
    x_reg_world = data.site_xpos[region_site_id].copy()

    x_foot = world_to_body_frame(data, arm_base_body_id, x_foot_world)
    x_reg = world_to_body_frame(data, arm_base_body_id, x_reg_world)

    e = x_reg - x_foot

    # Task-space velocity
    v = kp * e
    v_norm = float(np.linalg.norm(v))
    if v_norm > v_max and v_norm > 1e-12:
        v *= (v_max / v_norm)

    # Stop commanding if close
    if float(np.linalg.norm(e)) < POS_TOL:
        return q_cmd

    # Jacobian
    J = compute_pos_jacobian_analytical(theta1, theta2, d3)

    # Damped least-squares pseudoinverse
    JJt = J @ J.T
    A = JJt + (damping ** 2) * np.eye(3)
    try:
        qdot = J.T @ np.linalg.solve(A, v)
    except np.linalg.LinAlgError:
        qdot, *_ = np.linalg.lstsq(J, v, rcond=None)

    # Euler integrate into commanded pos
    dq = qdot * dt

    # Optional per-step clamp DURING RMRC
    if MAX_DTHETA_STEP is not None:
        dq[0] = float(np.clip(dq[0], -MAX_DTHETA_STEP, MAX_DTHETA_STEP))
        dq[1] = float(np.clip(dq[1], -MAX_DTHETA_STEP, MAX_DTHETA_STEP))
    if MAX_DD3_STEP is not None:
        dq[2] = float(np.clip(dq[2], -MAX_DD3_STEP, MAX_DD3_STEP))

    q_cmd = q_cmd + dq

    # Send position commands
    for i, aid in enumerate(act_ids):
        data.ctrl[aid] = q_cmd[i]

    return q_cmd


def try_attach(
    controller: RobotAttachmentController,
    model,
    data,
    foot_key: str,
    eq_ids,
    foot_site_id: int,
    region_site_id: int,
    region_geom_id: int,
    anchor_mocap_id: int,
    config: SimulationConfig,
) -> bool:
    """Auto-attach (anchor) if within region sphere. Returns True if attached on this call."""
    foot_pos = data.site_xpos[foot_site_id]
    region_pos = data.site_xpos[region_site_id]
    d = foot_pos - region_pos
    dist_sq = float(d @ d)

    if dist_sq < config.region_radius * config.region_radius:
        if not controller.is_foot_attached(foot_key):
            controller.attach_foot_to_region(foot_key, foot_key)

            # Teleport anchor mocap and activate equality
            data.mocap_pos[anchor_mocap_id] = region_pos
            data.eq_active[eq_ids[foot_key]] = 1

            # Visual
            set_region_color(model, region_geom_id, config.region_active_rgba)

            return True

    return False


def step_qcmd_toward_target(
    q_cmd: np.ndarray,
    q_tgt: np.ndarray,
    max_dtheta_step: float,
    max_dd3_step: float,
) -> np.ndarray:
    """Move q_cmd toward q_tgt with per-step increment limits. q=[theta1,theta2,d3]."""
    dq = q_tgt - q_cmd

    dq[0] = float(np.clip(dq[0], -max_dtheta_step, max_dtheta_step))
    dq[1] = float(np.clip(dq[1], -max_dtheta_step, max_dtheta_step))
    dq[2] = float(np.clip(dq[2], -max_dd3_step, max_dd3_step))

    return q_cmd + dq


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

    # Pre-cache per-arm info
    arm_base_body_ids = {k: get_arm_base_body_id(model, k) for k in ("1", "2", "3", "4")}
    arm_qpos_indices = {k: get_arm_joint_qpos_indices(model, k) for k in ("1", "2", "3", "4")}
    arm_act_ids = {k: get_arm_actuator_ids(model, k) for k in ("1", "2", "3", "4")}

    # Per-arm command states
    q_cmds = {}
    for k in ("1", "2", "3", "4"):
        q1, q2, q3 = arm_qpos_indices[k]
        q_cmds[k] = np.array([data.qpos[q1], data.qpos[q2], data.qpos[q3]], dtype=float)

    # Sequence definition
    sequence = ["1", "3", "2", "4"]
    seq_idx = 0

    # Input state
    state = {"sequence_active": False, "post_anchor_active": False, "quit": False}
    start_input_thread(state)

    print("Opening MuJoCo viewer.")
    print("Sequence: 1->1, 3->3, 2->2, 4->4")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if state.get("quit", False):
                break

            mujoco.mj_step(model, data)

            # Always allow attachment checks for all feet
            for k in ("1", "2", "3", "4"):
                attached_now = try_attach(
                    controller=controller,
                    model=model,
                    data=data,
                    foot_key=k,
                    eq_ids=eq_ids,
                    foot_site_id=foot_site_ids[k],
                    region_site_id=region_site_ids[k],
                    region_geom_id=region_geom_ids[k],
                    anchor_mocap_id=anchor_mocap_ids[k],
                    config=config,
                )
                if attached_now:
                    print(f"[SUCCESS] Foot {k} anchored to region {k}.")

            # If sequence is running, drive the current arm until it anchors
            if state["sequence_active"] and seq_idx < len(sequence):
                k = sequence[seq_idx]

                # Skip if already attached (e.g., if it attached early)
                if controller.is_foot_attached(k):
                    seq_idx += 1
                    if seq_idx < len(sequence):
                        nxt = sequence[seq_idx]
                        print(f"[RMRC] Advancing to arm{nxt} -> region{nxt}")
                    else:
                        print("[RMRC] Sequence complete. Entering post-anchor move-to-target mode.")
                        state["sequence_active"] = False
                        state["post_anchor_active"] = True
                else:
                    dt = float(model.opt.timestep)
                    q_cmds[k] = rmrc_step_arm(
                        model=model,
                        data=data,
                        arm_base_body_id=arm_base_body_ids[k],
                        foot_site_id=foot_site_ids[k],
                        region_site_id=region_site_ids[k],
                        qpos_indices=arm_qpos_indices[k],
                        act_ids=arm_act_ids[k],
                        q_cmd=q_cmds[k],
                        dt=dt,
                    )

            # Post-anchor: move ALL arms simultaneously to target joint pose
            if state.get("post_anchor_active", False):
                if all(controller.is_foot_attached(k) for k in ("1", "2", "3", "4")):
                    q_tgt = np.array([TARGET_THETA1, TARGET_THETA2, TARGET_D3], dtype=float)

                    all_done = True
                    for k in ("1", "2", "3", "4"):
                        q_cmds[k] = step_qcmd_toward_target(
                            q_cmd=q_cmds[k],
                            q_tgt=q_tgt,
                            max_dtheta_step=POST_MAX_DTHETA_STEP,
                            max_dd3_step=POST_MAX_DD3_STEP,
                        )

                        # Send commands
                        for i, aid in enumerate(arm_act_ids[k]):
                            data.ctrl[aid] = q_cmds[k][i]

                        if float(np.linalg.norm(q_tgt - q_cmds[k])) > POST_DONE_TOL:
                            all_done = False

                    if all_done:
                        print("[POST] All arms reached target. Holding.")
                        state["post_anchor_active"] = False

            viewer.sync()


def main():
    config = SimulationConfig(region_radius=0.1, target_rtf=1.0)

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
