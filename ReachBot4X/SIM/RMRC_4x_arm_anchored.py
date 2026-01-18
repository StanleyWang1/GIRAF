"""
Resolved Motion Rate Control (RMRC) - Sequential Anchoring + Synchronized Locomotion.

Behavior:
- Awaits joystick 'Y' press to begin sequential anchoring sequence
- Drives arms 1->3->2->4 one at a time until all anchored
- Once all anchored, enters locomotion control mode
- RT trigger: +Y velocity (task-space, all arms synchronized)
- LT trigger: -Y velocity (task-space, all arms synchronized)
- Both pressed simultaneously: no motion
- X button: E-STOP (instant quit)

Control law (sequencing phase):
  e = x_region_k - x_foot_k                  (all in arm-k base frame)
  v = Kp * e                                 (task-space velocity)
  qdot = J^+ v                               (damped least-squares)
  q_cmd <- q_cmd + qdot * dt                 (Euler)

Control law (locomotion phase):
  v = K * RT/LT trigger value  (task-space velocity in Y direction)
  qdot = J^+ v                 (damped least-squares per arm)
  q_cmd <- q_cmd + qdot * dt   (Euler integration)

Notes:
- All 4 arms move synchronously during locomotion with the same task-space velocity
- Uses measured foot/region positions from MuJoCo sites
- Uses analytical position Jacobian (RRP arm kinematics)
"""

import sys
import time
import threading
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# Import joystick driver
sys.path.insert(0, str(Path(__file__).parent))
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from DEV.attachment_controller import SimulationConfig, RobotAttachmentController


# ============================================================================
# Configuration
# ============================================================================

MODEL_REL_PATH = "./ReachBot4X/SIM/RB4X/env_flat_w_dynamic_anchors.xml"

# Locomotion Control Gains
K_RMRC = 5.0             # task-space proportional gain during anchoring (1/s)
V_MAX_RMRC = 0.05        # max task-space speed during anchoring (m/s)
K_LOCOMOTION = 0.01     # gain for trigger-to-velocity mapping (m/s per full trigger)
DAMPING = 1e-2          # damping for pseudoinverse
POS_TOL = 0.01          # position error tolerance (m)

# Per-step joint increment clamps
MAX_DTHETA_STEP = None  # rad/step (None = no limit)
MAX_DD3_STEP = None     # m/step (None = no limit)

# Global state for joystick control and E-stop
ESTOP_TRIGGERED = False
SEQUENCE_STARTED = False


# ============================================================================
# Model Loading & Setup
# ============================================================================

def load_model(path: Path):
    """Load MuJoCo model and data, exit if file not found."""
    model_path = path.resolve()
    if not model_path.exists():
        print(f"ERROR: Model file not found:\n{model_path}")
        sys.exit(1)

    print(f"Loading MuJoCo model:\n{model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    return model, data


def get_equalities(model):
    """Resolve equality constraint IDs for foot anchors."""
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
    """Get IDs for foot sites, regions, and anchor mocap bodies."""
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
    """Update region visualization color."""
    model.geom_rgba[geom_id] = rgba


# ============================================================================
# Per-Arm Model Lookups
# ============================================================================

def get_arm_base_body_id(model, arm_key: str) -> int:
    """Get the base body ID for an arm."""
    body_name = f"arm{arm_key}"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Could not find body '{body_name}'")
    return int(body_id)


def get_arm_joint_qpos_indices(model, arm_key: str) -> list[int]:
    """Get qpos indices for [theta1, theta2, d3] of a given arm."""
    revolver1_name = f"revolver{arm_key}1"
    revolver2_name = f"revolver{arm_key}2"
    prismatic_name = f"prismatic{arm_key}"

    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, revolver1_name)
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, revolver2_name)
    j3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prismatic_name)

    if j1 < 0 or j2 < 0 or j3 < 0:
        raise RuntimeError(f"Missing joints for arm {arm_key}")

    return [
        int(model.jnt_qposadr[j1]),
        int(model.jnt_qposadr[j2]),
        int(model.jnt_qposadr[j3]),
    ]


def get_arm_actuator_ids(model, arm_key: str) -> list[int]:
    """Get actuator IDs in order [motor{k}1, motor{k}2, boomMotor{k}]."""
    names = [f"motor{arm_key}1", f"motor{arm_key}2", f"boomMotor{arm_key}"]
    act_ids = []
    for n in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if aid < 0:
            raise RuntimeError(f"Missing actuator '{n}'")
        act_ids.append(int(aid))
    return act_ids


# ============================================================================
# Kinematics & Control
# ============================================================================

def compute_pos_jacobian_analytical(theta1_rad, theta2_rad, d3_m):
    """
    Compute position Jacobian J = d(x,y,z)/d(theta1,theta2,d3) in arm base frame.
    (RRP arm configuration)
    """
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
    """Transform world position to body frame."""
    R = data.xmat[body_id].reshape(3, 3)
    p0 = data.xpos[body_id]
    return R.T @ (p_world - p0)


def world_velocity_to_arm_frame(data, arm_base_body_id: int, v_world: np.ndarray) -> np.ndarray:
    """
    Transform a velocity command from world frame to arm base frame.
    
    Args:
        data: MuJoCo data object
        arm_base_body_id: ID of the arm's base body
        v_world: velocity in world frame [vx, vy, vz]
    
    Returns:
        velocity in arm base frame [vx_arm, vy_arm, vz_arm]
    """
    # Get arm base body's rotation matrix (world-to-arm orientation)
    R = data.xmat[arm_base_body_id].reshape(3, 3)
    
    # Transform: v_arm = R.T @ v_world
    v_arm = R.T @ v_world
    
    return v_arm


# ============================================================================
# Joystick Monitor (100 Hz)
# ============================================================================

class JoystickState:
    """Thread-safe container for joystick state."""
    def __init__(self):
        self.rt_value = 0.0
        self.lt_value = 0.0
        self.x_pressed = False
        self.lock = threading.Lock()

    def update(self, rt, lt, x):
        with self.lock:
            self.rt_value = rt
            self.lt_value = lt
            self.x_pressed = x

    def get(self):
        with self.lock:
            return self.rt_value, self.lt_value, self.x_pressed


joystick_state = JoystickState()


def joystick_monitor_thread():
    """
    Background thread: Monitor joystick at 100 Hz.
    - Y button: starts anchoring sequence
    - RT trigger: +Y locomotion (during locomotion phase)
    - LT trigger: -Y locomotion (during locomotion phase)
    - X button: E-STOP
    """
    global ESTOP_TRIGGERED, SEQUENCE_STARTED

    try:
        js = joystick_connect()
        print("[Joystick Monitor] Connected. Waiting for Y to start anchoring...")
    except RuntimeError as e:
        print(f"[Joystick Monitor] ERROR: {e}")
        return

    dt = 0.01  # 100 Hz

    while not ESTOP_TRIGGERED:
        try:
            data = joystick_read(js)

            # Y button starts sequence
            if data["YB"] and not SEQUENCE_STARTED:
                SEQUENCE_STARTED = True
                print("[Joystick] Y pressed → ANCHORING SEQUENCE STARTED")

            # X button triggers E-STOP
            if data["XB"]:
                ESTOP_TRIGGERED = True
                print("[Joystick] X pressed → E-STOP TRIGGERED!")
                break

            joystick_state.update(float(data["RT"]), float(data["LT"]), False)

        except Exception as e:
            print(f"[Joystick Monitor] Error reading input: {e}")
            break

        time.sleep(dt)

    try:
        joystick_disconnect(js)
    except:
        pass


# ============================================================================
# Attachment & Control Logic
# ============================================================================

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
    """Auto-attach (anchor) if within region sphere. Returns True if attached this step."""
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

            # Visual feedback
            set_region_color(model, region_geom_id, config.region_active_rgba)

            return True

    return False


def rmrc_step_arm_anchoring(
    model,
    data,
    arm_base_body_id: int,
    foot_site_id: int,
    region_site_id: int,
    qpos_indices: list[int],
    act_ids: list[int],
    q_cmd: np.ndarray,
    dt: float,
    kp: float = K_RMRC,
    v_max: float = V_MAX_RMRC,
    damping: float = DAMPING,
):
    """Execute one RMRC step for a single arm during anchoring phase (RRP kinematics)."""
    q1_idx, q2_idx, q3_idx = qpos_indices

    theta1 = float(data.qpos[q1_idx])
    theta2 = float(data.qpos[q2_idx])
    d3 = float(data.qpos[q3_idx])

    # Measured positions in arm's base frame
    x_foot_world = data.site_xpos[foot_site_id].copy()
    x_reg_world = data.site_xpos[region_site_id].copy()

    x_foot = world_to_body_frame(data, arm_base_body_id, x_foot_world)
    x_reg = world_to_body_frame(data, arm_base_body_id, x_reg_world)

    e = x_reg - x_foot

    # Task-space velocity with saturation
    v = kp * e
    v_norm = float(np.linalg.norm(v))
    if v_norm > v_max and v_norm > 1e-12:
        v *= (v_max / v_norm)

    # Stop if converged
    if float(np.linalg.norm(e)) < POS_TOL:
        return q_cmd

    # Compute Jacobian
    J = compute_pos_jacobian_analytical(theta1, theta2, d3)

    # Damped least-squares pseudoinverse
    JJt = J @ J.T
    A = JJt + (damping ** 2) * np.eye(3)
    try:
        qdot = J.T @ np.linalg.solve(A, v)
    except np.linalg.LinAlgError:
        qdot, *_ = np.linalg.lstsq(J, v, rcond=None)

    # Euler integration
    dq = qdot * dt

    # Optional per-step clamps
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

def rmrc_locomotion_step_arm(
    model,
    data,
    arm_base_body_id: int,
    foot_site_id: int,
    qpos_indices: list[int],
    act_ids: list[int],
    q_cmd: np.ndarray,
    dt: float,
    v_target_arm: np.ndarray,
    damping: float = DAMPING,
):
    """
    Execute one RMRC step for a single arm with task-space velocity command in arm frame.
    
    Args:
        v_target_arm: desired task-space velocity in arm base frame [vx, vy, vz] (m/s)
    """
    q1_idx, q2_idx, q3_idx = qpos_indices

    theta1 = float(data.qpos[q1_idx])
    theta2 = float(data.qpos[q2_idx])
    d3 = float(data.qpos[q3_idx])

    # Compute Jacobian in arm frame
    J = compute_pos_jacobian_analytical(theta1, theta2, d3)

    # Task-space velocity in arm frame
    v = v_target_arm

    # Damped least-squares pseudoinverse
    JJt = J @ J.T
    A = JJt + (damping ** 2) * np.eye(3)
    try:
        qdot = J.T @ np.linalg.solve(A, v)
    except np.linalg.LinAlgError:
        qdot, *_ = np.linalg.lstsq(J, v, rcond=None)

    # Euler integration
    dq = qdot * dt

    # Optional per-step clamps
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


# ============================================================================
# Main Controller
# ============================================================================

def run_controller(
    model,
    data,
    eq_ids,
    foot_site_ids,
    region_site_ids,
    region_geom_ids,
    anchor_mocap_ids,
    config=None,
):
    """Main control loop: sequential anchoring followed by synchronized locomotion."""
    global ESTOP_TRIGGERED, SEQUENCE_STARTED

    if config is None:
        config = SimulationConfig()

    controller = RobotAttachmentController(config)
    controller.reset()

    # Initialize: all equalities OFF, regions inactive
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
    anchoring_phase = True
    locomotion_active = False

    print("\n" + "=" * 60)
    print("RMRC 4-Arm Anchoring + Synchronized Locomotion")
    print("=" * 60)
    print("Anchoring Sequence: 1->1, 3->3, 2->2, 4->4")
    print("Press Y on joystick to start anchoring")
    print("After anchoring:")
    print("  RT → +Z (up in world frame)")
    print("  LT → -Z (down in world frame)")
    print("  Both → no motion")
    print("  X → E-STOP")
    print("=" * 60 + "\n")

    # Start joystick monitor thread
    monitor_thread = threading.Thread(target=joystick_monitor_thread, daemon=True)
    monitor_thread.start()

    print("Opening MuJoCo viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # E-STOP check
            if ESTOP_TRIGGERED:
                print("[MAIN] E-STOP triggered → exiting")
                break

            mujoco.mj_step(model, data)

            # ========== ANCHORING PHASE ==========
            if anchoring_phase:
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

                # Sequential anchoring phase
                if SEQUENCE_STARTED and seq_idx < len(sequence):
                    k = sequence[seq_idx]

                    # Skip if already attached
                    if controller.is_foot_attached(k):
                        seq_idx += 1
                        if seq_idx < len(sequence):
                            nxt = sequence[seq_idx]
                            print(f"[RMRC] Advancing to arm{nxt} -> region{nxt}")
                        else:
                            print("[RMRC] Anchoring sequence complete!")
                            print("[RMRC] Entering locomotion control mode...")
                            anchoring_phase = False
                            locomotion_active = True
                    else:
                        dt = float(model.opt.timestep)
                        q_cmds[k] = rmrc_step_arm_anchoring(
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

            # ========== LOCOMOTION PHASE ==========
            elif locomotion_active:
                if all(controller.is_foot_attached(k) for k in ("1", "2", "3", "4")):
                    # Get joystick state
                    rt, lt, _ = joystick_state.get()

                    # Compute desired Z velocity in WORLD frame
                    # RT → +Z (up), LT → -Z (down)
                    # Both pressed → no motion
                    if rt > 1e-6 and lt > 1e-6:
                        vz_world = 0.0
                    # RT only
                    elif rt > 1e-6:
                        vz_world = K_LOCOMOTION * rt
                    # LT only
                    elif lt > 1e-6:
                        vz_world = -K_LOCOMOTION * lt
                    # Neither pressed
                    else:
                        vz_world = 0.0

                    # Apply locomotion to all arms synchronously
                    if abs(vz_world) > 1e-9:
                        dt = float(model.opt.timestep)
                        # World-frame velocity command (Z direction)
                        v_world = np.array([0.0, 0.0, vz_world], dtype=float)
                        
                        for k in ("1", "2", "3", "4"):
                            # Transform world velocity to arm base frame
                            v_arm = world_velocity_to_arm_frame(
                                data,
                                arm_base_body_ids[k],
                                v_world
                            )
                            
                            q_cmds[k] = rmrc_locomotion_step_arm(
                                model=model,
                                data=data,
                                arm_base_body_id=arm_base_body_ids[k],
                                foot_site_id=foot_site_ids[k],
                                qpos_indices=arm_qpos_indices[k],
                                act_ids=arm_act_ids[k],
                                q_cmd=q_cmds[k],
                                dt=dt,
                                v_target_arm=v_arm,
                            )

            viewer.sync()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Load model and run controller."""
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
