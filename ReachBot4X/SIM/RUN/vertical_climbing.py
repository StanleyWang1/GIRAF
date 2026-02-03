"""
Resolved Motion Rate Control (RMRC) - Vertical Climbing Simulation.

Behavior:
- Awaits joystick 'Y' press to begin sequential anchoring sequence
- Drives arms 1->3->2->4 one at a time until all anchored (first 4 anchors)
- Once all anchored, enters locomotion control mode
- LX/LY: X/Y velocity (task-space, all arms synchronized)
- RT trigger: +Z velocity (up)
- LT trigger: -Z velocity (down)
- Both pressed simultaneously: no Z motion
- X button: E-STOP (instant quit)

Control law (sequencing phase):
  e = x_region_k - x_foot_k                  (all in arm-k base frame)
  v = Kp * e                                 (task-space velocity)
  qdot = J^+ v                               (damped least-squares)
  q_cmd <- q_cmd + qdot * dt                 (Euler)

Control law (locomotion phase):
  v = K * joystick value  (task-space velocity in world frame)
  qdot = J^+ v            (damped least-squares per arm)
  q_cmd <- q_cmd + qdot * dt   (Euler integration)

Notes:
- All 4 arms move synchronously during locomotion via QP-based control
- Uses measured foot/region positions from MuJoCo sites
- Uses analytical position Jacobian (RRP arm kinematics)
- Vertical climbing environment with 12 anchors at 3 height levels
"""

import sys
import time
import threading
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# Import joystick driver and attachment controller from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from DEV.attachment_controller import SimulationConfig, RobotAttachmentController


# ============================================================================
# Configuration
# ============================================================================

MODEL_REL_PATH = "./ReachBot4X/SIM/RB4X/vertical_climbing.xml"

# Locomotion Control Gains
K_RMRC = 5.0             # task-space proportional gain during anchoring (1/s)
V_MAX_RMRC = 0.05        # max task-space speed during anchoring (m/s)
K_LOCOMOTION = 0.05     # gain for trigger-to-velocity mapping (m/s per full trigger)
K_ORIENTATION = -10.0     # proportional gain for orientation regulation (1/s)
MAX_ANGULAR_VEL = 0.1   # max angular velocity command (rad/s)
DAMPING = 5e-2          # base damping for pseudoinverse (increased for stability)
ADAPTIVE_DAMPING = True  # use adaptive damping based on manipulability
MANIPULABILITY_THRESHOLD = 1e-3  # threshold for singularity detection
POS_TOL = 0.01          # position error tolerance (m)

# Per-step joint increment clamps
MAX_DTHETA_STEP = 0.05  # rad/step (limit joint velocities)
MAX_DD3_STEP = 0.01     # m/step (limit prismatic joint velocity)
MAX_JOINT_VELOCITY = 0.5  # rad/s or m/s (additional velocity limit)

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

    region_site_ids = {}
    for i in range(1, 13):  # regions 1-12
        region_site_ids[str(i)] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"region_site{i}")

    region_geom_ids = {}
    for i in range(1, 13):  # regions 1-12
        region_geom_ids[str(i)] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"region{i}_geom")

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



def detach_foot(
    controller: RobotAttachmentController,
    model,
    data,
    foot_key: str,
    eq_ids,
    region_geom_id: int,
    config: SimulationConfig,
):
    """Detach a foot from its current anchor."""
    if controller.is_foot_attached(foot_key):
        controller.detach_foot(foot_key)
        # Deactivate equality constraint
        data.eq_active[eq_ids[foot_key]] = 0
        # Reset region color to inactive
        set_region_color(model, region_geom_id, config.region_inactive_rgba)
        return True
    return False

def get_mainbody_pose(model, data):
    """
    Get the current SE(3) pose of the mainBody as a 4x4 homogeneous transformation matrix.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data (contains current state)
    
    Returns:
        np.ndarray: 4x4 SE(3) transformation matrix [R | p; 0 0 0 1]
                    where R is the 3x3 rotation matrix and p is the 3D position vector
    """
    # Get mainBody body ID - try both naming conventions
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mainBody")
    if body_id < 0:
        # Try alternative: look for the geom named mainBody and get its body
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mainBody")
        if geom_id >= 0:
            body_id = model.geom_bodyid[geom_id]
        else:
            raise RuntimeError("Could not find body or geom 'mainBody'")
    
    # Get position (3D vector)
    pos = data.xpos[body_id].copy()
    
    # Get rotation matrix (3x3) from quaternion
    # data.xquat[body_id] is [w, x, y, z] quaternion
    # Need to reshape data.xmat[body_id] which is flattened row-major 3x3
    rot_mat = data.xmat[body_id].reshape(3, 3).copy()
    
    # Construct 4x4 SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos
    
    return T


def get_arm_base_in_body_frame(model, data, arm_key: str):
    """
    Get the SE(3) transform from mainBody frame to arm base frame (T_body^base).
    This should be constant throughout the simulation.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data (contains current state)
        arm_key: "1", "2", "3", or "4"
    
    Returns:
        np.ndarray: 4x4 SE(3) transformation matrix T_BS (Body to Shoulder/base)
    """
    # Get mainBody body ID
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mainBody")
    if main_body_id < 0:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mainBody")
        if geom_id >= 0:
            main_body_id = model.geom_bodyid[geom_id]
        else:
            raise RuntimeError("Could not find body or geom 'mainBody'")
    
    # Get arm base body ID
    arm_base_body_id = get_arm_base_body_id(model, arm_key)
    
    # Get world poses
    # MainBody pose in world
    R_body = data.xmat[main_body_id].reshape(3, 3)
    p_body = data.xpos[main_body_id]
    
    # Arm base pose in world
    R_arm = data.xmat[arm_base_body_id].reshape(3, 3)
    p_arm = data.xpos[arm_base_body_id]
    
    # Compute relative transform: T_body^arm = T_world^body^-1 * T_world^arm
    # R_body^arm = R_body^T * R_arm
    # p_body^arm = R_body^T * (p_arm - p_body)
    R_rel = R_body.T @ R_arm
    p_rel = R_body.T @ (p_arm - p_body)
    
    # Construct 4x4 SE(3) matrix
    T_BS = np.eye(4)
    T_BS[:3, :3] = R_rel
    T_BS[:3, 3] = p_rel
    
    return T_BS


def get_all_arm_base_transforms(model, data):
    """
    Get all four arm base transforms in mainBody frame.
    
    Returns:
        dict: {"1": T_BS1, "2": T_BS2, "3": T_BS3, "4": T_BS4}
    """
    return {
        "1": get_arm_base_in_body_frame(model, data, "1"),
        "2": get_arm_base_in_body_frame(model, data, "2"),
        "3": get_arm_base_in_body_frame(model, data, "3"),
        "4": get_arm_base_in_body_frame(model, data, "4"),
    }


def print_mainbody_pose(model, data):
    """
    Print the current SE(3) pose of mainBody as a formatted 4x4 matrix.
    Uses ANSI escape codes to overwrite previous output in place.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data (contains current state)
    """
    T = get_mainbody_pose(model, data)
    
    # Move cursor to beginning of line and clear screen from cursor down
    if not hasattr(print_mainbody_pose, '_first_call'):
        print_mainbody_pose._first_call = False
        # First call: just print normally
        pass
    else:
        # Subsequent calls: move cursor up and clear
        print("\033[H\033[J", end="")  # Move to home and clear screen
    
    print("=== mainBody Pose (SE(3)) ===")
    # Format matrix with 3 decimal places, no scientific notation
    with np.printoptions(precision=3, suppress=True, formatter={'float': lambda x: f'{x:8.3f}'}):
        print(T)
    print(f"Position (x, y, z): [{T[0,3]:7.3f}, {T[1,3]:7.3f}, {T[2,3]:7.3f}]")
    print("=============================")
    print()  # Extra line for spacing


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

def rotation_matrix_log_map(R: np.ndarray) -> np.ndarray:
    """
    Compute the log map of a rotation matrix: so(3) ← SO(3).
    
    Returns the 3D axis-angle vector ω such that R = exp([ω]×).
    
    Args:
        R: 3×3 rotation matrix
    
    Returns:
        ω: 3D axis-angle vector (rotation axis scaled by rotation angle)
    """
    # Compute rotation angle
    trace_R = np.trace(R)
    theta = np.arccos(np.clip((trace_R - 1.0) / 2.0, -1.0, 1.0))
    
    # Handle small angles (Taylor expansion)
    if theta < 1e-6:
        return np.zeros(3)
    
    # Handle theta ≈ π (180° rotation)
    if abs(theta - np.pi) < 1e-6:
        # Find largest diagonal element to determine axis
        diag = np.diag(R)
        k = np.argmax(diag)
        axis = np.zeros(3)
        axis[k] = np.sqrt((R[k, k] + 1.0) / 2.0)
        for i in range(3):
            if i != k:
                axis[i] = R[k, i] / (2.0 * axis[k])
        return theta * axis
    
    # General case: extract axis from skew-symmetric part
    omega_hat = (R - R.T) / (2.0 * np.sin(theta))
    omega = np.array([
        omega_hat[2, 1],  # -ω_x from (3,2) element
        omega_hat[0, 2],  # -ω_y from (1,3) element
        omega_hat[1, 0]   # -ω_z from (2,1) element
    ])
    
    return theta * omega


def compute_orientation_error(R_current: np.ndarray, R_desired: np.ndarray) -> np.ndarray:
    """
    Compute orientation error as a 3D vector in world frame.
    
    The error represents the axis-angle needed to rotate from R_current to R_desired.
    
    Args:
        R_current: current 3×3 rotation matrix
        R_desired: desired 3×3 rotation matrix
    
    Returns:
        e_omega: 3D orientation error vector (axis-angle representation)
    """
    R_error = R_desired @ R_current.T
    return rotation_matrix_log_map(R_error)


def compute_manipulability(J: np.ndarray) -> float:
    """
    Compute manipulability measure (Yoshikawa) for a Jacobian.
    
    Manipulability μ = sqrt(det(J @ J.T))
    
    Near singularities, manipulability approaches zero.
    
    Args:
        J: Jacobian matrix (m×n)
    
    Returns:
        Manipulability measure (scalar >= 0)
    """
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    if det < 0:
        det = 0.0
    return np.sqrt(det)


def compute_local_jacobian(theta1_rad, theta2_rad, d3_m):
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


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector.
    
    For v = [x, y, z], returns:
    [[ 0, -z,  y],
     [ z,  0, -x],
     [-y,  x,  0]]
    
    This satisfies: Skew(v) @ w = v × w (cross product)
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)


def compute_body_and_joint_jacobians(
    model, 
    data, 
    arm_key: str, 
    foot_site_id: int,
    qpos_indices: list[int]
):
    """
    Compute body and joint Jacobians for foot velocity decomposition.
    
    The foot velocity in world frame decomposes as:
    v_F = Jb @ [v_B; ω_B] + Jq @ q̇
    
    where:
    - Jb (3×6): Body Jacobian - maps body twist to foot velocity
    - Jq (3×3): Joint Jacobian - maps joint velocities to foot velocity
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        arm_key: "1", "2", "3", or "4"
        foot_site_id: MuJoCo site ID for the foot
        qpos_indices: [theta1_idx, theta2_idx, d3_idx] in data.qpos
    
    Returns:
        Jb: (3×6) body Jacobian [I_{3×3}, -Skew(p_W^F - p_W^B)]
        Jq: (3×3) joint Jacobian R_W^B @ R_B^A @ J_local
    """
    # ========== Get current joint angles ==========
    q1_idx, q2_idx, q3_idx = qpos_indices
    theta1 = float(data.qpos[q1_idx])
    theta2 = float(data.qpos[q2_idx])
    d3 = float(data.qpos[q3_idx])
    
    # ========== Get T_WB (world to body transform) ==========
    # T_WB is the mainBody pose in world frame
    T_WB = get_mainbody_pose(model, data)
    R_WB = T_WB[:3, :3]  # Rotation: world to body
    p_WB = T_WB[:3, 3]   # Position: body origin in world frame
    
    # ========== Get T_BA (body to arm base transform) ==========
    # T_BA is the arm base pose in body frame (constant)
    T_BA = get_arm_base_in_body_frame(model, data, arm_key)
    R_BA = T_BA[:3, :3]  # Rotation: body to arm base
    # p_BA = T_BA[:3, 3]  # Not needed for Jacobians
    
    # ========== Get foot position in world frame ==========
    p_WF = data.site_xpos[foot_site_id].copy()  # Foot position in world frame
    
    # ========== Compute local Jacobian (in arm frame) ==========
    # J_local = ∂p_A^F/∂q where p_A^F is foot position in arm frame
    J_local = compute_local_jacobian(theta1, theta2, d3)  # 3×3 matrix
    
    # ========== Compute Body Jacobian Jb ==========
    # Jb maps body twist [v_B; ω_B] to foot velocity v_F
    # v_F_body = I @ v_B + (-Skew(p_W^F - p_W^B)) @ ω_B
    # Jb = [I_{3×3}, -Skew(p_W^F - p_W^B)]
    
    r_BF = p_WF - p_WB  # Vector from body to foot in world frame
    
    Jb = np.zeros((3, 6), dtype=float)
    Jb[:3, :3] = np.eye(3)           # Translational part: I_{3×3}
    Jb[:3, 3:6] = -skew_symmetric(r_BF)  # Rotational part: -Skew(r_BF)
    
    # ========== Compute Joint Jacobian Jq ==========
    # Jq maps joint velocities q̇ to foot velocity v_F
    # The local Jacobian J_local gives velocity in arm frame
    # We need to rotate it through arm base to body, then body to world:
    # Jq = R_W^B @ R_B^A @ J_local
    
    Jq = R_WB @ R_BA @ J_local  # 3×3 matrix
    
    return Jb, Jq


def compute_body_jacobian(
    model, 
    data, 
    foot_site_ids: dict, 
    arm_qpos_indices: dict, 
    damping: float = DAMPING,
    adaptive_damping: bool = ADAPTIVE_DAMPING
):
    """
    Compute the body Jacobian J_body that maps joint velocities to body twist.
    
    The relationship is:
        [v_B; ω_B] = J_body @ q̇
    
    where:
        - [v_B; ω_B] is the 6D body twist (linear + angular velocity)
        - q̇ is the 12×1 vector of all joint velocities [q̇1; q̇2; q̇3; q̇4]
        - Each q̇i = [θ̇1, θ̇2, ḋ3] for arm i
    
    The computation follows:
        J_body = [Jb1; Jb2; Jb3; Jb4]^† @ diag(Jq1, Jq2, Jq3, Jq4)
    
    where:
        - Jbi (3×6): body Jacobian for arm i
        - Jqi (3×3): joint Jacobian for arm i
        - Stacked Jb matrix is 12×6
        - Block diagonal Jq matrix is 12×12
        - Damped pseudoinverse produces 6×12 result
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        foot_site_ids: dict {"1": id1, "2": id2, "3": id3, "4": id4}
        arm_qpos_indices: dict {"1": [q1, q2, q3], "2": [...], ...}
        damping: damping factor for pseudoinverse (default: DAMPING)
        adaptive_damping: use adaptive damping based on manipulability (default: ADAPTIVE_DAMPING)
    
    Returns:
        J_body: (6×12) body Jacobian matrix
    """
    # ========== Build stacked body Jacobian and block diagonal joint Jacobian ==========
    Jb_stack = []  # Will stack to 12×6
    Jq_block = np.zeros((12, 12), dtype=float)  # Block diagonal matrix
    Jq_list = []  # Store individual Jq matrices for manipulability check
    
    for i, k in enumerate(["1", "2", "3", "4"]):
        # Compute Jb (3×6) and Jq (3×3) for arm k
        Jb, Jq = compute_body_and_joint_jacobians(
            model=model,
            data=data,
            arm_key=k,
            foot_site_id=foot_site_ids[k],
            qpos_indices=arm_qpos_indices[k]
        )
        
        # Stack Jb vertically: [Jb1; Jb2; Jb3; Jb4]
        Jb_stack.append(Jb)
        Jq_list.append(Jq)
        
        # Place Jq on diagonal block
        # Arm i occupies rows/cols [3i:3i+3, 3i:3i+3]
        row_start = i * 3
        col_start = i * 3
        Jq_block[row_start:row_start+3, col_start:col_start+3] = Jq
    
    # ========== Form stacked body Jacobian ==========
    # Jb_all: 12×6 (4 arms × 3 DOF per foot, 6 DOF body twist)
    Jb_all = np.vstack(Jb_stack)
    
    # ========== Adaptive damping based on manipulability ==========
    effective_damping = damping
    if adaptive_damping:
        # Check manipulability of stacked body Jacobian (12×6)
        # For non-square matrix, use smallest singular value as measure
        try:
            singular_values = np.linalg.svd(Jb_all, compute_uv=False)
            min_sv = np.min(singular_values)
            
            # Also check individual arm manipulabilities
            min_arm_manipulability = float('inf')
            for Jq in Jq_list:
                mu_arm = compute_manipulability(Jq)
                min_arm_manipulability = min(min_arm_manipulability, mu_arm)
            
            # Use the worse of the two conditions
            if min_sv < MANIPULABILITY_THRESHOLD or min_arm_manipulability < MANIPULABILITY_THRESHOLD:
                # Increase damping near singularities
                critical_value = min(min_sv, min_arm_manipulability)
                scale = MANIPULABILITY_THRESHOLD / max(critical_value, 1e-6)
                effective_damping = damping * min(scale, 10.0)  # cap at 10x base damping
        except:
            # Fallback to base damping if SVD fails
            pass
    
    # ========== Compute damped pseudoinverse of Jb_all ==========
    # For overdetermined system (12×6), use right pseudoinverse:
    # Jb^† = (Jb^T @ Jb + λ²I)^-1 @ Jb^T
    # This gives a 6×12 matrix
    JbT = Jb_all.T  # 6×12
    A = JbT @ Jb_all + (effective_damping ** 2) * np.eye(6)  # 6×6
    Jb_pinv = np.linalg.solve(A, JbT)  # 6×12
    
    # ========== Compute body Jacobian ==========
    # J_body = Jb^† @ Jq_block
    # (6×12) @ (12×12) = (6×12)
    J_body = Jb_pinv @ Jq_block
    
    return J_body


def compute_body_jacobian_subset(
    model,
    data,
    foot_site_ids_subset: dict,
    arm_qpos_indices_subset: dict,
    damping: float = DAMPING,
    adaptive_damping: bool = ADAPTIVE_DAMPING
):
    """
    Compute body Jacobian for a subset of arms (e.g., 3 anchored arms).
    Same logic as compute_body_jacobian but works with any number of arms.
    
    Returns:
        J_body: (6×3N) body Jacobian where N is number of arms in subset
    """
    Jb_stack = []
    n_arms = len(foot_site_ids_subset)
    n_joints = n_arms * 3
    Jq_block = np.zeros((n_joints, n_joints), dtype=float)
    Jq_list = []
    
    for i, k in enumerate(sorted(foot_site_ids_subset.keys())):
        # Compute Jb (3×6) and Jq (3×3) for arm k
        Jb, Jq = compute_body_and_joint_jacobians(
            model=model,
            data=data,
            arm_key=k,
            foot_site_id=foot_site_ids_subset[k],
            qpos_indices=arm_qpos_indices_subset[k]
        )
        
        Jb_stack.append(Jb)
        Jq_list.append(Jq)
        
        # Place Jq on diagonal
        row_start = i * 3
        col_start = i * 3
        Jq_block[row_start:row_start+3, col_start:col_start+3] = Jq
    
    # Stack body Jacobians: 3N×6
    Jb_all = np.vstack(Jb_stack)
    
    # Adaptive damping
    effective_damping = damping
    if adaptive_damping:
        try:
            singular_values = np.linalg.svd(Jb_all, compute_uv=False)
            min_sv = np.min(singular_values)
            
            min_arm_manipulability = float('inf')
            for Jq in Jq_list:
                mu_arm = compute_manipulability(Jq)
                min_arm_manipulability = min(min_arm_manipulability, mu_arm)
            
            if min_sv < MANIPULABILITY_THRESHOLD or min_arm_manipulability < MANIPULABILITY_THRESHOLD:
                critical_value = min(min_sv, min_arm_manipulability)
                scale = MANIPULABILITY_THRESHOLD / max(critical_value, 1e-6)
                effective_damping = damping * min(scale, 10.0)
        except:
            pass
    
    # Damped pseudoinverse: 6×3N
    JbT = Jb_all.T  # 6×3N
    A = JbT @ Jb_all + (effective_damping ** 2) * np.eye(6)  # 6×6
    Jb_pinv = np.linalg.solve(A, JbT)  # 6×3N
    
    # Body Jacobian: (6×3N) @ (3N×3N) = (6×3N)
    J_body = Jb_pinv @ Jq_block
    
    return J_body


# ============================================================================
# Joystick Monitor (100 Hz)
# ============================================================================

class JoystickState:
    """Thread-safe container for joystick state."""
    def __init__(self):
        self.rt_value = 0.0
        self.lt_value = 0.0
        self.lx_value = 0.0
        self.ly_value = 0.0
        self.x_pressed = False
        self.lock = threading.Lock()

    def update(self, rt, lt, lx, ly, x):
        with self.lock:
            self.rt_value = rt
            self.lt_value = lt
            self.lx_value = lx
            self.ly_value = ly
            self.x_pressed = x

    def get(self):
        with self.lock:
            return self.rt_value, self.lt_value, self.lx_value, self.ly_value, self.x_pressed


joystick_state = JoystickState()


# ============================================================================
# Diagnostics Thread (10 Hz)
# ============================================================================

class DiagnosticsState:
    """Thread-safe container for diagnostics data."""
    def __init__(self):
        self.model = None
        self.data = None
        self.lock = threading.Lock()
        self.running = True

    def update(self, model, data):
        with self.lock:
            self.model = model
            self.data = data

    def get(self):
        with self.lock:
            return self.model, self.data
    
    def stop(self):
        with self.lock:
            self.running = False
    
    def is_running(self):
        with self.lock:
            return self.running


diagnostics_state = DiagnosticsState()


def diagnostics_thread():
    """Background thread: placeholder for optional diagnostics."""
    global ESTOP_TRIGGERED
    while diagnostics_state.is_running() and not ESTOP_TRIGGERED:
        time.sleep(0.1)


def joystick_monitor_thread():
    """
    Background thread: Monitor joystick at 100 Hz.
    - Y button: starts anchoring sequence
    - RT trigger: +Z locomotion (during locomotion phase)
    - LT trigger: -Z locomotion (during locomotion phase)
    - LX/LY: X/Y locomotion (during locomotion phase)
    - X button: E-STOP
    """
    global ESTOP_TRIGGERED, SEQUENCE_STARTED

    try:
        js = joystick_connect()
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

            # X button triggers E-STOP
            if data["XB"]:
                ESTOP_TRIGGERED = True
                print("[Joystick] X pressed → E-STOP TRIGGERED!")
                break

            joystick_state.update(float(data["RT"]), float(data["LT"]), 
                                float(data["LX"]), float(data["LY"]), False)

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
    region_key: str = None,
) -> bool:
    """Auto-attach (anchor) if within region sphere. Returns True if attached this step."""
    foot_pos = data.site_xpos[foot_site_id]
    region_pos = data.site_xpos[region_site_id]
    d = foot_pos - region_pos
    dist_sq = float(d @ d)

    if dist_sq < config.region_radius * config.region_radius:
        if not controller.is_foot_attached(foot_key):
            # Always use foot_key for attachment tracking (controller only knows 4 regions)
            # The actual region is tracked separately via region_key parameter
            controller.attach_foot_to_region(foot_key, foot_key)

            # Teleport anchor mocap and activate equality
            data.mocap_pos[anchor_mocap_id] = region_pos
            data.eq_active[eq_ids[foot_key]] = 1

            # Visual feedback
            set_region_color(model, region_geom_id, config.region_active_rgba)

            return True

    return False

def solve_body_velocity_qp(
    J_body: np.ndarray,
    V_B_ref: np.ndarray,
    W: np.ndarray = None,
    R: np.ndarray = None,
    lambda_reg: float = 1e-2
):
    """
    Solve QP to find optimal joint velocities given desired body twist.
    
    Minimizes:
        (J_body*Q_dot - V_B_ref)^T * W * (J_body*Q_dot - V_B_ref) + lambda * Q_dot^T * R * Q_dot
    
    This balances:
        - Tracking error: how well J*Q_dot matches desired body velocity
        - Regularization: penalizes large joint velocities
    
    Analytical solution:
        Q_dot = (J^T*W*J + lambda*R)^{-1} * J^T*W*V_ref
    
    Args:
        J_body: (6×12) body Jacobian matrix
        V_B_ref: (6×1) desired body twist [vx, vy, vz, wx, wy, wz]
        W: (6×6) tracking weight matrix (default: higher weight on angular velocities)
        R: (12×12) regularization weight matrix (default: identity)
        lambda_reg: regularization strength (default: 1e-3)
    
    Returns:
        Q_dot: (12×1) optimal joint velocities
    """
    # Default weights
    if W is None:
        # Higher weight on angular velocities to prevent body drift/rotation
        # W = diag([1, 1, 1, w_rot, w_rot, w_rot])
        W = np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])  # 10x weight on angular velocities
    if R is None:
        R = np.eye(12)  # Equal regularization on all joints
    
    # Solve: (J^T*W*J + lambda*R) * Q_dot = J^T*W*V_ref
    JT = J_body.T  # 12×6
    A = JT @ W @ J_body + lambda_reg * R  # 12×12
    b = JT @ W @ V_B_ref  # 12×1
    
    # Solve linear system
    Q_dot = np.linalg.solve(A, b)
    
    # Apply joint velocity limits
    Q_dot = np.clip(Q_dot, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)
    
    return Q_dot


def qp_locomotion_step(
    model,
    data,
    J_body: np.ndarray,
    V_B_ref: np.ndarray,
    arm_qpos_indices: dict,
    arm_act_ids: dict,
    q_cmds: dict,
    dt: float
):
    """
    Execute one QP-based locomotion step for all arms simultaneously.
    
    Uses body Jacobian to compute optimal joint velocities that achieve
    desired body twist, then applies them to all 12 joints.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        J_body: (6×12) body Jacobian
        V_B_ref: (6×1) desired body twist [vx, vy, vz, wx, wy, wz]
        arm_qpos_indices: dict of joint indices for each arm
        arm_act_ids: dict of actuator IDs for each arm
        q_cmds: dict of current joint commands for each arm
        dt: timestep
    
    Returns:
        q_cmds: updated dict of joint commands
    """
    # Solve QP for optimal joint velocities (12×1)
    Q_dot = solve_body_velocity_qp(J_body, V_B_ref)
    
    # Apply velocities to each arm (3 joints per arm)
    for i, k in enumerate(["1", "2", "3", "4"]):
        # Extract joint velocities for this arm
        idx_start = i * 3
        q_dot_k = Q_dot[idx_start:idx_start+3]
        
        # Euler integration: q_new = q_old + q_dot * dt
        dq = q_dot_k * dt
        
        # Optional per-step clamps (same as before)
        if MAX_DTHETA_STEP is not None:
            dq[0] = float(np.clip(dq[0], -MAX_DTHETA_STEP, MAX_DTHETA_STEP))
            dq[1] = float(np.clip(dq[1], -MAX_DTHETA_STEP, MAX_DTHETA_STEP))
        if MAX_DD3_STEP is not None:
            dq[2] = float(np.clip(dq[2], -MAX_DD3_STEP, MAX_DD3_STEP))
        
        q_cmds[k] = q_cmds[k] + dq
        
        # Send position commands to actuators
        for j, aid in enumerate(arm_act_ids[k]):
            data.ctrl[aid] = q_cmds[k][j]
    
    return q_cmds


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
    adaptive_damping: bool = ADAPTIVE_DAMPING,
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
    J = compute_local_jacobian(theta1, theta2, d3)

    # Adaptive damping based on manipulability
    effective_damping = damping
    if adaptive_damping:
        mu = compute_manipulability(J)
        if mu < MANIPULABILITY_THRESHOLD:
            # Increase damping near singularities
            scale = MANIPULABILITY_THRESHOLD / max(mu, 1e-6)
            effective_damping = damping * min(scale, 10.0)  # cap at 10x base damping

    # Damped least-squares pseudoinverse
    JJt = J @ J.T
    A = JJt + (effective_damping ** 2) * np.eye(3)
    try:
        qdot = J.T @ np.linalg.solve(A, v)
    except np.linalg.LinAlgError:
        qdot, *_ = np.linalg.lstsq(J, v, rcond=None)
    
    # Limit joint velocities
    qdot = np.clip(qdot, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)

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

    # Extended sequence definition for vertical climbing
    # Format: (arm_key, region_key, needs_detach)
    sequence = [
        # Layer 1 (z=0.5): Initial anchoring
        ("1", "1", False),
        ("2", "2", False),
        ("3", "3", False),
        ("4", "4", False),
        # Layer 2 (z=1.0): Detach and re-anchor
        ("1", "5", True),
        ("2", "6", True),
        ("3", "7", True),
        ("4", "8", True),
        # Layer 3 (z=1.5): Detach and re-anchor
        ("1", "9", True),
        ("2", "10", True),
        ("3", "11", True),
        ("4", "12", True),
    ]
    
    # Track which region each arm is currently attached to
    current_arm_regions = {"1": None, "2": None, "3": None, "4": None}
    
    seq_idx = 0
    anchoring_phase = True
    detach_phase = False  # Flag to control detachment before anchoring
    qp_positioning_phase = False  # Flag for QP body positioning between anchors
    qp_position_converged = False  # Flag to track if QP positioning is done
    locomotion_active = False
    
    # QP positioning parameters
    QP_POS_TOL = 0.02  # 2cm tolerance for position convergence (m)
    QP_ORIENT_TOL = 0.05  # orientation error tolerance (rad)
    Z_OFFSET_FROM_ANCHORS = 0.25  # body should be 0.25m below average anchor position

    
    print("RMRC 4-Arm Vertical Climbing - Press Y to start, X to E-STOP\n")


    # Start joystick monitor thread
    monitor_thread = threading.Thread(target=joystick_monitor_thread, daemon=True)
    monitor_thread.start()

    # Start diagnostics thread (10 Hz pose printing)
    diag_thread = threading.Thread(target=diagnostics_thread, daemon=True)
    diag_thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # E-STOP check
            if ESTOP_TRIGGERED:
                print("[MAIN] E-STOP triggered → exiting")
                break

            mujoco.mj_step(model, data)

            # Update diagnostics thread with latest data
            diagnostics_state.update(model, data)

            # ========== ANCHORING PHASE ==========
            if anchoring_phase:
                if SEQUENCE_STARTED and seq_idx < len(sequence):
                    arm_key, region_key, needs_detach = sequence[seq_idx]
                    
                    # Step 1: Detach if needed
                    if needs_detach and not detach_phase:
                        # Find the old region to reset its color
                        if seq_idx >= 4:  # After first layer
                            old_region_key = sequence[seq_idx - 4][1]  # Get region from 4 steps ago
                            detach_foot(
                                controller=controller,
                                model=model,
                                data=data,
                                foot_key=arm_key,
                                eq_ids=eq_ids,
                                region_geom_id=region_geom_ids[old_region_key],
                                config=config,
                            )
                        detach_phase = True
                    
                    # Step 2: Check if already attached to target region
                    attached_now = try_attach(
                        controller=controller,
                        model=model,
                        data=data,
                        foot_key=arm_key,
                        eq_ids=eq_ids,
                        foot_site_id=foot_site_ids[arm_key],
                        region_site_id=region_site_ids[region_key],
                        region_geom_id=region_geom_ids[region_key],
                        anchor_mocap_id=anchor_mocap_ids[arm_key],
                        config=config,
                        region_key=region_key,
                    )
                    
                    if attached_now:
                        current_arm_regions[arm_key] = region_key
                        seq_idx += 1
                        detach_phase = False
                        
                        if seq_idx == 4 or (seq_idx > 4 and seq_idx <= len(sequence)):
                            anchoring_phase = False
                            qp_positioning_phase = True
                            qp_position_converged = False
                    else:
                        # Step 3: Drive arm toward target region
                        dt = float(model.opt.timestep)
                        q_cmds[arm_key] = rmrc_step_arm_anchoring(
                            model=model,
                            data=data,
                            arm_base_body_id=arm_base_body_ids[arm_key],
                            foot_site_id=foot_site_ids[arm_key],
                            region_site_id=region_site_ids[region_key],
                            qpos_indices=arm_qpos_indices[arm_key],
                            act_ids=arm_act_ids[arm_key],
                            q_cmd=q_cmds[arm_key],
                            dt=dt,
                        )
            
            # ========== QP BODY POSITIONING PHASE ==========
            elif qp_positioning_phase:
                # Compute target position: average of current anchor positions - 0.25z
                anchor_positions = []
                for arm_k, region_k in current_arm_regions.items():
                    if region_k is not None:
                        anchor_pos = data.site_xpos[region_site_ids[region_k]].copy()
                        anchor_positions.append(anchor_pos)
                
                if len(anchor_positions) == 4:
                    # Compute average anchor position
                    avg_anchor_pos = np.mean(anchor_positions, axis=0)
                    
                    # Target body position: average - 0.25 in z
                    target_body_pos = avg_anchor_pos.copy()
                    target_body_pos[2] -= Z_OFFSET_FROM_ANCHORS
                    
                    # Get current body pose
                    T_body = get_mainbody_pose(model, data)
                    current_body_pos = T_body[:3, 3]
                    R_current = T_body[:3, :3]
                    
                    # Compute position error
                    pos_error = target_body_pos - current_body_pos
                    pos_error_norm = np.linalg.norm(pos_error)
                    
                    # Compute orientation error (keep level)
                    R_desired = np.eye(3)
                    e_omega = compute_orientation_error(R_current, R_desired)
                    orient_error_norm = np.linalg.norm(e_omega)
                    
                    if pos_error_norm < QP_POS_TOL and orient_error_norm < QP_ORIENT_TOL:
                        if not qp_position_converged:
                            qp_position_converged = True
                            if seq_idx >= len(sequence):
                                print("✓ Climbing complete")
                                qp_positioning_phase = False
                                locomotion_active = True
                            else:
                                qp_positioning_phase = False
                                anchoring_phase = True
                    else:
                        # Compute desired body twist
                        # Proportional control on position
                        # NOTE: Negate error because feet are anchored - positive error requires negative velocity
                        v_world = -K_RMRC * pos_error
                        
                        # Saturate velocity
                        v_norm = np.linalg.norm(v_world)
                        if v_norm > V_MAX_RMRC:
                            v_world = v_world * (V_MAX_RMRC / v_norm)
                        
                        # Orientation feedback
                        omega_cmd = K_ORIENTATION * e_omega
                        omega_norm = np.linalg.norm(omega_cmd)
                        if omega_norm > MAX_ANGULAR_VEL:
                            omega_cmd = omega_cmd * (MAX_ANGULAR_VEL / omega_norm)
                        
                        # Form 6D body twist
                        V_B_ref = np.array([v_world[0], v_world[1], v_world[2], 
                                           omega_cmd[0], omega_cmd[1], omega_cmd[2]], dtype=float)
                        
                        # Compute body Jacobian and apply QP control
                        dt = float(model.opt.timestep)
                        J_body = compute_body_jacobian(model, data, foot_site_ids, arm_qpos_indices)
                        
                        q_cmds = qp_locomotion_step(
                            model=model,
                            data=data,
                            J_body=J_body,
                            V_B_ref=V_B_ref,
                            arm_qpos_indices=arm_qpos_indices,
                            arm_act_ids=arm_act_ids,
                            q_cmds=q_cmds,
                            dt=dt
                        )

            # ========== LOCOMOTION PHASE (JOYSTICK TELEOP) ==========
            elif locomotion_active:
                if all(controller.is_foot_attached(k) for k in ("1", "2", "3", "4")):
                    # Get joystick state
                    rt, lt, lx, ly, _ = joystick_state.get()

                    # Compute desired body velocities in WORLD frame
                    # LX → ±X, LY → ±Y
                    vx_world = -K_LOCOMOTION * lx
                    vy_world = -K_LOCOMOTION * ly

                    # Compute desired Z velocity in WORLD frame
                    # RT → +Z (up), LT → -Z (down)
                    if rt > 1e-6 and lt > 1e-6:
                        vz_world = 0.0
                    elif rt > 1e-6:
                        vz_world = -K_LOCOMOTION * rt
                    elif lt > 1e-6:
                        vz_world = K_LOCOMOTION * lt
                    else:
                        vz_world = 0.0

                    # Compute orientation feedback to keep body level
                    T_body = get_mainbody_pose(model, data)
                    R_current = T_body[:3, :3]
                    R_desired = np.eye(3)
                    e_omega = compute_orientation_error(R_current, R_desired)
                    omega_cmd = K_ORIENTATION * e_omega
                    
                    omega_norm = np.linalg.norm(omega_cmd)
                    if omega_norm > MAX_ANGULAR_VEL:
                        omega_cmd = omega_cmd * (MAX_ANGULAR_VEL / omega_norm)
                    
                    # Apply QP-based body velocity control
                    if abs(vx_world) > 1e-9 or abs(vy_world) > 1e-9 or abs(vz_world) > 1e-9 or np.linalg.norm(omega_cmd) > 1e-9:
                        dt = float(model.opt.timestep)
                        V_B_ref = np.array([vx_world, vy_world, vz_world, omega_cmd[0], omega_cmd[1], omega_cmd[2]], dtype=float)
                        J_body = compute_body_jacobian(model, data, foot_site_ids, arm_qpos_indices)
                        
                        q_cmds = qp_locomotion_step(
                            model=model,
                            data=data,
                            J_body=J_body,
                            V_B_ref=V_B_ref,
                            arm_qpos_indices=arm_qpos_indices,
                            arm_act_ids=arm_act_ids,
                            q_cmds=q_cmds,
                            dt=dt
                        )

            viewer.sync()
    
    # Cleanup: stop diagnostics thread
    diagnostics_state.stop()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Load model and run controller."""
    # Region radius set to 0.05m (half of original 0.1m)
    config = SimulationConfig(region_radius=0.05, target_rtf=1.0)

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
