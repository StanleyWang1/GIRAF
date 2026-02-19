"""
RB4X Vertical Climbing Simulation.

Behaviour
---------
Press Enter to begin the pre-programmed anchor sequence.
Arms are anchored one at a time (1 → 2 → 3 → 4) using RMRC.
After every group of four anchors the body is repositioned with a
QP controller before continuing to the next layer.
Simulation holds at completion (press the viewer close button to quit).

Control architecture
--------------------
Anchoring  (per-arm RMRC)
    e = p_region - p_foot             (in arm base frame)
    v = Kp * e                        (P-control, speed-saturated)
    q̇ = J⁺ v                         (damped least-squares)
    q_cmd += q̇ dt                    (Euler)

Body repositioning  (QP over anchored joints)
    V_B_ref = [-Kp * pos_err;  K_ori * ori_err]   (negate: feet are fixed)
    q̇ = (Jᵀ W J + λ R)⁻¹ Jᵀ W V_B_ref           (weighted least-squares)
    q_cmd += q̇ dt                                  (Euler)

    Body Jacobian derivation (anchored foot i, v_Fi = 0):
        v_Fi = Jb_i twist_B + Jq_i q̇_i = 0
        Jb_i = [I₃  | −skew(p_Fi − p_B)]    (3×6)
        Jq_i = R_WB R_BA J_local             (3×3, maps q̇ → foot vel in world)
        Stack → Jb_all (3N×6), Jq_block (3N×3N)
        J_body = Jb_all†  Jq_block           (6×3N, maps q̇_anchored → body twist)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from robot import RB4X


# ============================================================================
# Control gains & limits
# ============================================================================

# --- RMRC anchoring ---
K_RMRC      = 5.0    # task-space P-gain (1/s)
V_MAX_RMRC  = 0.05   # max task-space speed (m/s)
POS_TOL     = 0.01   # convergence tolerance (m)
DAMPING     = 5e-2   # DLS damping factor

# --- QP body repositioning ---
K_BODY      = 5.0    # body position P-gain (1/s)
K_ORI       = -10.0  # orientation P-gain (1/s)
V_MAX_BODY  = 0.05   # max body translation speed (m/s)
MAX_OMEGA   = 0.10   # max angular rate command (rad/s)
QP_POS_TOL  = 0.02   # body position convergence tolerance (m)
QP_ORI_TOL  = 0.05   # body orientation convergence tolerance (rad)
LAMBDA_REG  = 1e-2   # QP joint-velocity regularisation

W_TASK = np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])  # tracking weight (position / orientation)

# --- Universal joint limits ---
MAX_JOINT_VEL = 0.5   # rad/s or m/s
MAX_DTHETA    = 0.05  # rad per step
MAX_DD3       = 0.01  # m   per step

# --- Anchoring ---
ANCHOR_RADIUS = 0.05  # snap distance (m)
Z_OFFSET      = 0.20  # body sits this far below the lowest anchor (m)

# --- Climbing sequence: (arm, target_region, detach_first) ---
SEQUENCE: List[Tuple[int, int, bool]] = [
    # Layer 1  (z ≈ 0.5 m)
    (1,  1, False),
    (2,  2, False),
    (3,  3, False),
    (4,  4, False),
    # Layer 2  (z ≈ 1.0 m)
    (1,  5, True),
    (2,  6, True),
    (3,  7, True),
    (4,  8, True),
    # Layer 3  (z ≈ 1.5 m)
    (1,  9, True),
    (2, 10, True),
    (3, 11, True),
    (4, 12, True),
]


# ============================================================================
# Math utilities
# ============================================================================

def skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix  (skew(v) @ w == v × w)."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ],
    ], dtype=float)


def rotation_log(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector ω s.t. R ≈ exp([ω]×).  Log map SO(3) → ℝ³."""
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-6:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        k = int(np.argmax(np.diag(R)))
        axis = np.zeros(3)
        axis[k] = np.sqrt((R[k, k] + 1.0) / 2.0)
        for i in range(3):
            if i != k:
                axis[i] = R[k, i] / (2.0 * axis[k])
        return theta * axis
    S = (R - R.T) / (2.0 * np.sin(theta))
    return theta * np.array([S[2, 1], S[0, 2], S[1, 0]])


def orientation_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Axis-angle error: rotation needed to bring R_cur onto R_des."""
    return rotation_log(R_des @ R_cur.T)


# ============================================================================
# Per-arm RMRC  (anchoring phase)
# ============================================================================

def rmrc_step(
    robot: RB4X,
    arm: int,
    region: int,
    q_cmd: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    One RMRC step driving arm ``arm`` toward region ``region``.

    Error is computed in the arm base frame so the analytical Jacobian
    (also expressed in arm base frame) can be applied directly.

    Args:
        robot:  RB4X instance
        arm:    arm index 1–4
        region: target region index 1–12
        q_cmd:  current joint position command, shape (3,)

    Returns:
        q_cmd:     updated command
        converged: True when foot is within POS_TOL of target
    """
    R_WA = robot.get_arm_base_pose(arm)[:3, :3]

    p_foot   = robot.get_foot_pos(arm)
    p_region = robot.get_region_pos(region)

    # Error in arm base frame
    e = R_WA.T @ (p_region - p_foot)

    if float(np.linalg.norm(e)) < POS_TOL:
        return q_cmd, True

    # P-control with speed clamping
    v = K_RMRC * e
    v_norm = float(np.linalg.norm(v))
    if v_norm > V_MAX_RMRC:
        v *= V_MAX_RMRC / v_norm

    # Damped least-squares pseudoinverse  q̇ = Jᵀ (J Jᵀ + λ²I)⁻¹ v
    J   = robot.arm_jacobian(arm)
    JJt = J @ J.T
    qdot = J.T @ np.linalg.solve(JJt + (DAMPING ** 2) * np.eye(3), v)
    qdot = np.clip(qdot, -MAX_JOINT_VEL, MAX_JOINT_VEL)

    dq    = qdot * robot.dt
    dq[0] = float(np.clip(dq[0], -MAX_DTHETA, MAX_DTHETA))
    dq[1] = float(np.clip(dq[1], -MAX_DTHETA, MAX_DTHETA))
    dq[2] = float(np.clip(dq[2], -MAX_DD3,    MAX_DD3))

    q_cmd = q_cmd + dq
    robot.set_joint_commands(arm, q_cmd)
    return q_cmd, False


# ============================================================================
# QP body repositioning  (between anchor layers)
# ============================================================================

def _arm_jacobians(robot: RB4X, arm: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Jb (3×6) and Jq (3×3) for a single arm.

    Jb maps body twist [v_B; ω_B] to foot velocity in world frame:
        Jb = [I₃  |  −skew(p_F − p_B)]

    Jq maps joint velocities to foot velocity in world frame:
        Jq = R_WB  R_BA  J_local
    """
    T_WB = robot.get_body_pose()
    R_WB = T_WB[:3, :3]
    p_WB = T_WB[:3, 3]

    R_WA = robot.get_arm_base_pose(arm)[:3, :3]
    R_BA = R_WB.T @ R_WA          # arm base orientation in body frame

    p_F     = robot.get_foot_pos(arm)
    r_BF    = p_F - p_WB           # body-to-foot vector in world

    Jb = np.zeros((3, 6), dtype=float)
    Jb[:, :3] = np.eye(3)
    Jb[:, 3:] = -skew(r_BF)

    Jq = R_WB @ R_BA @ robot.arm_jacobian(arm)

    return Jb, Jq


def _body_jacobian(robot: RB4X, arms: List[int]) -> np.ndarray:
    """
    Stacked body Jacobian J (6 × 3N) for the subset of anchored ``arms``.

    For each anchored foot i the no-slip constraint gives  v_Fi = 0, so:
        0 = Jb_i twist_B + Jq_i q̇_i
        twist_B = −Jb_i⁻ Jq_i q̇_i

    Stacking N arms:
        Jb_all (3N×6),  Jq_block (3N×3N, block-diagonal)
        J_body = Jb_all†  Jq_block      [uses damped pseudoinverse of Jb_all]
    """
    N        = len(arms)
    Jb_all   = np.zeros((3 * N, 6),     dtype=float)
    Jq_block = np.zeros((3 * N, 3 * N), dtype=float)

    for i, arm in enumerate(arms):
        Jb, Jq = _arm_jacobians(robot, arm)
        Jb_all  [3*i:3*i+3, :]          = Jb
        Jq_block[3*i:3*i+3, 3*i:3*i+3] = Jq

    JbT     = Jb_all.T
    A       = JbT @ Jb_all + (DAMPING ** 2) * np.eye(6)
    Jb_pinv = np.linalg.solve(A, JbT)   # 6 × 3N

    return Jb_pinv @ Jq_block            # 6 × 3N


def qp_step(
    robot: RB4X,
    V_B_ref: np.ndarray,
    q_cmds: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    One QP velocity step for all currently-anchored arms.

    Minimises the weighted tracking + regularisation cost:
        min  (J q̇ − V_ref)ᵀ W (J q̇ − V_ref)  +  λ q̇ᵀ R q̇
    Analytical solution (weighted least-squares):
        q̇ = (Jᵀ W J + λ R)⁻¹ Jᵀ W V_ref

    Args:
        robot:   RB4X instance
        V_B_ref: desired body twist [vx, vy, vz, wx, wy, wz], shape (6,)
        q_cmds:  current joint commands per arm

    Returns:
        q_cmds: updated dict
    """
    anchored = robot.anchored_arms
    if not anchored:
        return q_cmds

    N = len(anchored)
    J = _body_jacobian(robot, anchored)  # 6 × 3N
    R = np.eye(3 * N)

    JT    = J.T
    A     = JT @ W_TASK @ J + LAMBDA_REG * R
    b     = JT @ W_TASK @ V_B_ref
    Q_dot = np.clip(np.linalg.solve(A, b), -MAX_JOINT_VEL, MAX_JOINT_VEL)

    dt = robot.dt
    for i, arm in enumerate(anchored):
        dq    = Q_dot[3*i:3*i+3] * dt
        dq[0] = float(np.clip(dq[0], -MAX_DTHETA, MAX_DTHETA))
        dq[1] = float(np.clip(dq[1], -MAX_DTHETA, MAX_DTHETA))
        dq[2] = float(np.clip(dq[2], -MAX_DD3,    MAX_DD3))
        q_cmds[arm] = q_cmds[arm] + dq
        robot.set_joint_commands(arm, q_cmds[arm])

    return q_cmds


def body_reposition_step(
    robot: RB4X,
    target_pos: np.ndarray,
    q_cmds: Dict[int, np.ndarray],
) -> Tuple[Dict[int, np.ndarray], bool]:
    """
    Drive body toward ``target_pos`` in world frame while keeping upright.

    Velocity sign is negated because the feet are fixed: to translate the
    body in +x the joints must effectively pull the body towards the feet.

    Returns:
        q_cmds:    updated commands
        converged: True when within QP_POS_TOL and QP_ORI_TOL
    """
    T   = robot.get_body_pose()
    R   = T[:3, :3]
    pos = T[:3, 3]

    pos_err = target_pos - pos
    ori_err = orientation_error(R, np.eye(3))

    if np.linalg.norm(pos_err) < QP_POS_TOL and np.linalg.norm(ori_err) < QP_ORI_TOL:
        return q_cmds, True

    # Position velocity command (negate: feet fixed, body moves opposite)
    v = -K_BODY * pos_err
    v_norm = np.linalg.norm(v)
    if v_norm > V_MAX_BODY:
        v *= V_MAX_BODY / v_norm

    # Orientation velocity command
    omega = K_ORI * ori_err
    omega_norm = np.linalg.norm(omega)
    if omega_norm > MAX_OMEGA:
        omega *= MAX_OMEGA / omega_norm

    V_B_ref = np.concatenate([v, omega])
    q_cmds  = qp_step(robot, V_B_ref, q_cmds)
    return q_cmds, False


# ============================================================================
# Anchoring helper
# ============================================================================

def try_anchor(robot: RB4X, arm: int, region: int) -> bool:
    """
    Snap arm ``arm`` to region ``region`` if the foot is within ANCHOR_RADIUS.

    Returns True if newly anchored this call.
    """
    if robot.is_anchored(arm):
        return False   # already anchored (to this region or another)

    p_foot   = robot.get_foot_pos(arm)
    p_region = robot.get_region_pos(region)
    if float(np.linalg.norm(p_foot - p_region)) < ANCHOR_RADIUS:
        robot.anchor(arm, region)
        return True
    return False


# ============================================================================
# Target body position helper
# ============================================================================

def target_body_pos(robot: RB4X) -> np.ndarray:
    """
    Desired body position: XY centroid of anchored regions, Z = min_anchor_z - Z_OFFSET.
    """
    positions = [
        robot.get_region_pos(r)
        for r in robot.anchor_state.values()
        if r is not None
    ]
    arr = np.array(positions)
    xy  = arr[:, :2].mean(axis=0)
    z   = arr[:, 2].min() - Z_OFFSET
    return np.array([xy[0], xy[1], z])


# ============================================================================
# Main simulation loop
# ============================================================================

def run(model_path: Path) -> None:
    """Initialise robot, await Enter, then execute climbing sequence."""
    robot = RB4X(model_path)

    # Seed joint commands from the initial qpos
    q_cmds: Dict[int, np.ndarray] = {
        arm: robot.get_joint_angles(arm).copy()
        for arm in range(1, RB4X.NUM_ARMS + 1)
    }

    print("=== RB4X Vertical Climbing Sim ===")
    print(f"Sequence: {len(SEQUENCE)} anchors across {len(SEQUENCE)//4} layers")
    print("Press Enter to begin...")
    input()

    seq_idx = 0
    phase   = "anchoring"     # "anchoring" | "repositioning" | "done"
    reposition_target: np.ndarray | None = None

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            robot.step()

            # ------------------------------------------------------------------
            if phase == "anchoring":
                if seq_idx >= len(SEQUENCE):
                    print("✓ Climbing sequence complete — holding position.")
                    phase = "done"
                    viewer.sync()
                    continue

                arm, region, detach_first = SEQUENCE[seq_idx]

                # Detach this arm from its previous region before moving
                if detach_first and robot.is_anchored(arm):
                    robot.detach(arm)
                    print(f"  detached  arm {arm}")

                # Attempt to snap foot to target region
                if try_anchor(robot, arm, region):
                    print(f"  anchored  arm {arm} → region {region}  "
                          f"(layer {(seq_idx // 4) + 1})")
                    seq_idx += 1

                    # After completing a full layer of 4 anchors, reposition
                    if seq_idx % 4 == 0 and seq_idx < len(SEQUENCE):
                        reposition_target = target_body_pos(robot)
                        print(f"  repositioning body → {reposition_target.round(3)}")
                        phase = "repositioning"
                else:
                    # Drive arm toward target region with RMRC
                    q_cmds[arm], _ = rmrc_step(robot, arm, region, q_cmds[arm])

            # ------------------------------------------------------------------
            elif phase == "repositioning":
                q_cmds, converged = body_reposition_step(robot, reposition_target, q_cmds)
                if converged:
                    print("  body repositioned.")
                    phase = "anchoring"

            # ------------------------------------------------------------------
            # "done": hold position, viewer stays open until closed manually

            viewer.sync()


def main() -> None:
    model_path = Path(__file__).parent / "models" / "vertical_climbing.xml"
    run(model_path)


if __name__ == "__main__":
    main()
