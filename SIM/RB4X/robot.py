"""
RB4X robot model.

Holds the MuJoCo model/data and all cached handles needed to query and
command the robot.  This class is intentionally free of any controller
logic -- RMRC, QP solvers, joystick reading, etc. all live elsewhere.

Public API summary
------------------
State queries
    robot.get_body_pose()               -> 4×4 SE(3) of mainBody in world
    robot.get_arm_base_pose(arm)        -> 4×4 SE(3) of arm base in world
    robot.get_T_body_to_arm(arm)        -> 4×4 SE(3) of arm base in body frame
    robot.get_foot_pos(arm)             -> (3,) foot position in world
    robot.get_region_pos(region)        -> (3,) region site position in world
    robot.get_joint_angles(arm)         -> (3,) [theta1, theta2, d3]
    robot.get_all_joint_angles()        -> {1..4: (3,)}
    robot.qpos_indices(arm)             -> [i1, i2, i3] into data.qpos
    robot.act_ids(arm)                  -> [a1, a2, a3] actuator indices

Kinematics
    robot.arm_jacobian(arm)             -> 3×3 analytical position Jacobian
                                          in arm base frame (RRP kinematics)

Actuation
    robot.set_joint_commands(arm, q)    -> write q to data.ctrl

Anchoring
    robot.anchor(arm, region)           -> activate equality + update state
    robot.detach(arm)                   -> deactivate equality + clear state
    robot.is_anchored(arm)              -> bool
    robot.anchored_region(arm)          -> int | None
    robot.anchor_state                  -> {1..4: region_int | None}
    robot.n_anchored                    -> int (convenience count)
    robot.anchored_arms                 -> list[int]
    robot.free_arms                     -> list[int]

Simulation
    robot.step()                        -> mj_step
    robot.reset()                       -> mj_resetData + mj_forward + clear anchors
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import numpy as np

from utils import load_model, get_rb4x_handles


# ---------------------------------------------------------------------------
# RRP arm kinematics  (shared constants)
# ---------------------------------------------------------------------------

# These offsets encode the fixed geometry of each arm's prismatic joint axis
# relative to its shoulder joint origins.  They must match the MJCF model.
_Y0 = 0.059837   # base y-offset (m)
_Z0 = -0.0525    # base z-offset (m)


def _rrp_jacobian(theta1: float, theta2: float, d3: float) -> np.ndarray:
    """
    Analytical 3×3 position Jacobian for the RRP arm in the arm base frame.

    Rows → [dx, dy, dz]
    Cols → [dtheta1, dtheta2, dd3]

    This is the pure kinematic calculation; no MuJoCo queries needed.
    """
    r = _Y0 + d3

    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)

    # A = c2*r - s2*z0   (composite scalar used in FK and Jacobian)
    A          =  c2 * r      - s2 * _Z0
    dA_dtheta2 = -s2 * r      - c2 * _Z0
    dA_dd3     =  c2

    # dz/dtheta2 = c2*r - s2*z0 = A  (same expression, no alias needed)
    return np.array([
        [-c1 * A,   -s1 * dA_dtheta2,  -s1 * dA_dd3],
        [-s1 * A,    c1 * dA_dtheta2,   c1 * dA_dd3],
        [ 0.0,       A,                  s2          ],
    ], dtype=float)


# ---------------------------------------------------------------------------
# RB4X class
# ---------------------------------------------------------------------------

class RB4X:
    """MuJoCo model wrapper for the ReachBot 4X (4-arm) robot."""

    NUM_ARMS    = 4
    NUM_REGIONS = 12

    # Name of the central floating body in the MJCF model.
    MAIN_BODY_NAME = "reachbot_mount"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, model_path: str | Path) -> None:
        # ---- load model ----
        self.model, self.data = load_model(model_path)

        # ---- resolve site / mocap handles from utils ----
        handles = get_rb4x_handles(
            self.model,
            data=self.data,
            num_regions=self.NUM_REGIONS,
        )
        # feet:    {1..4: {"site_id": int, "xpos": ndarray}}
        # regions: {1..12: {"site_id": int, "xpos": ndarray}}
        # anchors: {1..4: mocap_id (int)}
        self._foot_site_ids:   Dict[int, int] = {k: v["site_id"] for k, v in handles["feet"].items()}
        self._region_site_ids: Dict[int, int] = {k: v["site_id"] for k, v in handles["regions"].items()}
        self._anchor_mocap:    Dict[int, int] = handles["anchors"]

        # ---- resolve equality-constraint IDs ----
        self._eq_ids: Dict[int, int] = self._resolve_equality_ids()

        # ---- main body ----
        self._main_body_id: int = self._resolve_main_body_id()

        # ---- per-arm cached handles ----
        self._arm_body_ids:  Dict[int, int]       = {}
        self._arm_qpos_idx:  Dict[int, List[int]] = {}
        self._arm_act_ids:   Dict[int, List[int]] = {}

        for k in range(1, self.NUM_ARMS + 1):
            self._arm_body_ids[k] = self._resolve_arm_body_id(k)
            self._arm_qpos_idx[k] = self._resolve_arm_qpos_indices(k)
            self._arm_act_ids[k]  = self._resolve_arm_actuator_ids(k)

        # ---- anchoring state ----
        # Maps arm index → region index (or None if free)
        self.anchor_state: Dict[int, Optional[int]] = {k: None for k in range(1, self.NUM_ARMS + 1)}

        # ---- initialize sim: all equality constraints off ----
        for eq_id in self._eq_ids.values():
            self.data.eq_active[eq_id] = 0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """Simulation timestep (s)."""
        return float(self.model.opt.timestep)

    @property
    def n_anchored(self) -> int:
        """Number of feet currently anchored."""
        return sum(1 for v in self.anchor_state.values() if v is not None)

    @property
    def anchored_arms(self) -> List[int]:
        """List of arm indices that are currently anchored."""
        return [k for k, v in self.anchor_state.items() if v is not None]

    @property
    def free_arms(self) -> List[int]:
        """List of arm indices that are currently free (not anchored)."""
        return [k for k, v in self.anchor_state.items() if v is None]

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_body_pose(self) -> np.ndarray:
        """Return the 4×4 SE(3) pose of mainBody in world frame."""
        R = self.data.xmat[self._main_body_id].reshape(3, 3).copy()
        p = self.data.xpos[self._main_body_id].copy()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = p
        return T

    def get_arm_base_pose(self, arm: int) -> np.ndarray:
        """Return the 4×4 SE(3) pose of arm ``arm`` base in world frame."""
        bid = self._arm_body_ids[arm]
        R = self.data.xmat[bid].reshape(3, 3).copy()
        p = self.data.xpos[bid].copy()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = p
        return T

    def get_T_body_to_arm(self, arm: int) -> np.ndarray:
        """
        Return the 4×4 SE(3) transform from mainBody frame to arm base frame
        (T_BA).  This is effectively constant for a rigid body chain.
        """
        T_WB = self.get_body_pose()
        T_WA = self.get_arm_base_pose(arm)
        R_WB = T_WB[:3, :3]
        T_BA = np.eye(4)
        T_BA[:3, :3] = R_WB.T @ T_WA[:3, :3]
        T_BA[:3, 3]  = R_WB.T @ (T_WA[:3, 3] - T_WB[:3, 3])
        return T_BA

    def get_foot_pos(self, arm: int) -> np.ndarray:
        """Return foot end-effector position in world frame, shape (3,)."""
        return np.asarray(self.data.site_xpos[self._foot_site_ids[arm]], dtype=float).copy()

    def get_region_pos(self, region: int) -> np.ndarray:
        """Return anchor region site position in world frame, shape (3,)."""
        return np.asarray(self.data.site_xpos[self._region_site_ids[region]], dtype=float).copy()

    def get_joint_angles(self, arm: int) -> np.ndarray:
        """Return ``[theta1, theta2, d3]`` for arm ``arm``, shape (3,)."""
        i1, i2, i3 = self._arm_qpos_idx[arm]
        return np.array([self.data.qpos[i1], self.data.qpos[i2], self.data.qpos[i3]], dtype=float)

    def get_all_joint_angles(self) -> Dict[int, np.ndarray]:
        """Return ``{arm: [theta1, theta2, d3]}`` for all arms."""
        return {k: self.get_joint_angles(k) for k in range(1, self.NUM_ARMS + 1)}

    def qpos_indices(self, arm: int) -> List[int]:
        """Return ``[i1, i2, i3]`` — the ``data.qpos`` indices for arm ``arm``."""
        return self._arm_qpos_idx[arm]

    def act_ids(self, arm: int) -> List[int]:
        """Return ``[a1, a2, a3]`` — the ``data.ctrl`` indices for arm ``arm``."""
        return self._arm_act_ids[arm]

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def arm_jacobian(self, arm: int) -> np.ndarray:
        """
        Compute the 3×3 analytical position Jacobian for arm ``arm`` in the
        arm base frame (RRP kinematics).

        Returns:
            J: (3, 3) ndarray  -- ``v_arm = J @ qdot``
        """
        theta1, theta2, d3 = self.get_joint_angles(arm)
        return _rrp_jacobian(theta1, theta2, d3)

    # ------------------------------------------------------------------
    # Actuation
    # ------------------------------------------------------------------

    def set_joint_commands(self, arm: int, q_cmd: np.ndarray) -> None:
        """Write position commands ``[theta1_cmd, theta2_cmd, d3_cmd]`` to actuators."""
        for j, aid in enumerate(self._arm_act_ids[arm]):
            self.data.ctrl[aid] = float(q_cmd[j])

    # ------------------------------------------------------------------
    # Anchoring
    # ------------------------------------------------------------------

    def anchor(self, arm: int, region: int) -> None:
        """
        Anchor foot ``arm`` to region ``region``.

        Teleports the mocap anchor body to the region site position and
        activates the corresponding equality constraint.
        """
        region_pos = self.get_region_pos(region)
        self.data.mocap_pos[self._anchor_mocap[arm]] = region_pos
        self.data.eq_active[self._eq_ids[arm]] = 1
        self.anchor_state[arm] = region

    def detach(self, arm: int) -> None:
        """
        Detach foot ``arm`` from its current anchor.

        Deactivates the equality constraint and clears the anchor state.
        No-op if the arm is already free.
        """
        self.data.eq_active[self._eq_ids[arm]] = 0
        self.anchor_state[arm] = None

    def is_anchored(self, arm: int) -> bool:
        """Return True if arm ``arm`` is currently anchored."""
        return self.anchor_state[arm] is not None

    def anchored_region(self, arm: int) -> Optional[int]:
        """Return the region index that arm ``arm`` is anchored to, or None."""
        return self.anchor_state[arm]

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        """
        Reset simulation to initial state.

        Clears all equality constraints and anchor state, then runs a
        forward pass to update derived quantities.
        """
        mujoco.mj_resetData(self.model, self.data)
        for eq_id in self._eq_ids.values():
            self.data.eq_active[eq_id] = 0
        self.anchor_state = {k: None for k in range(1, self.NUM_ARMS + 1)}
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for k in range(1, self.NUM_ARMS + 1):
            r = self.anchor_state[k]
            parts.append(f"arm{k}→{'free' if r is None else f'region{r}'}")
        return f"RB4X({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_main_body_id(self) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.MAIN_BODY_NAME)
        if bid < 0:
            raise RuntimeError(f"Could not find body '{self.MAIN_BODY_NAME}' in model")
        return int(bid)

    def _resolve_equality_ids(self) -> Dict[int, int]:
        eq_ids = {}
        for k in range(1, self.NUM_ARMS + 1):
            name = f"foot{k}_anchor"
            eid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
            if eid < 0:
                raise RuntimeError(f"Missing equality constraint '{name}'")
            eq_ids[k] = int(eid)
        return eq_ids

    def _resolve_arm_body_id(self, arm: int) -> int:
        name = f"arm{arm}"
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"Missing body '{name}'")
        return int(bid)

    def _resolve_arm_qpos_indices(self, arm: int) -> List[int]:
        joint_names = [f"revolver{arm}1", f"revolver{arm}2", f"prismatic{arm}"]
        indices = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Missing joint '{name}'")
            indices.append(int(self.model.jnt_qposadr[jid]))
        return indices

    def _resolve_arm_actuator_ids(self, arm: int) -> List[int]:
        # Naming in the MJCF: motor{arm}{joint}, e.g. motor11, motor12, boomMotor1
        act_names = [f"motor{arm}1", f"motor{arm}2", f"boomMotor{arm}"]
        ids = []
        for name in act_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"Missing actuator '{name}'")
            ids.append(int(aid))
        return ids
