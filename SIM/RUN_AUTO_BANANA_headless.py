"""
GIRAF Headless Auto-Grasp ALL Bananas (Speed-Optimized)
Same control logic as RUN_AUTO_BANANAS.py but optimized for raw speed.
Uses sim time (data.time) for all waits => fully deterministic.

Speed optimizations over base headless:
  - Inline rotation math (no scipy Rotation object overhead)
  - Pre-allocated arrays (no np.vstack / np.array each iteration)
  - Array-based actuator IDs (no dict lookups)
  - Control decimation: --ctrl-skip N reuses Jacobian inverse for N physics steps
"""

import numpy as np
import mujoco
import time
import os
import sys
import argparse

from kinematic_model import num_jacobian, num_forward_kinematics

## ----------------------------------------------------------------------------------------------------
# Inline rotation math (replaces scipy.spatial.transform.Rotation)
## ----------------------------------------------------------------------------------------------------
def quat_wxyz_to_matrix(q):
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix. Pure numpy."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    tx, ty, tz = x + x, y + y, z + z
    return np.array([
        [1.0 - (y*ty + z*tz), x*ty - w*tz,         x*tz + w*ty        ],
        [x*ty + w*tz,         1.0 - (x*tx + z*tz), y*tz - w*tx        ],
        [x*tz - w*ty,         y*tz + w*tx,         1.0 - (x*tx + y*ty)]
    ])

def rotmat_to_rotvec(R):
    """Rotation matrix -> rotation vector (axis * angle). Pure numpy."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_a = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    angle = np.arccos(cos_a)
    if angle < 1e-10:
        return np.zeros(3)
    if abs(angle - np.pi) < 1e-6:
        col = np.argmax(np.diag(R))
        v = R[:, col] + np.eye(3)[col]
        return v * (angle / np.linalg.norm(v))
    return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) * (angle / (2.0 * np.sin(angle)))

## ----------------------------------------------------------------------------------------------------
# Banana Tracking Helpers
## ----------------------------------------------------------------------------------------------------
BOX_CENTER = np.array([1.0, 0.5])
BOX_HALF_SIZE = 0.14
BOX_TOP_Z = 0.16

def get_banana_pos(model, data, body_id):
    jnt_id = model.body_jntadr[body_id]
    qpos_addr = model.jnt_qposadr[jnt_id]
    return data.qpos[qpos_addr:qpos_addr+3].copy()

def get_banana_quat(model, data, body_id):
    jnt_id = model.body_jntadr[body_id]
    qpos_addr = model.jnt_qposadr[jnt_id]
    return data.qpos[qpos_addr+3:qpos_addr+7].copy()

def is_banana_in_box(model, data, body_id):
    pos = get_banana_pos(model, data, body_id)
    in_x = abs(pos[0] - BOX_CENTER[0]) < BOX_HALF_SIZE
    in_y = abs(pos[1] - BOX_CENTER[1]) < BOX_HALF_SIZE
    in_z = pos[2] < BOX_TOP_Z + 0.1
    return in_x and in_y and in_z

def find_next_banana(model, data, banana_body_ids):
    best_id = None
    best_dist = float('inf')
    for bid in banana_body_ids:
        if not is_banana_in_box(model, data, bid):
            pos = get_banana_pos(model, data, bid)
            dist = np.linalg.norm(pos[:2])
            if dist < best_dist:
                best_dist = dist
                best_id = bid
    return best_id

## ----------------------------------------------------------------------------------------------------
# Main
## ----------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctrl-skip', type=int, default=1,
                        help='Reuse Jacobian inverse for N physics steps (default: 1, try 5-10)')
    args = parser.parse_args()
    CTRL_SKIP = max(1, args.ctrl_skip)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "GIRAF_bananas.xml")

    print(f"Loading model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    print(f"Model loaded. Joints: {model.njnt}, Actuators: {model.nu}, Timestep: {model.opt.timestep}s")

    # Joint / actuator mappings
    joint_names = ['R1', 'R2', 'P3', 'R4', 'R5', 'R6', 'left_grip_joint', 'right_grip_joint']
    joint_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}

    actuator_names = ['actuator_R1', 'actuator_R2', 'actuator_P3', 'actuator_R4', 'actuator_R5', 'actuator_R6', 'actuator_left_grip', 'actuator_right_grip']
    actuator_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names}
    aid_arr = np.array([actuator_ids[n] for n in actuator_names], dtype=np.intp)

    # Find banana bodies
    banana_body_ids = []
    for i in range(1, 11):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"banana_{i}")
        if bid >= 0:
            banana_body_ids.append(bid)
    print(f"Found {len(banana_body_ids)} bananas")

    def inverse_jacobian(joint_coords):
        J = num_jacobian(joint_coords)
        return np.linalg.pinv(J)

    # Initialize joint positions (same as RUN_AUTO_BANANAS.py)
    roll_pos = 0.0
    roll_offset = 0.0
    pitch_pos = 0.0
    d3_pos = 0.25
    theta4_pos = 0.0
    theta5_pos = 0.0
    theta6_pos = 0.0
    gripper_pos = 0.0

    data.qpos[joint_ids['R1']] = roll_pos
    data.qpos[joint_ids['R2']] = pitch_pos
    data.qpos[joint_ids['P3']] = d3_pos
    data.qpos[joint_ids['R4']] = theta4_pos
    data.qpos[joint_ids['R5']] = theta5_pos
    data.qpos[joint_ids['R6']] = theta6_pos
    data.qpos[joint_ids['left_grip_joint']] = gripper_pos
    data.qpos[joint_ids['right_grip_joint']] = gripper_pos

    mujoco.mj_forward(model, data)

    # Control params (identical to RUN_AUTO_BANANAS.py)
    AUTO_KP = 5.0
    AUTO_KP_ROT = 2.0
    RELEASE_WAIT_TIME = 0.5
    AUTO_MAX_VEL = 0.45
    AUTO_MAX_OMEGA = 1.0
    AUTO_HOVER_HEIGHT = 0.1
    AUTO_GRASP_HEIGHT = 0.025
    AUTO_GRASP_HEIGHT_DEFAULT = 0.02
    AUTO_GRASP_HEIGHT_RETRY = 0.01
    AUTO_POS_TOL = 0.015
    AUTO_ROT_TOL = 0.3
    BOX_DROP_HEIGHT = 0.35
    SAFE_LIFT_HEIGHT = 0.25
    STABILIZE_TIME = 0.5

    # Rate-limit deltas (identical)
    MAX_CTRL_DELTA = np.array([0.05, 0.1, 0.05, 0.01, 0.01, 0.01, 0.025, 0.025])

    # Pre-computed FK offset constants
    HALF_PI = np.pi * 0.5
    FIVE_SIXTH_PI = np.pi * 5.0 / 6.0
    dt = 0.0025  # integration timestep

    auto_state = "select"
    stabilize_start_simtime = 0.0
    release_wait_start_simtime = 0.0
    current_target_body = None
    bananas_collected = 0
    velocity = np.zeros((6, 1))
    joint_velocity = np.zeros((6, 1))
    ctrl_desired = np.zeros(8)
    gripper_velocity = 0
    total_iterations = 0
    ctrl_countdown = 1  # triggers full update on first iteration
    all_done = False

    wall_start = time.perf_counter()
    if CTRL_SKIP > 1:
        print(f"Control decimation: computing IK every {CTRL_SKIP} physics steps")
    print("Starting headless autonomous collection...\n")

    while not all_done:
        total_iterations += 1

        ## ------ Fast path: skip FK/IK, reuse cached joint velocity ------
        ctrl_countdown -= 1
        if ctrl_countdown > 0:
            roll_pos += dt * joint_velocity[0, 0]
            pitch_pos += dt * joint_velocity[1, 0]
            d3_pos += dt * joint_velocity[2, 0]
            theta4_pos += dt * joint_velocity[3, 0]
            theta5_pos += dt * joint_velocity[4, 0]
            theta6_pos += dt * joint_velocity[5, 0]
            gripper_pos += gripper_velocity
            roll_pos = np.arctan2(np.sin(roll_pos), np.cos(roll_pos))
            pitch_pos = np.arctan2(np.sin(pitch_pos), np.cos(pitch_pos))
            theta4_pos = np.arctan2(np.sin(theta4_pos), np.cos(theta4_pos))
            theta5_pos = np.arctan2(np.sin(theta5_pos), np.cos(theta5_pos))
            theta6_pos = np.arctan2(np.sin(theta6_pos), np.cos(theta6_pos))
            roll_pos = np.clip(roll_pos, -np.pi/2, np.pi/2)
            pitch_pos = np.clip(pitch_pos, -np.pi/4, np.pi/2)
            d3_pos = np.clip(d3_pos, 0.2, 3.0)
            theta5_pos = max(theta5_pos, -1.7)
            gripper_pos = np.clip(gripper_pos, 0.0, 0.05)
            ctrl_desired[0] = roll_pos + roll_offset
            ctrl_desired[1] = pitch_pos
            ctrl_desired[2] = d3_pos
            ctrl_desired[3] = theta4_pos
            ctrl_desired[4] = theta5_pos
            ctrl_desired[5] = theta6_pos
            ctrl_desired[6] = gripper_pos
            ctrl_desired[7] = gripper_pos
            for i in range(8):
                a = aid_arr[i]
                prev = data.ctrl[a]
                d = ctrl_desired[i] - prev
                md = MAX_CTRL_DELTA[i]
                if d > md: d = md
                elif d < -md: d = -md
                data.ctrl[a] = prev + d
            mujoco.mj_step(model, data)
            continue
        ctrl_countdown = CTRL_SKIP

        # Read actual joint positions
        actual_roll = data.qpos[joint_ids['R1']]
        actual_pitch = data.qpos[joint_ids['R2']]
        actual_d3 = data.qpos[joint_ids['P3']]
        actual_theta4 = data.qpos[joint_ids['R4']]
        actual_theta5 = data.qpos[joint_ids['R5']]
        actual_theta6 = data.qpos[joint_ids['R6']]

        FK_mat = num_forward_kinematics([
            actual_roll,
            actual_pitch + HALF_PI,
            actual_d3,
            actual_theta4 + HALF_PI,
            actual_theta5 + FIVE_SIXTH_PI,
            actual_theta6
        ])

        current_pos = FK_mat[:3, 3]
        current_rot = FK_mat[:3, :3]
        banana_pos = np.zeros(3)
        if current_target_body is not None:
            banana_pos = get_banana_pos(model, data, current_target_body)

        sim_time = data.time

        ## ------ STATE MACHINE (identical logic) ------
        if auto_state == "select":
            current_target_body = find_next_banana(model, data, banana_body_ids)

            if current_target_body is None:
                bananas_in_box = sum(1 for bid in banana_body_ids if is_banana_in_box(model, data, bid))
                print(f"All bananas collected! ({bananas_in_box}/{len(banana_body_ids)} in box)")
                all_done = True
                break
            else:
                banana_pos = get_banana_pos(model, data, current_target_body)
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current_target_body)
                bananas_in_box = sum(1 for bid in banana_body_ids if is_banana_in_box(model, data, bid))
                print(f"[{sim_time:7.2f}s] Targeting {body_name} at [{banana_pos[0]:.2f}, {banana_pos[1]:.2f}, {banana_pos[2]:.2f}] ({bananas_in_box}/{len(banana_body_ids)} in box)")
                auto_state = "approach"
            velocity = np.zeros((6, 1))
            gripper_velocity = 0

        if auto_state != "select" and not all_done:
            banana_pos = get_banana_pos(model, data, current_target_body)

            # Desired orientation from banana (inline, no scipy)
            banana_quat_wxyz = get_banana_quat(model, data, current_target_body)
            banana_R = quat_wxyz_to_matrix(banana_quat_wxyz)

            bx, by = banana_R[0, 0], banana_R[1, 0]
            bn = bx * bx + by * by
            if bn > 1e-12:
                bn = 1.0 / np.sqrt(bn)
                bx *= bn; by *= bn
            else:
                bx, by = 1.0, 0.0

            # Resolve 180° ambiguity (inline dot product)
            if bx * current_rot[0, 1] + by * current_rot[1, 1] < 0:
                bx, by = -bx, -by

            # R_desired: x_ee=[-by,bx,0], y_ee=[bx,by,0], z_ee=[0,0,-1]
            R_desired = np.array([[-by, bx, 0.0], [bx, by, 0.0], [0.0, 0.0, -1.0]])

            # Orientation error (inline, no scipy)
            R_error = R_desired @ current_rot.T
            omega_error = rotmat_to_rotvec(R_error)
            rot_error_mag = np.linalg.norm(omega_error)
            v_angular = (AUTO_KP_ROT * omega_error).reshape(3, 1)
            omega_norm = np.linalg.norm(v_angular)
            if omega_norm > AUTO_MAX_OMEGA:
                v_angular = v_angular * (AUTO_MAX_OMEGA / omega_norm)

            if auto_state == "approach":
                target_pos = banana_pos.copy()
                target_pos[2] += AUTO_HOVER_HEIGHT

                pos_error = target_pos - current_pos
                pos_error_mag = np.linalg.norm(pos_error)
                v_linear = AUTO_KP * pos_error.reshape(3, 1)
                v_norm = np.linalg.norm(v_linear)
                if v_norm > AUTO_MAX_VEL:
                    v_linear = v_linear * (AUTO_MAX_VEL / v_norm)

                velocity = np.vstack([v_linear, v_angular])

                if gripper_pos < 0.045:
                    gripper_velocity = 0.002
                else:
                    gripper_velocity = 0

                if pos_error_mag < AUTO_POS_TOL and rot_error_mag < AUTO_ROT_TOL:
                    auto_state = "descend"
                    print(f"[{sim_time:7.2f}s]   Aligned - descending")

            elif auto_state == "descend":
                target_pos = banana_pos.copy()
                target_pos[2] += AUTO_GRASP_HEIGHT

                pos_error = target_pos - current_pos
                pos_error_mag = np.linalg.norm(pos_error)

                v_linear = AUTO_KP * pos_error.reshape(3, 1)
                v_norm = np.linalg.norm(v_linear)
                if v_norm > AUTO_MAX_VEL * 0.5:
                    v_linear = v_linear * (AUTO_MAX_VEL * 0.5 / v_norm)

                velocity = np.vstack([v_linear, v_angular])
                gripper_velocity = 0

                if pos_error_mag < AUTO_POS_TOL:
                    auto_state = "grasp"
                    print(f"[{sim_time:7.2f}s]   Closing gripper")

            elif auto_state == "grasp":
                target_pos = banana_pos.copy()
                target_pos[2] += AUTO_GRASP_HEIGHT

                pos_error = target_pos - current_pos
                v_linear = AUTO_KP * pos_error.reshape(3, 1)
                v_norm = np.linalg.norm(v_linear)
                if v_norm > AUTO_MAX_VEL * 0.3:
                    v_linear = v_linear * (AUTO_MAX_VEL * 0.3 / v_norm)

                velocity = np.vstack([v_linear, v_angular])

                if gripper_pos > 0.005:
                    gripper_velocity = -0.002
                else:
                    gripper_velocity = 0
                    auto_state = "stabilize"
                    stabilize_start_simtime = sim_time
                    print(f"[{sim_time:7.2f}s]   Stabilizing")

            elif auto_state == "stabilize":
                target_pos = banana_pos.copy()
                target_pos[2] += AUTO_GRASP_HEIGHT

                pos_error = target_pos - current_pos
                v_linear = AUTO_KP * pos_error.reshape(3, 1)
                v_norm = np.linalg.norm(v_linear)
                if v_norm > AUTO_MAX_VEL * 0.3:
                    v_linear = v_linear * (AUTO_MAX_VEL * 0.3 / v_norm)

                velocity = np.vstack([v_linear, v_angular])
                gripper_velocity = 0

                if sim_time - stabilize_start_simtime >= STABILIZE_TIME:
                    auto_state = "lift_clear"
                    print(f"[{sim_time:7.2f}s]   Lifting clear")

            elif auto_state == "lift_clear":
                target_pos = np.array([current_pos[0], current_pos[1], SAFE_LIFT_HEIGHT])

                pos_error = target_pos - current_pos
                pos_error_mag = np.linalg.norm(pos_error)
                v_linear = AUTO_KP * pos_error.reshape(3, 1)
                v_norm = np.linalg.norm(v_linear)
                if v_norm > AUTO_MAX_VEL:
                    v_linear = v_linear * (AUTO_MAX_VEL / v_norm)

                velocity = np.vstack([v_linear, v_angular])
                gripper_velocity = 0

                if pos_error_mag < AUTO_POS_TOL:
                    # Check if banana still grasped
                    banana_check = get_banana_pos(model, data, current_target_body)
                    grasp_dist = np.linalg.norm(banana_check - current_pos)
                    if grasp_dist > 0.08:
                        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current_target_body)
                        print(f"[{sim_time:7.2f}s]   LOST {body_name} (dist={grasp_dist:.3f}m) - retrying lower")
                        AUTO_GRASP_HEIGHT = AUTO_GRASP_HEIGHT_RETRY
                        current_target_body = None
                        auto_state = "select"
                    else:
                        if AUTO_GRASP_HEIGHT != AUTO_GRASP_HEIGHT_DEFAULT:
                            print(f"[{sim_time:7.2f}s]   Retry succeeded - restoring default height")
                            AUTO_GRASP_HEIGHT = AUTO_GRASP_HEIGHT_DEFAULT
                        auto_state = "transit"
                        print(f"[{sim_time:7.2f}s]   Transit to box")

            elif auto_state == "transit":
                target_pos = np.array([BOX_CENTER[0], BOX_CENTER[1], BOX_DROP_HEIGHT])

                pos_error = target_pos - current_pos
                pos_error_mag = np.linalg.norm(pos_error)
                v_linear = AUTO_KP * pos_error.reshape(3, 1)
                v_norm = np.linalg.norm(v_linear)
                if v_norm > AUTO_MAX_VEL:
                    v_linear = v_linear * (AUTO_MAX_VEL / v_norm)

                velocity = np.vstack([v_linear, v_angular])
                gripper_velocity = 0

                if pos_error_mag < AUTO_POS_TOL:
                    auto_state = "drop"
                    print(f"[{sim_time:7.2f}s]   Above box - releasing")

            elif auto_state == "drop":
                target_pos = np.array([BOX_CENTER[0], BOX_CENTER[1], BOX_DROP_HEIGHT])

                pos_error = target_pos - current_pos
                v_linear = AUTO_KP * pos_error.reshape(3, 1)
                v_norm = np.linalg.norm(v_linear)
                if v_norm > AUTO_MAX_VEL * 0.3:
                    v_linear = v_linear * (AUTO_MAX_VEL * 0.3 / v_norm)

                velocity = np.vstack([v_linear, v_angular])

                if gripper_pos < 0.045:
                    gripper_velocity = 0.002
                else:
                    gripper_velocity = 0
                    auto_state = "release_wait"
                    release_wait_start_simtime = sim_time
                    print(f"[{sim_time:7.2f}s]   Waiting for drop")

            elif auto_state == "release_wait":
                target_pos = np.array([BOX_CENTER[0], BOX_CENTER[1], BOX_DROP_HEIGHT])
                pos_error = target_pos - current_pos
                v_linear = AUTO_KP * pos_error.reshape(3, 1)
                v_norm = np.linalg.norm(v_linear)
                if v_norm > AUTO_MAX_VEL * 0.3:
                    v_linear = v_linear * (AUTO_MAX_VEL * 0.3 / v_norm)

                velocity = np.vstack([v_linear, v_angular])
                gripper_velocity = 0

                if sim_time - release_wait_start_simtime >= RELEASE_WAIT_TIME:
                    bananas_collected += 1
                    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current_target_body)
                    print(f"[{sim_time:7.2f}s]   {body_name} deposited! ({bananas_collected}/10)")
                    current_target_body = None
                    auto_state = "select"

        ## ------ Inverse kinematics + integration ------
        Jv_inv = inverse_jacobian([
            actual_roll,
            actual_pitch + HALF_PI,
            actual_d3,
            actual_theta4 + HALF_PI,
            actual_theta5 + FIVE_SIXTH_PI,
            actual_theta6
        ])
        joint_velocity = Jv_inv @ velocity

        roll_pos += dt * joint_velocity[0, 0]
        pitch_pos += dt * joint_velocity[1, 0]
        d3_pos += dt * joint_velocity[2, 0]
        theta4_pos += dt * joint_velocity[3, 0]
        theta5_pos += dt * joint_velocity[4, 0]
        theta6_pos += dt * joint_velocity[5, 0]
        gripper_pos += gripper_velocity

        # Smooth angle wrapping
        roll_pos = np.arctan2(np.sin(roll_pos), np.cos(roll_pos))
        pitch_pos = np.arctan2(np.sin(pitch_pos), np.cos(pitch_pos))
        theta4_pos = np.arctan2(np.sin(theta4_pos), np.cos(theta4_pos))
        theta5_pos = np.arctan2(np.sin(theta5_pos), np.cos(theta5_pos))
        theta6_pos = np.arctan2(np.sin(theta6_pos), np.cos(theta6_pos))

        # Joint limits
        roll_pos = np.clip(roll_pos, -np.pi/2, np.pi/2)
        pitch_pos = np.clip(pitch_pos, -np.pi/4, np.pi/2)
        d3_pos = np.clip(d3_pos, 0.2, 3.0)
        theta5_pos = max(theta5_pos, -1.7)
        gripper_pos = np.clip(gripper_pos, 0.0, 0.05)

        # Rate-limited control targets
        ctrl_desired[0] = roll_pos + roll_offset
        ctrl_desired[1] = pitch_pos
        ctrl_desired[2] = d3_pos
        ctrl_desired[3] = theta4_pos
        ctrl_desired[4] = theta5_pos
        ctrl_desired[5] = theta6_pos
        ctrl_desired[6] = gripper_pos
        ctrl_desired[7] = gripper_pos
        for i in range(8):
            a = aid_arr[i]
            prev = data.ctrl[a]
            d = ctrl_desired[i] - prev
            md = MAX_CTRL_DELTA[i]
            if d > md: d = md
            elif d < -md: d = -md
            data.ctrl[a] = prev + d

        # Step physics (identical mj_step call)
        mujoco.mj_step(model, data)

    ## ------ Summary ------
    wall_elapsed = time.perf_counter() - wall_start
    sim_elapsed = data.time
    bananas_in_box = sum(1 for bid in banana_body_ids if is_banana_in_box(model, data, bid))

    print(f"\n{'='*60}")
    print(f"  HEADLESS RUN COMPLETE (OPTIMIZED)")
    print(f"{'='*60}")
    print(f"  Ctrl skip:         {CTRL_SKIP}")
    print(f"  Bananas in box:    {bananas_in_box}/{len(banana_body_ids)}")
    print(f"  Total iterations:  {total_iterations:,}")
    print(f"  Sim time:          {sim_elapsed:.2f}s")
    print(f"  Wall time:         {wall_elapsed:.2f}s")
    print(f"  Speedup:           {sim_elapsed / wall_elapsed:.1f}x real-time")
    print(f"  Avg iter/sec:      {total_iterations / wall_elapsed:,.0f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
