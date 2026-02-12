"""
GIRAF Auto-Grasp Simulation
Loads a single banana at a random position and runs joystick teleop.
Banana spawns at random (x, y, yaw) with z = 0.25m each execution.
"""

import numpy as np
import mujoco
import mujoco.viewer
import threading
import time
import os
import sys
import cv2
from scipy.spatial.transform import Rotation

from kinematic_model import num_jacobian, num_forward_kinematics
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect

## ----------------------------------------------------------------------------------------------------
# Global Variables
## ----------------------------------------------------------------------------------------------------
joystick_data = {"LX":0, "LY":0, "RX":0, "RY":0, "LT":0, "RT":0, "AB":0, "BB":0, "XB":0, "YB":0, "LB":0, "RB":0, "MENULEFT":0, "MENURIGHT":0}
joystick_lock = threading.Lock()

velocity = np.zeros((6, 1))
velocity_lock = threading.Lock()

data_lock = threading.Lock()

running = True
running_lock = threading.Lock()

autonomous_mode = False
autonomous_lock = threading.Lock()

## ----------------------------------------------------------------------------------------------------
# Joystick Monitoring Thread
## ----------------------------------------------------------------------------------------------------
def joystick_monitor():
    global joystick_data, running

    js = joystick_connect()
    print("\033[93mSIM: Joystick Connected!\033[0m")

    while running:
        with joystick_lock:
            joystick_data = joystick_read(js)
        time.sleep(0.005)

    joystick_disconnect(js)
    print("\033[93mSIM: Joystick Disconnected!\033[0m")

## ----------------------------------------------------------------------------------------------------
# Camera Rendering Thread
## ----------------------------------------------------------------------------------------------------
def camera_render_thread(model, data):
    global running
    
    width = 1280
    height = 720
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    print(f"\033[96mSIM: Camera thread started ({width}x{height})\033[0m")
    
    loop_count = 0
    start_time = time.perf_counter()
    fps = 0.0
    
    try:
        while running:
            with data_lock:
                renderer.update_scene(data, camera="wrist_cam")
                rgb = renderer.render()
            
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            loop_count += 1
            if loop_count >= 30:
                elapsed = time.perf_counter() - start_time
                fps = loop_count / elapsed
                loop_count = 0
                start_time = time.perf_counter()
            
            cv2.putText(bgr, f"Wrist Camera - {fps:.1f} FPS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("GIRAF Wrist Camera", bgr)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            time.sleep(1/100)
            
    except Exception as e:
        print(f"\033[91mCamera thread error: {e}\033[0m")
    finally:
        cv2.destroyAllWindows()
        print("\033[96mSIM: Camera thread stopped\033[0m")

## ----------------------------------------------------------------------------------------------------
# Randomize Banana Pose
## ----------------------------------------------------------------------------------------------------
def randomize_banana(model, data, banana_joint_id):
    """
    Set banana to random position and orientation.
    - x: uniform [0.75, 1.25]
    - y: uniform [-0.25, 0.25]
    - z: 0.25m (fixed)
    - orientation: random yaw (rotation about z-axis)
    """
    x = np.random.uniform(0.75, 1.25)
    y = np.random.uniform(-0.25, 0.25)
    z = 0.1
    
    # Random orientation: random rotation about all axes for full random orientation
    yaw = np.random.uniform(0, 2 * np.pi)
    pitch = np.random.uniform(-np.pi/6, np.pi/6)  # Slight tilt variation
    roll = np.random.uniform(-np.pi/6, np.pi/6)
    
    # Convert euler angles to quaternion (MuJoCo uses w, x, y, z order)
    rot = Rotation.from_euler('zyx', [yaw, pitch, roll])
    quat_xyzw = rot.as_quat()  # scipy returns x, y, z, w
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    # freejoint qpos: [x, y, z, qw, qx, qy, qz]
    qpos_addr = model.jnt_qposadr[banana_joint_id]
    data.qpos[qpos_addr:qpos_addr+3] = [x, y, z]
    data.qpos[qpos_addr+3:qpos_addr+7] = quat_wxyz
    
    # Zero velocity
    qvel_addr = model.jnt_dofadr[banana_joint_id]
    data.qvel[qvel_addr:qvel_addr+6] = 0
    
    print(f"\033[93mBanana spawned at: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={np.degrees(yaw):.1f}°\033[0m")

## ----------------------------------------------------------------------------------------------------
# Main Simulation
## ----------------------------------------------------------------------------------------------------
def main():
    global joystick_data, velocity, running, autonomous_mode

    # Load MuJoCo model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "GIRAF_banana.xml")
    
    print(f"Loading model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print("\033[92mSIM: Model loaded successfully!\033[0m")
    print(f"  Joints: {model.njnt}")
    print(f"  Actuators: {model.nu}")
    
    # Joint name to index mapping
    joint_names = ['R1', 'R2', 'P3', 'R4', 'R5', 'R6', 'left_grip_joint', 'right_grip_joint']
    joint_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}
    
    # Actuator name to index mapping  
    actuator_names = ['actuator_R1', 'actuator_R2', 'actuator_P3', 'actuator_R4', 'actuator_R5', 'actuator_R6', 'actuator_left_grip', 'actuator_right_grip']
    actuator_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names}
    
    # Banana joint
    banana_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "banana_joint")
    
    def inverse_jacobian(joint_coords):
        J = num_jacobian(joint_coords)
        J_inv = np.linalg.pinv(J)
        return J_inv
    
    # Initialize joint positions
    roll_pos = 0.0
    roll_offset = 0.0
    pitch_pos = 0.0
    d3_pos = 0.25
    theta4_pos = 0.0
    theta5_pos = 0.0
    theta6_pos = 0.0
    gripper_pos = 0.0
    
    # Set initial robot state
    data.qpos[joint_ids['R1']] = roll_pos
    data.qpos[joint_ids['R2']] = pitch_pos
    data.qpos[joint_ids['P3']] = d3_pos
    data.qpos[joint_ids['R4']] = theta4_pos
    data.qpos[joint_ids['R5']] = theta5_pos
    data.qpos[joint_ids['R6']] = theta6_pos
    data.qpos[joint_ids['left_grip_joint']] = gripper_pos
    data.qpos[joint_ids['right_grip_joint']] = gripper_pos
    
    # Randomize banana position
    randomize_banana(model, data, banana_joint_id)
    
    mujoco.mj_forward(model, data)
    
    print("\033[92mSIM: Ready! Controls:\033[0m")
    print("  LB + RB: Enable control (teleop)")
    print("  Left stick: XY translation")
    print("  Right stick: Wrist rotation")
    print("  LT/RT: Z translation")
    print("  A/B: Close/Open gripper")
    print("  Menu Left: Roll offset +")
    print("  Y: Toggle autonomous mode")
    print("  X: Exit")
    print("\n" * 5)
    
    # Start joystick thread
    joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
    joystick_thread.start()
    
    # Start camera rendering thread
    camera_thread = threading.Thread(target=camera_render_thread, args=(model, data), daemon=True)
    camera_thread.start()
    
    # Performance monitoring
    loop_count = 0
    start_time = time.perf_counter()
    loop_hz = 0.0
    last_print_time = time.perf_counter()
    PRINT_INTERVAL = 0.1  # 10 Hz status display
    
    # Autonomous mode PD gains
    AUTO_KP = 5.0
    AUTO_KP_ROT = 2.0  # Lower gain for orientation
    RELEASE_WAIT_TIME = 0.5             # Seconds to hold after opening gripper
    AUTO_KD = 0.1
    AUTO_MAX_VEL = 0.45
    AUTO_MAX_OMEGA = 1.0  # Max angular velocity (rad/s)
    AUTO_HOVER_HEIGHT = 0.1   # 0.1m above banana (approach phase)
    AUTO_GRASP_HEIGHT = 0.025
    AUTO_GRASP_HEIGHT_DEFAULT = 0.025
    AUTO_GRASP_HEIGHT_RETRY = 0.02  # 0.03m above banana (descend phase)
    AUTO_POS_TOL = 0.015      # Position tolerance (m)
    AUTO_ROT_TOL = 0.3       # Orientation tolerance (rad)
    BOX_CENTER = np.array([1.0, 0.5])  # XY center of collection bin
    BOX_DROP_HEIGHT = 0.35              # Drop from 0.35m above ground
    SAFE_LIFT_HEIGHT = 0.25             # Lift straight up to this Z before translating (above box walls at 0.16m)
    STABILIZE_TIME = 0.5                # Seconds to hold after gripper closes
    auto_state = "approach"    # "approach" -> "descend" -> "grasp" -> "stabilize" -> "lift_clear" -> "transit" -> "drop" -> "done"
    stabilize_start = 0.0
    yb_prev = 0  # For Y button edge detection
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and running:
            # Unpack joystick data
            with joystick_lock:
                LX = joystick_data["LX"]
                LY = joystick_data["LY"]
                RX = joystick_data["RX"]
                RY = joystick_data["RY"]
                LT = joystick_data["LT"]
                RT = joystick_data["RT"]
                AB = joystick_data["AB"]
                BB = joystick_data["BB"]
                XB = joystick_data["XB"]
                YB = joystick_data["YB"]
                LB = joystick_data["LB"]
                RB = joystick_data["RB"]
                MENULEFT = joystick_data["MENULEFT"]
            
            if XB:  # Exit
                with running_lock:
                    running = False
                break
            
            # Y button toggle: autonomous <-> teleop
            if YB and not yb_prev:
                with autonomous_lock:
                    autonomous_mode = not autonomous_mode
                if autonomous_mode:
                    auto_state = "approach"
                    print("\n\033[95m[AUTO] Autonomous mode ENABLED - approaching banana\033[0m")
                else:
                    print("\n\033[93m[AUTO] Autonomous mode DISABLED - teleop active\033[0m")
            yb_prev = YB
            
            # Read actual joint positions from simulation
            actual_roll = data.qpos[joint_ids['R1']]
            actual_pitch = data.qpos[joint_ids['R2']]
            actual_d3 = data.qpos[joint_ids['P3']]
            actual_theta4 = data.qpos[joint_ids['R4']]
            actual_theta5 = data.qpos[joint_ids['R5']]
            actual_theta6 = data.qpos[joint_ids['R6']]

            FK_mat = num_forward_kinematics([
                actual_roll, 
                actual_pitch + np.pi/2, 
                actual_d3,
                actual_theta4 + np.pi/2, 
                actual_theta5 + 5*np.pi/6, 
                actual_theta6
            ])
            
            # Get banana position
            banana_qpos_addr = model.jnt_qposadr[banana_joint_id]
            banana_pos = data.qpos[banana_qpos_addr:banana_qpos_addr+3].copy()
            
            if autonomous_mode:
                ## ------ AUTONOMOUS MODE STATE MACHINE ------
                current_pos = FK_mat[:3, 3]
                current_rot = FK_mat[:3, :3]
                
                # --- Compute desired orientation (shared across states) ---
                banana_quat_wxyz = data.qpos[banana_qpos_addr+3:banana_qpos_addr+7].copy()
                banana_quat_xyzw = np.array([banana_quat_wxyz[1], banana_quat_wxyz[2], banana_quat_wxyz[3], banana_quat_wxyz[0]])
                banana_rot = Rotation.from_quat(banana_quat_xyzw)
                banana_R = banana_rot.as_matrix()
                
                banana_long_axis = banana_R[:, 0]
                banana_xy = banana_long_axis[:2].copy()
                banana_xy_norm = np.linalg.norm(banana_xy)
                if banana_xy_norm > 1e-6:
                    banana_xy /= banana_xy_norm
                else:
                    banana_xy = np.array([1.0, 0.0])
                
                # Resolve 180° ambiguity: pick direction closest to current gripper y-axis
                current_y_ee = current_rot[:2, 1]
                if np.dot(banana_xy, current_y_ee) < 0:
                    banana_xy = -banana_xy
                
                y_ee = np.array([banana_xy[0], banana_xy[1], 0.0])
                z_ee = np.array([0.0, 0.0, -1.0])
                x_ee = np.cross(y_ee, z_ee)
                x_ee /= np.linalg.norm(x_ee)
                R_desired = np.column_stack([x_ee, y_ee, z_ee])
                
                # Orientation error
                R_error = R_desired @ current_rot.T
                rot_error = Rotation.from_matrix(R_error)
                omega_error = rot_error.as_rotvec()
                rot_error_mag = np.linalg.norm(omega_error)
                v_angular = (AUTO_KP_ROT * omega_error).reshape(3, 1)
                omega_norm = np.linalg.norm(v_angular)
                if omega_norm > AUTO_MAX_OMEGA:
                    v_angular = v_angular * (AUTO_MAX_OMEGA / omega_norm)
                
                if auto_state == "approach":
                    # Move to 0.1m above banana, align orientation, open gripper
                    target_pos = banana_pos.copy()
                    target_pos[2] += AUTO_HOVER_HEIGHT
                    
                    pos_error = target_pos - current_pos
                    pos_error_mag = np.linalg.norm(pos_error)
                    v_linear = AUTO_KP * pos_error.reshape(3, 1)
                    v_norm = np.linalg.norm(v_linear)
                    if v_norm > AUTO_MAX_VEL:
                        v_linear = v_linear * (AUTO_MAX_VEL / v_norm)
                    
                    with velocity_lock:
                        velocity = np.vstack([v_linear, v_angular])
                    
                    # Open gripper
                    if gripper_pos < 0.045:
                        gripper_velocity = 0.002
                    else:
                        gripper_velocity = 0
                    
                    # Transition when close enough in position AND orientation
                    if pos_error_mag < AUTO_POS_TOL and rot_error_mag < AUTO_ROT_TOL:
                        auto_state = "descend"
                        print("\n\033[95m[AUTO] Aligned above banana - descending\033[0m")
                
                elif auto_state == "descend":
                    # Move down to 0.03m above banana (Z only), maintain orientation
                    target_pos = banana_pos.copy()
                    target_pos[2] += AUTO_GRASP_HEIGHT
                    
                    pos_error = target_pos - current_pos
                    pos_error_mag = np.linalg.norm(pos_error)
                    
                    # Only command Z velocity, hold XY with PD
                    v_linear = AUTO_KP * pos_error.reshape(3, 1)
                    # Slower descent
                    v_norm = np.linalg.norm(v_linear)
                    if v_norm > AUTO_MAX_VEL * 0.5:
                        v_linear = v_linear * (AUTO_MAX_VEL * 0.5 / v_norm)
                    
                    with velocity_lock:
                        velocity = np.vstack([v_linear, v_angular])
                    
                    # Keep gripper open
                    gripper_velocity = 0
                    
                    # Transition when close to grasp height
                    if pos_error_mag < AUTO_POS_TOL:
                        auto_state = "grasp"
                        print("\n\033[95m[AUTO] At grasp height - closing gripper\033[0m")
                
                elif auto_state == "grasp":
                    # Hold position, close gripper
                    target_pos = banana_pos.copy()
                    target_pos[2] += AUTO_GRASP_HEIGHT
                    
                    pos_error = target_pos - current_pos
                    v_linear = AUTO_KP * pos_error.reshape(3, 1)
                    v_norm = np.linalg.norm(v_linear)
                    if v_norm > AUTO_MAX_VEL * 0.3:
                        v_linear = v_linear * (AUTO_MAX_VEL * 0.3 / v_norm)
                    
                    with velocity_lock:
                        velocity = np.vstack([v_linear, v_angular])
                    
                    # Close gripper
                    if gripper_pos > 0.005:
                        gripper_velocity = -0.002
                    else:
                        gripper_velocity = 0
                        auto_state = "stabilize"
                        stabilize_start = time.perf_counter()
                        print("\n\033[95m[AUTO] Grasp complete - stabilizing\033[0m")
                
                elif auto_state == "stabilize":
                    # Hold position for STABILIZE_TIME seconds to let grasp settle
                    target_pos = banana_pos.copy()
                    target_pos[2] += AUTO_GRASP_HEIGHT
                    
                    pos_error = target_pos - current_pos
                    v_linear = AUTO_KP * pos_error.reshape(3, 1)
                    v_norm = np.linalg.norm(v_linear)
                    if v_norm > AUTO_MAX_VEL * 0.3:
                        v_linear = v_linear * (AUTO_MAX_VEL * 0.3 / v_norm)
                    
                    with velocity_lock:
                        velocity = np.vstack([v_linear, v_angular])
                    
                    gripper_velocity = 0  # Keep closed
                    
                    if time.perf_counter() - stabilize_start >= STABILIZE_TIME:
                        auto_state = "lift_clear"
                        print("\n\033[95m[AUTO] Lifting clear of obstacles\033[0m")
                
                elif auto_state == "lift_clear":
                    # Lift straight up to safe height (above box walls) before translating
                    target_pos = np.array([current_pos[0], current_pos[1], SAFE_LIFT_HEIGHT])
                    
                    pos_error = target_pos - current_pos
                    pos_error_mag = np.linalg.norm(pos_error)
                    v_linear = AUTO_KP * pos_error.reshape(3, 1)
                    v_norm = np.linalg.norm(v_linear)
                    if v_norm > AUTO_MAX_VEL:
                        v_linear = v_linear * (AUTO_MAX_VEL / v_norm)
                    
                    with velocity_lock:
                        velocity = np.vstack([v_linear, v_angular])
                    
                    gripper_velocity = 0  # Keep closed
                    
                    if pos_error_mag < AUTO_POS_TOL:
                        # Check if banana is still grasped
                        banana_check = data.qpos[banana_qpos_addr:banana_qpos_addr+3].copy()
                        grasp_dist = np.linalg.norm(banana_check - current_pos)
                        if grasp_dist > 0.08:
                            print(f"\n\033[91m[AUTO] Lost banana during lift (dist={grasp_dist:.3f}m) - retrying at lower height\033[0m")
                            AUTO_GRASP_HEIGHT = AUTO_GRASP_HEIGHT_RETRY
                            auto_state = "approach"
                        else:
                            if AUTO_GRASP_HEIGHT != AUTO_GRASP_HEIGHT_DEFAULT:
                                print(f"\n\033[92m[AUTO] Retry grasp succeeded - restoring default height\033[0m")
                                AUTO_GRASP_HEIGHT = AUTO_GRASP_HEIGHT_DEFAULT
                            auto_state = "transit"
                            print("\n\033[95m[AUTO] Clear - moving to box\033[0m")
                
                elif auto_state == "transit":
                    # Move to above box center at drop height, keep gripper closed
                    target_pos = np.array([BOX_CENTER[0], BOX_CENTER[1], BOX_DROP_HEIGHT])
                    
                    pos_error = target_pos - current_pos
                    pos_error_mag = np.linalg.norm(pos_error)
                    v_linear = AUTO_KP * pos_error.reshape(3, 1)
                    v_norm = np.linalg.norm(v_linear)
                    if v_norm > AUTO_MAX_VEL:
                        v_linear = v_linear * (AUTO_MAX_VEL / v_norm)
                    
                    with velocity_lock:
                        velocity = np.vstack([v_linear, v_angular])
                    
                    gripper_velocity = 0  # Keep closed
                    
                    if pos_error_mag < AUTO_POS_TOL:
                        auto_state = "drop"
                        print("\n\033[95m[AUTO] Above box - releasing\033[0m")
                
                elif auto_state == "drop":
                    # Hold position, open gripper
                    target_pos = np.array([BOX_CENTER[0], BOX_CENTER[1], BOX_DROP_HEIGHT])
                    
                    pos_error = target_pos - current_pos
                    v_linear = AUTO_KP * pos_error.reshape(3, 1)
                    v_norm = np.linalg.norm(v_linear)
                    if v_norm > AUTO_MAX_VEL * 0.3:
                        v_linear = v_linear * (AUTO_MAX_VEL * 0.3 / v_norm)
                    
                    with velocity_lock:
                        velocity = np.vstack([v_linear, v_angular])
                    
                    # Open gripper
                    if gripper_pos < 0.045:
                        gripper_velocity = 0.002
                    else:
                        gripper_velocity = 0
                        auto_state = "release_wait"
                        release_wait_start = time.perf_counter()
                        print("\n\033[95m[AUTO] Gripper open - waiting for banana to drop\033[0m")
                
                elif auto_state == "release_wait":
                    # Hold position above box while banana falls clear
                    target_pos = np.array([BOX_CENTER[0], BOX_CENTER[1], BOX_DROP_HEIGHT])
                    pos_error = target_pos - current_pos
                    v_linear = AUTO_KP * pos_error.reshape(3, 1)
                    v_norm = np.linalg.norm(v_linear)
                    if v_norm > AUTO_MAX_VEL * 0.3:
                        v_linear = v_linear * (AUTO_MAX_VEL * 0.3 / v_norm)
                    
                    with velocity_lock:
                        velocity = np.vstack([v_linear, v_angular])
                    gripper_velocity = 0
                    
                    if time.perf_counter() - release_wait_start >= RELEASE_WAIT_TIME:
                        auto_state = "done"
                        print("\n\033[92m[AUTO] Banana deposited! Returning to teleop.\033[0m")
                        with autonomous_lock:
                            autonomous_mode = False
                
                elif auto_state == "done":
                    # Idle, zero velocity
                    with velocity_lock:
                        velocity = np.zeros((6, 1))
                    gripper_velocity = 0
            
            else:
                ## ------ TELEOP MODE ------
                # Safety interlock
                if LB and RB:
                    with velocity_lock:
                        velocity[0] = 0.25 * LY
                        velocity[1] = 0.25 * -LX
                        velocity[4] = -0.5 * RY
                        velocity[5] = -0.5 * RX
                    
                    if RT and not LT:
                        with velocity_lock:
                            velocity[2] = 0.25 * RT
                    elif LT and not RT and (pitch_pos > 0):
                        with velocity_lock:
                            velocity[2] = -0.25 * LT
                    else:
                        with velocity_lock:
                            velocity[2] = 0
                    
                    # Gripper control
                    if AB and not BB:
                        gripper_velocity = 0.001
                    elif BB and not AB:
                        gripper_velocity = -0.001
                    else:
                        gripper_velocity = 0
                    
                    # Roll offset
                    if MENULEFT:
                        roll_offset += 0.0025
                else:
                    with velocity_lock:
                        velocity = np.zeros((6, 1))
                    gripper_velocity = 0
            
            # Compute joint velocities via inverse Jacobian
            velocity_world = velocity
            
            Jv_inv = inverse_jacobian([
                actual_roll, 
                actual_pitch + np.pi/2, 
                actual_d3,
                actual_theta4 + np.pi/2, 
                actual_theta5 + 5*np.pi/6, 
                actual_theta6
            ])
            joint_velocity = Jv_inv @ velocity_world
            
            # Integrate joint positions
            dt = 0.0025
            roll_pos += dt * joint_velocity[0, 0]
            pitch_pos += dt * joint_velocity[1, 0]
            d3_pos += dt * joint_velocity[2, 0]
            theta4_pos += dt * joint_velocity[3, 0]
            theta5_pos += dt * joint_velocity[4, 0]
            theta6_pos += dt * joint_velocity[5, 0]
            gripper_pos += gripper_velocity
            
            # Smooth angle wrapping to [-pi, pi] for revolute joints
            roll_pos = np.arctan2(np.sin(roll_pos), np.cos(roll_pos))
            pitch_pos = np.arctan2(np.sin(pitch_pos), np.cos(pitch_pos))
            theta4_pos = np.arctan2(np.sin(theta4_pos), np.cos(theta4_pos))
            theta5_pos = np.arctan2(np.sin(theta5_pos), np.cos(theta5_pos))
            theta6_pos = np.arctan2(np.sin(theta6_pos), np.cos(theta6_pos))
            
            # Apply joint limits
            roll_pos = np.clip(roll_pos, -np.pi/2, np.pi/2)
            pitch_pos = np.clip(pitch_pos, -np.pi/4, np.pi/2)
            d3_pos = np.clip(d3_pos, 0.2, 3.0)
            theta5_pos = max(theta5_pos, -1.7)
            gripper_pos = np.clip(gripper_pos, 0.0, 0.05)
            
            # Compute desired control targets
            ctrl_desired = np.array([
                roll_pos + roll_offset,  # R1
                pitch_pos,               # R2
                d3_pos,                  # P3
                theta4_pos,              # R4
                theta5_pos,              # R5
                theta6_pos,              # R6
                gripper_pos,             # left grip
                gripper_pos,             # right grip
            ])
            ctrl_keys = ['actuator_R1', 'actuator_R2', 'actuator_P3', 'actuator_R4',
                         'actuator_R5', 'actuator_R6', 'actuator_left_grip', 'actuator_right_grip']
            
            # Rate-limit: clamp max change per step to prevent jumps (e.g. ±pi wrap)
            MAX_CTRL_DELTA = 0.05  # max change per control step (rad or m)
            for i, key in enumerate(ctrl_keys):
                aid = actuator_ids[key]
                prev = data.ctrl[aid]
                delta = ctrl_desired[i] - prev
                delta = np.clip(delta, -MAX_CTRL_DELTA, MAX_CTRL_DELTA)
                data.ctrl[aid] = prev + delta
            
            # Step simulation
            with data_lock:
                mujoco.mj_step(model, data)
                viewer.sync()
            
            # Performance monitoring
            loop_count += 1
            if loop_count >= 500:
                elapsed = time.perf_counter() - start_time
                loop_hz = loop_count / elapsed
                loop_count = 0
                start_time = time.perf_counter()
            
            # Status display (throttled to 10 Hz)
            now = time.perf_counter()
            if now - last_print_time >= PRINT_INTERVAL:
                last_print_time = now
                ee_pos = FK_mat[:3, 3]
                dist_to_banana = np.linalg.norm(ee_pos - banana_pos)
                mode_str = f"\033[95mAUTO:{auto_state}\033[0m" if autonomous_mode else "\033[93mTELEOP\033[0m"
                
                # Save cursor, move to bottom, print, restore cursor
                sys.stdout.write(f"\033[s\033[999;1H\033[2K"  # save cursor, move to last row, clear line
                               f"  {mode_str} | Loop: {loop_hz:.0f} Hz | "
                               f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] | "
                               f"Banana: [{banana_pos[0]:.3f}, {banana_pos[1]:.3f}, {banana_pos[2]:.3f}] | "
                               f"Dist: {dist_to_banana:.3f}m | Grip: {gripper_pos:.4f}"
                               f"\033[u")  # restore cursor
                sys.stdout.flush()
            
            time.sleep(0.001)
    
    print("\n\033[93mSIM: Simulation ended\033[0m")

if __name__ == "__main__":
    main()
