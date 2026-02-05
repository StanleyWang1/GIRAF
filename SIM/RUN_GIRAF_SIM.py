"""
GIRAF Manipulator Simulation with Joystick Teleop
Simplified simulation version of RUN_task_space_teleop.py
"""

import numpy as np
import mujoco
import mujoco.viewer
import threading
import time
import os
import sys

from kinematic_model import num_jacobian
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect

## ----------------------------------------------------------------------------------------------------
# Global Variables
## ----------------------------------------------------------------------------------------------------
joystick_data = {"LX":0, "LY":0, "RX":0, "RY":0, "LT":0, "RT":0, "AB":0, "BB":0, "XB":0, "LB":0, "RB":0, "MENULEFT":0, "MENURIGHT":0}
joystick_lock = threading.Lock()

velocity = np.zeros((6, 1))
velocity_lock = threading.Lock()

running = True
running_lock = threading.Lock()

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
# Main Simulation
## ----------------------------------------------------------------------------------------------------
def main():
    global joystick_data, velocity, running

    # Load MuJoCo model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "GIRAF.xml")
    
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
    
    def inverse_jacobian(joint_coords):
        """Compute inverse Jacobian using kinematic model"""
        J = num_jacobian(joint_coords)
        J_inv = np.linalg.pinv(J)
        return J_inv
    
    # Initialize joint positions (matching hardware teleop initial values)
    roll_pos = 0.0
    roll_offset = 0.0
    pitch_pos = 0.0
    d3_pos = 0.5  # Start at 0.5m extension (simplified from hardware's (55+255+80)/1000)
    theta4_pos = 0.0
    theta5_pos = 0.0
    theta6_pos = 0.0
    gripper_pos = 0.0  # 0 = open, 0.05 = closed
    
    # Set initial state
    data.qpos[joint_ids['R1']] = roll_pos
    data.qpos[joint_ids['R2']] = pitch_pos
    data.qpos[joint_ids['P3']] = d3_pos
    data.qpos[joint_ids['R4']] = theta4_pos
    data.qpos[joint_ids['R5']] = theta5_pos
    data.qpos[joint_ids['R6']] = theta6_pos
    data.qpos[joint_ids['left_grip_joint']] = gripper_pos
    data.qpos[joint_ids['right_grip_joint']] = gripper_pos
    
    mujoco.mj_forward(model, data)
    
    print("\033[92mSIM: Ready! Controls:\033[0m")
    print("  LB + RB: Enable control")
    print("  Left stick: XY translation")
    print("  Right stick: Wrist rotation")
    print("  LT/RT: Z translation")
    print("  A/B: Close/Open gripper")
    print("  Menu Left/Right: Roll offset")
    print("  X: Exit")
    
    # Start joystick thread
    joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
    joystick_thread.start()
    
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
                LB = joystick_data["LB"]
                RB = joystick_data["RB"]
                MENULEFT = joystick_data["MENULEFT"]
                MENURIGHT = joystick_data["MENURIGHT"]
            
            if XB:  # Exit
                with running_lock:
                    running = False
                break
            
            # Safety interlock
            if LB and RB:
                with velocity_lock:
                    velocity[0] = -0.25 * LY  # X velocity
                    velocity[1] = -0.25 * LX  # Y velocity
                    velocity[4] = 0.5 * RY    # WY angular velocity
                    velocity[5] = -0.5 * RX   # WZ angular velocity
                
                if RT and not LT:  # Z up
                    with velocity_lock:
                        velocity[2] = 0.1 * RT
                elif LT and not RT and (pitch_pos > 0):  # Z down
                    with velocity_lock:
                        velocity[2] = -0.1 * LT
                else:
                    with velocity_lock:
                        velocity[2] = 0
                
                # Gripper control
                if AB and not BB:  # Close
                    gripper_velocity = 0.001  # m/s
                elif BB and not AB:  # Open
                    gripper_velocity = -0.001
                else:
                    gripper_velocity = 0
                
                # Roll offset
                if MENULEFT and not MENURIGHT:
                    roll_offset += 0.0025
                elif MENURIGHT and not MENULEFT:
                    roll_offset -= 0.0025
            else:
                with velocity_lock:
                    velocity = np.zeros((6, 1))
                gripper_velocity = 0
            
            print(velocity)
            # Compute inverse Jacobian and joint velocities
            # Note: Adding offsets to match kinematic model frame conventions
            Jv_inv = inverse_jacobian([
                roll_pos, 
                pitch_pos + np.pi/2, 
                d3_pos,
                theta4_pos + np.pi/2, 
                theta5_pos + 5*np.pi/6, 
                theta6_pos
            ])
            joint_velocity = Jv_inv @ velocity
            
            # Integrate joint positions
            dt = 0.0075
            roll_pos += dt * joint_velocity[0, 0]
            pitch_pos += dt * joint_velocity[1, 0]
            d3_pos += dt * joint_velocity[2, 0]
            theta4_pos += dt * joint_velocity[3, 0]
            theta5_pos += dt * joint_velocity[4, 0]
            theta6_pos += dt * joint_velocity[5, 0]
            gripper_pos += gripper_velocity * 10  # Scale for responsiveness
            
            # Apply joint limits
            roll_pos = np.clip(roll_pos, -np.pi/2, np.pi/2)
            pitch_pos = np.clip(pitch_pos, 0, np.pi/2)
            d3_pos = np.clip(d3_pos, 0.2, 3.0)
            theta5_pos = max(theta5_pos, -1.7)  # Wrist pitch limit
            gripper_pos = np.clip(gripper_pos, 0.0, 0.05)
            
            # Set control targets (position control via actuators)
            data.ctrl[actuator_ids['actuator_R1']] = roll_pos + roll_offset
            data.ctrl[actuator_ids['actuator_R2']] = pitch_pos
            data.ctrl[actuator_ids['actuator_P3']] = d3_pos
            data.ctrl[actuator_ids['actuator_R4']] = theta4_pos
            data.ctrl[actuator_ids['actuator_R5']] = theta5_pos
            data.ctrl[actuator_ids['actuator_R6']] = theta6_pos
            data.ctrl[actuator_ids['actuator_left_grip']] = gripper_pos
            data.ctrl[actuator_ids['actuator_right_grip']] = gripper_pos
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Match hardware loop rate
            time.sleep(0.005)
    
    print("\033[93mSIM: Simulation ended\033[0m")

if __name__ == "__main__":
    main()
