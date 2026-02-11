"""
GIRAF Vibration Test - Minimal Simulation Skeleton
"""

import numpy as np
import mujoco
import mujoco.viewer
import os

def main():
    # Load MuJoCo model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "GIRAF_stiffness_test.xml")
    
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
    actuator_names = ['actuator_R1', 'actuator_R2', 'actuator_P3', 'actuator_R4', 
                     'actuator_R5', 'actuator_R6', 'actuator_left_grip', 'actuator_right_grip']
    actuator_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                   for name in actuator_names}
    
    # Initialize joint positions
    data.qpos[joint_ids['R1']] = 0.0
    data.qpos[joint_ids['R2']] = 0.0
    data.qpos[joint_ids['P3']] = 0.5
    data.qpos[joint_ids['R4']] = 0.0
    data.qpos[joint_ids['R5']] = 0.0
    data.qpos[joint_ids['R6']] = 0.0
    data.qpos[joint_ids['left_grip_joint']] = 0.0
    data.qpos[joint_ids['right_grip_joint']] = 0.0
    
    # Set actuator controls to match initial positions
    data.ctrl[actuator_ids['actuator_R1']] = 0.0
    data.ctrl[actuator_ids['actuator_R2']] = 0.0
    data.ctrl[actuator_ids['actuator_P3']] = 0.5
    data.ctrl[actuator_ids['actuator_R4']] = 0.0
    data.ctrl[actuator_ids['actuator_R5']] = 0.0
    data.ctrl[actuator_ids['actuator_R6']] = 0.0
    data.ctrl[actuator_ids['actuator_left_grip']] = 0.0
    data.ctrl[actuator_ids['actuator_right_grip']] = 0.0
    
    mujoco.mj_forward(model, data)
    
    print("\033[92mSIM: Ready!\033[0m")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = data.time
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()

if __name__ == "__main__":
    main()
