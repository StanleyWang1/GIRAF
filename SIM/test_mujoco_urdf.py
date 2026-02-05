"""
Test script to load and visualize the cable manipulator in MuJoCo
"""

import mujoco
import mujoco.viewer
import numpy as np
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Use the MJCF XML file instead (has lighting and ground)
xml_path = os.path.join(script_dir, "./models/GIRAF.xml")

print(f"Loading model from: {xml_path}")

# Load the MJCF model
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    print("✓ Model loaded successfully!")
    
    # Print model information
    print(f"\nModel Info:")
    print(f"  Number of bodies: {model.nbody}")
    print(f"  Number of joints: {model.njnt}")
    print(f"  Number of DOFs: {model.nv}")
    print(f"  Number of actuators: {model.nu}")
    
    # Print joint information
    print(f"\nJoint Information:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]
        type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
        print(f"  Joint {i}: {joint_name} (type: {type_names.get(joint_type, 'unknown')})")
    
    # Create data structure
    data = mujoco.MjData(model)
    
    # Set initial joint positions (home configuration)
    # Adjust these based on your robot's configuration
    if model.nq >= 6:
        data.qpos[0] = 0.0      # joint1 (theta1)
        data.qpos[1] = 0.0      # joint2 (theta2)
        data.qpos[2] = 0.5      # joint3 (d3) - prismatic extension
        data.qpos[3] = 0.0      # joint4 (theta4)
        data.qpos[4] = 0.0      # joint5 (theta5)
        data.qpos[5] = 0.0      # joint6 (theta6)
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    print(f"\nInitial Configuration:")
    print(f"  q = {data.qpos[:model.nq]}")
    
    # Get end effector position (last body)
    ee_body_id = model.nbody - 1
    ee_pos = data.xpos[ee_body_id]
    print(f"  End effector position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
    
    print("\nLaunching interactive viewer...")
    print("Controls:")
    print("  - Right-click drag to rotate view")
    print("  - Left-click drag to move view")
    print("  - Scroll to zoom")
    print("  - Double-click body to track it")
    print("  - Ctrl+Right-click joint to perturb")
    print("  - Press SPACE to pause/resume")
    print("  - Press TAB to show/hide controls UI")
    print("  - Press ESC to exit")
    
    # Launch interactive viewer with control
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simple animation loop
        t = 0
        while viewer.is_running():
            # Set control targets (position control via actuators)
            if model.nu >= 6:
                # data.ctrl[0] = 0.3 * np.sin(t * 0.5)           # joint1
                # data.ctrl[1] = 0.3 * np.sin(t * 0.7)           # joint2
                # data.ctrl[2] = 0.8 + 0.3 * np.sin(t * 0.3)     # joint3 (prismatic)
                # data.ctrl[3] = 0.2 * np.sin(t * 0.9)           # joint4
                # data.ctrl[4] = 0.2 * np.sin(t * 1.1)           # joint5
                # data.ctrl[5] = 0.2 * np.sin(t * 1.3)           # joint6
                pass
            # Step physics simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Small time increment
            t += 0.01
            
            # Slow down the loop to real-time
            import time
            time.sleep(0.01)

except Exception as e:
    print(f"✗ Error loading URDF: {e}")
    import traceback
    traceback.print_exc()
