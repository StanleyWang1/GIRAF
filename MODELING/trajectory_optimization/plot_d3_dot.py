import sympy as sp
import numpy as np
import time
import sys
import os
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

# --- Import robot modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from CABLE_MANIP_CONTROL.square_traj import trajectory
from CABLE_MANIP_CONTROL.kinematic_model import num_forward_kinematics, num_jacobian

# --- Global State for Visualization ---
EE_xyz = np.zeros(3)
target_xyz = np.zeros(3)
EE_trace = []
d3_dot_trace = deque(maxlen=500)  # store the latest N points for smooth live plot
xyz_lock = threading.Lock()
visualizer_running = True

def run_control_loop():
    global EE_xyz, target_xyz, EE_trace, d3_dot_trace, visualizer_running

    joint_coords = [0, np.pi/2, 0.5, np.pi/2, 5*np.pi/6 - np.pi/4, 0]  
    dt = 0.0075

    for i in range(500):
        goal = trajectory[0]
        EE_pose = num_forward_kinematics(joint_coords)
        EE_pos = EE_pose[:3, 3].flatten()
        # EE_pos += np.random.normal(0, np.sqrt(0.001), size=3)

        # --- PI Controller ---
        if i == 0:
            integral_error = np.zeros(3)
        error = goal - EE_pos
        integral_error += error * dt
        Kp = 10.0
        Ki = 5.0
        P_velocity = Kp * error + Ki * integral_error
        P_velocity = np.clip(P_velocity, -0.5, 0.5)
        task_velocity = np.concatenate([P_velocity, [0, 0, 0]])
        J_pinv = np.linalg.pinv(num_jacobian(joint_coords))
        joint_velocity = J_pinv @ task_velocity
        joint_coords = joint_coords + dt * joint_velocity

        with xyz_lock:
            EE_xyz[:] = EE_pos
            target_xyz[:] = goal
            if i % 10 == 0:
                d3_dot_trace.append(joint_velocity[2])

        time.sleep(dt)

    for idx, traj_point in enumerate(trajectory):
        goal = traj_point
        # joint_coords[2] += np.random.normal(0, np.sqrt(0.0001))
        EE_pose = num_forward_kinematics(joint_coords)
        EE_pos = EE_pose[:3, 3].flatten()
        # EE_pos += np.random.normal(0, np.sqrt(0.001), size=3)

        P_velocity = 3.0 * (goal - EE_pos)
        if idx > 0:
            P_velocity += 2.0*(trajectory[idx] - trajectory[idx - 1]) / dt

        P_velocity = np.clip(P_velocity, -0.5, 0.5)
        task_velocity = np.concatenate([P_velocity, [0, 0, 0]])
        J_pinv = np.linalg.pinv(num_jacobian(joint_coords))
        joint_velocity = J_pinv @ task_velocity
        joint_coords = joint_coords + dt * joint_velocity

        with xyz_lock:
            EE_xyz[:] = EE_pos
            target_xyz[:] = goal
            if idx % 5 == 0:
                d3_dot_trace.append(joint_velocity[2])

        time.sleep(dt)

    time.sleep(2)
    visualizer_running = False

# --- Visualizer (Main Thread) ---
def visualizer_main():
    global EE_xyz, target_xyz, EE_trace, d3_dot_trace, visualizer_running

    plt.ion()
    fig = plt.figure(figsize=(12, 6))

    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    while visualizer_running:
        with xyz_lock:
            current_xyz = EE_xyz.copy()
            goal_xyz = target_xyz.copy()
            EE_trace.append(current_xyz)
            d3_vals = list(d3_dot_trace)

        # --- 3D subplot ---
        ax3d.cla()
        ax3d.set_xlim(-1, 1)
        ax3d.set_ylim(-1, 1)
        ax3d.set_zlim(-0.2, 1.2)
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        if len(EE_trace) > 1:
            trace_np = np.array(EE_trace)
            ax3d.plot(trace_np[:, 0], trace_np[:, 1], trace_np[:, 2], color='gray', alpha=0.3)

        ax3d.scatter(current_xyz[0], current_xyz[1], current_xyz[2], c='red', s=50)
        ax3d.scatter(goal_xyz[0], goal_xyz[1], goal_xyz[2], c='blue', s=50)
        ax3d.set_title("EE Pose (3D)")

                # Draw thick black line from base to EE
        base = np.array([0, 0, 0.15])
        ax3d.plot(
            [base[0], current_xyz[0]],
            [base[1], current_xyz[1]],
            [base[2], current_xyz[2]],
            color='black',
            linewidth=3
        )

        # --- 2D subplot ---
        ax2d.cla()
        ax2d.plot(d3_vals, color='purple')
        ax2d.set_ylim(-0.5, 0.5)
        ax2d.set_title("Joint 3 Velocity (d3_dot)")
        ax2d.set_xlabel("Time (frames)")
        ax2d.set_ylabel("rad/s")

        plt.tight_layout()
        plt.pause(0.01)

    plt.ioff()
    plt.show()

# --- Entry Point ---
if __name__ == "__main__":
    control_thread = threading.Thread(target=run_control_loop)
    control_thread.start()

    visualizer_main()
