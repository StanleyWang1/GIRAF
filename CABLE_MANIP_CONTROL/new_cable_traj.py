import numpy as np

def generate_custom_trajectory():
    # Define key points
    p0 = np.array([-0.25,  0.00, -0.1])
    p0A = np.array([-0.25,  -0.2, -0.1])
    p1 = np.array([-0.25, -0.05, -0.1])
    p2 = np.array([-0.1, -0.05, -0.085])
    p3 = np.array([-0.1,  0.05, -0.085])
    p4 = np.array([-0.175,  0.02, -0.1])
    p5 = np.array([-0.25, -0.1, -0.2])
    p6 = np.array([-0.15, -0.1, -0.1])
    # 400-point constant segment
    traj0 = np.tile(p0, (400, 1))

    # 400-point linear interpolations
    traj1 = np.linspace(p0, p0A, 400)
    traj1A = np.linspace(p0A, p1, 400)
    traj2 = np.linspace(p1, p2, 400)
    traj3 = np.linspace(p2, p3, 400)
    traj4 = np.linspace(p3, p4, 400)
    traj5 = np.linspace(p4, p5, 400)
    traj6 = np.linspace(p5, p6, 400)

    # Combine all segments
    trajectory = np.vstack([traj0, traj1, traj1A, traj2, traj3, traj4, traj5, traj6])
    return trajectory

# Example usage
trajectory = generate_custom_trajectory()
# print(trajectory.shape)  # (2400, 3)
