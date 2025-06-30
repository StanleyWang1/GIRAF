import numpy as np

def generate_custom_trajectory():
    # Define key points
    p0 = np.array([-0.25,  0.00, -0.1])
    p1 = np.array([-0.25, -0.05, -0.1])
    p2 = np.array([-0.1, -0.05, -0.075])
    p3 = np.array([-0.1,  0.05, -0.075])
    p4 = np.array([-0.17,  0.02, -0.1])
    p5 = np.array([-0.17, -0.1, -0.1])

    # 400-point constant segment
    traj0 = np.tile(p0, (400, 1))

    # 400-point linear interpolations
    traj1 = np.linspace(p0, p1, 400)
    traj2 = np.linspace(p1, p2, 400)
    traj3 = np.linspace(p2, p3, 400)
    traj4 = np.linspace(p3, p4, 400)
    traj5 = np.linspace(p4, p5, 400)

    # Combine all segments
    trajectory = np.vstack([traj0, traj1, traj2, traj3, traj4, traj5])
    return trajectory

# Example usage
trajectory = generate_custom_trajectory()
# print(trajectory.shape)  # (2400, 3)
