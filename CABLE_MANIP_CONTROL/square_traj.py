import numpy as np

def generate_square_xy_trajectory():
    z = -0.2  # Constant height
    offset = np.array([-0.00757, -0.00202, -0.00403])  # Offset to shift the square in space
    # 0.00125, 0.00005, 0.0016

    # Define square corners in clockwise or counterclockwise order
    p1 = np.array([-0.25,  -0.05, z]) + offset
    p2 = np.array([-0.15,  -0.05, z]) + offset
    p3 = np.array([-0.15,   0.05, z]) + offset
    p4 = np.array([-0.25,   0.05, z]) + offset
    # z = 0.25  # Constant height

    # z = 0.25
    # # # Define square corners in clockwise or counterclockwise order
    # p1 = np.array([0.5,  -0.25, z])
    # p2 = np.array([1.0,  -0.25, z])
    # p3 = np.array([1.0,   0.25, z])
    # p4 = np.array([0.5,   0.25, z])

    # Create linear segments
    traj1 = np.linspace(p1, p2, 600)
    traj2 = np.linspace(p2, p3, 600)
    traj3 = np.linspace(p3, p4, 600)
    traj4 = np.linspace(p4, p1, 600)

    # Combine all into one trajectory
    trajectory = np.vstack([traj1, traj2, traj3, traj4])
    return trajectory

# Example usage
trajectory = generate_square_xy_trajectory()
# print(trajectory.shape)  # (1600, 3)
