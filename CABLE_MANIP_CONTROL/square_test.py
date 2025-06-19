import numpy as np

def generate_square_trajectory():
    # Define points
    p1 = np.array([-0.2, -0.2, 0.1])
    p2 = np.array([-0.2,  0.2, 0.1])
    p3 = np.array([ 0.2,  0.2, 0.1])
    p4 = np.array([ 0.2, -0.2, 0.1])
    p5 = np.array([-0.2, -0.2, 0.1])

    # Repeat each point for 500 steps
    traj1 = np.tile(p1, (500, 1))
    traj2 = np.tile(p2, (500, 1))
    traj3 = np.tile(p3, (500, 1))
    traj4 = np.tile(p4, (500, 1))
    traj5 = np.tile(p5, (500, 1))

    # Stack the trajectory
    trajectory = np.vstack([traj1, traj2, traj3, traj4, traj5])
    return trajectory

# Example usage
trajectory = generate_square_trajectory()
# print(trajectory.shape)  # (2500, 3)
