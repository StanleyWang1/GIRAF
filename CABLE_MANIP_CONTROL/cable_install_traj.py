import numpy as np

def generate_custom_trajectory():
    # Define points
    p1 = np.array([-0.1,  0.0,  -0.02])   # hold for 500 steps
    p2 = np.array([-0.1, -0.15, -0.02])
    p3 = np.array([ 0.1, -0.15, -0.02])
    p4 = np.array([ 0.1,  0.0,  -0.02])
    p5 = np.array([ 0.0,  0.0,  -0.02])
    p6 = np.array([ 0.0,  0.0,  -0.15])
    p7 = np.array([ 0.0, -0.2,  -0.15])

    # Build trajectory
    traj1 = np.tile(p1, (500, 1))
    traj2 = np.array([p2, p3, p4, p5, p6, p7])  # 1 step each

    trajectory = np.vstack([traj1, traj2])
    return trajectory

# Example usage
trajectory = generate_custom_trajectory()
# print(trajectory.shape)  # (506, 3)
