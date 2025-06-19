import numpy as np

def generate_custom_trajectory():
    # Define points
    p1 = np.array([-0.2,  0.0,  -0.01])   # hold for 500 steps
    p2 = np.array([-0.2, -0.2, -0.01])
    p3 = np.array([ 0.2, -0.2, -0.01])
    p4 = np.array([ 0.2,  0.0,  -0.01])
    p5 = np.array([ 0.0,  0.0,  -0.01])
    p6 = np.array([ 0.0,  0.0,  -0.2])
    p7 = np.array([ 0.0, -0.2,  -0.2])

    # Build trajectory
    traj1 = np.tile(p1, (500, 1))
    traj2 = np.tile(p2, (500, 1))
    traj3 = np.tile(p3, (500, 1))
    traj4 = np.tile(p4, (500, 1))
    traj5 = np.tile(p5, (500, 1))
    traj6 = np.tile(p6, (500, 1))
    traj7 = np.tile(p7, (500, 1))

    trajectory = np.vstack([traj1, traj2, traj3, traj4, traj5, traj6, traj7])
    return trajectory

# Example usage
trajectory = generate_custom_trajectory()
# print(trajectory.shape)  # (506, 3)
