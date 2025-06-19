import numpy as np

def generate_trajectory():
    # (1) Hold at start
    p1 = np.array([-0.08, 0.08, 0.02])
    traj1 = np.tile(p1, (500, 1))

    # (2) Arc from 180° to 360° around center (0, 0.08, 0.02), radius 0.08
    theta = np.linspace(np.pi, 2 * np.pi, 500)
    center = np.array([0.0, 0.08, 0.02])
    radius = 0.08
    x_arc = -radius * np.cos(theta)
    y_arc = center[1] - radius * np.sin(theta)
    z_arc = np.full_like(x_arc, center[2])
    traj2 = np.stack([x_arc, y_arc, z_arc], axis=1)

    # (3) Raise Z from 0.02 to 0.15
    z_lift = np.linspace(0.02, 0.15, 200)
    traj3 = np.stack([np.zeros(200), np.zeros(200), z_lift], axis=1)

    # (4) Move Y from 0 to 0.15 at fixed Z = 0.15
    y_slide = np.linspace(0, 0.15, 200)
    traj4 = np.stack([np.zeros(200), y_slide, np.full(200, 0.15)], axis=1)

    # Concatenate all parts
    trajectory = np.vstack([traj1, traj2, traj3, traj4])
    return trajectory

# Example usage
trajectory = generate_trajectory()
# print(trajectory.shape)  # Should be (300, 3)
