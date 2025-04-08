import numpy as np

# Number of points on the circle
N = 1000

# Parameterize angle
theta = np.linspace(0, 2 * np.pi, N)

# Coordinates
x = np.full_like(theta, 0.5)                      # Constant x = 0.5
y = 0.25 * np.cos(theta)
z = 0.5 + 0.25 * np.sin(theta)

# Stack into (N, 3) array
circle_traj = np.stack((x, y, z), axis=-1)

dt = 1 / 200
circle_velocity = np.vstack(((circle_traj[1:] - circle_traj[:-1]) / dt, [(circle_traj[-1] - circle_traj[-2]) / dt]))
