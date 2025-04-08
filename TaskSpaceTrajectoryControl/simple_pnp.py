import numpy as np

# Constants
dt = 1 / 200  # 200 Hz
pause_time = 2.0     # seconds at each endpoint
traj_time = 2.0      # time to traverse the semicircle

pause_steps = int(pause_time / dt)
traj_steps = int(traj_time / dt)

# Semicircle (upper half, CCW)
theta_fwd = np.linspace(np.pi, 0, traj_steps)
theta_bwd = np.linspace(0, np.pi, traj_steps)  # reverse

def semicircle(theta):
    x = np.full_like(theta, 0.75)
    y = 0.25 * np.cos(theta)
    z = 0.05 + 0.25 * np.sin(theta)
    return np.stack((x, y, z), axis=-1)

# Core paths
semi_fwd = semicircle(theta_fwd)
semi_bwd = semicircle(theta_bwd)

# Pause points
start_pt = semi_fwd[0]
end_pt = semi_fwd[-1]
pause_start = np.tile(start_pt, (pause_steps, 1))
pause_end = np.tile(end_pt, (pause_steps, 1))

# Concatenate full cyclic trajectory: pause → fwd → pause → bwd
semicirc_traj = np.vstack((
    pause_start,
    semi_fwd,
    pause_end,
    semi_bwd
))

# Velocity via finite difference
vel = (semicirc_traj[1:] - semicirc_traj[:-1]) / dt
semicirc_velocity = np.vstack((vel, [vel[-1]]))  # pad final row

# Grasp logic:
# - 0 during first half of pause_start
# - 1 during second half of pause_start
# - 1 during fwd
# - 1 during first half of pause_end
# - 0 during second half of pause_end
# - 1 during bwd

g0 = np.array([0] * (pause_steps // 2) + [1] * (pause_steps - pause_steps // 2))   # start pause
g1 = np.ones(traj_steps)                                                           # forward
g2 = np.array([1] * (pause_steps // 2) + [0] * (pause_steps - pause_steps // 2))   # end pause
g3 = np.ones(traj_steps)                                                           # backward

semicirc_grasp = np.concatenate((g0, g1, g2, g3))
