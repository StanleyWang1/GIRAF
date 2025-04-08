import numpy as np

# Constants
dt = 1 / 200  # 200 Hz
def steps(seconds):
    return int(seconds / dt)

# Positions (customize these)
pick = np.array([0.5, -0.25, 0.05])     # example pick location
place = np.array([0.5,  0.25, 0.05])    # example place location
above_offset = np.array([0, 0, 0.1])  # 10 cm above

above_pick = pick + above_offset
above_place = place + above_offset

# Segment generator
def linear_segment(start, end, duration, grasp_val):
    n = steps(duration)
    pos = np.linspace(start, end, n)
    grasp = np.full(n, grasp_val)
    return pos, grasp

# Sequence
segments = []

# (1) open, go to 10cm above pick (2s)
seg1, g1 = linear_segment(above_place, above_pick, 2.0, 0)  # assuming you start at above_place

# (2) open, go to pick (1s)
seg2, g2 = linear_segment(above_pick, pick, 1.0, 0)

# (3) close (2s) — stay still
seg3, g3 = linear_segment(pick, pick, 2.0, 1)

# (4) close, go to 10cm above pick (1s)
seg4, g4 = linear_segment(pick, above_pick, 1.0, 1)

# (5) close, go to 10cm above place (2s)
seg5, g5 = linear_segment(above_pick, above_place, 2.0, 1)

# (6) close, go to place (1s)
seg6, g6 = linear_segment(above_place, place, 1.0, 1)

# (7) open (2s) — stay still
seg7, g7 = linear_segment(place, place, 2.0, 0)

# (8) open, go to 10cm above place (1s)
seg8, g8 = linear_segment(place, above_place, 1.0, 0)

# Concatenate all segments
pnp_traj = np.vstack((seg1, seg2, seg3, seg4, seg5, seg6, seg7, seg8))
pnp_grasp = np.concatenate((g1, g2, g3, g4, g5, g6, g7, g8))

# Velocity via finite differences
vel = (pnp_traj[1:] - pnp_traj[:-1]) / dt
pnp_velocity = np.vstack((vel, [vel[-1]]))  # pad final row

# Optional print to verify
# print(f"Trajectory shape: {pnp_traj.shape}, Velocity shape: {pnp_velocity.shape}, Grasp shape: {pnp_grasp.shape}")
