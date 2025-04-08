import numpy as np

# Constants
dt = 1 / 200
def steps(seconds): return int(seconds / dt)

# Timing
t_above = 2.0
t_down  = 1.5
t_grasp = 2.0
t_up    = 1.5
t_move  = 2.0

# Positions (customize these!)
A = np.array([1.5, -0.5, 0.28])
B = np.array([1.5,  0.5, 0.28])
C = np.array([2,  0, 0.37])
above_offset = np.array([0, 0, 0.25])
above = lambda pt: pt + above_offset

# Segment generator
def linear_segment(start, end, duration, grasp_val):
    n = steps(duration)
    pos = np.linspace(start, end, n)
    grasp = np.full(n, grasp_val)
    return pos, grasp

def pick_and_place(pick_pt, place_pt):
    segs, grasps = [], []

    # 1. Move above pick (2s)
    s, g = linear_segment(above(place_pt), above(pick_pt), t_above, 0)
    segs.append(s); grasps.append(g)

    # 2. Move down to pick (1s)
    s, g = linear_segment(above(pick_pt), pick_pt, t_down, 0)
    segs.append(s); grasps.append(g)

    # 3. Grasp (1s)
    s, g = linear_segment(pick_pt, pick_pt, t_grasp, 1)
    segs.append(s); grasps.append(g)

    # 4. Move up (1s)
    s, g = linear_segment(pick_pt, above(pick_pt), t_up, 1)
    segs.append(s); grasps.append(g)

    # 5. Move above place (2s)
    s, g = linear_segment(above(pick_pt), above(place_pt), t_move, 1)
    segs.append(s); grasps.append(g)

    # 6. Move down to place (1s)
    s, g = linear_segment(above(place_pt), place_pt, t_down, 1)
    segs.append(s); grasps.append(g)

    # 7. Release (1s)
    s, g = linear_segment(place_pt, place_pt, t_grasp, 0)
    segs.append(s); grasps.append(g)

    # 8. Move up (1s)
    s, g = linear_segment(place_pt, above(place_pt), t_up, 0)
    segs.append(s); grasps.append(g)

    return np.vstack(segs), np.concatenate(grasps)

# Sequence: define the 6 motions in your task
traj_parts = []
grasp_parts = []

# 1. Move object 1: A → B
t1, g1 = pick_and_place(A, B)
traj_parts.append(t1); grasp_parts.append(g1)

# 2. Move object 2: C → A
t2, g2 = pick_and_place(C, A)
traj_parts.append(t2); grasp_parts.append(g2)

# 3. Move object 1: B → C
t3, g3 = pick_and_place(B, C)
traj_parts.append(t3); grasp_parts.append(g3)

# 4. Move object 2: A → B
t4, g4 = pick_and_place(A, B)
traj_parts.append(t4); grasp_parts.append(g4)

# 5. Move object 1: C → A
t5, g5 = pick_and_place(C, A)
traj_parts.append(t5); grasp_parts.append(g5)

# 6. Move object 2: B → C
t6, g6 = pick_and_place(B, C)
traj_parts.append(t6); grasp_parts.append(g6)

# Final full trajectory and grasp sequence
pnp_traj = np.vstack(traj_parts)
pnp_grasp = np.concatenate(grasp_parts)

# Velocity
vel = (pnp_traj[1:] - pnp_traj[:-1]) / dt
pnp_velocity = np.vstack((vel, [vel[-1]]))
