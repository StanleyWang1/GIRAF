# ISPARO_DATA/main/generate_goal.py

import numpy as np
import os

def generate_square_loop(center, side_length, height_offset=0.0, num_points_per_side=25):
    """Generate a square loop in 3D space centered at `center`, lying in the XY plane."""
    cx, cy, cz = center
    half = side_length / 2
    z = cz + height_offset  # fixed height for XY plane

    # 4 corners in XY plane
    A = np.array([cx - half, cy - half, z])
    B = np.array([cx + half, cy - half, z])
    C = np.array([cx + half, cy + half, z])
    D = np.array([cx - half, cy + half, z])

    # Interpolate edges
    edge1 = np.linspace(A, B, num_points_per_side, endpoint=False)
    edge2 = np.linspace(B, C, num_points_per_side, endpoint=False)
    edge3 = np.linspace(C, D, num_points_per_side, endpoint=False)
    edge4 = np.linspace(D, A, num_points_per_side, endpoint=False)

    return np.vstack([edge1, edge2, edge3, edge4])

if __name__ == "__main__":
    center = []

    # actual square: -0.2126606  -0.01505871 -0.19918473
    # differences     0.00513     0.01304    -0.004846
    # actual        : -0.20842242 -0.10804782  0.26851626

    # desired square:-0.20757,  -0.00202,    -0.20403


    center = [-0.20842242, -0.10804782,  0.26851626]  # for vertical data
    # center = [-0.20021098, -0.00074972, -0.27799192] for horizontal data
    side_length = 0.1          # 0.5 meters
    height_offset = 0          # Z height of the square
    points_per_edge = 50  

    traj = generate_square_loop(center, side_length, height_offset, num_points_per_side=points_per_edge)

    out_dir = "data/goals"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "0724_CLIVE_2_OPTIMAL_SQUARE.csv")
    np.savetxt(out_path, traj, delimiter=',')

    print(f"✅ Saved square goal trajectory to {out_path}")
