import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import os
from utils.optimal_square import optimize_square_center, optimize_square_center_hierarchical

# === CONFIG ===
CSV_PATH = "data/processed/0724_CLIVE_SQUARE_2_CONVERTED.csv"  # <-- replace with your real file
PLANE = "xy"                                    # "xy", "xz", or "yz"
SIDE_LENGTH = 0.1                              # in meters
COARSE_STEPS = 10                              # how coarse to search
FINE_STEPS = 10                              # how fine to search
REFINE_FACTOR = 0.1                            # how much to refine the search box

# === LOAD TRAJECTORY ===
robot_traj = np.loadtxt(CSV_PATH, delimiter=',')

# === COMPUTE BEST SQUARE CENTER ===
# center, error = optimize_square_center(
#    robot_points=robot_traj,
#    side=SIDE_LENGTH,
#    plane=PLANE,
#    num_steps=NUM_STEPS
#)

center, error = optimize_square_center_hierarchical(
    robot_points=robot_traj,
    side=SIDE_LENGTH,
    plane=PLANE,
    coarse_steps=COARSE_STEPS,  # Coarse search grid size
    fine_steps=FINE_STEPS,
    refine_factor=REFINE_FACTOR  # Refine in a box 1/10 the size of full region
)


# === OUTPUT ===
print(f"✅ Best-fit center for square ({PLANE}-plane): {center}")
print(f"📉 Min RMSE: {error:.6f}")
