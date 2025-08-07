import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from utils.plot import set_axes_equal, hide_axes

# ---- CONFIGURE HERE ----
files_and_colors = [
    #("data/processed/0722_corrected_kff_10X_100_slow_CONVERTED.csv", "mediumslateblue", "NEW"),
    ("data/processed/0724_CLIVE_SQUARE_2_CONVERTED.csv", "black", "NEW"),
    ("data/goals/0724_CLIVE_2_OPTIMAL_SQUARE.csv", "lightseagreen", "NEW"),
    # ("data/goals/0724_45_OPTIMAL_SQUARE.csv", "orange", "NEW"),
    # ("data/processed/0722_length_36in_speed_10X_CONVERTED.csv", "lightseagreen", "AVG"),
    #("data/processed/0722_length_72in_speed_10X_CONVERTED.csv", "orange", "OLD"),
    #("data/processed/0722_length_24in_speed_10X_CONVERTED.csv", "red", "OLD"),
    #("data/processed/0721_square_with_kd_ki_75_CONVERTED.csv", "orange", "75% Kff YES KI/KD"),
    #("data/processed/0721_short_fast.csv", "red", "50% Kff YES KI/KD"),
    # ("data/goals/0722_OPTIMAL_SQUARE.csv", "BLACK", "GOAL TRAJECTORY"),
]

plot_path = "plots/0.5_1_2x_speeds.png"
# ------------------------

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

for filepath, color, label in files_and_colors:
    traj = np.loadtxt(filepath, delimiter=',')
    ax.plot(*traj.T, label=label, color=color, linewidth=2)

set_axes_equal(ax)
# hide_axes(ax)
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.savefig(plot_path)
plt.show()


print(f"✅ Saved plot to {plot_path}")
