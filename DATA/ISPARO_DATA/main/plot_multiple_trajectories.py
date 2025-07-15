import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from utils.plot import set_axes_equal, hide_axes

root = True

# ---- CONFIGURE HERE ----
files_and_colors = [
    ("data/processed/0.5X_SPEED_BOOM_CONVERTED.csv", "orange", "0.5x speed"),
    ("data/processed/1X_SPEED_BOOM_CONVERTED.csv", "lightseagreen", "1x speed"),
    ("data/processed/2X_SPEED_BOOM_CONVERTED.csv", "mediumslateblue", "2x speed"),
]

plot_path = "plots/0.5_1_2x_speeds.png"

if root:
    # Prepend the base directory to all file paths
    base_dir = "./DATA/ISPARO_DATA/"
    files_and_colors = [
        (os.path.join(base_dir, filepath), color, label)
        for filepath, color, label in files_and_colors
    ]
    plot_path = os.path.join(base_dir, plot_path)

# ------------------------

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

for filepath, color, label in files_and_colors:
    traj = np.loadtxt(filepath, delimiter=',')
    ax.plot(*traj.T, label=label, color=color, linewidth=1.5)

set_axes_equal(ax)
hide_axes(ax)
ax.legend()
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"✅ Saved plot to {plot_path}")
