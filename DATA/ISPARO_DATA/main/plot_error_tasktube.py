# ISPARO_DATA/main/plot_error_tasktube.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import local_max_errors, compute_error_stats
from utils.plot import plot_task_tube_spheres, plot_task_tube_ellipsoids

EXPERIMENT_NAME = "0721_square_without_kd_ki"
# Paths to robot and goal trajectories
robot_path = f"data/processed/{EXPERIMENT_NAME}_CONVERTED.csv"

# Customize colors
goal_color = "red"
robot_color = "mediumslateblue"
tube_color = "mediumslateblue"


# Use the average trajectory as the goal
# goal_path = f"data/processed/{EXPERIMENT_NAME}_AVG.csv"
#plot_path = f"plots/{EXPERIMENT_NAME}_vs_AVG.png"

# Use a separate goal file if needed
goal_path = "data/goals/OPTIMAL_SQUARE.csv"
plot_path = f"plots/{EXPERIMENT_NAME}_vs_GOAL.png"

robot_traj = np.loadtxt(robot_path, delimiter=',')
goal_traj = np.loadtxt(goal_path, delimiter=',')

# Optional downsampling
DOWNSAMPLE_GOAL_TO = 200  # or None
if DOWNSAMPLE_GOAL_TO is not None:
    idx = np.linspace(0, len(goal_traj) - 1, DOWNSAMPLE_GOAL_TO).astype(int)
    goal_traj = goal_traj[idx]

# Error sphere radii per goal point
radii = local_max_errors(goal_traj, robot_traj)

# Compute stats based on nearest distances (not 1-to-1)
stats = compute_error_stats(goal_traj, robot_traj)

print("📊 Error Metrics (nearest-neighbor-based):")
print(f"Mean Error       : {stats['mean']:.4f} m")
print(f"Max Error        : {stats['max']:.4f} m")
print(f"RMSE             : {stats['rmse']:.4f} m")
print(f"90th Percentile  : {stats['perc_90']:.4f} m")
print(f"95th Percentile  : {stats['perc_95']:.4f} m")

# Plot and save
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
# Remove axis panes (background)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Remove grid lines
ax.grid(False)

# Optional: turn off axes box entirely
ax.set_axis_off()

# Plot the task tube with spheres
plot_task_tube_spheres(goal_traj, robot_traj, radii, ax=ax,
               goal_color=goal_color,
               robot_color=robot_color,
               tube_color=tube_color)

ax.view_init(elev=90, azim=-90)


# Plot the task tube with ellipsoids
# plot_task_tube_ellipsoids(goal_traj, robot_traj, ax=ax, scale=1.0,
#                         goal_color=goal_color, robot_color=robot_color, tube_color=tube_color)

plt.tight_layout()
plt.savefig(plot_path)
plt.show()


print(f"✅ Plot saved to {plot_path}")
