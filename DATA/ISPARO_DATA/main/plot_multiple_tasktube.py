import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import local_max_errors, compute_error_stats
from utils.plot import plot_task_tube_spheres

# === CONFIGURATION ===
robot_filenames = [
    # "0722_length_24in_speed_2X_CONVERTED.csv",
    # "0722_length_24in_speed_4X_CONVERTED.csv",
    # "0722_length_24in_speed_6X_CONVERTED.csv",
    # "0722_length_24in_speed_8X_CONVERTED.csv",
    # "0722_length_24in_speed_10X_CONVERTED.csv",
    # "0722_length_36in_speed_2X_CONVERTED.csv",
    # "0722_length_36in_speed_4X_CONVERTED.csv",
    # "0722_length_36in_speed_6X_CONVERTED.csv",
    # "0722_length_36in_speed_8X_CONVERTED.csv",
    # "0722_length_36in_speed_10X_CONVERTED.csv",
    # "0722_length_48in_speed_2X_CONVERTED.csv",
    # "0722_length_48in_speed_4X_CONVERTED.csv",
    # "0722_length_48in_speed_6X_CONVERTED.csv",
    # "0722_length_48in_speed_8X_CONVERTED.csv",
    # "0722_length_48in_speed_10X_CONVERTED.csv",
    # "0722_length_60in_speed_2X_CONVERTED.csv",
    # "0722_length_60in_speed_4X_CONVERTED.csv",
    # "0722_length_60in_speed_6X_CONVERTED.csv",
    # "0722_length_60in_speed_8X_CONVERTED.csv",
    # "0722_length_60in_speed_10X_CONVERTED.csv",
    # "0722_length_72in_speed_2X_CONVERTED.csv",
    # "0722_length_72in_speed_4X_CONVERTED.csv",
    # "0722_length_72in_speed_6X_CONVERTED.csv",
    # "0722_length_72in_speed_8X_CONVERTED.csv",
    # "0722_length_72in_speed_10X_CONVERTED.csv",
    # "0723_length_36in_speed_2X_vert_CONVERTED.csv",
    # "0723_length_36in_speed_4X_vert_CONVERTED.csv",
    # "0723_length_36in_speed_6X_vert_CONVERTED.csv",
    # "0723_length_36in_speed_8X_vert_CONVERTED.csv",
    "0723_length_36in_speed_10X_vert_CONVERTED.csv",

]

goal_path = "data/goals/0723_VERT_OPTIMAL_SQUARE.csv"
goal_traj = np.loadtxt(goal_path, delimiter=',')

# Optional: Downsample goal trajectory
DOWNSAMPLE_GOAL_TO = 150  # or None
if DOWNSAMPLE_GOAL_TO is not None:
    idx = np.linspace(0, len(goal_traj) - 1, DOWNSAMPLE_GOAL_TO).astype(int)
    goal_traj = goal_traj[idx]

# === COLORS ===
goal_color = "black"
robot_color = "mediumslateblue"
tube_color = "violet"

# === SETUP SUBPLOTS ===
num_trials = len(robot_filenames)
cols = math.ceil(np.sqrt(num_trials))
rows = math.ceil(num_trials / cols)

fig = plt.figure(figsize=(6 * cols, 6 * rows))

for i, robot_filename in enumerate(robot_filenames):
    robot_path = os.path.join("data/processed", robot_filename)
    robot_traj = np.loadtxt(robot_path, delimiter=',')
    radii = local_max_errors(goal_traj, robot_traj)
    stats = compute_error_stats(goal_traj, robot_traj)

    #print("\n📊 Error Metrics for", robot_filename)
    #print(f"Max Error        : {stats['max']:.4f} m")
    # print(f"RMSE             : {stats['rmse']:.4f} m")
    #print(f"95th Percentile  : {stats['perc_95']:.4f} m")
    print(f"95th Percentile  : {stats['perc_95']:.4f} m")

    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
    ax.set_title(robot_filename.replace(".csv", ""), fontsize=10)
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=90, azim=-90)

    plot_task_tube_spheres(goal_traj, robot_traj, radii, ax=ax,
                        goal_color=goal_color, robot_color=robot_color, tube_color=tube_color)

plt.tight_layout()
output_path = "plots/test_ALL_TRIALS_subplots.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\n✅ Combined subplot figure saved to {output_path}")
