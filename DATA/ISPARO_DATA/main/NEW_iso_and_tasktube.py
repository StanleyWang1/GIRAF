import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors

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
    # "0723_length_24in_speed_2X_vert_CONVERTED.csv",
    # "0723_length_24in_speed_4X_vert_CONVERTED.csv",
    # "0723_length_24in_speed_6X_vert_CONVERTED.csv",
    # "0723_length_24in_speed_8X_vert_CONVERTED.csv",
    # "0723_length_24in_speed_10X_vert_CONVERTED.csv",
    # "0723_length_36in_speed_2X_vert_CONVERTED.csv",
    # "0723_length_36in_speed_4X_vert_CONVERTED.csv",
    # "0723_length_36in_speed_6X_vert_CONVERTED.csv",
    # "0723_length_36in_speed_8X_vert_CONVERTED.csv",
    # "0723_length_36in_speed_10X_vert_CONVERTED.csv",
    # "0723_length_48in_speed_2X_vert_CONVERTED.csv",
    # "0723_length_48in_speed_4X_vert_CONVERTED.csv",
    # "0723_length_48in_speed_6X_vert_CONVERTED.csv",
    # "0723_length_48in_speed_8X_vert_CONVERTED.csv",
    # "0723_length_48in_speed_10X_vert_CONVERTED.csv",
    # "0723_length_60in_speed_2X_vert_CONVERTED.csv",
    # "0723_length_60in_speed_4X_vert_CONVERTED.csv",
    # "0723_length_60in_speed_6X_vert_CONVERTED.csv",
    # "0723_length_60in_speed_8X_vert_CONVERTED.csv",
    # "0723_length_60in_speed_10X_vert_CONVERTED.csv",
    # "0723_length_72in_speed_2X_vert_CONVERTED.csv",
    # "0723_length_72in_speed_4X_vert_CONVERTED.csv",
    # "0723_length_72in_speed_6X_vert_CONVERTED.csv",
    # "0723_length_72in_speed_8X_vert_CONVERTED.csv",
    # "0723_length_72in_speed_10X_vert_CONVERTED.csv",
    "0724_length_24in_speed_2X_45_CONVERTED.csv",
    "0724_length_24in_speed_4X_45_CONVERTED.csv",
    "0724_length_24in_speed_6X_45_CONVERTED.csv",
    "0724_length_24in_speed_8X_45_CONVERTED.csv",
    "0724_length_24in_speed_10X_45_CONVERTED.csv",
    "0724_length_36in_speed_2X_45_CONVERTED.csv",
    "0724_length_36in_speed_4X_45_CONVERTED.csv",
    "0724_length_36in_speed_6X_45_CONVERTED.csv",
    "0724_length_36in_speed_8X_45_CONVERTED.csv",
    "0724_length_36in_speed_10X_45_CONVERTED.csv",
    "0724_length_48in_speed_2X_45_CONVERTED.csv",
    "0724_length_48in_speed_4X_45_CONVERTED.csv",
    "0724_length_48in_speed_6X_45_CONVERTED.csv",
    "0724_length_48in_speed_8X_45_CONVERTED.csv",
    "0724_length_48in_speed_10X_45_CONVERTED.csv",
    "0724_length_60in_speed_2X_45_CONVERTED.csv",
    "0724_length_60in_speed_4X_45_CONVERTED.csv",
    "0724_length_60in_speed_6X_45_CONVERTED.csv",
    "0724_length_60in_speed_8X_45_CONVERTED.csv",
    "0724_length_60in_speed_10X_45_CONVERTED.csv",
    "0724_length_72in_speed_2X_45_CONVERTED.csv",
    "0724_length_72in_speed_4X_45_CONVERTED.csv",
    "0724_length_72in_speed_6X_45_CONVERTED.csv",
    "0724_length_72in_speed_8X_45_CONVERTED.csv",
    "0724_length_72in_speed_10X_45_CONVERTED.csv",
]

speeds = np.array([17, 33, 50, 67, 80])
lengths = np.array([0.6, 0.9, 1.2, 1.5, 1.8])

# === Manual colormap bounds ===
vmin = 0   # in mm
vmax = 50  # in mm

# === Load goal trajectory and downsample ===
# goal_traj = np.loadtxt("data/goals/0723_VERT_OPTIMAL_SQUARE.csv", delimiter=',')
goal_traj = np.loadtxt("data/goals/0724_45_OPTIMAL_SQUARE.csv", delimiter=',')
goal_traj24 = np.loadtxt("data/goals/0724_45_24ONLY_OPTIMAL_SQUARE.csv", delimiter=',')
DOWNSAMPLE_GOAL_TO = 200
if DOWNSAMPLE_GOAL_TO:
    idx = np.linspace(0, len(goal_traj) - 1, DOWNSAMPLE_GOAL_TO).astype(int)
    goal_traj = goal_traj[idx]
    
    # FOR 24 ONLY
    idx24 = np.linspace(0, len(goal_traj24) - 1, DOWNSAMPLE_GOAL_TO).astype(int)
    goal_traj24 = goal_traj24[idx24]


# === Compute 95th percentile errors ===
perc95_errors = []
for filename in robot_filenames:
    robot_path = os.path.join("data/processed", filename)
    robot_traj = np.loadtxt(robot_path, delimiter=',')  
    if filename.startswith("0724_length_24in"):
        stats = compute_error_stats(goal_traj24, robot_traj)
        perc95_errors.append(stats['perc_95'] * 1000)  # mm
    else:
        # For other lengths, use the full goal trajectory
        stats = compute_error_stats(goal_traj, robot_traj)
        perc95_errors.append(stats['perc_95'] * 1000)
    # stats = compute_error_stats(goal_traj, robot_traj)
    # perc95_errors.append(stats['perc_95'] * 1000)  # mm
    print(stats['perc_95'] * 1000)

error_matrix = np.array(perc95_errors).reshape(5, 5)  # rows=lengths, cols=speeds

# === Fit spline surface to error matrix ===
X, Y = np.meshgrid(speeds, lengths)
spline = SmoothBivariateSpline(X.ravel(), Y.ravel(), error_matrix.ravel(), s=1)

speeds_fine = np.linspace(speeds.min(), speeds.max(), 100)
lengths_fine = np.linspace(lengths.min(), lengths.max(), 100)
S_fine, L_fine = np.meshgrid(speeds_fine, lengths_fine)
error_fine = spline.ev(S_fine.ravel(), L_fine.ravel()).reshape(100, 100)

# === Colormap setup with normalization and clipping ===
cmap = LinearSegmentedColormap.from_list("custom", [
    "lightseagreen", "orange", "indianred", "deeppink"
], N=256)

norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

# === Normalize color values for task tube plotting ===
normed_errors = (error_matrix - vmin) / (vmax - vmin)
color_indices = np.clip((normed_errors * 255).astype(int), 0, 255)
colors_for_tubes = [cmap(i) for row in color_indices for i in row]

# === Plot error contour map ===
plt.figure(figsize=(8, 6))
contourf = plt.contourf(L_fine, S_fine, error_fine,
                        levels=np.linspace(vmin, vmax, 31),
                        cmap=cmap, norm=norm, alpha=0.9)
contour = plt.contour(L_fine, S_fine, error_fine,
                      levels=np.linspace(vmin, vmax, 31),
                      colors='grey', linewidths=1, norm=norm)
cbar = plt.colorbar(contourf, pad=0.02)
cbar.set_label('95th Percentile Error (mm)', fontsize=20)
cbar.ax.tick_params(labelsize=16)
plt.xlabel('Boom Length (m)', fontsize=20)
plt.ylabel('Speed (mm/s)', fontsize=20)
plt.title('95th Percentile Error Contours', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(False)
plt.tight_layout()
plt.savefig("plots/fig_error_contours_colored_45.png", dpi=300)
plt.close()

# === Plot individual task tubes ===
for i, filename in enumerate(robot_filenames):
    color = colors_for_tubes[i]
    robot_path = os.path.join("data/processed", filename)
    robot_traj = np.loadtxt(robot_path, delimiter=',')
    if filename.startswith("0724_length_24in"):
        
        radii = local_max_errors(goal_traj24, robot_traj)
    else:
          
        radii = local_max_errors(goal_traj, robot_traj)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(filename.replace(".csv", ""), fontsize=10)
    ax.set_axis_off()
    ax.view_init(elev=90, azim=-90)

    # center = [0.0716353, -0.15443102, -0.27722796]  # vertical center
    # enter = [-0.20021098, -0.00074972, -0.27799192] # horizontal data
    # ax.set_xlim([center[0] - 0.056, center[0] + 0.056])
    # ax.set_ylim([center[1] - 0.056, center[1] + 0.056])
    if filename.startswith("0724_length_24in"):
        ax.set_xlim([goal_traj24[:, 0].min() - 0.015, goal_traj24[:, 0].max() + 0.015])
        ax.set_ylim([goal_traj24[:, 1].min() - 0.015, goal_traj24[:, 1].max() + 0.015])
        plot_task_tube_spheres(goal_traj24, robot_traj, radii, ax=ax,
                           goal_color='black', robot_color=color, tube_color=color)
    else:
        ax.set_xlim([goal_traj[:, 0].min() - 0.015, goal_traj[:, 0].max() + 0.015])
        ax.set_ylim([goal_traj[:, 1].min() - 0.015, goal_traj[:, 1].max() + 0.015])
        plot_task_tube_spheres(goal_traj, robot_traj, radii, ax=ax,
                           goal_color='black', robot_color=color, tube_color=color)
    
    
    plt.tight_layout()
    save_path = f"plots/individual_tasktube_45_{i+1:02d}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")
