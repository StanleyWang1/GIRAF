import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import RectBivariateSpline

# === CONFIGURATION ===
speeds = np.array([17, 33, 50, 67, 80])
speed_labels = np.array([2, 4, 6, 8, 10])  # in X

lengths = np.array([0.6, 0.9, 1.2, 1.5, 1.8])  # m
vmin, vmax = 0, 90  # mm

# === Utility ===
def load_goal(path):
    goal = np.loadtxt(path, delimiter=',')
    DOWNSAMPLE_TO = 200
    if DOWNSAMPLE_TO:
        idx = np.linspace(0, len(goal) - 1, DOWNSAMPLE_TO).astype(int)
        goal = goal[idx]
    return goal

def load_errors(filenames, goal_traj):
    from utils.metrics import compute_error_stats
    errors = []
    for fname in filenames:
        path = os.path.join("data/processed", fname)
        data = np.loadtxt(path, delimiter=',')
        stats = compute_error_stats(goal_traj, data)
        errors.append(stats['perc_95'] * 1000)  # mm
    return np.array(errors).reshape(len(lengths), len(speeds))

# === Filename generators ===
def make_filenames(prefix, postfix):
    return [f"{prefix}_length_{int(l*40)}in_speed_{s}X{postfix}_CONVERTED.csv"
            for l in lengths for s in speed_labels]

horizontal_filenames = make_filenames("0722", "")
vertical_filenames   = make_filenames("0723", "_vert")

# === Load goal trajectories ===
goal_horiz = load_goal("data/goals/0722_OPTIMAL_SQUARE.csv")
goal_vert  = load_goal("data/goals/0723_VERT_OPTIMAL_SQUARE.csv")

# === Load errors ===
horz_errors = load_errors(horizontal_filenames, goal_horiz)  # shape (5,5)
vert_errors = load_errors(vertical_filenames, goal_vert)     # shape (5,5)

# === Colormap ===
cmap = LinearSegmentedColormap.from_list("custom", [
    "lightseagreen", "orange", "orangered", "firebrick", "deeppink"
], N=256)
norm = Normalize(vmin=vmin, vmax=vmax)

# === Plot setup ===
theta_res = 200
r_res = 200
theta = np.linspace(0, np.pi/2, theta_res)  # 0° to 90°
angle_vals = np.rad2deg(theta)
radii = np.linspace(lengths.min(), lengths.max(), r_res)
T, R = np.meshgrid(theta, radii)

# === Store interpolators ===
error_surfaces = []

# === Generate quarter-circle plots for each speed ===
for speed_idx, speed in enumerate(speeds):
    H = horz_errors[:, speed_idx]
    V = vert_errors[:, speed_idx]
    E = np.zeros((r_res, theta_res))
    for i in range(r_res):
        r = radii[i]
        h_val = np.interp(r, lengths, H)
        v_val = np.interp(r, lengths, V)
        E[i, :] = h_val * np.cos(T[i, :]) + v_val * np.sin(T[i, :])

    # Create interpolator (RectBivariateSpline: y=length, x=angle)
    interp_func = RectBivariateSpline(radii, angle_vals, E)
    error_surfaces.append(interp_func)

    # Convert polar to Cartesian
    X = R * np.cos(T)
    Y = R * np.sin(T)

    # Plot filled gradient
    plt.figure(figsize=(6, 6))
    mesh = plt.pcolormesh(X, Y, E, cmap=cmap, norm=norm, shading='auto', alpha=0.6)

    # Add contour lines
    contour_levels = np.linspace(vmin, vmax, 31)
    contours = plt.contour(X, Y, E, levels=contour_levels, colors='black', linewidths=0.5)
    label_levels = contour_levels[::3]  # every 3rd level
    plt.clabel(contours, levels=label_levels, inline=True, fontsize=8, fmt='%1.0f')

    # Colorbar and labels
    cbar = plt.colorbar(mesh, pad=0.02)
    cbar.set_label('95th Percentile Error (mm)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.title(f'Speed = {speed} mm/s', fontsize=16)
    plt.xlabel("Horizontal →", fontsize=12)
    plt.ylabel("↑ Vertical", fontsize=12)
    plt.axis('equal')
    plt.xlim(0, lengths.max())
    plt.ylim(0, lengths.max())
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/quarter_circle_error_speed{speed}_contour.png", dpi=300)
    plt.close()
    print(f"✅ Saved: plots/quarter_circle_error_speed{speed}_contour.png")

# === Max Allowable Speed Query ===
def get_max_allowable_speed(length_m, angle_deg, max_error_mm):
    """
    Returns the max allowable speed (in mm/s) such that error at (length, angle) ≤ max_error_mm.
    Returns None if no speed meets the requirement.
    """
    assert 0 <= length_m <= 2.0, "Length must be between 0 and 2 meters."
    assert 0 <= angle_deg <= 90.0, "Angle must be between 0 and 90 degrees."
    assert 0 <= max_error_mm <= 90.0, "Max error must be between 0 and 90 mm."

    valid_speeds = []
    for i, interp in enumerate(error_surfaces):
        err = float(interp(length_m, angle_deg))  # scalar
        if err <= max_error_mm:
            valid_speeds.append(speeds[i])

    return max(valid_speeds) if valid_speeds else None

from scipy.interpolate import RegularGridInterpolator

# === Build 3D error grid: shape (length, angle, speed) ===
length_vals_fine = radii                  # (200,)
angle_vals_fine = angle_vals              # (200,)
speed_vals_fine = speeds                  # (5,)

error_volume = np.zeros((len(length_vals_fine), len(angle_vals_fine), len(speed_vals_fine)))

for i, interp in enumerate(error_surfaces):
    for li, l in enumerate(length_vals_fine):
        error_volume[li, :, i] = interp(l, angle_vals_fine).flatten()

# === Create 3D interpolator ===
error_interp_3d = RegularGridInterpolator(
    (length_vals_fine, angle_vals_fine, speed_vals_fine),
    error_volume,
    bounds_error=False,
    fill_value=None
)


def get_max_allowable_speed_fine(length_m, angle_deg, max_error_mm, step_mm_per_s=1):
    """
    Checks all speeds from 17 to 80 mm/s and returns the highest one such that
    error at (length, angle, speed) <= max_error_mm.
    """
    speeds_fine = np.arange(17, 81, step_mm_per_s)
    valid_speeds = []

    for s in speeds_fine:
        err = error_interp_3d((length_m, angle_deg, s))
        if err <= max_error_mm:
            valid_speeds.append(s)

    return max(valid_speeds) if valid_speeds else None


# === Example usage ===
result = get_max_allowable_speed_fine(1, 45, 10)
print(f"Max allowable speed: {result} mm/s")




