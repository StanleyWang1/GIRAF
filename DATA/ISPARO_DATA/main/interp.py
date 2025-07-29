# error_interpolator.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Grid axes ===
lengths = np.array([0.6, 0.9, 1.2, 1.5, 1.8])         # meters
speeds = np.array([17, 33, 50, 67, 80])              # mm/s
angles = np.array([0, 45, 90])                       # degrees

# === Error values (length x speed x angle) ===
error_data = np.array([
    [[3.23, 3.049, 3.79],
     [4.62, 4.23, 5.26],
     [5.86, 6.21, 7.14],
     [7.21, 8.76, 10.14],
     [8.69, 10.82, 11.84]],

    [[4.14, 4.68, 5.85],
     [6.08, 8.00, 7.71],
     [8.84, 12.97, 9.97],
     [10.68, 12.80, 15.21],
     [15.67, 20.03, 19.14]],

    [[5.95, 6.68, 8.45],
     [8.79, 9.78, 14.61],
     [11.93, 12.54, 19.49],
     [16.77, 14.58, 22.02],
     [19.24, 18.58, 25.32]],

    [[11.20, 11.62, 12.12],
     [14.53, 18.62, 17.36],
     [16.25, 20.33, 25.43],
     [20.66, 20.79, 32.05],
     [21.91, 27.26, 36.01]],

    # [[13.42, 15.44, 19.67],
    #  [16.38, 23.68, 33.68],
    #  [21.92, 29.65, 50.36],
    #  [22.73, 34.81, 55.24],
    #  [29.10, 39.66, 90.73]]

    [[13.42, 15.44, 19.67],
     [16.38, 23.68, 33.68],
     [21.92, 29.65, 50.00],
     [22.73, 34.81, 50.00],
     [29.10, 39.66, 50.00]]
])

# === Interpolator ===
_interpolator = RegularGridInterpolator(
    (lengths, speeds, angles), error_data, bounds_error=False, fill_value=None
)

def get_error(length_m, speed_mmps, angle_deg):
    """Interpolated 95th percentile error in mm"""
    point = np.array([[length_m, speed_mmps, angle_deg]])
    return _interpolator(point)[0]

# def plot_circle_with_hemisphere(speed_mmps=50, resolution=100):
#     r_min, r_max = lengths.min(), lengths.max()

#     # === Hemisphere ===
#     theta = np.linspace(0, 2 * np.pi, 100)
#     phi = np.linspace(0, np.pi / 2, 100)
#     theta, phi = np.meshgrid(theta, phi)

#     x = r_max * np.sin(phi) * np.cos(theta)
#     y = r_max * np.sin(phi) * np.sin(theta)
#     z = r_max * np.cos(phi)

#     x_inner = r_min * np.sin(phi) * np.cos(theta)
#     y_inner = r_min * np.sin(phi) * np.sin(theta)
#     z_inner = r_min * np.cos(phi)

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x, y, z, color='lightblue', alpha=0.2, edgecolor='none')
#     ax.plot_surface(x_inner, y_inner, z_inner, color='lightblue', alpha=0.2, edgecolor='none')

#     # === Gradient slice from 2D quarter-circle ===
#     cmap = LinearSegmentedColormap.from_list("custom", ["lightseagreen", "orange", "orangered", "deeppink"], N=256)
#     norm = Normalize(vmin=0, vmax=50)

#     theta_2d = np.linspace(0, np.pi / 2, resolution)
#     radius = np.linspace(r_min, r_max, resolution)
#     T, R = np.meshgrid(theta_2d, radius)
#     angles_deg = np.degrees(T)

#     points = np.stack([R.ravel(), np.full_like(R.ravel(), speed_mmps), angles_deg.ravel()], axis=1)
#     E = _interpolator(points).reshape(R.shape)

#     for i in range(resolution - 1):
#         for j in range(resolution - 1):
#             quad = [
#                 [R[i, j]*np.cos(T[i, j]), 0, R[i, j]*np.sin(T[i, j])],
#                 [R[i+1, j]*np.cos(T[i+1, j]), 0, R[i+1, j]*np.sin(T[i+1, j])],
#                 [R[i+1, j+1]*np.cos(T[i+1, j+1]), 0, R[i+1, j+1]*np.sin(T[i+1, j+1])],
#                 [R[i, j+1]*np.cos(T[i, j+1]), 0, R[i, j+1]*np.sin(T[i, j+1])]
#             ]
#             val = np.mean([E[i, j], E[i+1, j], E[i+1, j+1], E[i, j+1]])
#             poly = Poly3DCollection([quad], facecolor=cmap(norm(val)), alpha=0.6, edgecolor='none')
#             ax.add_collection3d(poly)

#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")
#     ax.set_zlabel("Z (m)")
#     ax.set_title(f"3D Hemisphere with Gradient Quarter Circle Slice (Speed = {speed_mmps} mm/s)")
#     ax.set_box_aspect([1, 1, 0.6])
#     # === Hide axes, ticks, grid, and labels ===
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.set_zlabel('')
#     ax.grid(False)
    

#     # Turn off the axis entirely (removes everything including panes)
#     ax.axis('off')

#     ax.view_init(elev=30, azim=315)
#     ax.grid(False)               # Removes the 3D grid


#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# === Grid axes ===
lengths = np.array([0.6, 0.9, 1.2, 1.5, 1.8])  # meters
speeds = np.array([17, 33, 50, 67, 80])       # mm/s
angles = np.array([0, 45, 90])                # degrees

# === Error values (length x speed x angle) ===
error_data = np.array([
    [[3.23, 3.049, 3.79], [4.62, 4.23, 5.26], [5.86, 6.21, 7.14], [7.21, 8.76, 10.14], [8.69, 10.82, 11.84]],
    [[4.14, 4.68, 5.85], [6.08, 8.00, 7.71], [8.84, 12.97, 9.97], [10.68, 12.80, 15.21], [15.67, 20.03, 19.14]],
    [[5.95, 6.68, 8.45], [8.79, 9.78, 14.61], [11.93, 12.54, 19.49], [16.77, 14.58, 22.02], [19.24, 18.58, 25.32]],
    [[11.20, 11.62, 12.12], [14.53, 18.62, 17.36], [16.25, 20.33, 25.43], [20.66, 20.79, 32.05], [21.91, 27.26, 36.01]],
    [[13.42, 15.44, 19.67], [16.38, 23.68, 33.68], [21.92, 29.65, 50.00], [22.73, 34.81, 50.00], [29.10, 39.66, 50.00]]
])

_interpolator = RegularGridInterpolator((lengths, speeds, angles), error_data, bounds_error=False, fill_value=None)

def get_error(length_m, speed_mmps, angle_deg):
    return _interpolator([[length_m, speed_mmps, angle_deg]])[0]

def plot_circle_with_hemisphere(speed_mmps=50, resolution=25):
    r_min, r_max = lengths.min(), lengths.max()

    # === Hemisphere ===
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi / 2, 100)
    theta, phi = np.meshgrid(theta, phi)
    x = r_max * np.sin(phi) * np.cos(theta)
    y = r_max * np.sin(phi) * np.sin(theta)
    z = r_max * np.cos(phi)
    x_inner = r_min * np.sin(phi) * np.cos(theta)
    y_inner = r_min * np.sin(phi) * np.sin(theta)
    z_inner = r_min * np.cos(phi)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.2, edgecolor='none')
    ax.plot_surface(x_inner, y_inner, z_inner, color='lightblue', alpha=0.2, edgecolor='none')

    # === Gradient quarter circle slice ===
    cmap = LinearSegmentedColormap.from_list("custom", ["lightseagreen", "orange", "orangered", "deeppink"], N=256)
    norm = Normalize(vmin=0, vmax=50)

    theta_2d = np.linspace(0, np.pi / 2, resolution)
    radius = np.linspace(r_min, r_max, resolution)
    T, R = np.meshgrid(theta_2d, radius)
    angles_deg = np.degrees(T)
    points = np.stack([R.ravel(), np.full_like(R.ravel(), speed_mmps), angles_deg.ravel()], axis=1)
    E = _interpolator(points).reshape(R.shape)

    for i in range(resolution - 1):
        for j in range(resolution - 1):
            quad = [
                [R[i, j]*np.cos(T[i, j]), 0, R[i, j]*np.sin(T[i, j])],
                [R[i+1, j]*np.cos(T[i+1, j]), 0, R[i+1, j]*np.sin(T[i+1, j])],
                [R[i+1, j+1]*np.cos(T[i+1, j+1]), 0, R[i+1, j+1]*np.sin(T[i+1, j+1])],
                [R[i, j+1]*np.cos(T[i, j+1]), 0, R[i, j+1]*np.sin(T[i, j+1])]
            ]
            val = np.mean([E[i, j], E[i+1, j], E[i+1, j+1], E[i, j+1]])
            poly = Poly3DCollection([quad], facecolor=cmap(norm(val)), edgecolor='none', alpha=1)
            ax.add_collection3d(poly)

    # === Outline the quarter circle ===
    arc_outer = np.array([[r_max * np.cos(t), 0, r_max * np.sin(t)] for t in theta_2d])
    arc_inner = np.array([[r_min * np.cos(t), 0, r_min * np.sin(t)] for t in theta_2d])
    radial0 = np.array([[r_min, 0, 0], [r_max, 0, 0]])
    radial90 = np.array([[0, 0, r_min], [0, 0, r_max]])

    ax.plot(arc_outer[:, 0], arc_outer[:, 1], arc_outer[:, 2], color='darkblue', linewidth=2.5)
    ax.plot(arc_inner[:, 0], arc_inner[:, 1], arc_inner[:, 2], color='darkblue', linewidth=2.5)
    ax.plot(radial0[:, 0], radial0[:, 1], radial0[:, 2], color='darkblue', linewidth=2.5)
    ax.plot(radial90[:, 0], radial90[:, 1], radial90[:, 2], color='darkblue', linewidth=2.5)

    # === Final view and cleanup ===
    ax.view_init(elev=30, azim=315)
    ax.set_box_aspect([1, 1, 0.6])
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# def plot_quarter_circle_for_speed(speed_mmps, resolution=25):
#     """Plot quarter-circle of interpolated error values for fixed speed"""
#     theta = np.linspace(0, np.pi / 2, resolution)
#     radius = np.linspace(lengths.min(), lengths.max(), resolution)
#     T, R = np.meshgrid(theta, radius)
#     X = R * np.cos(T)
#     Y = R * np.sin(T)
#     angles_deg = np.degrees(T)

#     points = np.stack([R.ravel(), np.full_like(R.ravel(), speed_mmps), angles_deg.ravel()], axis=1)
#     E = _interpolator(points).reshape(R.shape)

#     fig, ax = plt.subplots(figsize=(6, 6))
#     cmap = LinearSegmentedColormap.from_list("custom", [
#         "lightseagreen", "orange", "orangered", "deeppink"
#     ], N=256)
#     norm = Normalize(vmin=0, vmax=50)
#     mesh = ax.pcolormesh(X, Y, E, shading='auto', alpha=0.5, cmap=cmap, norm=norm)
#     cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
#     cbar.set_label('95th Percentile Error (mm)', fontsize=12)

#     ax.set_title(f'Quarter Circle Error Plot\nSpeed = {speed_mmps} mm/s', fontsize=14)
#     ax.set_xlabel('Horizontal (m)')
#     ax.set_ylabel('Vertical (m)')
#     ax.axis('equal')
#     ax.set_xlim(0, lengths.max())
#     ax.set_ylim(0, lengths.max())
#     plt.tight_layout()
#     return fig


def plot_quarter_circle_for_speed(speed_mmps, resolution=100):
    """Plot quarter-circle of interpolated error values for fixed speed with contour lines"""
    theta = np.linspace(0, np.pi / 2, resolution)
    radius = np.linspace(lengths.min(), lengths.max(), resolution)
    T, R = np.meshgrid(theta, radius)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    angles_deg = np.degrees(T)

    points = np.stack([R.ravel(), np.full_like(R.ravel(), speed_mmps), angles_deg.ravel()], axis=1)
    E = _interpolator(points).reshape(R.shape)

    fig, ax = plt.subplots(figsize=(6, 6))

    # === Color map ===
    cmap = LinearSegmentedColormap.from_list("custom", [
        "lightseagreen",  # teal (low error)
        "orange",  # yellow (medium)
        "hotpink"   # red-orange (high error)
    ], N=256)
    norm = Normalize(vmin=0, vmax=50)

    # === Main heatmap
    mesh = ax.pcolormesh(X, Y, E, shading='auto', alpha=0.9, cmap=cmap, norm=norm)

    # === Contour lines ===
    contour_levels = [5, 10, 15, 20, 25, 30, 40, 50]  # adjust as needed
    contour = ax.contour(X, Y, E, levels=contour_levels, colors='black', linewidths=1.0)
    ax.clabel(contour, inline=True, fontsize=12, fmt="%d mm")

    # === Colorbar
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label('95th Percentile Error (mm)', fontsize=12)

    # === Add black outline
    arc_outer_x = lengths.max() * np.cos(theta)
    arc_outer_y = lengths.max() * np.sin(theta)
    arc_inner_x = lengths.min() * np.cos(theta)
    arc_inner_y = lengths.min() * np.sin(theta)

    ax.plot(arc_outer_x, arc_outer_y, color='darkblue', linewidth=2.5)
    ax.plot(arc_inner_x, arc_inner_y, color='darkblue', linewidth=2.5)
    ax.plot([lengths.min(), lengths.max()], [0, 0], color='darkblue', linewidth=2.5)
    ax.plot([0, 0], [lengths.min(), lengths.max()], color='darkblue', linewidth=2.5)

    ax.set_title(f'Quarter Circle Error Plot\nSpeed = {speed_mmps} mm/s', fontsize=14)
    ax.set_xlabel('Horizontal (m)')
    ax.set_ylabel('Vertical (m)')
    ax.axis('equal')
    ax.set_xlim(0, lengths.max())
    ax.set_ylim(0, lengths.max())
    plt.tight_layout()
    return fig




# def plot_quarter_circle_for_speed(speed_mmps, resolution=100):
#     """Plot quarter-circle of interpolated error values for fixed speed"""
#     theta = np.linspace(0, np.pi / 2, resolution)
#     radius = np.linspace(lengths.min(), lengths.max(), resolution)
#     T, R = np.meshgrid(theta, radius)
#     X = R * np.cos(T)
#     Y = R * np.sin(T)
#     angles_deg = np.degrees(T)

#     points = np.stack([R.ravel(), np.full_like(R.ravel(), speed_mmps), angles_deg.ravel()], axis=1)
#     E = _interpolator(points).reshape(R.shape)

#     fig, ax = plt.subplots(figsize=(6, 6))
#     # cmap = LinearSegmentedColormap.from_list("custom", [
#     #     "lightseagreen", "orange", "orangered", "deeppink"
#     # ], N=256)
#     cmap = plt.get_cmap("magma_r")
#     norm = Normalize(vmin=0, vmax=50)
#     mesh = ax.pcolormesh(X, Y, E, shading='auto', alpha=0.75, cmap=cmap, norm=norm)
#     cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
#     cbar.set_label('95th Percentile Error (mm)', fontsize=12)

#     # === Add black outline ===
#     arc_outer_x = lengths.max() * np.cos(theta)
#     arc_outer_y = lengths.max() * np.sin(theta)
#     arc_inner_x = lengths.min() * np.cos(theta)
#     arc_inner_y = lengths.min() * np.sin(theta)

#     # Arcs
#     ax.plot(arc_outer_x, arc_outer_y, color='darkblue', linewidth=2.5)
#     ax.plot(arc_inner_x, arc_inner_y, color='darkblue', linewidth=2.5)

#     # Radial lines
#     ax.plot([lengths.min(), lengths.max()], [0, 0], color='darkblue', linewidth=2.5)  # 0 deg line
#     ax.plot([0, 0], [lengths.min(), lengths.max()], color='darkblue', linewidth=2.5)  # 90 deg line

#     ax.set_title(f'Quarter Circle Error Plot\nSpeed = {speed_mmps} mm/s', fontsize=14)
#     ax.set_xlabel('Horizontal (m)')
#     ax.set_ylabel('Vertical (m)')
#     ax.axis('equal')
#     ax.set_xlim(0, lengths.max())
#     ax.set_ylim(0, lengths.max())
#     plt.tight_layout()
#     return fig
