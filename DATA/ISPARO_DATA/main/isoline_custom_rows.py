import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline
from matplotlib.colors import LinearSegmentedColormap, Normalize

# === Define Speeds (X-axis / columns of matrix) ===
speeds = np.array([17, 33, 50, 67, 80])   # Speed settings (mm/s)

# === Define Boom Lengths (Y-axis / rows of matrix) ===
lengths = np.array([0.6, 0.9, 1.2, 1.5, 1.8])  # Boom lengths in meters

# === Input Error Matrix row by row (rows = lengths, columns = speeds) ===
error_row_length_1 = [3.3, 4.7, 5.9, 7.2, 8.7]
error_row_length_2 = [4.2, 6.1, 8.8, 10.7, 15.7]
error_row_length_3 = [6.0, 8.9, 11.9, 16.8, 19.2]
error_row_length_4 = [11.2, 14.5, 16.3, 20.7, 21.9]
error_row_length_5 = [13.4, 16.4, 21.9, 22.7, 29.1]

# Combine into full error matrix and transpose
error_matrix = np.array([
    error_row_length_1,
    error_row_length_2,
    error_row_length_3,
    error_row_length_4,
    error_row_length_5
]).T

# === Fit smooth spline surface ===
X, Y = np.meshgrid(lengths, speeds)
x_flat = X.ravel()
y_flat = Y.ravel()
z_flat = error_matrix.ravel()
spline = SmoothBivariateSpline(x_flat, y_flat, z_flat, s=1)

# === Create fine evaluation grid ===
lengths_fine = np.linspace(lengths.min(), lengths.max(), 100)
speeds_fine = np.linspace(speeds.min(), speeds.max(), 100)
L_fine, S_fine = np.meshgrid(lengths_fine, speeds_fine)
error_fine = spline.ev(L_fine.ravel(), S_fine.ravel()).reshape(100, 100)

# === Custom colormap ===
custom_cmap = LinearSegmentedColormap.from_list("custom", ["lightseagreen", "orange", "red"], N=256)
norm = Normalize(vmin=np.min(z_flat), vmax=np.max(z_flat))

# === Plot ===
plt.figure(figsize=(8, 6))

# Contours
# contourf = plt.contourf(L_fine, S_fine, error_fine, levels=20, cmap=custom_cmap, alpha=0.5)
# contour = plt.contour(L_fine, S_fine, error_fine, levels=20, colors='grey', linewidths=1)
# Contours
contourf = plt.contourf(L_fine, S_fine, error_fine, levels=30, cmap=custom_cmap, alpha=0.5)

# === Define specific contour levels and assign custom colors ===
contour_levels = [5, 10, 15, 20, 25, 30]
contour_colors = ['grey' if lvl not in [2, 3] else 'red' for lvl in contour_levels]

# Draw custom-colored contour lines
contour = plt.contour(
    L_fine, S_fine, error_fine,
    levels=contour_levels,
    colors=contour_colors,
    linewidths=1.5
)

# Add labels to the contour lines
plt.clabel(contour, fmt='%d', colors='black', fontsize=16)



# Overlay colored data points
for xi, yi, zi in zip(x_flat, y_flat, z_flat):
    plt.scatter(xi, yi, color=custom_cmap(norm(zi)), edgecolor='k', s=70, zorder=10, clip_on=False)

# Colorbar and labels
cbar = plt.colorbar(contourf, pad=0.02)
# cbar.set_label('RMSE (mm)', fontsize=20)
cbar.ax.tick_params(labelsize=16)

# plt.xlim(0.5, 2.0)     # Set desired x-axis limits (Boom Length)
# plt.ylim(10, 90)       # Set desired y-axis limits (Speed)

# plt.xlabel('Boom Length (m)', fontsize=20)
# plt.ylabel('Speed (mm/s)', fontsize=20)
# plt.title('95th Perc Error Contours (Spline Fit)', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(False)

plt.tight_layout()
plt.savefig("fig_error_contours_with_points.png", dpi=300)
plt.show()
