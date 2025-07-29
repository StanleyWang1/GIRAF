import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline
from matplotlib.colors import LinearSegmentedColormap

# === Original grid ===
lengths = np.array([1, 2, 3, 4])         # Boom length
speeds = np.array([1, 2, 3, 4])            # Speed

# Error matrix
error_matrix = np.array([
    [5.0, 6.7, 8.5, 9.5],
    [7.3, 8.5, 8.3, 10.3],
    [8.9, 10.1, 11.9, 12.9],
    [12.0, 12.5, 13.8, 14.5]
])

# Flatten the grid and values for fitting
X, Y = np.meshgrid(lengths, speeds)
x_flat = X.ravel()
y_flat = Y.ravel()
z_flat = error_matrix.ravel()

# === Fit smooth surface ===
spline = SmoothBivariateSpline(x_flat, y_flat, z_flat, s=1)

# Generate fine grid
lengths_fine = np.linspace(lengths.min(), lengths.max(), 100)
speeds_fine = np.linspace(speeds.min(), speeds.max(), 100)
L_fine, S_fine = np.meshgrid(lengths_fine, speeds_fine)

# Evaluate spline on fine grid
error_fine = spline.ev(L_fine.ravel(), S_fine.ravel()).reshape(100, 100)

# === Custom colormap: replace colors below with any 3 you like
custom_cmap = LinearSegmentedColormap.from_list("custom", ["orange", "lightseagreen", "mediumslateblue"], N=256)

# === Plot ===
plt.figure(figsize=(8, 6))

# Filled contours (smooth) with custom colormap
contourf = plt.contourf(L_fine, S_fine, error_fine, levels=55, cmap=custom_cmap, alpha=0.5)

# Contour lines
contour = plt.contour(L_fine, S_fine, error_fine, levels=55, colors='grey', linewidths=1)

# Colorbar
cbar = plt.colorbar(contourf, pad=0.02)
cbar.set_label('RMSE (mm)', fontsize=20)
cbar.ax.tick_params(labelsize=16)

# Labels and formatting
plt.xlabel('Boom Length (m)', fontsize=20, labelpad=10)
plt.ylabel('Speed (idk)', fontsize=20, labelpad=10)
plt.title('RMSE Error Contours (Spline Fit)', fontsize=22, pad=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(False)

# Save + show
plt.tight_layout()
plt.savefig("fig_error_contours_custom_cmap.png", dpi=300)
plt.show()
