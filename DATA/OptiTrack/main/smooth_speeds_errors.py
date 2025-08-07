import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# === Original grid ===
lengths = np.array([1, 2, 3, 4])         # Boom length
speeds = np.array([1, 2, 3, 4])          # Speed

# Error matrix
error_matrix = np.array([
    [5.0, 6.7, 8.5, 9.5],
    [7.3, 8.5, 8.3, 10.3],
    [8.9, 10.1, 11.9, 12.9],
    [12.0, 12.5, 13.8, 14.5]
])

# Create meshgrid for original data
L, S = np.meshgrid(lengths, speeds)

# === Custom colormap ===
custom_cmap = LinearSegmentedColormap.from_list("custom", ["orange", "lightseagreen", "mediumslateblue"], N=256)

# === Plot ===
plt.figure(figsize=(8, 6))

# Filled contours using raw data
contourf = plt.contourf(L, S, error_matrix, levels=20, cmap=custom_cmap, alpha=0.6)

# Contour lines
contour = plt.contour(L, S, error_matrix, levels=20, colors='grey', linewidths=1)

# Colorbar
cbar = plt.colorbar(contourf, pad=0.02)
cbar.set_label('RMSE (mm)', fontsize=20)
cbar.ax.tick_params(labelsize=16)

# Labels and formatting
plt.xlabel('Boom Length (m)', fontsize=20, labelpad=10)
plt.ylabel('Speed', fontsize=20, labelpad=10)
plt.title('RMSE Error Contours (No Spline)', fontsize=22, pad=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)

# Save + show
plt.tight_layout()
plt.savefig("fig_error_contours_no_spline.png", dpi=300)
plt.show()
