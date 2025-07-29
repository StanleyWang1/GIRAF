import numpy as np
import matplotlib.pyplot as plt

# Define speeds and lengths
speeds = np.array([17, 33, 50, 67, 80])
lengths = np.array([0.6, 0.9, 1.2, 1.5, 1.8])  # in meters

# Error matrix: rows = lengths, columns = speeds
errors = np.array([
    [1.6, 1.9, 2.3, 2.8, 3.1],   # 24in (0.6m)
    [1.7, 3.1, 5.6, 4.6, 9.9],   # 36in (0.9m)
    [3.2, 4.8, 7.5, 10.0, 10.5], # 48in (1.2m)
    [2.9, 5.2, 6.8, 9.2, 10.5],  # 60in (1.5m)
    [4.5, 7.5, 8.5, 10.6, 13.9]  # 72in (1.8m)
])

# Flatten both dimensions
speed_grid, length_grid = np.meshgrid(speeds, lengths)
speedxlength = (speed_grid * length_grid).flatten()
error_flat = errors.flatten()

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(speedxlength, error_flat, color='mediumslateblue', label='Data Points')

# Line of best fit (linear)
coeffs = np.polyfit(speedxlength, error_flat, deg=1)
fit_line = np.poly1d(coeffs)
x_fit = np.linspace(min(speedxlength), max(speedxlength), 100)
plt.plot(x_fit, fit_line(x_fit), color='crimson', linestyle='--', label=f'Best Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}', linewidth=2)

# Labels and formatting
plt.xlabel("Speed × Length (mm·m/s)", fontsize=14)
plt.ylabel("95th Percentile Error (mm)", fontsize=14)
plt.title("Error vs. Speed × Length", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
