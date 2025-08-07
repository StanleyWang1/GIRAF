import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

# Load CSV
df = pd.read_csv("./DATA/deployer_blossoming/task_logs/boom_log_20250722_162913.csv")  # Replace with your actual filename

# Trim time between 50 and 200 seconds
# df = df[(df["timestamp"] >= 50) & (df["timestamp"] <= 200)].copy()
# df["timestamp"] -= 50  # Reset 50s to 0s

# Compute baselines from first 10 rows (after trimming)
baseline_d3_pos = df["d3_pos"][:10].mean()
baseline_d3_real = df["d3_real"][:10].mean()

# Normalize
df["d3_pos_norm"] = df["d3_pos"] - baseline_d3_pos
df["d3_real_norm"] = df["d3_real"] - baseline_d3_real

# Compute discrete-time derivatives (velocity)
dt = np.diff(df["timestamp"])
dd3_pos = np.diff(df["d3_pos"])
dd3_real = np.diff(df["d3_real"])
vel_d3_pos = dd3_pos / dt
vel_d3_real = dd3_real / dt

# Apply median filter (kernel size = 9)
vel_d3_pos_filtered = median_filter(vel_d3_pos, size=5)
vel_d3_real_filtered = median_filter(vel_d3_real, size=5)

# Trim timestamp for velocity plot (one shorter than original)
timestamp_vel = df["timestamp"].iloc[1:]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Subplot 1: Normalized positions
ax1.plot(df["timestamp"], df["d3_pos_norm"], label="d3 (commanded)")
ax1.plot(df["timestamp"], df["d3_real_norm"], label="d3 (actual)")
ax1.set_ylabel("Boom Length (m)")
ax1.set_title("Boom Length [m]")
ax1.legend()
ax1.grid(True)

# Subplot 2: Filtered velocities
ax2.plot(timestamp_vel, vel_d3_pos_filtered, label="d3_dot (commanded)", linestyle="-", color="tab:blue")
ax2.plot(timestamp_vel, vel_d3_real_filtered, label="d3_dot (actual)", linestyle="-", color="tab:red")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity (m/s)")
ax2.set_title("Boom Velocity [m/s]")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
