import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("./DATA/observer_damping/no_damping_boom_150cm_trial1.csv")  # Replace with your actual file path

# Time
t = df["t"]

# Extract signals
accel = df["accel_x"]
pitch_pos = df["pitch_pos"]
x1_hat = df["x1_hat"]  # Estimated position
x2_hat = df["x2_hat"]  # Estimated velocity

# Plot acceleration
plt.figure(figsize=(10, 4))
plt.plot(t, accel, label="Measured Acceleration", alpha=0.6)
plt.xlabel("Time (s)")
plt.ylabel("Accel X (m/s²)")
plt.grid(True)
plt.legend()
plt.title("Accelerometer Reading")

# Plot estimated state vs command
plt.figure(figsize=(10, 4))
plt.plot(t, pitch_pos, label="Base Pitch Pos (integrated)")
plt.plot(t, x1_hat, label="Estimated Tip Displacement")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (rad or m)")
plt.grid(True)
plt.legend()
plt.title("Pitch Command vs Estimated Tip Displacement")

# Plot estimated velocity
plt.figure(figsize=(10, 4))
plt.plot(t, x2_hat, label="Estimated Tip Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s or m/s)")
plt.grid(True)
plt.legend()
plt.title("Estimated Tip Velocity (Observer)")

plt.tight_layout()
plt.show()
