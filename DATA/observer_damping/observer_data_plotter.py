import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV
df = pd.read_csv("./DATA/observer_damping/full_damping_boom_150cm_trial2.csv")  # Replace with your path

# Calculate energy-like quantity for instability check
df["energy"] = df["x1_hat"]**2 + df["x2_hat"]**2

# Plot acceleration
plt.figure(figsize=(10, 4))
plt.plot(df["t"], df["accel_x"], label="accel_x", alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration Signal")
plt.grid(True)
plt.legend()

# Plot observer states
plt.figure(figsize=(10, 4))
plt.plot(df["t"], df["x1_hat"], label="x̂₁ (Displacement)", alpha=0.8)
plt.plot(df["t"], df["x2_hat"], label="x̂₂ (Velocity)", alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Estimate")
plt.title("Observer State Estimates")
plt.grid(True)
plt.legend()

# Plot pitch command (optional)
plt.figure(figsize=(10, 4))
plt.plot(df["t"], df["pitch_pos"], label="Pitch Pos (rad)", color='purple')
plt.xlabel("Time (s)")
plt.ylabel("Pitch Position (rad)")
plt.title("Pitch Command vs Time")
plt.grid(True)
plt.legend()

# Plot energy
plt.figure(figsize=(10, 4))
plt.plot(df["t"], df["energy"], label="x̂₁² + x̂₂²", color='darkred')
plt.xlabel("Time (s)")
plt.ylabel("Energy (arb. units)")
plt.title("Observer Estimated Energy Over Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
