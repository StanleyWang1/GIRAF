import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("./DATA/resonance_step_velocity/boom_11p25in_trial1.csv")  # Replace with your actual file path

# Extract columns
t = df['t']
accel_x = df['accel_x']
pitch_pos = df['pitch_pos']

# Create the figure and axis objects
fig, ax1 = plt.subplots()

# Plot accel_x on left y-axis
color1 = 'tab:red'
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Accel X (m/s²)", color=color1)
ax1.plot(t, accel_x, color=color1, label="Accel X")
ax1.tick_params(axis='y', labelcolor=color1)

# Create second y-axis for pitch_pos
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel("Pitch Position", color=color2)
ax2.plot(t, pitch_pos, color=color2, linestyle='--', label="Pitch Pos")
ax2.tick_params(axis='y', labelcolor=color2)

# Optional: Add legends
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# Show plot
plt.title("Acceleration and Pitch Position Over Time")
plt.tight_layout()
plt.show()
