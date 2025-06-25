# Compare active injected damping from observer with no damping

import pandas as pd
import matplotlib.pyplot as plt

# Load data
df1 = pd.read_csv("./DATA/observer_damping/no_damping_boom_150cm_trial2.csv")
df2 = pd.read_csv("./DATA/observer_damping/quarter_damping_boom_150cm_trial2.csv")

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Pitch position
axs[0].plot(df1['t'], df1['pitch_pos'], label='No Damping', alpha=0.7)
axs[0].plot(df2['t'], df2['pitch_pos'], label='Damping', alpha=0.7)
axs[0].set_ylabel("Pitch Position (rad)")
axs[0].set_title("Commanded Pitch Position")
axs[0].legend()
axs[0].grid(True)

# Measured acceleration
axs[1].plot(df1['t'], df1['accel_x'], label='No Damping', alpha=0.7)
axs[1].plot(df2['t'], df2['accel_x'], label='Damping', alpha=0.7)
axs[1].set_ylabel("Accel X (m/s²)")
axs[1].set_title("Measured Tip Acceleration")
axs[1].legend()
axs[1].grid(True)

# Observer estimated velocity
axs[2].plot(df1['t'], df1['x1_hat'], label='No Damping', alpha=0.7)
axs[2].plot(df2['t'], df2['x1_hat'], label='Damping', alpha=0.7)
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Estimated Displacement (m)")
axs[2].set_title("Observer Displacement Estimate x̂₂")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
