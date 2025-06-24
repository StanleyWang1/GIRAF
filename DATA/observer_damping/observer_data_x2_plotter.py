import pandas as pd
import matplotlib.pyplot as plt

# Load both files
df1 = pd.read_csv("./DATA/observer_damping/no_damping_boom_150cm_trial2.csv")
df2 = pd.read_csv("./DATA/observer_damping/quarter_damping_boom_150cm_trial1.csv")

# Plot acceleration
plt.figure(figsize=(10, 4))
plt.plot(df1['t'], df1['accel_x'], label='No Damping', alpha=0.7)
plt.plot(df2['t'], df2['accel_x'], label='Quarter Damping', alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Accel X (m/s²)")
plt.title("Tip Acceleration Comparison")
plt.legend()
plt.grid(True)

# Plot estimated displacement x̂₁
plt.figure(figsize=(10, 4))
plt.plot(df1['t'], df1['x1_hat'], label='No Damping', alpha=0.7)
plt.plot(df2['t'], df2['x1_hat'], label='Quarter Damping', alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Estimated Tip Displacement")
plt.title("Estimated x̂₁ (Displacement) Comparison")
plt.legend()
plt.grid(True)

# Plot estimated velocity x̂₂
plt.figure(figsize=(10, 4))
plt.plot(df1['t'], df1['x2_hat'], label='No Damping', alpha=0.7)
plt.plot(df2['t'], df2['x2_hat'], label='Quarter Damping', alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Estimated Tip Velocity")
plt.title("Estimated x̂₂ (Velocity) Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
