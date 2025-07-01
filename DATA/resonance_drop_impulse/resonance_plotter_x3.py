import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
df1 = pd.read_csv("./DATA/cantilever_resonance/boom_50cm_trial1.csv")
df2 = pd.read_csv("./DATA/cantilever_resonance/boom_100cm_trial1.csv")
df3 = pd.read_csv("./DATA/cantilever_resonance/boom_150cm_trial1.csv")

# Plot accel_x side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, df, title in zip(axes, [df1, df2, df3], ['50 cm', '100 cm', '150 cm']):
    ax.plot(df['t'], df['accel_x'], label='Accel X')
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.grid(True)

axes[0].set_ylabel("Acceleration (m/s²)")
plt.suptitle("X-Axis Acceleration Across Trials")
plt.tight_layout()
plt.show()
