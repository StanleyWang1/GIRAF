import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
df1 = pd.read_csv("./DATA/cantilever_resonance/boom_100cm_trial1.csv")
df2 = pd.read_csv("./DATA/cantilever_resonance/boom_100cm_trial2.csv")
df3 = pd.read_csv("./DATA/cantilever_resonance/boom_100cm_trial3.csv")

# Compute timesteps
df1['dt'] = df1['t'].diff()
df2['dt'] = df2['t'].diff()
df3['dt'] = df3['t'].diff()

# Plot timestep vs index
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, df, title in zip(axes, [df1, df2, df3], ['Trial 1', 'Trial 2', 'Trial 3']):
    ax.plot(df.index, df['dt'])
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.grid(True)

axes[0].set_ylabel("Timestep (s)")
plt.suptitle("Timestep (Δt) Between Samples Across Trials")
plt.tight_layout()
plt.show()
