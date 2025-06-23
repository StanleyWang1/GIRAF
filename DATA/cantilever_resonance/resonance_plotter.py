import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("./DATA/cantilever_resonance/boom_150cm_trial3.csv")

# Plot accelerometer axes
plt.figure()
plt.plot(df['t'], df['accel_x'], label='Accel X')
plt.plot(df['t'], df['accel_y'], label='Accel Y')
plt.plot(df['t'], df['accel_z'], label='Accel Z')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.title("Accelerometer Readings")
plt.legend()
plt.grid(True)

plt.show()
