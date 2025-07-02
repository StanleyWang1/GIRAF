import pandas as pd
import matplotlib.pyplot as plt

# Load the frequency data CSV
df = pd.read_csv("./DATA/resonance_step_velocity/frequency_data.csv")

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(df["Boom Length (m)"], df["Sine Fit Freq (Hz)"], color='red', label="Sine Fit", alpha=0.7)
plt.scatter(df["Boom Length (m)"], df["FFT Peak Freq (Hz)"], color='blue', label="FFT Peak", alpha=0.7)

plt.xlabel("Boom Length (m)")
plt.ylabel("Frequency (Hz)")
plt.title("Frequency vs Boom Length")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
