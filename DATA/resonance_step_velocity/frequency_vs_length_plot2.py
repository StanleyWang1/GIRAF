import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load CSV
df = pd.read_csv("./DATA/resonance_step_velocity/frequency_data.csv")

x = df["Boom Length (m)"].values + 0.265
y_fft = df["FFT Peak Freq (Hz)"].values

# Define model: y = a/x + c
def inv(x, a, b, c):
    return a / x**2 + b / x + c

# Fit model
params, _ = curve_fit(inv, x, y_fft)

# Generate fit line
x_fit = np.linspace(min(x), max(x), 300)
y_fit = inv(x_fit, *params)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y_fft, color='blue', label="FFT Peak Data", alpha=0.7)
plt.plot(x_fit, y_fit, 'r--', label=f"Fit: a/x^2 + b/x + c, a={params[0]:.2f}, b={params[1]:.2f}, c={params[2]:.2f}")

plt.xlabel("Boom Length (m)")
plt.ylabel("Frequency (Hz)")
plt.title("FFT Peak Frequency vs Boom Length")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
