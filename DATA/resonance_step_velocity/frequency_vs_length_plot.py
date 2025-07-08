import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load CSV
df = pd.read_csv("./DATA/resonance_step_velocity/frequency_data.csv")

x = df["Boom Length (m)"].values + 0.265
y_sine = df["Sine Fit Freq (Hz)"].values
y_fft = df["FFT Peak Freq (Hz)"].values

# Define model functions
def inv(x, a, b, c):        # y = a / x
    return a / x + c

def inv_sq(x, b, d):     # y = b / x^2
    return b / x**2 + d

# Fit both models for sine fit
a_sine, _ = curve_fit(inv, x, y_sine)
b_sine, _ = curve_fit(inv_sq, x, y_sine)

# Fit both models for FFT peak
a_fft, _ = curve_fit(inv, x, y_fft)
b_fft, _ = curve_fit(inv_sq, x, y_fft)

# Generate fit curves
x_fit = np.linspace(min(x), max(x), 300)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y_sine, color='red', label="Sine Fit Data", alpha=0.6)
plt.scatter(x, y_fft, color='blue', label="FFT Peak Data", alpha=0.6)

# Overlay fits
plt.plot(x_fit, inv(x_fit, *a_sine), 'r--', label=f"Sine Fit: a/x, a={a_sine[0]:.2f}")
plt.plot(x_fit, inv_sq(x_fit, *b_sine), 'r:', label=f"Sine Fit: b/x², b={b_sine[0]:.2f}")

plt.plot(x_fit, inv(x_fit, *a_fft), 'b--', label=f"FFT Fit: a/x, a={a_fft[0]:.2f}")
plt.plot(x_fit, inv_sq(x_fit, *b_fft), 'b:', label=f"FFT Fit: b/x², b={b_fft[0]:.2f}")

plt.xlabel("Boom Length (m)")
plt.ylabel("Frequency (Hz)")
plt.title("Frequency vs Boom Length with 1/x and 1/x² Fits")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
