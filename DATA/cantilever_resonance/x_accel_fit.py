import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load CSV
df = pd.read_csv("./DATA/cantilever_resonance/boom_150cm_trial3.csv")

# Normalize accel_x using last 100 samples
x_offset = df['accel_x'].iloc[-100:].mean()
df['accel_x_norm'] = df['accel_x'] - x_offset

# Crop to t >= 4 seconds
df = df[df['t'] >= 4.0]

# Extract time and normalized accel
t = df['t'].values
x = df['accel_x_norm'].values

# Damped sinusoid model
def damped_sine(t, A, zeta, wn, phi, C):
    wd = wn * np.sqrt(1 - zeta**2)
    return A * np.exp(-zeta * wn * t) * np.cos(wd * t + phi) + C

# Initial guess
p0 = [np.max(np.abs(x)), 0.02, 2*np.pi*2, 0, 0]

# Fit curve
params, _ = curve_fit(damped_sine, t, x, p0=p0, maxfev=10000)
A_fit, zeta_fit, wn_fit, phi_fit, C_fit = params

# Compute fitted curve
t_fit = np.linspace(t[0], t[-1], 1000)
x_fit = damped_sine(t_fit, *params)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(t, x, s=8, color='grey', alpha=0.5, label='Accel X (normalized)')
plt.plot(t_fit, x_fit, color='red', label='Fitted Damped Sinusoid')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (normalized)")
plt.title("Damped Sinusoid Fit to Impulse Response")
plt.grid(True)
plt.legend()
plt.show()

# Print extracted parameters
print(f"Estimated ω_n (rad/s): {wn_fit:.3f}")
print(f"Estimated ζ (damping ratio): {zeta_fit:.4f}")
