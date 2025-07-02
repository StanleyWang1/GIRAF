import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def damped_sine(t, A, zeta, wn, phi, C):
    """Damped sinusoidal model."""
    wd = wn * np.sqrt(1 - zeta**2)
    return A * np.exp(-zeta * wn * t) * np.cos(wd * t + phi) + C

def process_and_fit(csv_path, n_last=4000):
    """
    Load CSV, crop to pitch_pos >= 0.45, zero accel_x, fit damped sine,
    and extract natural frequency and damping ratio.
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Step 1: Crop to pitch_pos >= 0.45
    first_idx = df[df['pitch_pos'] >= 0.45].index.min()
    if pd.isna(first_idx):
        print("No pitch_pos >= 0.45 found.")
        return
    df = df.loc[first_idx:].reset_index(drop=True)

    # Step 2: Zero accel_x using full mean after crop
    a_raw = df['accel_x'].values
    a_zeroed = a_raw - np.mean(a_raw)
    df['accel_x_zeroed'] = a_zeroed

    # Step 3: Prepare data for fitting
    t = df['t'].values
    a = df['accel_x_zeroed'].values

    # Initial parameter guess: [A, zeta, wn, phi, C]
    A0 = np.max(np.abs(a))
    zeta0 = 0.02
    wn0 = 2 * np.pi * 2  # Assume ~2 Hz
    phi0 = 0
    C0 = 0
    p0 = [A0, zeta0, wn0, phi0, C0]

    try:
        popt, _ = curve_fit(damped_sine, t, a, p0=p0)
    except RuntimeError:
        print("Fit failed. Try adjusting the initial guess or crop.")
        return

    A_fit, zeta_fit, wn_fit, phi_fit, C_fit = popt
    wd_fit = wn_fit * np.sqrt(1 - zeta_fit**2)
    freq_natural = wn_fit / (2 * np.pi)

    # Step 4: Plot results
    plt.figure(figsize=(10, 5))
    plt.scatter(t, a, s=4, label="Accel X (zeroed)", color="tab:red")
    plt.plot(t, damped_sine(t, *popt), label="Fitted Damped Sine", color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Accel X (zeroed, m/s²)")
    plt.title("Damped Oscillation Fit")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 5: Report results
    print(f"Fit Results:")
    print(f"  Amplitude (A):       {A_fit:.4f}")
    print(f"  Damping ratio (ζ):   {zeta_fit:.4f}")
    print(f"  Natural freq (Hz):   {freq_natural:.4f}")
    print(f"  Damped freq (Hz):    {wd_fit / (2 * np.pi):.4f}")

def main():
    csv_path = "./DATA/resonance_step_velocity/boom_31p5in_trial2.csv"
    process_and_fit(csv_path)

if __name__ == "__main__":
    main()
