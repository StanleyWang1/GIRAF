import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import os
import re

# --- Damped sinusoidal model ---
def damped_sine(t, A, zeta, wn, phi, C):
    wd = wn * np.sqrt(1 - zeta**2)
    return A * np.exp(-zeta * wn * t) * np.cos(wd * t + phi) + C

# --- Fit decaying sine ---
def fit_decaying_sine(t, a):
    A0 = np.max(np.abs(a))
    zeta0 = 0.02
    wn0 = 2 * np.pi * 2  # ~2 Hz
    phi0 = 0
    C0 = 0
    p0 = [A0, zeta0, wn0, phi0, C0]
    try:
        popt, _ = curve_fit(damped_sine, t, a, p0=p0, maxfev=10000)
        wn_fit = popt[2]
        freq_natural = wn_fit / (2 * np.pi)
        return freq_natural, popt
    except RuntimeError:
        return None, None

# --- FFT peak frequency ---
def fft_peak_frequency(t, a):
    dt_array = np.diff(t)
    if not np.allclose(dt_array, dt_array[0], rtol=1e-3):
        fs = 1 / np.median(dt_array)
        t_uniform = np.linspace(t[0], t[-1], len(t))
        a_uniform = np.interp(t_uniform, t, a)
    else:
        fs = 1 / dt_array[0]
        t_uniform = t
        a_uniform = a

    window = np.hanning(len(a_uniform))
    a_windowed = a_uniform * window
    N = len(a_windowed)
    fft_vals = fft(a_windowed)
    freqs = fftfreq(N, d=1/fs)
    pos_mask = freqs > 0
    spectrum = 2.0 / N * np.abs(fft_vals[pos_mask])
    freqs = freqs[pos_mask]
    peak_idx = np.argmax(spectrum)
    return freqs[peak_idx], freqs, spectrum

# --- Crop and zero accel_x ---
def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    first_idx = df[df['pitch_pos'] >= 0.45].index.min()
    if pd.isna(first_idx):
        return None
    df = df.loc[first_idx:].reset_index(drop=True)
    t = df['t'].values
    a_raw = df['accel_x'].values
    a_zeroed = a_raw - np.mean(a_raw)
    return t, a_zeroed

# --- Main batch analysis ---
def main():
    base_dir = "./DATA/resonance_step_velocity/"
    filenames = [
        "boom_2p0in_trial1.csv", "boom_2p0in_trial2.csv", "boom_2p0in_trial3.csv",
        "boom_11p25in_trial1.csv", "boom_11p25in_trial2.csv", "boom_11p25in_trial3.csv",
        "boom_21p3in_trial1.csv", "boom_21p3in_trial2.csv", "boom_21p3in_trial3.csv",
        "boom_31p5in_trial1.csv", "boom_31p5in_trial2.csv", "boom_31p5in_trial3.csv",
        "boom_42p6in_trial1.csv", "boom_42p6in_trial2.csv", "boom_42p6in_trial3.csv",
        "boom_54p1in_trial1.csv", "boom_54p1in_trial2.csv", "boom_54p1in_trial3.csv",
        "boom_65p7in_trial1.csv", "boom_65p7in_trial2.csv", "boom_65p7in_trial3.csv"
    ]

    results = []

    for fname in filenames:
        csv_path = os.path.join(base_dir, fname)
        data = process_csv(csv_path)
        if data is None:
            print(f"Skipping {fname} (no pitch trigger)")
            continue
        t, a = data

        # Run both analyses
        sine_freq, popt = fit_decaying_sine(t, a)
        fft_freq, freqs, spectrum = fft_peak_frequency(t, a)

        # Extract boom length in meters
        boom_match = re.search(r'boom_(\d+)p(\d+)', fname)
        if boom_match:
            inches = float(boom_match.group(1) + "." + boom_match.group(2))
            meters = inches * 0.0254
        else:
            meters = np.nan

        # Append results
        results.append([meters, sine_freq, fft_freq])

        # Plot and save
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].scatter(t, a, s=4, label="Accel X (zeroed)", color="tab:red")
        if popt is not None:
            axs[0].plot(t, damped_sine(t, *popt), color='black', label="Fit")
        axs[0].set_title(f"{meters:.2f} m | Sine Fit: {sine_freq:.2f} Hz")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Accel X (m/s²)")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(freqs, spectrum, color="tab:blue")
        axs[1].set_title(f"FFT: {fft_freq:.2f} Hz")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Amplitude")
        axs[1].grid(True)

        fig.suptitle(fname)
        fig.tight_layout()
        png_path = os.path.join(base_dir, fname.replace(".csv", ".png"))
        plt.savefig(png_path, dpi=300)
        plt.close(fig)

    # Convert results to NumPy array and print
    results_array = np.array(results)
    print("\n[ Boom Length (m), Sine Fit Freq (Hz), FFT Peak Freq (Hz) ]")
    print(results_array)

if __name__ == "__main__":
    main()
