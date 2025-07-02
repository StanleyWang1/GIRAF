import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def fft_analysis(csv_path):
    # --- Load and crop data ---
    df = pd.read_csv(csv_path)
    first_idx = df[df['pitch_pos'] >= 0.45].index.min()
    if pd.isna(first_idx):
        print("No pitch_pos >= 0.45 found.")
        return
    df = df.loc[first_idx:].reset_index(drop=True)

    # --- Zero accel_x ---
    a_raw = df['accel_x'].values
    a_zeroed = a_raw - np.mean(a_raw)
    t = df['t'].values

    # --- Preprocessing: Ensure uniform time steps ---
    dt_array = np.diff(t)
    if not np.allclose(dt_array, dt_array[0], rtol=1e-3):
        print("Non-uniform timesteps detected. Interpolating...")
        # Interpolate to uniform sampling
        fs = 1 / np.median(dt_array)
        t_uniform = np.linspace(t[0], t[-1], len(t))
        a_uniform = np.interp(t_uniform, t, a_zeroed)
    else:
        fs = 1 / dt_array[0]
        t_uniform = t
        a_uniform = a_zeroed

    # --- Optional: Apply windowing to reduce spectral leakage ---
    window = np.hanning(len(a_uniform))
    a_windowed = a_uniform * window

    # --- FFT ---
    N = len(a_windowed)
    fft_vals = fft(a_windowed)
    freqs = fftfreq(N, d=1/fs)

    # Only keep positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    spectrum = 2.0 / N * np.abs(fft_vals[pos_mask])

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT of Zeroed Accel X")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Peak frequency ---
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]
    print(f"Peak frequency: {peak_freq:.3f} Hz")

def main():
    csv_path = "./DATA/resonance_step_velocity/boom_31p5in_trial2.csv"
    fft_analysis(csv_path)

if __name__ == "__main__":
    main()
