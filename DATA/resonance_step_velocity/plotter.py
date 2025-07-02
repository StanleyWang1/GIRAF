import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def process_and_plot(csv_path, n_last=5502):
    """
    Process and plot acceleration and estimated displacement from a CSV.

    Steps:
    1. Trim data to first zero-crossing of accel_x_zeroed after pitch_pos >= 0.45
    2. Zero accelerations using the mean of the last `n_last` samples
    3. Estimate displacement by double integrating accel_x
    4. Plot accel_x and estimated velocity vs time with dual y-axes

    Parameters:
    - csv_path (str): Path to the input CSV
    - n_last (int): Number of final samples to compute accel offset
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Step 1: Find where pitch_pos first exceeds 0.45
    pitch_idx = df[df['pitch_pos'] >= 0.45].index.min()
    if pd.isna(pitch_idx):
        print(f"No pitch_pos >= 0.45 found in {csv_path}")
        return

    # Step 2: Zero accelerometer data first (needed to find zero crossing)
    accel_cols = ['accel_x', 'accel_y', 'accel_z']
    offsets = df[accel_cols].iloc[-n_last:].mean()
    for col in accel_cols:
        df[col + '_zeroed'] = df[col] - offsets[col]

    # Step 3: Find zero-crossing of accel_x_zeroed after pitch threshold
    a = df['accel_x_zeroed'].values
    signs = np.sign(a)
    zero_crossings = np.where((signs[:-1] * signs[1:] < 0))[0]  # indices before sign change
    valid_crossings = zero_crossings[zero_crossings >= pitch_idx]

    if len(valid_crossings) == 0:
        print(f"No zero-crossing found after pitch_pos >= 0.45 in {csv_path}")
        return

    first_zero_cross_idx = valid_crossings[0]
    df = df.loc[first_zero_cross_idx:].reset_index(drop=True)

    # Step 4: Recompute accel after trimming
    t = df['t'].values
    a = df['accel_x_zeroed'].values
    dt = np.diff(t, prepend=t[0])

    # Step 5: Integrate to get velocity and displacement
    v = np.cumsum(a * dt)
    x_disp = np.cumsum(v * dt)

    df['velocity_est'] = v
    df['x_disp_est'] = x_disp

    # Step 6: Plot accel_x and estimated velocity vs time
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Accel X (zeroed, m/s²)", color='tab:red')
    ax1.plot(t, a, color='tab:red', label='Accel X (zeroed)')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Estimated Velocity (m/s)", color='tab:blue')
    ax2.plot(t, v, color='tab:blue', linestyle='--', label='Velocity')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Accel X (Zeroed) and Estimated Velocity vs Time")
    fig.tight_layout()
    plt.grid(True)
    plt.show()

def main():
    csv_path = "./DATA/resonance_step_velocity/boom_31p5in_trial2.csv"
    process_and_plot(csv_path)

if __name__ == "__main__":
    main()
