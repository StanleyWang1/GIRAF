import pandas as pd
import numpy as np

def compute_speeds_from_csv(filename, rigid_body_id, num_intervals=5):
    # Load the CSV
    df = pd.read_csv(filename)

    # Filter for specific rigid body
    df = df[df['rigid_body_id'] == rigid_body_id].copy()

    # Sort by timestamp
    df.sort_values('timestamp', inplace=True)

    # Compute time and position differences
    df['dt'] = df['timestamp'].diff()
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dz'] = df['z'].diff()

    # Compute instantaneous speed (magnitude of velocity)
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2) / df['dt']

    # Drop NaNs from diff
    df = df.dropna()

    # Split into equal intervals in time
    time_start = df['timestamp'].iloc[0]
    time_end = df['timestamp'].iloc[-1]
    interval_edges = np.linspace(time_start, time_end, num_intervals + 1)

    avg_speeds = []
    for i in range(num_intervals):
        t0, t1 = interval_edges[i], interval_edges[i + 1]
        segment = df[(df['timestamp'] >= t0) & (df['timestamp'] < t1)]
        if not segment.empty:
            avg_speed = segment['speed'].mean()
            avg_speeds.append((t0, t1, avg_speed))
        else:
            avg_speeds.append((t0, t1, np.nan))

    return avg_speeds

# === Example usage ===
if __name__ == "__main__":
    filename = "data/raw_optitrack/0724_CLIVE_DEMO_2.csv"  # Change this to your actual CSV path
    rigid_body_id = 16
    num_intervals = 20

    speeds = compute_speeds_from_csv(filename, rigid_body_id, num_intervals)
    for t0, t1, v in speeds:
        print(f"Interval {t0:.2f} to {t1:.2f} → Avg Speed: {v:.4f} m/s")
