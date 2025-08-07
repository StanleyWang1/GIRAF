# ISPARO_DATA/main/compute_all_averages.py
import sys
import os
import numpy as np

# Add the utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.trajectory import average_loop

# === Settings ===
PROCESSED_DIR = "data/processed"

# === List of experiment names (without .csv/_CONVERTED) ===
EXPERIMENT_NAMES = [
    # "0722_length_24in_speed_2X",
    # "0722_length_24in_speed_4X",
    # "0722_length_24in_speed_6X",
    # "0722_length_24in_speed_8X",
    # "0722_length_24in_speed_10X",
    # "0722_length_36in_speed_2X",
    # "0722_length_36in_speed_4X",
    # "0722_length_36in_speed_6X",
    # "0722_length_36in_speed_8X",
    # "0722_length_36in_speed_10X",
    # "0722_length_48in_speed_2X",
    # "0722_length_48in_speed_4X",
    # "0722_length_48in_speed_6X",
    # "0722_length_48in_speed_8X",
    # "0722_length_48in_speed_10X",
    # "0722_length_60in_speed_2X",
    # "0722_length_60in_speed_4X",
    # "0722_length_60in_speed_6X",
    # "0722_length_60in_speed_8X",
    # "0722_length_60in_speed_10X",
    # "0722_length_72in_speed_2X",
    # "0722_length_72in_speed_4X",
    # "0722_length_72in_speed_6X",
    # "0722_length_72in_speed_8X",
    # "0722_length_72in_speed_10X",
    "0723_length_24in_speed_2X_vert",
    "0723_length_24in_speed_4X_vert",
    "0723_length_24in_speed_6X_vert",
    "0723_length_24in_speed_8X_vert",
    "0723_length_24in_speed_10X_vert",
    "0723_length_36in_speed_2X_vert",
    "0723_length_36in_speed_4X_vert",
    "0723_length_36in_speed_6X_vert",
    "0723_length_36in_speed_8X_vert",
    "0723_length_36in_speed_10X_vert",
    "0723_length_48in_speed_2X_vert",
    "0723_length_48in_speed_4X_vert",
    "0723_length_48in_speed_6X_vert",
    "0723_length_48in_speed_8X_vert",
    "0723_length_48in_speed_10X_vert",
    "0723_length_60in_speed_2X_vert",
    "0723_length_60in_speed_4X_vert",
    "0723_length_60in_speed_6X_vert",
    "0723_length_60in_speed_8X_vert",
    "0723_length_60in_speed_10X_vert",
    "0723_length_72in_speed_2X_vert",
    "0723_length_72in_speed_4X_vert",
    "0723_length_72in_speed_6X_vert",
    "0723_length_72in_speed_8X_vert",
    "0723_length_72in_speed_10X_vert",
]

# === Compute and save average trajectories ===
for name in EXPERIMENT_NAMES:
    converted_path = os.path.join(PROCESSED_DIR, f"{name}_CONVERTED.csv")
    avg_path = os.path.join(PROCESSED_DIR, f"{name}_AVG.csv")

    try:
        traj = np.loadtxt(converted_path, delimiter=',')
        avg_traj = average_loop(traj)
        np.savetxt(avg_path, avg_traj, delimiter=',')
        print(f"✅ Averaged: {converted_path} → {avg_path}")
    except Exception as e:
        print(f"❌ Failed to process {converted_path}: {e}")
