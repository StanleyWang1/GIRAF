# ISPARO_DATA/main/compute_average.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.trajectory import average_loop, sliding_pca_average, average_loop_3d, crop_motion_segment, average_linear_bins
import numpy as np

EXPERIMENT_NAME = "0722_length_36in_speed_10X"  # Name of the experiment
converted_path = f"data/processed/{EXPERIMENT_NAME}_CONVERTED.csv"
avg_path = f"data/processed/{EXPERIMENT_NAME}_AVG.csv"

traj = np.loadtxt(converted_path, delimiter=',')

avg_traj = average_loop(traj)
#avg_traj = sliding_pca_average([traj])

np.savetxt(avg_path, avg_traj, delimiter=',')
print(f"✅ Saved average trajectory to {avg_path}")