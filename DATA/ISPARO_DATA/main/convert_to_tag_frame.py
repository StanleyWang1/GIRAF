# ISPARO_DATA/main/convert_to_tag_frame.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.transforms import convert_gripper_to_tag
import numpy as np

EXPERIMENT_NAME = "ROBOT_STANLEY_TRAJ_1"
raw_path = f"data/raw_optitrack/{EXPERIMENT_NAME}.csv"
converted_path = f"data/processed/{EXPERIMENT_NAME}_CONVERTED.csv"

converted = convert_gripper_to_tag(
    csv_path=raw_path,
    table_id=10,
    gripper_id=6
)
np.savetxt(converted_path, converted, delimiter=',')
print(f"✅ Saved converted trajectory to {converted_path}")
