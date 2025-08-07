import sys
import os
import numpy as np

# Add the utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.transforms import convert_gripper_to_tag

# === Settings ===
RAW_DIR = "data/raw_optitrack"
PROCESSED_DIR = "data/processed"
TABLE_ID = 10
GRIPPER_ID = 6

# === List of filenames (without .csv) ===
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
    # "0723_length_24in_speed_2X_vert",
    # "0723_length_24in_speed_4X_vert",
    # "0723_length_24in_speed_6X_vert",
    # "0723_length_24in_speed_8X_vert",
    # "0723_length_24in_speed_10X_vert",
    # "0723_length_36in_speed_2X_vert",
    # "0723_length_36in_speed_4X_vert",
    # "0723_length_36in_speed_6X_vert",
    # "0723_length_36in_speed_8X_vert",
    # "0723_length_36in_speed_10X_vert",
    # "0723_length_48in_speed_2X_vert",
    # "0723_length_48in_speed_4X_vert",
    # "0723_length_48in_speed_6X_vert",
    # "0723_length_48in_speed_8X_vert",
    # "0723_length_48in_speed_10X_vert",
    # "0723_length_60in_speed_2X_vert",
    # "0723_length_60in_speed_4X_vert",
    # "0723_length_60in_speed_6X_vert",
    # "0723_length_60in_speed_8X_vert",
    # "0723_length_60in_speed_10X_vert",
    # "0723_length_72in_speed_2X_vert",
    # "0723_length_72in_speed_4X_vert",
    # "0723_length_72in_speed_6X_vert",
    # "0723_length_72in_speed_8X_vert",
    # "0723_length_72in_speed_10X_vert",
    "0724_length_24in_speed_2X_45",
    "0724_length_24in_speed_4X_45",
    "0724_length_24in_speed_6X_45",
    "0724_length_24in_speed_8X_45",
    "0724_length_24in_speed_10X_45",
    "0724_length_36in_speed_2X_45",
    "0724_length_36in_speed_4X_45",
    "0724_length_36in_speed_6X_45",
    "0724_length_36in_speed_8X_45",
    "0724_length_36in_speed_10X_45",
    "0724_length_48in_speed_2X_45",
    "0724_length_48in_speed_4X_45",
    "0724_length_48in_speed_6X_45",
    "0724_length_48in_speed_8X_45",
    "0724_length_48in_speed_10X_45",
    "0724_length_60in_speed_2X_45",
    "0724_length_60in_speed_4X_45",
    "0724_length_60in_speed_6X_45",
    "0724_length_60in_speed_8X_45",
    "0724_length_60in_speed_10X_45",
    "0724_length_72in_speed_2X_45",
    "0724_length_72in_speed_4X_45",
    "0724_length_72in_speed_6X_45",
    "0724_length_72in_speed_8X_45",
    "0724_length_72in_speed_10X_45",
]

# === Run conversion ===
for name in EXPERIMENT_NAMES:
    raw_path = os.path.join(RAW_DIR, f"{name}.csv")
    converted_path = os.path.join(PROCESSED_DIR, f"{name}_CONVERTED.csv")

    try:
        converted = convert_gripper_to_tag(
            csv_path=raw_path,
            table_id=TABLE_ID,
            gripper_id=GRIPPER_ID
        )
        np.savetxt(converted_path, converted, delimiter=',')
        print(f"✅ Converted: {raw_path} → {converted_path}")
    except Exception as e:
        print(f"❌ Failed to convert {raw_path}: {e}")
