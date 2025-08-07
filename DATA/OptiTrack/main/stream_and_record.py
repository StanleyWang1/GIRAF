# ISPARO_DATA/main/stream_and_record.py
import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.streaming import stream_and_log
# speeds are slow mid fast
# lengths are short medium long
# 0.4m / 24 sec = 0.0167 m/s

# 2X: 400 mm / 24 sec = 16.67 mm/s
# 4X: 400 mm / 12 sec = 33.33 mm/s
# 6X: 400 mm / 8 sec = 50 mm/s
# 8X: 400 mm / 6 sec = 66.67 mm/s
# 10X: 400 mm / 5 sec = 80 mm/s

# 24in = 0.6096 m
# 36in = 0.9144 m
# 48in = 1.2192 m
# 60in = 1.524 m
# 72in = 1.8288 m

LENGTH = "XXin"
SPEED = "XX"
EXPERIMENT_NAME = "0724_CLIVE_3_HOOKS_2"
# EXPERIMENT_NAME = f"0724_length_{LENGTH}_speed_{SPEED}_45"

#EXPERIMENT_NAME = "0722_corrected_kff_10X_100_slow"
CONFIG_PATH = "config/streaming.yaml"

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

stream_and_log(
    client_ip=cfg["client_ip"],
    server_ip=cfg["server_ip"],
    table_id=cfg["table_id"],
    gripper_id=cfg["gripper_id"],
    csv_filename=f"data/raw_optitrack/{EXPERIMENT_NAME}.csv"
)