# ISPARO_DATA/main/stream_and_record.py
import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.streaming import stream_and_log

EXPERIMENT_NAME = "TEST_TRIAL"
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