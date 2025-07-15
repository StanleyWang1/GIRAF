# ISPARO_DATA/utils/transforms.py
import numpy as np
from scipy.spatial.transform import Rotation as R

# Fixed offset and rotation to tag frame
CUSTOM_T = np.eye(4)
CUSTOM_T[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
CUSTOM_T[:3, 3] = np.array([0.056, 0.193, -0.035])

def pose_to_transform(position, quaternion):
    T = np.eye(4)
    rot = R.from_quat(quaternion)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = position
    return T

def convert_gripper_to_tag(csv_path, table_id, gripper_id):
    import csv
    gripper_world = None
    transformed = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rb_id = int(row["rigid_body_id"])
            pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            quat = np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])])

            if rb_id == gripper_id:
                gripper_world = pos
            elif rb_id == table_id and gripper_world is not None:
                T_table = pose_to_transform(pos, quat)
                T_world_table = np.linalg.inv(T_table)
                p_hom = np.append(gripper_world, 1.0)
                p_tag = CUSTOM_T @ (T_world_table @ p_hom)
                transformed.append(p_tag[:3])
                gripper_world = None

    return np.array(transformed)