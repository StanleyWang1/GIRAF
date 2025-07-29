import numpy as np
from scipy.spatial.transform import Rotation as R

# === 1. Define arbitrary Euler angles for current camera and tag orientations ===
# Format: (roll [x], pitch [y], yaw [z]) in degrees
euler_cam_deg = [10, 20, 30]  # Starting orientation of the camera in global frame
euler_tag_deg = [-15, 5, 60]  # Starting orientation of the tag in global frame

# Convert to rotation matrices
R_CG_cur = R.from_euler('xyz', euler_cam_deg, degrees=True).as_matrix()
R_TG_cur = R.from_euler('xyz', euler_tag_deg, degrees=True).as_matrix()

# === 2. Compute current camera-in-tag frame: R_CT_cur = R_CG_cur^T * R_TG_cur ===
R_CT_cur = R_CG_cur.T @ R_TG_cur

# === 3. Define desired camera-in-tag rotation (e.g., 45 deg about X axis) ===
# You can change this to test different relative goals
desired_relative_euler_deg = [-45, 0, 0]  # e.g., pitch down 45 deg
R_CT_des = R.from_euler('xyz', desired_relative_euler_deg, degrees=True).as_matrix()

# === 4. Compute desired camera global orientation ===
R_CG_des = R_CG_cur @ R_CT_cur.T @ R_CT_des

# === 5. Extract final Euler angles (global XYZ) to command robot wrist ===
euler_des = R.from_matrix(R_CG_des).as_euler('xyz', degrees=True)

# === Output ===
np.set_printoptions(precision=3, suppress=True)
print("Initial Camera Euler XYZ (deg):", euler_cam_deg)
print("Initial Tag Euler XYZ (deg):   ", euler_tag_deg)
print("Desired Camera-in-Tag Euler XYZ (deg):", desired_relative_euler_deg)
print("\nR_CG_des (final camera orientation in global frame):\n", R_CG_des)
print("Final Euler angles to send to wrist (XYZ, degrees):", np.round(euler_des, 2))
