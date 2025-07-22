import cv2
import numpy as np
import glob
import os

# --- Configuration ---
image_dir = './CAMERA/CALIBRATION/DATA'
chessboard_size = (8, 5)  # number of inner corners (columns, rows)
square_size = 9.56/1000  # in m
# square_size = 1.0

# --- Prepare object points ---
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                       0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Lists to store object points and image points
objpoints = []  # 3D real world points
imgpoints = []  # 2D image points

# Get image files
image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))

# --- Checkerboard Detection + Visualization ---
for img_path in image_paths:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size,
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    display = img.copy()

    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners)
        cv2.drawChessboardCorners(display, chessboard_size, corners, ret)
        status = f"✔ Checkerboard found: {os.path.basename(img_path)}"
    else:
        status = f"✘ Checkerboard NOT found: {os.path.basename(img_path)}"

    # Show image until space is pressed
    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0) if ret else (0, 0, 255), 2)
    cv2.imshow('Checkerboard Detection', display)

    while True:
        key = cv2.waitKey(0)
        if key == 32:  # space bar
            break
        elif key == 27:  # esc to quit
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()

# --- Calibration ---
if len(objpoints) < 3:
    print("❌ Not enough valid checkerboards detected. At least 3 are recommended.")
    exit()

# flags = cv2.CALIB_RATIONAL_MODEL
flags = 0
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)

# --- Compute and print full projection matrix ---
# Use the first rotation and translation vectors
R, _ = cv2.Rodrigues(rvecs[0])
t = tvecs[0].reshape(3, 1)
Rt = np.hstack((R, t))
P = K @ Rt
print("Full Projection Matrix (P):\n", P)

# --- Output ---
print("\n✅ Calibration successful!")
print("Camera Intrinsic Matrix (K):\n", K)
print("Distortion Coefficients:\n", dist.ravel())
print(f"Reprojection error: {ret:.4f}")
