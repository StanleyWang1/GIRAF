import cv2
import numpy as np
import glob
import os
import argparse

def calibrate_camera(image_folder, output_file, chessboard_size=(8, 5), square_size=0.00956):
    print("Starting calibration...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(image_folder, '*.png'))
    print(f"Found {len(images)} images in {image_folder}")

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            window_title = f"✔ {os.path.basename(fname)} - Corners Found"
            cv2.imshow(window_title, img)
            cv2.waitKey(300)
            cv2.destroyWindow(window_title)
        else:
            print(f"❌ Chessboard not found in {fname}")

    print(f"✔️ {len(objpoints)} valid calibration images")

    if len(objpoints) < 10:
        print("⚠️ Not enough valid calibration images (need at least 10).")
        return

    flags = cv2.CALIB_RATIONAL_MODEL
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)

    if ret:
        np.savez(output_file, camera_matrix=mtx, dist_coeffs=dist)
        print(f"✅ Calibration successful! Saved to {output_file}.npz")
        print("Camera Matrix:\n", mtx)
        print("Distortion Coefficients:\n", dist.ravel())

        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        print(f"Mean reprojection error: {total_error / len(objpoints):.4f}")
    else:
        print("❌ Calibration failed.")


def main():
    parser = argparse.ArgumentParser(description='Camera Calibration from chessboard images')
    parser.add_argument('--folder', type=str, required=True, help='Folder containing calibration .jpg images')
    parser.add_argument('--output', type=str, default='camera_intrinsics', help='Output file prefix (without .npz)')
    parser.add_argument('--rows', type=int, default=8, help='Number of inner corners along rows')
    parser.add_argument('--cols', type=int, default=5, help='Number of inner corners along columns')
    parser.add_argument('--square_size', type=float, default=0.00956, help='Size of a square in meters')

    args = parser.parse_args()
    chessboard_size = (args.rows, args.cols)

    calibrate_camera(args.folder, args.output, chessboard_size, args.square_size)


if __name__ == '__main__':
    main()
