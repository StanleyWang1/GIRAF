import socket, struct, cv2, depthai as dai
import numpy as np
import apriltag
import queue
import time
from scipy.spatial.transform import Rotation as R, Slerp

def get_camera_intrinsics(device):
    # Alternatively, use actual calibration:
        # calib = device.readCalibration()
        # return np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 640, 480))
    # MATLAB Calibration:
        # return np.array([[704.584, 0.0,     325.885],
        #                  [0.0,    704.761, 245.785],
        #                  [0.0,    0.0,     1.0]])
    # OpenCV Calibration:
    return np.array([[689.31326763, 0.0, 326.24435805],
                     [0.0, 687.91253208, 245.33734652],
                     [0.0, 0.0, 1.0]])

def get_camera_distortion():
    return np.array([ 2.44017168e-01, -1.76348449e+00, -2.33728618e-03, -1.30995964e-03, 5.22669668e+00])

def draw_pose(frame, rvec, tvec):
    text = f"T: {tvec.ravel()}\nR: {rvec.ravel()}"
    y0 = 30
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (10, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def draw_fps(frame, fps):
    text = f"FPS: {fps:.1f}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = 30
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def estimate_pose(tag, camera_matrix, dist_coeffs, tag_size):
    object_pts = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)
    image_pts = np.array(tag.corners, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_pts, image_pts, camera_matrix, dist_coeffs)

    if not success:
        return None, None, None

    # Compute reprojection error
    projected_pts, _ = cv2.projectPoints(object_pts, rvec, tvec, camera_matrix, None)
    error = np.linalg.norm(image_pts - projected_pts.squeeze(), axis=1)
    mean_error = np.mean(error)
    
    # Convert to weight (lower error = higher weight)
    epsilon = 1e-6
    weight = 1.0 / (mean_error + epsilon)

    return rvec, tvec, weight

def weighted_average_transforms(prev_matrix, new_matrix, weight=0.5):
    """
    Compute weighted average of two 4x4 transformation matrices.
    Args:
        prev_matrix (np.ndarray): Previous 4x4 transform matrix.
        new_matrix (np.ndarray): New 4x4 transform matrix.
        weight (float): Weight for new_matrix; in [0,1].
                        0 returns prev_matrix, 1 returns new_matrix.
    Returns:
        np.ndarray: Averaged 4x4 transform matrix.
    """
    prev_matrix = np.array(prev_matrix)
    new_matrix = np.array(new_matrix)

    # Weighted average of translations
    prev_trans = prev_matrix[:3, 3]
    new_trans = new_matrix[:3, 3]
    avg_trans = (1 - weight) * prev_trans + weight * new_trans

    # Convert rotation parts to Rotation objects
    prev_rot = R.from_matrix(prev_matrix[:3, :3])
    new_rot = R.from_matrix(new_matrix[:3, :3])

    # Prepare times and rotations for Slerp
    key_times = [0, 1]  # Correspond to prev_rot and new_rot
    key_rots = R.concatenate([prev_rot, new_rot])
    slerp = Slerp(key_times, key_rots)

    # Interpolate rotation at given weight (time)
    avg_rot = slerp([weight]).as_matrix()[0]

    # Assemble averaged transform matrix
    avg_transform = np.eye(4)
    avg_transform[:3, :3] = avg_rot
    avg_transform[:3, 3] = avg_trans

    return avg_transform

def run_camera_server(params=None, output_queue=None):
    if params is None:
        params = {}
    HOST = params.get("host", "0.0.0.0")
    PORT = params.get("port", 8485)
    TAG_SIZE = params.get("tag_size", 0.04901)  # in meters

    if output_queue is not None and output_queue.maxsize != 1:
        raise ValueError("output_queue must have maxsize=1")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"[camera_driver] Server listening on {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    print(f"[camera_driver] Client connected from {addr}")

    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam.setFps(30)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    detector = apriltag.Detector()

    with dai.Device(pipeline) as device:
        intrinsics = get_camera_intrinsics(device)
        dist = get_camera_distortion()

        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

        prev_time = time.time()
        fps = 0.0

        # tag_hierarchy = {11:1, 12:2, 13:5, 14:8, 15:3, 16:4, 17:9, 18:6, 19:7, 20:10}
        tag_hierarchy = {11:5, 12:4, 13:3, 14:2, 15:1}
        try:
            while True:
                in_rgb = rgb_queue.get()
                frame = in_rgb.getCvFrame()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Compute FPS
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time))
                prev_time = curr_time
                draw_fps(frame, fps)

                pose_list = []
                tags = detector.detect(gray)

                if tags:
                    # Find the tag with the lowest id
                    min_tag = min(tags, key=lambda t: tag_hierarchy.get(t.tag_id))
                    tag = min_tag
                    rvec, tvec, weight = estimate_pose(tag, intrinsics, dist, TAG_SIZE)
                    if rvec is not None:
                        draw_pose(frame, rvec, tvec)
                        for pt in tag.corners:
                            cv2.circle(frame, tuple(map(int, pt)), 5, (0, 255, 0), -1)

                        pose_list.append({
                            "id": tag.tag_id,
                            "rvec": rvec.ravel().tolist(),
                            "tvec": tvec.ravel().tolist(),
                            "weight": weight
                        })

                # Send full list of tag poses to the output queue
                if output_queue is not None:
                    if output_queue.full():
                        try: output_queue.get_nowait()
                        except queue.Empty: pass
                    output_queue.put_nowait(pose_list)  # [] if no tags

                # JPEG encode and send over TCP
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                conn.sendall(struct.pack(">L", len(jpeg)) + jpeg.tobytes())

        except Exception as e:
            print(f"[camera_driver] Error: {e}")
        finally:
            conn.close()
            server_socket.close()
