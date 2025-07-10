import socket, struct, cv2, depthai as dai
import numpy as np
import queue
import time

def get_camera_intrinsics(device):
    return np.array([[704.584, 0.0,     325.885],
                     [0.0,    704.761, 245.785],
                     [0.0,    0.0,     1.0]])

def draw_pose(frame, rvec, tvec):
    text = f"T: {tvec.ravel()}\nR: {rvec.ravel()}"
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def draw_fps(frame, fps):
    text = f"FPS: {fps:.1f}"
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, text, (frame.shape[1] - size[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def run_camera_server(params=None, output_queue=None):
    if params is None:
        params = {}
    HOST = params.get("host", "0.0.0.0")
    PORT = params.get("port", 8485)
    TAG_SIZE = params.get("tag_size", 0.037)  # 37 mm

    if output_queue is not None and output_queue.maxsize != 1:
        raise ValueError("output_queue must have maxsize=1.")

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

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    with dai.Device(pipeline) as device:
        intrinsics = get_camera_intrinsics(device)
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

        fps = 0.0
        prev_time = time.time()

        try:
            while True:
                in_rgb = rgb_queue.get()
                frame = in_rgb.getCvFrame()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # FPS update
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1 / (now - prev_time))
                prev_time = now
                draw_fps(frame, fps)

                # Detect ArUco markers
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, TAG_SIZE, intrinsics, None
                    )

                    for i, marker_id in enumerate(ids.flatten()):
                        rvec = rvecs[i]
                        tvec = tvecs[i]

                        cv2.drawFrameAxes(frame, intrinsics, None, rvec, tvec, 0.05)
                        cv2.putText(frame, f"ID: {marker_id}",
                                    tuple(corners[i][0][0].astype(int) + np.array([5, -5])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        draw_pose(frame, rvec, tvec)

                        if output_queue is not None:
                            pose_data = {
                                "id": int(marker_id),
                                "rvec": rvec.ravel().tolist(),
                                "tvec": tvec.ravel().tolist()
                            }
                            if output_queue.full():
                                try: output_queue.get_nowait()
                                except queue.Empty: pass
                            output_queue.put_nowait(pose_data)
                else:
                    if output_queue is not None:
                        if output_queue.full():
                            try: output_queue.get_nowait()
                            except queue.Empty: pass
                        output_queue.put_nowait({"id": None})

                # Encode and stream frame
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                conn.sendall(struct.pack(">L", len(jpeg)) + jpeg)

        except Exception as e:
            print(f"[camera_driver] Error: {e}")
        finally:
            conn.close()
            server_socket.close()
