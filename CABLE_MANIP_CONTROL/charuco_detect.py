import socket, struct, cv2, depthai as dai
import numpy as np
import apriltag
import queue
import time

def get_camera_intrinsics(device):
    return np.array([[704.584, 0.0,     325.885],
                     [0.0,    704.761, 245.785],
                     [0.0,    0.0,     1.0]])

def draw_pose(frame, rvec, tvec, origin):
    text = f"T: {tvec.ravel()}\nR: {rvec.ravel()}"
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.drawFrameAxes(frame, CAMERA_MATRIX, None, rvec, tvec, 0.05)

def estimate_pose(tag, camera_matrix, tag_size):
    object_pts = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)
    image_pts = np.array(tag.corners, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_pts, image_pts, camera_matrix, None)
    return (rvec, tvec) if success else (None, None)

def run_camera_server(params=None, output_queue=None):
    global CAMERA_MATRIX
    if params is None:
        params = {}
    HOST = params.get("host", "0.0.0.0")
    PORT = params.get("port", 8485)
    TAG_SIZE = params.get("tag_size", 0.037)  # in meters

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
        CAMERA_MATRIX = get_camera_intrinsics(device)
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

        fps = 0.0
        prev_time = time.time()

        try:
            while True:
                in_rgb = rgb_queue.get()
                frame = in_rgb.getCvFrame()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Update FPS
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1 / (now - prev_time))
                prev_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                tags = detector.detect(gray)
                if tags:
                    for tag in tags[:10]:  # limit to 10 tags
                        rvec, tvec = estimate_pose(tag, CAMERA_MATRIX, TAG_SIZE)
                        if rvec is not None:
                            # Draw pose axes
                            draw_pose(frame, rvec, tvec, tag.center)

                            # Draw corners and ID
                            for pt in tag.corners:
                                cv2.circle(frame, tuple(map(int, pt)), 4, (0, 255, 0), -1)
                            c = np.mean(tag.corners, axis=0).astype(int)
                            cv2.putText(frame, f"ID: {tag.tag_id}",
                                        tuple(c + np.array([5, -5])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                            # Send pose data
                            if output_queue is not None:
                                pose_data = {
                                    "id": tag.tag_id,
                                    "rvec": rvec.ravel().tolist(),
                                    "tvec": tvec.ravel().tolist(),
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

                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                conn.sendall(struct.pack(">L", len(jpeg)) + jpeg)

        except Exception as e:
            print(f"[camera_driver] Error: {e}")
        finally:
            conn.close()
            server_socket.close()
