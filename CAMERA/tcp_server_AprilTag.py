import socket, struct, cv2, depthai as dai
import numpy as np
import apriltag

def get_camera_intrinsics(device):
    calib = device.readCalibration()
    intrinsics = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 640, 480))
    return intrinsics

def draw_pose(frame, rvec, tvec):
    text = f"T: {tvec.ravel()}\nR: {rvec.ravel()}"
    y0 = 30
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (10, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def estimate_pose(tag, camera_matrix, tag_size):
    object_pts = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)

    image_pts = np.array(tag.corners, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_pts, image_pts, camera_matrix, None)
    return rvec, tvec if success else (None, None)

def main():
    HOST, PORT = '0.0.0.0', 8485

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Server listening on {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    print(f"Client connected from {addr}")

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
    TAG_SIZE = 0.065  # meters, adjust as needed

    with dai.Device(pipeline) as device:
        intrinsics = get_camera_intrinsics(device)
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

        try:
            while True:
                in_rgb = rgb_queue.get()
                frame = in_rgb.getCvFrame()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                tags = detector.detect(gray)
                for tag in tags:
                    rvec, tvec = estimate_pose(tag, intrinsics, TAG_SIZE)
                    if rvec is not None:
                        draw_pose(frame, rvec, tvec)
                        for pt in tag.corners:
                            cv2.circle(frame, tuple(map(int, pt)), 5, (0, 255, 0), -1)

                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                data = jpeg.tobytes()
                conn.sendall(struct.pack(">L", len(data)) + data)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
            server_socket.close()

if __name__ == "__main__":
    main()
