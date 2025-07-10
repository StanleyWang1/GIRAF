import socket, struct, cv2, depthai as dai
import numpy as np
import queue
import time

def get_camera_intrinsics(device):
    calib = device.readCalibration()
    intrinsics = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 640, 480))
    return intrinsics

# Global snapshot trigger function
snapshot_request = queue.Queue(maxsize=1)

def get_frame(path):
    """Trigger a snapshot to be saved at the given path (grayscale)."""
    if snapshot_request.full():
        try: snapshot_request.get_nowait()
        except queue.Empty: pass
    snapshot_request.put_nowait(path)

def draw_fps(frame, fps):
    text = f"FPS: {fps:.1f}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = 30
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def run_camera_server(params=None):
    if params is None:
        params = {}
    HOST = params.get("host", "0.0.0.0")
    PORT = params.get("port", 8485)

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

    with dai.Device(pipeline) as device:
        get_camera_intrinsics(device)  # Intrinsics available if needed later
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                in_rgb = rgb_queue.get()
                frame = in_rgb.getCvFrame()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Save snapshot if requested
                if not snapshot_request.empty():
                    try:
                        path = snapshot_request.get_nowait()
                        cv2.imwrite(path, gray)
                        print(f"[camera_driver] Snapshot saved to: {path}")
                    except Exception as e:
                        print(f"[camera_driver] Failed to save snapshot: {e}")

                # Compute FPS
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time))
                prev_time = curr_time
                draw_fps(frame, fps)

                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                data = jpeg.tobytes()
                conn.sendall(struct.pack(">L", len(data)) + data)

        except Exception as e:
            print(f"[camera_driver] Error: {e}")
        finally:
            conn.close()
            server_socket.close()

# --- MAIN: Print Intrinsics ---
def main():
    print("[camera_driver] Reading intrinsics...")
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam.setFps(30)

    with dai.Device(pipeline) as device:
        K = get_camera_intrinsics(device)
        print("Intrinsic Matrix (K):")
        print(K)

if __name__ == "__main__":
    main()
