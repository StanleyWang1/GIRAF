# jetson_camera_server_depthai.py
import socket
import cv2
import struct
import depthai as dai

def main():
    HOST = '0.0.0.0'
    PORT = 8485

    # Setup socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Server listening on {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    print(f"Client connected from {addr}")

    # Setup DepthAI pipeline
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setFps(30)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

        try:
            while True:
                in_rgb = rgb_queue.get()
                frame = in_rgb.getCvFrame()

                # Compress frame as JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # Send the length of the jpeg first
                data = jpeg.tobytes()
                size = len(data)
                conn.sendall(struct.pack(">L", size) + data)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
            server_socket.close()

if __name__ == "__main__":
    main()
