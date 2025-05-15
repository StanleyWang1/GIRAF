# jetson_camera_server.py
import socket
import cv2
import struct

def main():
    HOST = '0.0.0.0'  # Listen on all network interfaces
    PORT = 8485       # Choose any free port

    # Setup socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Server listening on {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    print(f"Client connected from {addr}")

    # Setup camera (DepthAI, webcam, etc.)
    cap = cv2.VideoCapture(0)  # Change this if you use DepthAI pipeline instead

    if not cap.isOpened():
        print("Camera could not be opened")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

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
        cap.release()

if __name__ == "__main__":
    main()
