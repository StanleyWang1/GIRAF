import socket
import cv2
import numpy as np
import struct

def recv_all(sock, size):
    """Receive exactly `size` bytes from the socket."""
    data = b""
    while len(data) < size:
        more = sock.recv(size - len(data))
        if not more:
            raise EOFError(f"Expected {size} bytes but got {len(data)} before socket closed")
        data += more
    return data

def main():
    HOST = 'localhost'   # Or remote IP, e.g., '192.168.1.100'
    PORT = 8485

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}")

    payload_size = struct.calcsize(">L")

    try:
        while True:
            # --- Read 4-byte frame length header ---
            packed_size = recv_all(client_socket, payload_size)
            frame_size = struct.unpack(">L", packed_size)[0]

            # --- Read full JPEG frame ---
            frame_data = recv_all(client_socket, frame_size)

            # --- Decode image ---
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.namedWindow('Live Feed', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Live Feed', 1920, 1080)
                cv2.imshow('Live Feed', frame)
            else:
                print("Failed to decode frame")

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
