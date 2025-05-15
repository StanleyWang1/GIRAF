# laptop_camera_client.py
import socket
import cv2
import numpy as np
import struct

def main():
    HOST = 'localhost'   # After SSH tunnel, this will be localhost
    PORT = 8485

    # Connect socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}")

    data = b""
    payload_size = struct.calcsize(">L")

    try:
        while True:
            # Read message length
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                data += packet

            packed_size = data[:payload_size]
            data = data[payload_size:]
            frame_size = struct.unpack(">L", packed_size)[0]

            # Read frame data
            while len(data) < frame_size:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                data += packet

            frame_data = data[:frame_size]
            data = data[frame_size:]

            # Decode and show frame
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow('Live Feed', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
