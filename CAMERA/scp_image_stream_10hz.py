import subprocess
import time
import cv2
import os

def main():
    remote_path = "giraf@giraf-desktop:~/Documents/MAB/CAMERA/photo_1080p.png"
    local_path = "photo_1080p.png"

    print("\033[93mStarting 10Hz SCP fetch and display loop. Press Ctrl+C to stop.\033[0m")

    while True:
        start_time = time.time()

        try:
            # Run SCP command to fetch the image
            subprocess.run(["scp", remote_path, local_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Check if file exists and load
            if os.path.exists(local_path):
                img = cv2.imread(local_path)
                if img is not None:
                    cv2.imshow("Live SCP Image", img)

            # Check for quit key
            if cv2.waitKey(1) == ord('q'):
                break

        except subprocess.CalledProcessError:
            print("\033[91mFailed to fetch image via SCP.\033[0m")

        # Timing control to maintain ~10Hz
        elapsed = time.time() - start_time
        time.sleep(max(0, 0.1 - elapsed))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
