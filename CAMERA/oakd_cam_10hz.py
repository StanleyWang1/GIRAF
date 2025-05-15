import depthai as dai
import cv2
import time

def main():
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define a color camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # Create output
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)  # Use .video for full 1080p frames

    # Start device
    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

        print("\033[93mStarting 10Hz photo capture. Press Ctrl+C to stop.\033[0m")
        while True:
            start_time = time.time()

            in_rgb = rgb_queue.get()  # Blocking wait
            frame = in_rgb.getCvFrame()

            cv2.imwrite("photo_1080p.png", frame)
            print("Photo saved as 'photo_1080p.png'.")

            # Wait so total loop time is ~0.1 sec
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.1 - elapsed))

if __name__ == "__main__":
    main()
