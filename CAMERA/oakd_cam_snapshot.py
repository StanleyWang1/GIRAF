# oakd_take_photo.py

import depthai as dai
import cv2

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
    cam_rgb.video.link(xout_rgb.input)  # <-- use `.video` instead of `.preview` to get full 1080p frames

    # Start device
    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

        # Grab a single frame
        in_rgb = rgb_queue.get()  # Blocking wait
        frame = in_rgb.getCvFrame()

        # Save the frame
        cv2.imwrite("photo_1080p.png", frame)
        print("Photo saved as 'photo_1080p.png'.")

if __name__ == "__main__":
    main()
