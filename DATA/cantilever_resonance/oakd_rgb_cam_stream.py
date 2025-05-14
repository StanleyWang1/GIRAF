# oakd_rgb_cam_stream.py

import depthai as dai
import cv2

def main():
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define a color camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)  # Max sensor res
    cam_rgb.setPreviewSize(640, 480)  # Output small preview (480p)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setFps(30)

    # Create output
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # Start device
    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            in_rgb = rgb_queue.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                cv2.imshow("OAK-D S2 RGB Stream", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
