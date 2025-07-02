import depthai as dai
import cv2
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define color camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam_rgb.setFps(30)

# Create output
xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
cam_rgb.video.link(xout_video.input)

# Start device
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=30, blocking=False)

    # Get first frame to determine size
    frame = video_queue.get().getCvFrame()
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter('./CAMERA/recording2.mp4', fourcc, 30, (width, height))

    input("Press Enter to start recording...")
    print("Recording from OAK-D... Press Ctrl+C to stop.")

    try:
        while True:
            in_frame = video_queue.get()
            frame = in_frame.getCvFrame()
            out.write(frame)
            time.sleep(1.0 / 30)  # Match the FPS set in VideoWriter

            # Optional preview
            # cv2.imshow("OAK-D Preview", frame)
            # if cv2.waitKey(1) == ord('q'):
            #     break

    except KeyboardInterrupt:
        print("Recording stopped.")

    finally:
        out.release()
        cv2.destroyAllWindows()
        print("Video saved as 'oakd_recording.mp4'")
