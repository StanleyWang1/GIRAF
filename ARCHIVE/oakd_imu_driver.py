# oakd_imu_driver.py

import depthai as dai
import time
import threading
import os

def imu_data_collector(running_flag, csv_path="./DATA/cantilever_resonance/test.csv"):
    # Set up pipeline
    pipeline = dai.Pipeline()

    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)  # Only accelerometer
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    xout_imu = pipeline.create(dai.node.XLinkOut)
    xout_imu.setStreamName("imu")
    imu.out.link(xout_imu.input)

    # Make sure the folder exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Start device
    with dai.Device(pipeline) as device:
        imu_queue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)

        # Prepare CSV
        with open(csv_path, 'w') as f:
            f.write("timestamp,accel_x,accel_y,accel_z\n")

        buffer = []
        buffer_size = 250  # ~0.5 sec worth

        while running_flag():
            imuData = imu_queue.tryGet()
            if imuData is not None:
                for imuPacket in imuData.packets:
                    acceleroValues = imuPacket.acceleroMeter
                    if acceleroValues is not None:
                        ts = acceleroValues.getTimestampDevice().total_seconds()
                        buffer.append((ts, acceleroValues.x, acceleroValues.y, acceleroValues.z))

                if len(buffer) >= buffer_size:
                    with open(csv_path, 'a') as f:
                        for row in buffer:
                            f.write(','.join(map(str, row)) + '\n')
                    buffer = []

            time.sleep(0.001)  # Light sleep (~1ms)

        # Final flush
        if buffer:
            with open(csv_path, 'a') as f:
                for row in buffer:
                    f.write(','.join(map(str, row)) + '\n')

## ----------------------------------------------------------------------------------------------------
# Main Entry
## ----------------------------------------------------------------------------------------------------
def main():
    running = False

    def running_flag():
        return running

    print("\033[93mPress Enter to start 10-second IMU recording...\033[0m")
    input()

    running = True
    imu_thread = threading.Thread(target=imu_data_collector, args=(running_flag,))
    imu_thread.start()

    time.sleep(10)  # Record for exactly 10 seconds

    running = False
    imu_thread.join()

    print("\033[92mRecording complete! Data saved.\033[0m")

if __name__ == "__main__":
    main()
