import numpy as np
import pyCandle
import threading
import time
import os
import depthai as dai

from control_table import MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR14_OPEN
from dynamixel_driver import dynamixel_connect, dynamixel_drive, dynamixel_disconnect
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from motor_driver import motor_connect, motor_status, motor_drive, motor_disconnect

joystick_data = {"LX":0, "LY":0, "RX":0, "RY":0, "LT":0, "RT":0, "AB":0, "BB":0, "XB":0, "LB":0, "RB":0}
joystick_lock = threading.Lock()
running = True
running_lock = threading.Lock()

AMPLITUDE = 0.1  # rad
FREQUENCY = 0.75  # Hz
CSV_PATH = "./DATA/cantilever_resonance/boom_50cm_trial1.csv"

def joystick_monitor():
    global joystick_data, running
    js = joystick_connect()
    print("\033[93mTELEOP: Joystick Connected!\033[0m")
    while running:
        with joystick_lock:
            joystick_data = joystick_read(js)
        time.sleep(0.005)
    joystick_disconnect(js)
    print("\033[93mTELEOP: Joystick Disconnected!\033[0m")

def imu_setup():
    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    xout_imu = pipeline.create(dai.node.XLinkOut)
    xout_imu.setStreamName("imu")
    imu.out.link(xout_imu.input)
    device = dai.Device(pipeline)
    queue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
    return device, queue

def motor_control_with_logging():
    global joystick_data, running

    roll_pos, pitch_pos, d3_pos = 0, 0, (55+255+80)/1000
    candle, motors = motor_connect()
    dmx_ctrl, dmx_GSW = dynamixel_connect()
    print("\033[93mTELEOP: Motors Connected and Dynamixel Homed!\033[0m")
    time.sleep(0.5)
    dynamixel_drive(dmx_ctrl, dmx_GSW, [MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR14_OPEN])
    time.sleep(0.5)

    print("\033[93mPress Enter to jog Boom Up\033[0m")
    input()
    pitch_sweep = np.linspace(0, 0.5, 500)
    for pitch_val in pitch_sweep:
        motor_drive(candle, motors, 0.0, pitch_val, 0.0)
        time.sleep(0.005)

    print("\033[93mPress Enter to begin Resonance Test\033[0m")
    input()

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, 'w') as f:
        f.write("t,accel_x,accel_y,accel_z\n")

    imu_device, imu_queue = imu_setup()
    t_start = time.time()
    buffer = []
    buffer_size = 250  # ~0.5 sec at 500 Hz

    try:
        while running:
            with joystick_lock:
                if joystick_data["XB"]:
                    with running_lock:
                        running = False
                    break

            t_now = time.time() - t_start

            if t_now > 10.0:
                running = False
            elif t_now > 1.0:
                motor_disconnect(candle)
            
            imuData = imu_queue.tryGet()
            if imuData is not None:
                for pkt in imuData.packets:
                    acc = pkt.acceleroMeter
                    if acc is not None:
                        t_now = time.time() - t_start
                        buffer.append((t_now, acc.x, acc.y, acc.z))

                if len(buffer) >= buffer_size:
                    with open(CSV_PATH, 'a') as f:
                        for row in buffer:
                            f.write(','.join(map(str, row)) + '\n')
                    buffer = []

            time.sleep(0.001)
    finally:
        if buffer:
            with open(CSV_PATH, 'a') as f:
                for row in buffer:
                    f.write(','.join(map(str, row)) + '\n')
        motor_disconnect(candle)
        dynamixel_disconnect(dmx_ctrl)
        imu_device.close()
        print("\033[93mTELEOP: Motors and IMU Disconnected.\033[0m")

if __name__ == "__main__":
    joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
    motor_thread = threading.Thread(target=motor_control_with_logging, daemon=True)
    joystick_thread.start()
    motor_thread.start()
    while running:
        time.sleep(1)
