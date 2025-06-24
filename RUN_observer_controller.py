import numpy as np
import threading
import time
import os
import depthai as dai
import pyCandle

from control_table import MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR14_OPEN
from dynamixel_driver import dynamixel_connect, dynamixel_drive, dynamixel_disconnect
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from motor_driver import motor_connect, motor_status, motor_drive, motor_disconnect

# Globals
joystick_data = {"LX":0, "LY":0, "RX":0, "RY":0, "LT":0, "RT":0, "AB":0, "BB":0, "XB":0, "LB":0, "RB":0}
joystick_lock = threading.Lock()
running = True
running_lock = threading.Lock()

pitch_pos = 0.0
x_hat = np.zeros((2, 1))
latest_accel = 0.0
csv_buffer = []
buffer_lock = threading.Lock()

CSV_PATH = "./DATA/observer_damping/quarter_damping_boom_150cm_trial3.csv"

wn = 12.763
z = 0.0194
beta = 1.0

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

def imu_loop():
    global latest_accel, running
    device, queue = imu_setup()
    while running:
        imuData = queue.tryGet()
        if imuData:
            for pkt in imuData.packets:
                acc = pkt.acceleroMeter
                if acc:
                    with buffer_lock:
                        latest_accel = acc.x + 8.25
        time.sleep(0.001)
    device.close()

def csv_logger():
    global csv_buffer, running
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, 'w') as f:
        f.write("t,accel_x,pitch_pos,x1_hat,x2_hat\n")
        while running:
            time.sleep(0.5)
            with buffer_lock:
                for row in csv_buffer:
                    f.write(','.join(map(str, row)) + '\n')
                csv_buffer = []

def observer_loop(candle, motors):
    global x_hat, latest_accel, pitch_pos, csv_buffer, running

    A = np.array([[0, 1], [-wn**2, -2*z*wn]])
    B = np.array([[0], [beta]])
    C = np.array([[0, 1]])
    D = np.array([[beta]])
    L = np.array([[-0.01668588], [0.9887092]])

    # Observer damping term
    Kd = 2*z*wn / 4

    prev_time = time.time()
    # dt = 1.0 / 500.0
    # duration = 11
    i = 0

    u_series = np.concatenate([
        np.zeros(int(0.5 * 500)),
        np.full(int(0.5 * 500), 0.5),
        np.zeros(int(10 * 500))
    ])
    t_start = time.time()
    pitch_pos = 0.2

    while running:
        if i < len(u_series):
            u_task = u_series[i]
            u = u_task + Kd * x_hat[1, 0]
        else:
            u = 0
            with running_lock:
                running = False

        with buffer_lock:
            y = latest_accel

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time

        pitch_pos += dt * u
        motor_drive(candle, motors, 0.0, pitch_pos, 0.0)

        y_hat = C @ x_hat + D * u
        x_hat_dot = A @ x_hat + B * u + L @ (y - y_hat)
        x_hat += x_hat_dot * dt

        t_now = time.time() - t_start
        with buffer_lock:
            csv_buffer.append((t_now, y, pitch_pos, x_hat[0,0], x_hat[1,0]))

        i += 1
        time.sleep(dt)

def initialize_motors():
    candle, motors = motor_connect()
    dmx_ctrl, dmx_GSW = dynamixel_connect()
    print("\033[93mTELEOP: Motors Connected and Dynamixel Homed!\033[0m")
    time.sleep(0.5)
    dynamixel_drive(dmx_ctrl, dmx_GSW, [MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR14_OPEN])
    time.sleep(0.5)

    print("\033[93mPress Enter to jog Boom Up\033[0m")
    input()
    pitch_sweep = np.linspace(0, 0.2, 500)
    for pitch_val in pitch_sweep:
        motor_drive(candle, motors, 0.0, pitch_val, 0.0)
        time.sleep(0.005)

    print("\033[93mPress Enter to begin Resonance Test\033[0m")
    input()
    return candle, motors, dmx_ctrl

def main():
    global running
    joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
    imu_thread = threading.Thread(target=imu_loop, daemon=True)
    logger_thread = threading.Thread(target=csv_logger, daemon=True)

    joystick_thread.start()
    imu_thread.start()

    candle, motors, dmx_ctrl = initialize_motors()
    observer_thread = threading.Thread(target=observer_loop, args=(candle, motors), daemon=True)
    logger_thread.start()
    print("\033[93mSTARTING OBSERVER CONTROLLER\033[0m")
    observer_thread.start()

    try:
        while running:
            with joystick_lock:
                if joystick_data["XB"]:
                    with running_lock:
                        running = False
            time.sleep(0.5)
    finally:
        motor_disconnect(candle)
        dynamixel_disconnect(dmx_ctrl)
        print("\033[93mTELEOP: Shutdown complete.\033[0m")

if __name__ == "__main__":
    main()
