import numpy as np
import pyCandle
import threading
import time

from control_table import MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR14_OPEN, MOTOR14_CLOSED
from dynamixel_driver import dynamixel_connect, dynamixel_drive, dynamixel_disconnect, radians_to_ticks
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from motor_driver import motor_connect, motor_status, motor_drive, motor_disconnect
# from kinematic_model import num_jacobian

## ----------------------------------------------------------------------------------------------------
# Joystick Controller Teleoperation
## ----------------------------------------------------------------------------------------------------

# Global Variables
joystick_data = {"LX":0, "LY":0, "RX":0, "RY":0, "LT":0, "RT":0, "AB":0, "BB":0, "XB":0, "LB":0, "RB":0}
joystick_lock = threading.Lock()

running = True
running_lock = threading.Lock()

# Test Parameters
AMPLITUDE = 0.1 # rad
FREQUENCY = 1.0 # Hz

## ----------------------------------------------------------------------------------------------------
# Joystick Monitoring Thread
## ----------------------------------------------------------------------------------------------------
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

## ----------------------------------------------------------------------------------------------------
# Motor Control Thread
## ----------------------------------------------------------------------------------------------------
def motor_control():
    global joystick_data, velocity, running

    # Joint Coords            
    roll_pos = 0
    pitch_pos = 0
    d3_pos = (55+255+80)/1000
    boom_pos = 0
    theta4_pos = 0
    theta5_pos = 0
    theta6_pos = 0
    gripper_pos = MOTOR14_OPEN # already in ticks!

    candle, motors = motor_connect()
    dmx_controller, dmx_GSW = dynamixel_connect()
    print("\033[93mTELEOP: Motors Connected!\033[0m")
    time.sleep(0.5)

    dynamixel_drive(dmx_controller, dmx_GSW, [MOTOR11_HOME,
                                                MOTOR12_HOME,
                                                MOTOR13_HOME,
                                                MOTOR14_OPEN])
    print("\033[93mTELEOP: Dynamixel Active and Homed!\033[0m")
    time.sleep(0.5)
    
    print("\033[93mTELEOP: Press Enter to Begin Resonance Test!\033[0m")
    input() # BLOCK and wait for user to press Enter
    
    try:
        t_start = time.time()
        while running:
            # unpack joystick data from dict
            with joystick_lock:
                XB = joystick_data["XB"]

            if XB: # stop button engaged - abort process!
                with running_lock:
                    running = False
                break
            else: # engage oscillation trajectory
                t = time.time() - t_start
                pitch_pos = AMPLITUDE/2 * (1 - np.cos(2 * np.pi * FREQUENCY * t))  # 0 to 0.1 rad at 1 Hz

            # check status then drive motors
            motor_status(candle, motors)
            motor_drive(candle, motors, 0.0, pitch_pos, 0.0)
            time.sleep(0.005)
    finally:
        motor_disconnect(candle)
        dynamixel_disconnect(dmx_controller)
        print("\033[93mTELEOP: Motors Disconnected!\033[0m")


joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
motor_thread = threading.Thread(target=motor_control, daemon=True)
joystick_thread.start()
motor_thread.start()

# Keep main thread alive
while running:
    time.sleep(1)
