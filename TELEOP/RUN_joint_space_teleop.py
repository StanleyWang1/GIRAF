import pyCandle
import threading
import time

from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from motor_driver import motor_connect, motor_status, motor_drive, motor_disconnect

## ----------------------------------------------------------------------------------------------------
# Joystick Controller Teleoperation
## ----------------------------------------------------------------------------------------------------

# Global Variables
joystick_data = {"LX":0, "LY":0, "LT":0, "RT":0, "XB":0, "LB":0, "RB":0}
joystick_lock = threading.Lock()

running = True
running_lock = threading.Lock()

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
    global joystick_data, running

    roll_pos = 0
    pitch_pos = 0
    boom_pos = 0

    candle, motors = motor_connect()
    print("\033[93mTELEOP: Motors Connected!\033[0m")

    try:
        while running:
            # unpack joystick data from dict
            with joystick_lock: 
                LX = joystick_data["LX"]
                LY = joystick_data["LY"]
                LT = joystick_data["LT"]
                RT = joystick_data["RT"]
                XB = joystick_data["XB"]
                LB = joystick_data["LB"]
                RB = joystick_data["RB"]
            
            if XB: # stop button engaged - abort process!
                with running_lock:
                    running = False
                break

            # dynamically adjust teleop drive ratio
            roll_drive_ratio = 0.0075/(-(boom_pos-4)/4)
            pitch_drive_ratio = 0.0075/(-(boom_pos-4)/4)
            boom_drive_ratio = 0.025

            # safety interlock
            if LB and RB:
                roll_pos = roll_pos - roll_drive_ratio*LX
                pitch_pos = pitch_pos - pitch_drive_ratio*LY
                if RT and not LT: # extend boom
                    boom_pos = boom_pos - boom_drive_ratio*RT
                elif LT and not RT: # retract boom
                    boom_pos = boom_pos + boom_drive_ratio*LT

            # joint limits
            boom_pos = max(min(boom_pos, 0), -30)
            print(round(boom_pos, 3))
            
            # check status then drive motors
            motor_status(candle, motors)
            motor_drive(candle, motors, roll_pos, pitch_pos, boom_pos)
            time.sleep(0.005)
    finally:
        motor_disconnect(candle)
        print("\033[93mTELEOP: Motors Disconnected!\033[0m")
        

joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
motor_thread = threading.Thread(target=motor_control, daemon=True)
joystick_thread.start()
motor_thread.start()

# Keep main thread alive
while running:
    time.sleep(1)

