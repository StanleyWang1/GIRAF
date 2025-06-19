import numpy as np
import threading
import queue
import time

from camera_driver import run_camera_server
from control_table import MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR14_OPEN, MOTOR14_CLOSED
from dynamixel_driver import dynamixel_connect, dynamixel_drive, dynamixel_disconnect, radians_to_ticks
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from motor_driver import motor_connect, motor_status, motor_drive, motor_disconnect
from kinematic_model import num_jacobian

## ----------------------------------------------------------------------------------------------------
# Joystick Controller Teleoperation
## ----------------------------------------------------------------------------------------------------

# Global Variables
joystick_data = {"LX":0, "LY":0, "RX":0, "RY":0, "LT":0, "RT":0, "AB":0, "BB":0, "XB":0, "LB":0, "RB":0}
joystick_lock = threading.Lock()

velocity = np.zeros((6, 1))
velocity_lock = threading.Lock()

running = True
running_lock = threading.Lock()

pose_queue = queue.Queue(maxsize=1)

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

    def inverse_jacobian(joint_coords):
        J = num_jacobian(joint_coords)
        J_inv = np.linalg.pinv(J)
        return J_inv

    def get_boom_pos(d3, d3_dot):
        use_blossoming_cal = False
        
        d3 = d3 - 80/1000
        if d3_dot > 0 and d3 > 1 and use_blossoming_cal: # extending
            # cubic approximation
            p1 = -0.5455
            p2 = 2.9053
            p3 = -21.8727
            p4 = 6.5349
            return p1 * d3**3 + p2 * d3**2 + p3 * d3 + p4 
            # linear approximation
            # return (1000*d3 - 55 - 255) / (-56) # [rad]
        else: # retracting
            # cubic approximation
            p1 = -0.0508
            p2 = -0.4122
            p3 = -15.2992
            p4 = 4.7840
            return p1 * d3**3 + p2 * d3**2 + p3 * d3 + p4
            # linear approximation
            # return (1000*d3 - 55 - 255) / (-60.5) # [rad]

    # Joint Coords            
    roll_pos = 0
    pitch_pos = 0
    d3_pos = (55+255+80)/1000
    boom_pos = 0
    theta4_pos = 0
    theta5_pos = 0
    theta6_pos = 0
    gripper_pos = MOTOR14_OPEN # already in ticks!

    joint_velocity = np.zeros((6,1)) # initialize as zero
    gripper_velocity = 0

    candle, motors = motor_connect()
    dmx_controller, dmx_GSW = dynamixel_connect()
    print("\033[93mTELEOP: Motors Connected!\033[0m")

    try:
        while running:
            # unpack joystick data from dict
            with joystick_lock:
                LX = joystick_data["LX"]
                LY = joystick_data["LY"]
                RX = joystick_data["RX"]
                RY = joystick_data["RY"]
                LT = joystick_data["LT"]
                RT = joystick_data["RT"]
                AB = joystick_data["AB"]
                BB = joystick_data["BB"]
                XB = joystick_data["XB"]
                LB = joystick_data["LB"]
                RB = joystick_data["RB"]

            if XB: # stop button engaged - abort process!
                with running_lock:
                    running = False
                break

            # safety interlock
            if LB and RB:
                with velocity_lock:
                    velocity[0] = -0.25*LY # X velocity
                    velocity[1] = -0.25*LX # Y velocity

                    velocity[4] = 0.5*RY # WY angular velocity
                    velocity[5] = -0.5*RX # WZ angular velocity

                if RT and not LT: # Z up
                    with velocity_lock:
                        velocity[2] = 0.1*RT # Z velocity up
                elif LT and not RT and (pitch_pos > 0): # Z down
                    with velocity_lock:
                        velocity[2] = -0.1*LT # Z velocity down
                else:
                    with velocity_lock:
                        velocity[2] = 0 # no Z velocity

                if AB and not BB: # close
                    gripper_velocity = 20
                elif BB and not AB:
                    gripper_velocity = -20
                else:
                    gripper_velocity = 0
            else:
                with velocity_lock:
                    velocity = np.zeros((6, 1))
                    gripper_velocity = 0

            Jv_inv = inverse_jacobian([roll_pos, pitch_pos + np.pi/2, d3_pos, 
                                       theta4_pos + np.pi/2, theta5_pos + 5*np.pi/6, theta6_pos])
            joint_velocity = Jv_inv @ velocity

            roll_pos = roll_pos + 0.0075*joint_velocity[0, 0]
            pitch_pos = pitch_pos + 0.0075*joint_velocity[1, 0]
            d3_pos = d3_pos + 0.0075*joint_velocity[2, 0]

            boom_pos = get_boom_pos(d3_pos, joint_velocity[2, 0]) # convert linear d3 to motor angle
            print(boom_pos)

            theta4_pos = theta4_pos + 0.0075*joint_velocity[3, 0]
            theta5_pos = theta5_pos + 0.0075*joint_velocity[4, 0]
            theta6_pos = theta6_pos + 0.0075*joint_velocity[5, 0]
            
            theta5_pos = max(theta5_pos, -1.7) # wrist pitch limit

            gripper_pos = gripper_pos + gripper_velocity

            # joint limits
            roll_pos = max(min(roll_pos, np.pi/2), -np.pi/2)
            pitch_pos = max(min(pitch_pos, np.pi/2), 0)
            boom_pos = max(min(boom_pos, 0), -25)
            gripper_pos = int(max(min(gripper_pos, MOTOR14_CLOSED), MOTOR14_OPEN))
            
            # check status then drive motors
            motor_status(candle, motors)
            motor_drive(candle, motors, roll_pos, pitch_pos, boom_pos)
            dynamixel_drive(dmx_controller, dmx_GSW, [radians_to_ticks(theta4_pos) + MOTOR11_HOME,
                                                      radians_to_ticks(theta5_pos) + MOTOR12_HOME,
                                                      radians_to_ticks(theta6_pos) + MOTOR13_HOME,
                                                      gripper_pos])
            time.sleep(0.005)
    finally:
        motor_disconnect(candle)
        dynamixel_disconnect(dmx_controller)
        print("\033[93mTELEOP: Motors Disconnected!\033[0m")

## ----------------------------------------------------------------------------------------------------
# Camera Server Thread
## ----------------------------------------------------------------------------------------------------
def camera_server():
    global pose_queue, running
    params = {
        "port": 8485,
        "tag_size": 0.065  # meters
    }
    run_camera_server(params=params, output_queue=pose_queue)

## ----------------------------------------------------------------------------------------------------
# Camera Server Thread
## ----------------------------------------------------------------------------------------------------
def pose_handler():
    global pose_queue, running
    while running:
        try:
            pose = pose_queue.get(timeout=1.0)
            if pose["id"] is not None:
                print(f"Pose received: {pose}")
        except queue.Empty:
            continue
        time.sleep(0.01)

joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
motor_thread = threading.Thread(target=motor_control, daemon=True)
camera_thread = threading.Thread(target=camera_server, daemon=True)
pose_thread = threading.Thread(target=pose_handler, daemon=True)

joystick_thread.start()
motor_thread.start()
camera_thread.start()
pose_thread.start()

# Keep main thread alive
while running:
    time.sleep(1)
