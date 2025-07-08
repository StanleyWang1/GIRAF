import numpy as np
import threading
import queue
import time
import cv2

from camera_driver import run_camera_server
from control_table import MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR14_OPEN, MOTOR14_CLOSED
from dynamixel_driver import dynamixel_connect, dynamixel_drive, dynamixel_boom_ticks, dynamixel_boom_meters, dynamixel_disconnect, radians_to_ticks
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from motor_driver import motor_connect, motor_status, motor_drive, motor_disconnect
from kinematic_model import num_jacobian, num_forward_kinematics

from new_cable_traj import trajectory
# from square_test import trajectory

## ----------------------------------------------------------------------------------------------------
# Joystick Controller Teleoperation
## ----------------------------------------------------------------------------------------------------

# Global Variables
joystick_data = {"LX":0, "LY":0, "RX":0, "RY":0, "LT":0, "RT":0, "AB":0, "BB":0, "XB":0, "YB":0, "LB":0, "RB":0}
joystick_lock = threading.Lock()

velocity = np.zeros((6, 1))
velocity_lock = threading.Lock()

FK_num = np.zeros((4, 4))
FK_num_lock = threading.Lock()

T_world_tag = np.zeros((4, 4))
T_world_tag_lock = threading.Lock()

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
    global joystick_data, velocity, FK_num, T_world_tag, running

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
    d3_pos = 0 # [m]
    boom_pos = 0 # [rad]
    theta4_pos = 0
    theta5_pos = 0
    theta6_pos = 0
    gripper_pos = MOTOR14_OPEN # already in ticks!

    joint_velocity = np.zeros((6,1)) # initialize as zero
    gripper_velocity = 0

    candle, motors = motor_connect()
    dmx_controller, dmx_GSW = dynamixel_connect()
    homing_offset = dynamixel_boom_ticks(dmx_controller)
    d3_pos = dynamixel_boom_meters(dmx_controller, homing_offset)
    print("\033[93mTELEOP: Motors Connected!\033[0m")

    tag_read = False
    waypoint_id = 0
    T_world_tag_temp = np.zeros((4, 4))
    T_world_tag_latest = np.zeros((4, 4))
    
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
                YB = joystick_data["YB"]
                LB = joystick_data["LB"]
                RB = joystick_data["RB"]

            if XB: # stop button engaged - abort process!
                with running_lock:
                    running = False
                break

            # safety interlock
            if LB and RB:
                if YB:
                    # Y BUTTON -- enter autonomous mode!

                    # Get x,y,z point from loaded trajectory
                    if waypoint_id < len(trajectory):
                        x, y, z = trajectory[waypoint_id]
                        waypoint_id += 1
                    else:
                        x, y, z = trajectory[-1]

                    T_tag_target = np.array([[1, 0, 0, x],
                                            [0, 1, 0, y],
                                            [0, 0, 1, z],
                                            [0, 0, 0, 1]])
                    
                    # Update tag pose if available
                    with T_world_tag_lock:
                        T_world_tag_temp = T_world_tag
                    if T_world_tag_temp is not None: # valid tag being read
                        tag_read = True # tag has been seen
                        T_world_tag_latest = T_world_tag_temp # save latest tag pose
                    
                    # If tag pose available, plan trajectory:
                    if tag_read:
                        T_world_target = T_world_tag_latest @ T_tag_target
                        target_pose = T_world_target[:3, 3]

                        with FK_num_lock:
                            EE_pose = FK_num[:3, 3]
                        P_velocity = 5 * (target_pose - EE_pose) # move towards target pose
                        P_velocity = np.clip(P_velocity, -0.5, 0.5) # set velocity limits
                        with velocity_lock:
                            velocity[0] = P_velocity[0] # X velocity
                            velocity[1] = P_velocity[1] # Y velocity
                            velocity[2] = P_velocity[2] # Z velocity

                elif LY or LX or RY or RX or LT or RT or AB or BB: # manual control  
                    tag_read = False
                    waypoint_id = 0

                    with velocity_lock:
                        velocity[0] = -0.25*LY # X velocity
                        velocity[1] = -0.25*LX # Y velocity

                        velocity[4] = 0.5*RY # WY angular velocity
                        velocity[5] = -0.5*RX # WZ angular velocity

                    if RT and not LT: # Z up
                        with velocity_lock:
                            velocity[2] = 0.25*RT # Z velocity up
                    elif LT and not RT and (pitch_pos > 0): # Z down
                        with velocity_lock:
                            velocity[2] = -0.25*LT # Z velocity down
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
            else:
                with velocity_lock:
                    velocity = np.zeros((6, 1))
                    gripper_velocity = 0

            d3_pos = dynamixel_boom_meters(dmx_controller, homing_offset)
            print(dynamixel_boom_ticks(dmx_controller))

            Jv_inv = inverse_jacobian([roll_pos, pitch_pos + np.pi/2, d3_pos, 
                                       theta4_pos + np.pi/2, theta5_pos + 5*np.pi/6, theta6_pos])
            with FK_num_lock:
                FK_num = num_forward_kinematics([roll_pos, pitch_pos + np.pi/2, d3_pos, 
                                             theta4_pos + np.pi/2, theta5_pos + 5*np.pi/6, theta6_pos])
            
            joint_velocity = Jv_inv @ velocity

            roll_pos = roll_pos + 0.005*joint_velocity[0, 0]
            pitch_pos = pitch_pos + 0.005*joint_velocity[1, 0]
            # d3_pos = d3_pos + 0.0075*joint_velocity[2, 0]
            # boom_pos = get_boom_pos(d3_pos, joint_velocity[2, 0]) # convert linear d3 to motor angle
            # print(boom_pos)
            boom_pos = boom_pos - 100.0*0.005*joint_velocity[2, 0]

            theta4_pos = theta4_pos + 0.005*joint_velocity[3, 0]
            theta5_pos = theta5_pos + 0.005*joint_velocity[4, 0]
            theta6_pos = theta6_pos + 0.005*joint_velocity[5, 0]
            
            theta5_pos = max(theta5_pos, -1.7) # wrist pitch limit

            gripper_pos = gripper_pos + gripper_velocity

            # joint limits
            roll_pos = max(min(roll_pos, np.pi/2), -np.pi/2)
            pitch_pos = max(min(pitch_pos, np.pi/2), 0)
            boom_pos = max(min(boom_pos, 0), -35)
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
        "tag_size": 0.048  # meters
    }
    run_camera_server(params=params, output_queue=pose_queue)

## ----------------------------------------------------------------------------------------------------
# Camera Server Thread
## ----------------------------------------------------------------------------------------------------
def pose_handler():
    global pose_queue, FK_num, T_world_tag, running

    T_ee_cam = np.array([[0, -0.984808, -0.173648, 0.045404],
                         [1, 0, 0, -0.00705],
                         [0, -0.173648, 0.984808, -0.076365],
                         [0, 0, 0, 1]])

    while running:
        try:
            pose = pose_queue.get(timeout=1.0)
            if pose["id"] is not None:
                rvec = np.array(pose["rvec"]).reshape(3, 1)
                tvec = np.array(pose["tvec"]).reshape(3, 1)
                R, _ = cv2.Rodrigues(rvec)
                T_cam_tag = np.eye(4)
                T_cam_tag[:3, :3] = R
                T_cam_tag[:3, 3] = tvec.ravel()

                # Compute the pose of tag in world frame
                with FK_num_lock:
                    T_world_ee = FK_num
                with T_world_tag_lock:
                    T_world_tag = T_world_ee @ T_ee_cam @ T_cam_tag
                # T_tag_cam = np.linalg.inv(T_world_tag) @ T_world_ee @ T_ee_cam
                # print(T_tag_cam[:3, 3])
            else:
                with T_world_tag_lock:
                    T_world_tag = None
        except queue.Empty:
            with T_world_tag_lock:
                    T_world_tag = None
            continue
        time.sleep(0.01)

camera_thread = threading.Thread(target=camera_server, daemon=True)
pose_thread = threading.Thread(target=pose_handler, daemon=True)
joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
motor_thread = threading.Thread(target=motor_control, daemon=True)

camera_thread.start()
pose_thread.start()
joystick_thread.start()
motor_thread.start()

# Keep main thread alive
while running:
    time.sleep(1)
