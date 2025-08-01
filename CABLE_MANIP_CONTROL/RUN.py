import numpy as np
import threading
import queue
import time
import cv2

from camera_driver import weighted_average_transforms, run_camera_server
from control_table import MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR14_OPEN, MOTOR14_CLOSED
from dynamixel_driver import dynamixel_connect, dynamixel_drive, dynamixel_boom_meters, dynamixel_disconnect, radians_to_ticks
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from motor_driver import motor_connect, motor_status, motor_drive, motor_disconnect
from kinematic_model import num_jacobian, num_forward_kinematics

# from new_cable_traj import trajectory
# from square_traj import trajectory

import pandas as pd
trajectory_df = pd.read_csv("CLIVE_TRAJ_SMOOTH.csv")  # Replace with actual path
# trajectory_df = pd.read_csv("STAN_TRAJ.csv")  # Replace with actual path
trajectory = trajectory_df[["x", "y", "z"]].values  # Convert to numpy array of shape (N, 3)
offset = np.array([0.00406, 0.012895, 0.02])  # Offset to align with tag center
trajectory = trajectory + offset  # Apply offset to all points

## ----------------------------------------------------------------------------------------------------
# Joystick Controller Teleoperation
## ----------------------------------------------------------------------------------------------------

# Global Variables
joystick_data = {"LX":0, "LY":0, "RX":0, "RY":0, "LT":0, "RT":0, "AB":0, "BB":0, "XB":0, "YB":0, "LB":0, "RB":0, "MENULEFT":0, "MENURIGHT":0}
joystick_lock = threading.Lock()

velocity = np.zeros((6, 1))
velocity_lock = threading.Lock()

wrist_tag_angle = 0.0
wrist_tag_angle_lock = threading.Lock()

FK_num = np.zeros((4, 4))
FK_num_lock = threading.Lock()

T_world_tag = np.zeros((4, 4))
T_world_tag_lock = threading.Lock()

input_mode = False
input_lock = threading.Lock()

running = True
autonomous_mode = False
running_lock = threading.Lock()

pose_queue = queue.Queue(maxsize=1)
log_queue = queue.Queue()

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
    global joystick_data, velocity, FK_num, T_world_tag, running, autonomous_mode, input_mode

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
    roll_offset = 0.0
    pitch_pos = 0
    d3_pos = (55+255+80)/1000
    d3_real = 0
    d3_dot_sum = 0
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

    tag_read = False
    # autonomous_mode = False
    waypoint_id = 0
    T_world_tag_temp = np.zeros((4, 4))
    T_world_tag_latest = np.zeros((4, 4))
    
    try:
        start_time = time.time()
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
                MENULEFT = joystick_data["MENULEFT"]
                MENURIGHT = joystick_data["MENURIGHT"]

            if XB: # stop button engaged - abort process!
                with running_lock:
                    running = False
                break

            # safety interlock
            if (LB and RB) or autonomous_mode:
                if YB:
                    with input_lock:
                        input_mode = True
                    speed = float(input("Enter speed multiplier (1, 2, 4x): "))
                    with input_lock:
                        input_mode = False
                        autonomous_mode = True
                    cycle_count = 0
                    feed_forward_velocity = np.zeros((3,))
                    # Y BUTTON -- enter autonomous mode!

                elif LY or LX or RY or RX or LT or RT or AB or BB or MENULEFT or MENURIGHT: # manual control  
                    tag_read = False
                    with input_lock:
                        autonomous_mode = False
                    waypoint_id = 0.0

                    with velocity_lock:
                        velocity[0] = -0.2*LY # X velocity
                        velocity[1] = -0.2*LX # Y velocity

                        velocity[4] = 0.5*RY # WY angular velocity
                        velocity[3] = -0.5*RX # WZ angular velocity

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
                        # gripper_velocity = 20
                        with velocity_lock:
                            velocity[5] = -0.5
                    elif BB and not AB:
                        # gripper_velocity = -20
                        with velocity_lock:
                            velocity[5] = 0.5
                    else:
                        # gripper_velocity = 0
                        with velocity_lock:
                            velocity[5] = 0.0
                    
                    print(f"Debug: MENULEFT={MENULEFT}, MENURIGHT={MENURIGHT}, roll_offset={roll_offset:.3f}")
                    if MENULEFT and not MENURIGHT:  
                        roll_offset += 0.0025
                    elif MENURIGHT and not MENULEFT:
                        roll_offset -= 0.0025

                elif autonomous_mode:
                    # Get x,y,z point from loaded trajectory
                    if waypoint_id < len(trajectory):
                        x, y, z = trajectory[int(waypoint_id)]
                        if waypoint_id > 0:
                            feed_forward_velocity = trajectory[int(waypoint_id)] - trajectory[int(waypoint_id - 1)]
                        waypoint_id += speed

                    else:
                        x, y, z = trajectory[-1]
                        waypoint_id = 0 # loop back to start
                        cycle_count += 1
                        if cycle_count >= 1:
                            with input_lock:
                                autonomous_mode = False
                            with velocity_lock:
                                velocity = np.zeros((6, 1))
                                gripper_velocity = 0
                        print(f"\033[93mTELEOP: Completed {cycle_count} cycles!\033[0m")

                    T_tag_target = np.array([[1, 0, 0, x],
                                            [0, 1, 0, y],
                                            [0, 0, 1, z - 0.075],
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

                        # transform feedforward velocity to world frame
                        with FK_num_lock:
                            EE_pose = FK_num[:3, 3]
                        P_velocity = 3.0 * (target_pose - EE_pose) + 0.9*speed/0.01 * (T_world_target[:3, :3]@feed_forward_velocity) # move towards target pose
                        P_velocity = np.clip(P_velocity, -0.5, 0.5) # set velocity limits
                        with velocity_lock:
                            velocity[0] = P_velocity[0] # X velocity
                            velocity[1] = P_velocity[1] # Y velocity
                            velocity[2] = P_velocity[2] # Z velocity
                            with wrist_tag_angle_lock:
                                velocity[4] = np.clip(0.05 * (75.0 - wrist_tag_angle), -0.5, 0.5)

                else:
                    with velocity_lock:
                        velocity = np.zeros((6, 1))
                        gripper_velocity = 0
            else:
                with velocity_lock:
                    velocity = np.zeros((6, 1))
                    gripper_velocity = 0

            Jv_inv = inverse_jacobian([roll_pos, pitch_pos + np.pi/2, d3_pos, 
                                       theta4_pos + np.pi/2, theta5_pos + 5*np.pi/6, theta6_pos])
            with FK_num_lock:
                FK_num = num_forward_kinematics([roll_pos, pitch_pos + np.pi/2, d3_pos, 
                                             theta4_pos + np.pi/2, theta5_pos + 5*np.pi/6, theta6_pos])
            
            joint_velocity = Jv_inv @ velocity

            # Dynamixel Boom Encoder Read
            if d3_real == 0: # first run
                d3_dot_real = 0
                d3_real = dynamixel_boom_meters(dmx_controller) # read boom length from encoder
            else:
                d3_prev = d3_real
                d3_real = dynamixel_boom_meters(dmx_controller) # read boom length from encoder
                d3_dot_real = (d3_real - d3_prev) / 0.0075 # compute velocity
                # Maintain a finite sum of the 50 most recent (joint_velocity[2, 0] - d3_dot_real) values
                if not hasattr(motor_control, "d3_dot_history"):
                    motor_control.d3_dot_history = []
                motor_control.d3_dot_history.append(joint_velocity[2, 0] - d3_dot_real)
                if len(motor_control.d3_dot_history) > 10:
                    motor_control.d3_dot_history.pop(0)
                d3_dot_sum = sum(motor_control.d3_dot_history) * 0.0075

            roll_pos = roll_pos + 0.0075*joint_velocity[0, 0]
            pitch_pos = pitch_pos + 0.0075*joint_velocity[1, 0]
            d3_pos = d3_pos + 0.0075*joint_velocity[2, 0] + 0.25*d3_dot_sum + 0.0075*(joint_velocity[2, 0] - d3_dot_real)
            d3_pos = max(d3_pos, (55+255+80)/1000)

            # log_queue.put([time.time()-start_time, d3_pos, d3_real])

            boom_pos = get_boom_pos(d3_pos, joint_velocity[2, 0]) # convert linear d3 to motor angle
            # print(boom_pos)

            theta4_pos = theta4_pos + 0.0075*joint_velocity[3, 0]
            theta5_pos = theta5_pos + 0.0075*joint_velocity[4, 0]
            theta6_pos = theta6_pos + 0.0075*joint_velocity[5, 0]
            
            theta5_pos = max(theta5_pos, -1.7) # wrist pitch limit

            gripper_pos = gripper_pos + gripper_velocity

            # joint limits
            roll_pos = max(min(roll_pos, np.pi/2), -np.pi/2)
            pitch_pos = max(min(pitch_pos, np.pi/2), 0)
            boom_pos = max(min(boom_pos, 0), -35)
            gripper_pos = int(max(min(gripper_pos, MOTOR14_CLOSED), MOTOR14_OPEN))
            
            # check status then drive motors
            motor_status(candle, motors)
            motor_drive(candle, motors, roll_pos + roll_offset, pitch_pos, boom_pos)
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
        "tag_size": 0.04901  # meters
    }
    run_camera_server(params=params, output_queue=pose_queue)

## ----------------------------------------------------------------------------------------------------
# Visual Servoing Thread
## ----------------------------------------------------------------------------------------------------
def pose_handler():
    global pose_queue, FK_num, wrist_tag_angle, T_world_tag, running, input_mode

    T_ee_cam = np.array([[0, -0.984808, -0.173648, 0.045404],
                         [1, 0, 0, -0.00705],
                         [0, -0.173648, 0.984808, -0.076365],
                         [0, 0, 0, 1]])
    # Version 1 Multitag Mount
    x1_dist = 104.5/1000
    x2_dist = 110.0/1000
    y_dist = 119.0/1000
    # x_dist = 106.0/1000
    # y_dist = 106.0/1000
    # x_dist = 73.0/1000
    # y_dist = 79.0/1000
    tag_to_15_transforms = {11: np.array([[1, 0, 0, x1_dist+x2_dist],
                                          [0, 1, 0, y_dist],
                                          [0, 0, 1, 0.0],
                                          [0, 0, 0, 1]]),
                            12: np.array([[1, 0, 0, x1_dist],
                                          [0, 1, 0, y_dist],
                                          [0, 0, 1, 0.0],
                                          [0, 0, 0, 1]]),
                            13: np.array([[1, 0, 0, 0.0],
                                          [0, 1, 0, y_dist],
                                          [0, 0, 1, 0.0],
                                          [0, 0, 0, 1]]),
                            14: np.array([[1, 0, 0, x1_dist],
                                          [0, 1, 0, 0.0],
                                          [0, 0, 1, 0.0],
                                          [0, 0, 0, 1]]),
                            15: np.eye(4)}
    while running:
        try:
            pose_list = pose_queue.get(timeout=1.0)
            if len(pose_list) > 0:
                # Find the pose with the highest weight
                # best_pose = min(pose_list, key=lambda pose: pose.get("id", 16))
                best_pose = pose_list[0] # first element
                tag_id = best_pose["id"]
                # print(tag_id)

                rvec = np.array(best_pose["rvec"]).reshape(3, 1)
                tvec = np.array(best_pose["tvec"]).reshape(3, 1)
                R, _ = cv2.Rodrigues(rvec)

                T_cam_tag = np.eye(4)
                T_cam_tag[:3, :3] = R
                T_cam_tag[:3, 3] = tvec.ravel()

                # Extract camera angle
                T_tag_cam = np.linalg.inv(T_cam_tag)
                R_tag_cam = T_tag_cam[:3, :3]
                y_cam_in_tag = R_tag_cam[:, 1]  # Second column of rotation matrix
                z_tag = np.array([0, 0, 1])  # Tag Z-axis in its own frame
                cos_theta = np.dot(y_cam_in_tag, z_tag)
                angle_rad = np.arccos(np.clip(np.abs(cos_theta), -1.0, 1.0))
                with wrist_tag_angle_lock:
                    wrist_tag_angle = np.degrees(angle_rad)
                with input_lock:
                    if not input_mode and not autonomous_mode:
                        print(f"Camera/tag angle: {np.degrees(angle_rad):.2f} deg")
                        pass

                T_tag_15 = tag_to_15_transforms.get(tag_id, np.eye(4))
                T_cam_15_best = T_cam_tag @ T_tag_15

                with FK_num_lock:
                    T_world_ee = FK_num
                T_world_tag_new = T_world_ee @ T_ee_cam @ T_cam_15_best
                with T_world_tag_lock:
                    if T_world_tag is not None:
                        T_world_tag = weighted_average_transforms(T_world_tag, T_world_tag_new, 0.5)
                    else:
                        T_world_tag = T_world_tag_new

                # Debug printout (pos of EE in tag 15 frame)
                # T_15_ee = np.linalg.inv(T_ee_cam @ T_cam_15_best)
                # print(T_15_ee[:3,3])

            else:
                with T_world_tag_lock:
                    T_world_tag = None
        except queue.Empty:
            with T_world_tag_lock:
                    T_world_tag = None
            continue
        time.sleep(0.01)

## ----------------------------------------------------------------------------------------------------
# Debug Logger Thread
## ----------------------------------------------------------------------------------------------------
def debug_logger():
    global running, autonomous_mode
    import csv
    from datetime import datetime

    buffer = []
    batch_size = 100  # Adjust as needed
    log_file = f"boom_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    while True: # block until autonomous mode is set
        with input_lock:
            if autonomous_mode:
                break
        time.sleep(0.1)

    print("\033[93mLOG: Debug Logger Started!\033[0m")
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "d3_pos", "d3_real"])  # Header

        while running or not log_queue.empty():
            try:
                entry = log_queue.get(timeout=0.5)
                buffer.append(entry)
                if len(buffer) >= batch_size:
                    writer.writerows(buffer)
                    buffer.clear()
            except queue.Empty:
                continue
        
        # Write remaining entries on shutdown
        if buffer:
            writer.writerows(buffer)

camera_thread = threading.Thread(target=camera_server, daemon=True)
pose_thread = threading.Thread(target=pose_handler, daemon=True)
joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
motor_thread = threading.Thread(target=motor_control, daemon=True)
logger_thread = threading.Thread(target=debug_logger, daemon=True)

camera_thread.start()
pose_thread.start()
joystick_thread.start()
logger_thread.start()
motor_thread.start()

# Keep main thread alive
while running:
    time.sleep(1)
