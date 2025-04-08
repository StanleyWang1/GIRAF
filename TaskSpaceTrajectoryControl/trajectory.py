import numpy as np
import pyCandle
import threading
import time

# from generate_circle import circle_traj, circle_velocity
from simple_pnp import pnp_grasp, pnp_traj, pnp_velocity
traj_grasp = pnp_grasp
traj_position = pnp_traj
traj_velocity = pnp_velocity

from dynamixel_driver import dynamixel_connect, dynamixel_drive, dynamixel_disconnect, radians_to_ticks
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from motor_driver import motor_connect, motor_status, motor_drive, motor_disconnect
from kinematic_model import num_forward_kinematics, num_jacobian

## ----------------------------------------------------------------------------------------------------
# Joystick Controller Teleoperation
## ----------------------------------------------------------------------------------------------------

# Global Variables
joystick_data = {"LX":0, "LY":0, "RX":0, "RY":0, "LT":0, "RT":0, "XB":0, "LB":0, "RB":0}
joystick_lock = threading.Lock()

velocity = np.zeros((6, 1))
velocity_lock = threading.Lock()

start = False # safety interlock to begin motion trajectory
start_lock = threading.Lock()

running = True # safety interlock to run all threads
running_lock = threading.Lock()

## ----------------------------------------------------------------------------------------------------
# Joystick Monitoring Thread
## ----------------------------------------------------------------------------------------------------
def joystick_monitor():
    global joystick_data, start, running

    js = joystick_connect()
    print("\033[93mCONTROL: Joystick Connected!\033[0m")

    while running:
        # Get joystick data
        with joystick_lock:
            joystick_data = joystick_read(js)
        # Check if e-stop pressed
        if joystick_data['XB']:
            with running_lock:
                running = False
        # Else check if trajectory should be started
        elif joystick_data['LB'] and joystick_data['RB']:
            with start_lock:
                start = True
        time.sleep(0.005)

    joystick_disconnect(js)
    print("\033[93mTELEOP: Joystick Disconnected!\033[0m")

## ----------------------------------------------------------------------------------------------------
# Motor Control Thread
## ----------------------------------------------------------------------------------------------------
def motor_control():
    global joystick_data, velocity, start, running, circle_traj, circle_velocity

    def inverse_jacobian(joint_coords):
        J = num_jacobian(joint_coords)
        J_inv = np.linalg.pinv(J)
        return J_inv

    def get_boom_pos(d3, d3_dot):
        d3 = d3 - 80/1000
        if d3_dot > 0: # extending
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
    joint_coords = np.array([0, 0, (55+255+80)/1000, 0, -np.pi/2, 0]) # roll, pitch, d3, th4, th5, th6     
    boom_pos = 0 # revolute position of boom motor (w/ blossoming cal)        
    joint_velocity = np.zeros((6,1)) # initialize as zero

    joint_offset = np.array([0, np.pi/2, 0, np.pi/2, 5*np.pi/6, 0])

    # Proportional position error gain
    K = np.diag([2.0, 2.0, 2.0])

    # Start Motors
    candle, motors = motor_connect()
    print("\033[93mCONTROL: MAB Motors Connected!\033[0m")
    dmx_controller, dmx_GSW = dynamixel_connect()
    print("\033[93mCONTROL: Dynamixels Connected!\033[0m")

    try:
        ix = 0 # id of current control point in trajectory
        # Wait for start trigger
        while running and not start:
            time.sleep(0.1)
        # Begin trajectory
        while running:
            x_curr = num_forward_kinematics(joint_coords + joint_offset).reshape((3,1))
            x_ref = traj_position[ix].reshape((3,1))
            v_ref = traj_velocity[ix].reshape((3,1))
            grasp = traj_grasp[ix]

            v_command = v_ref + K @ (x_ref - x_curr)
            v_command = np.vstack((v_command, np.zeros((3,1))))
            Jv_inv = inverse_jacobian(joint_coords + joint_offset)

            joint_velocity = Jv_inv @ v_command
            joint_velocity[[0,1,3,4,5], 0] = np.clip(joint_velocity[[0,1,3,4,5], 0], -1.0, 1.0)

            joint_coords[0] = joint_coords[0] + 0.005*joint_velocity[0, 0]
            joint_coords[1] = joint_coords[1] + 0.005*joint_velocity[1, 0]
            joint_coords[2] = joint_coords[2] + 0.005*joint_velocity[2, 0]
            boom_pos = get_boom_pos(joint_coords[2], joint_velocity[2, 0]) # convert linear d3 to motor angle
            # print(boom_pos)
            joint_coords[3] = joint_coords[3] + 0.005*joint_velocity[3, 0]
            joint_coords[4] = joint_coords[4] + 0.005*joint_velocity[4, 0]
            joint_coords[5] = joint_coords[5] + 0.005*joint_velocity[5, 0]

            # joint limits
            joint_coords[0] = max(min(joint_coords[0], np.pi/2), -np.pi/2)
            joint_coords[1] = max(min(joint_coords[1], np.pi/2), 0)
            boom_pos = max(min(boom_pos, 0), -36)
            
            if grasp:
                grasp_pos = 5000
            else:
                grasp_pos = 3000

            # check status then drive motors
            motor_status(candle, motors)
            motor_drive(candle, motors, joint_coords[0], joint_coords[1], boom_pos)
            dynamixel_drive(dmx_controller, dmx_GSW, [radians_to_ticks(joint_coords[3]) + 50,
                                                      radians_to_ticks(joint_coords[4]) + 1750,
                                                      radians_to_ticks(joint_coords[5]) + 2050,
                                                      grasp_pos])
            
            ix = (ix + 1) % len(traj_grasp)
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
