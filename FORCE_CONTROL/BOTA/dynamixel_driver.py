from dynamixel_sdk import *
import numpy as np
import time

from control_table import *
from dynamixel_controller import DynamixelController

# Motor IDs
ROLL_MOTOR = 11
PITCH_MOTOR = 12
BOOM_MOTOR = 13

# MOTOR_IDS = [ROLL_MOTOR, PITCH_MOTOR, BOOM_MOTOR]
MOTOR_IDS = [PITCH_MOTOR, BOOM_MOTOR]

def dynamixel_connect():
    # Initialize controller
    controller = DynamixelController('/dev/ttyUSB0', 2000000, 2.0)
    group_sync_write = GroupSyncWrite(controller.port_handler, controller.packet_handler, GOAL_POSITION[0], GOAL_POSITION[1])

    # --------------------------------------------------
    # Reboot WRIST motors to ensure clean startup
    for motor_id in MOTOR_IDS:
        dxl_comm_result, dxl_error = controller.packet_handler.reboot(controller.port_handler, motor_id)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to reboot Motor {motor_id}: {controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Error rebooting Motor {motor_id}: {controller.packet_handler.getRxPacketError(dxl_error)}")
        else:
            print(f"Motor {motor_id} rebooted successfully.")

    # Give motors time to reboot
    time.sleep(2)

    # Set Control Mode
    for motor_id in MOTOR_IDS:
        controller.write(motor_id, OPERATING_MODE, 4)  # extended position control
        controller.write(motor_id, PROFILE_VELOCITY, 400) # velocity limit
        controller.write(motor_id, PROFILE_ACCELERATION, 500) # acceleration limit
        controller.write(motor_id, TORQUE_ENABLE, 1) # torque enable
 
    # Optional: Force Limit on Gripper
    # controller.write(GRIPPER, PWM_LIMIT, 250)

    return controller, group_sync_write

def dynamixel_drive(controller, group_sync_write, ticks):
    param_success = group_sync_write.addParam(PITCH_MOTOR, ticks[0].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(BOOM_MOTOR, ticks[1].to_bytes(4, 'little', signed=True))
    # param_success &= group_sync_write.addParam(JOINT3, ticks[2].to_bytes(4, 'little', signed=True))
    # param_success &= group_sync_write.addParam(GRIPPER, ticks[3].to_bytes(4, 'little', signed=True))

    if not param_success:
        print("Failed to add parameters for Syncwrite")
        return False

    dxl_comm_result = group_sync_write.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Syncwrite communication error: {controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        return False

    group_sync_write.clearParam()
    return True

def dynamixel_disconnect(controller):
    # Torque OFF all motors individually (simple)
    for motor_id in MOTOR_IDS:
        controller.write(motor_id, TORQUE_ENABLE, 0) # torque disable

def radians_to_ticks(rad):
    return int(rad / (2 * np.pi) * 4096)

def main():
    controller, group_sync_write = dynamixel_connect()
    print("\033[93mDYNAMIXEL: Motors Connected, Driving to Home (5 sec)\033[0m")
    dynamixel_drive(controller, group_sync_write, [MOTOR12_HOME - 500])
    time.sleep(5)
    dynamixel_disconnect(controller)
    print("\033[93mDYNAMIXEL: Motors Disconnected, Torque Off\033[0m")

if __name__ == "__main__":
    main()

