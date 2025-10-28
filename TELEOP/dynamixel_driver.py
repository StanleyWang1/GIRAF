from dynamixel_sdk import *
import numpy as np
import time

from control_table import *
from dynamixel_controller import DynamixelController

# Motor IDs
JOINT1 = 11
JOINT2 = 12
JOINT3 = 13
GRIPPER1 = 100
GRIPPER2 = 101
GRIPPER3 = 102

def dynamixel_connect():
    # Initialize controller
    controller = DynamixelController('/dev/ttyUSB0', 57600, 2.0)
    group_sync_write = GroupSyncWrite(controller.port_handler, controller.packet_handler, GOAL_POSITION[0], GOAL_POSITION[1])

    # Reboot all motors to ensure clean startup
    for motor_id in [JOINT1, JOINT2, JOINT3, GRIPPER1, GRIPPER2, GRIPPER3]:
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
    controller.WRITE(JOINT1, OPERATING_MODE, 4)  # extended position control
    controller.WRITE(JOINT2, OPERATING_MODE, 4)
    controller.WRITE(JOINT3, OPERATING_MODE, 4)
    controller.WRITE(GRIPPER1, OPERATING_MODE, 4)
    controller.WRITE(GRIPPER2, OPERATING_MODE, 4)
    controller.WRITE(GRIPPER3, OPERATING_MODE, 4)

    # Optional: Force Limit on Gripper
    controller.WRITE(GRIPPER1, PWM_LIMIT, 200)
    controller.WRITE(GRIPPER2, PWM_LIMIT, 200)
    controller.WRITE(GRIPPER3, PWM_LIMIT, 200)

    # Torque Enable
    controller.WRITE(JOINT1, TORQUE_ENABLE, 1)
    controller.WRITE(JOINT2, TORQUE_ENABLE, 1)
    controller.WRITE(JOINT3, TORQUE_ENABLE, 1)
    controller.WRITE(GRIPPER1, TORQUE_ENABLE, 1)
    controller.WRITE(GRIPPER2, TORQUE_ENABLE, 1)
    controller.WRITE(GRIPPER3, TORQUE_ENABLE, 1)

    return controller, group_sync_write

def dynamixel_drive(controller, group_sync_write, ticks):
    param_success = group_sync_write.addParam(JOINT1, ticks[0].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(JOINT2, ticks[1].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(JOINT3, ticks[2].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(GRIPPER1, ticks[3].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(GRIPPER2, ticks[4].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(GRIPPER3, ticks[5].to_bytes(4, 'little', signed=True))

    if not param_success:
        print("Failed to add parameters for SyncWrite")
        return False

    dxl_comm_result = group_sync_write.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"SyncWrite communication error: {controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        return False

    group_sync_write.clearParam()
    return True

def dynamixel_disconnect(controller):
    # Torque OFF all motors individually (simple)
    controller.WRITE(JOINT1, TORQUE_ENABLE, 0)
    controller.WRITE(JOINT2, TORQUE_ENABLE, 0)
    controller.WRITE(JOINT3, TORQUE_ENABLE, 0)
    controller.WRITE(GRIPPER1, TORQUE_ENABLE, 0)
    controller.WRITE(GRIPPER2, TORQUE_ENABLE, 0)
    controller.WRITE(GRIPPER3, TORQUE_ENABLE, 0)

def radians_to_ticks(rad):
    return int(rad / (2 * np.pi) * 4096)

def main():
    controller, group_sync_write = dynamixel_connect()
    print("\033[93mDYNAMIXEL: Motors Connected, Driving to Home (5 sec)\033[0m")
    dynamixel_drive(controller, group_sync_write, [MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR100_OPEN, MOTOR101_OPEN, MOTOR102_OPEN])
    time.sleep(5)
    dynamixel_disconnect(controller)
    print("\033[93mDYNAMIXEL: Motors Disconnected, Torque Off\033[0m")

if __name__ == "__main__":
    main()

