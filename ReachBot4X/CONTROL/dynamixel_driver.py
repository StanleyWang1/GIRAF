from dynamixel_sdk import *
import numpy as np
import time

from control_table import *
from dynamixel_controller import DynamixelController

# Motor IDs
ARM1_ROLL = 21
ARM1_PITCH = 22
ARM1_BOOM = 23

ARM2_ROLL = 31
ARM2_PITCH = 32
ARM2_BOOM = 33

ARM3_ROLL = 41
ARM3_PITCH = 42
ARM3_BOOM = 43

ARM4_ROLL = 51
ARM4_PITCH = 52
ARM4_BOOM = 53

def dynamixel_connect():
    # Initialize controller
    controller = DynamixelController('COM3', 57600, 2.0)
    group_sync_write = GroupSyncWrite(controller.port_handler, controller.packet_handler, GOAL_POSITION[0], GOAL_POSITION[1])

    # --------------------------------------------------
    for motor_id in [ARM1_ROLL, ARM1_PITCH, ARM1_BOOM, ARM2_ROLL, ARM2_PITCH, ARM2_BOOM, ARM4_ROLL, ARM4_PITCH, ARM4_BOOM]:
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
    for motor_id in [ARM1_ROLL, ARM1_PITCH, ARM1_BOOM, ARM2_ROLL, ARM2_PITCH, ARM2_BOOM, ARM3_ROLL, ARM3_PITCH, ARM3_BOOM, ARM4_ROLL, ARM4_PITCH, ARM4_BOOM]:
        controller.write(motor_id, OPERATING_MODE, 4)  # extended position control
        controller.write(motor_id, PROFILE_VELOCITY, 200) # velocity limit
        controller.write(motor_id, PROFILE_ACCELERATION, 100) # acceleration limit
        controller.write(motor_id, TORQUE_ENABLE, 1) # torque enable

    return controller, group_sync_write

def dynamixel_drive(controller, group_sync_write, ticks):
    param_success = group_sync_write.addParam(ARM1_ROLL, ticks[0].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM1_PITCH, ticks[1].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM1_BOOM, ticks[2].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM2_ROLL, ticks[3].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM2_PITCH, ticks[4].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM2_BOOM, ticks[5].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM3_ROLL, ticks[6].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM3_PITCH, ticks[7].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM3_BOOM, ticks[8].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM4_ROLL, ticks[9].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM4_PITCH, ticks[10].to_bytes(4, 'little', signed=True))
    param_success &= group_sync_write.addParam(ARM4_BOOM, ticks[11].to_bytes(4, 'little', signed=True))

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
    controller.write(ARM1_ROLL, TORQUE_ENABLE, 0)
    controller.write(ARM1_PITCH, TORQUE_ENABLE, 0)
    controller.write(ARM1_BOOM, TORQUE_ENABLE, 0)

    controller.write(ARM2_ROLL, TORQUE_ENABLE, 0)
    controller.write(ARM2_PITCH, TORQUE_ENABLE, 0)
    controller.write(ARM2_BOOM, TORQUE_ENABLE, 0)

    controller.write(ARM3_ROLL, TORQUE_ENABLE, 0)
    controller.write(ARM3_PITCH, TORQUE_ENABLE, 0)
    controller.write(ARM3_BOOM, TORQUE_ENABLE, 0)
    
    controller.write(ARM4_ROLL, TORQUE_ENABLE, 0)
    controller.write(ARM4_PITCH, TORQUE_ENABLE, 0)
    controller.write(ARM4_BOOM, TORQUE_ENABLE, 0)

def radians_to_ticks(rad):
    return int(rad / (2 * np.pi) * 4096)

def main():
    controller, group_sync_write = dynamixel_connect()
    print("\033[93mDYNAMIXEL: Motors Connected, Driving to Home (5 sec)\033[0m")
    dynamixel_drive(controller, group_sync_write, [MOTOR21_HOME, MOTOR22_HOME, MOTOR23_HOME,
                                                   MOTOR31_HOME, MOTOR32_HOME, MOTOR33_HOME,
                                                   MOTOR41_HOME, MOTOR42_HOME, MOTOR43_HOME,
                                                   MOTOR51_HOME, MOTOR52_HOME, MOTOR53_HOME])
    time.sleep(5)
    dynamixel_disconnect(controller)
    print("\033[93mDYNAMIXEL: Motors Disconnected, Torque Off\033[0m")

if __name__ == "__main__":
    main()

