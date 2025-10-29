#!/usr/bin/env python3
# dynamixel_driver.py  — fixed to use controller.write/read (lowercase)

from dynamixel_sdk import *  # GroupSyncWrite, COMM_SUCCESS
import numpy as np
import time

from control_table import *  # e.g., OPERATING_MODE, TORQUE_ENABLE, GOAL_POSITION, PWM_LIMIT, etc.
from dynamixel_controller import DynamixelController

# Motor IDs
JOINT1   = 11
JOINT2   = 12
JOINT3   = 13
GRIPPER1 = 100
GRIPPER2 = 101
GRIPPER3 = 102

ALL_IDS = [JOINT1, JOINT2, JOINT3, GRIPPER1, GRIPPER2, GRIPPER3]

def _w(ctrl: DynamixelController, dxl_id: int, item: tuple[int, int], val: int, label: str) -> bool:
    """Write with a short log on failure."""
    ok = ctrl.write(dxl_id, item, val)
    if not ok:
        print(f"[WARN] write failed: id={dxl_id} {label} -> {val}")
    return ok

def _r(ctrl: DynamixelController, dxl_id: int, item: tuple[int, int], label: str):
    val = ctrl.read(dxl_id, item)
    if val is False:
        print(f"[WARN] read failed: id={dxl_id} {label}")
    return val

def dynamixel_connect(port: str = "/dev/ttyUSB0", baud: int = 57600, proto: float = 2.0):
    """
    Returns:
        controller: DynamixelController
        group_sync_write: GroupSyncWrite configured for GOAL_POSITION (4 bytes)
    """
    controller = DynamixelController(port, baud, proto)

    # GroupSyncWrite for 4-byte GOAL_POSITION
    group_sync_write = GroupSyncWrite(
        controller.port_handler, controller.packet_handler, GOAL_POSITION[0], GOAL_POSITION[1]
    )

    # Reboot all (best-effort)
    for motor_id in ALL_IDS:
        dxl_comm_result, dxl_error = controller.packet_handler.reboot(controller.port_handler, motor_id)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[WARN] Reboot Motor {motor_id}: {controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"[WARN] Reboot Motor {motor_id} error: {controller.packet_handler.getRxPacketError(dxl_error)}")
        else:
            print(f"[OK] Motor {motor_id} rebooted")

    time.sleep(1.0)  # brief settle after reboot

    # Configure motors: Extended Position Mode (4), optional PWM limit for grippers, then Torque Enable
    for motor_id in ALL_IDS:
        _w(controller, motor_id, TORQUE_ENABLE, 0, "TORQUE_ENABLE=0")  # safe to change mode

        ok_mode = _w(controller, motor_id, OPERATING_MODE, 4, "OPERATING_MODE=4 (Extended Position)")
        if not ok_mode:
            print(f"[HINT] If this fails, verify your model’s control table and address for OPERATING_MODE.")

        # Optional: force limit grippers if your control table includes PWM_LIMIT
        if motor_id in (GRIPPER1, GRIPPER2, GRIPPER3):
            try:
                _w(controller, motor_id, PWM_LIMIT, 300, "PWM_LIMIT=300")
            except NameError:
                pass  # PWM_LIMIT not defined in your control_table

        _w(controller, motor_id, TORQUE_ENABLE, 1, "TORQUE_ENABLE=1")

    # Show present positions (sanity)
    for motor_id in ALL_IDS:
        pv = _r(controller, motor_id, PRESENT_POSITION, "PRESENT_POSITION")
        if pv is not False:
            print(f"[INFO] id={motor_id} present={pv}")

    return controller, group_sync_write

def dynamixel_drive(controller: DynamixelController, group_sync_write: GroupSyncWrite, ticks):
    """
    ticks: list/tuple of 6 ints [JOINT1, JOINT2, JOINT3, GRIPPER1, GRIPPER2, GRIPPER3]
    Returns: True if txPacket() succeeded
    """
    if len(ticks) != 6:
        raise ValueError("ticks must be length 6")

    # Build params
    ok = True
    ok &= group_sync_write.addParam(JOINT1,   int(ticks[0]).to_bytes(4, 'little', signed=True))
    ok &= group_sync_write.addParam(JOINT2,   int(ticks[1]).to_bytes(4, 'little', signed=True))
    ok &= group_sync_write.addParam(JOINT3,   int(ticks[2]).to_bytes(4, 'little', signed=True))
    ok &= group_sync_write.addParam(GRIPPER1, int(ticks[3]).to_bytes(4, 'little', signed=True))
    ok &= group_sync_write.addParam(GRIPPER2, int(ticks[4]).to_bytes(4, 'little', signed=True))
    ok &= group_sync_write.addParam(GRIPPER3, int(ticks[5]).to_bytes(4, 'little', signed=True))

    if not ok:
        print("[WARN] GroupSyncWrite addParam failed")
        group_sync_write.clearParam()
        return False

    # Transmit once for all motors
    dxl_comm_result = group_sync_write.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[WARN] SyncWrite comm error: {controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        group_sync_write.clearParam()
        return False

    group_sync_write.clearParam()
    return True

def dynamixel_disconnect(controller: DynamixelController):
    """Torque off each motor and close port."""
    for motor_id in ALL_IDS:
        _w(controller, motor_id, TORQUE_ENABLE, 0, "TORQUE_ENABLE=0")
    try:
        controller.close_port()
    except Exception:
        pass

def radians_to_ticks(rad):
    return int(rad / (2 * np.pi) * 4096)

# Simple manual test
if __name__ == "__main__":
    ctrl, gsw = dynamixel_connect()
    print("\033[93mDYNAMIXEL: Connected. Stepping JOINT2 (ID=12) a bit...\033[0m")
    ticks = [0, 0, 0, 0, 0, 0]
    # Seed with present positions so we add steps from current pose
    for i, mid in enumerate(ALL_IDS):
        pv = ctrl.read(mid, PRESENT_POSITION)
        ticks[i] = int(pv) if pv is not False else 0

    try:
        for k in range(50):
            ticks[1] += 64  # JOINT2 (ID 12)
            dynamixel_drive(ctrl, gsw, ticks)
            time.sleep(0.02)
    finally:
        dynamixel_disconnect(ctrl)
        print("\033[93mDYNAMIXEL: Disconnected\033[0m")
