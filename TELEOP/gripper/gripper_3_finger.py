import time
import numpy as np
from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite

from control_table import *
from dynamixel_controller import DynamixelController

# =========================
# Connection
# =========================
PORT = "COM8"  # change to "COMx" on Windows if needed
BAUD = 57600
PROTO = 2.0

# =========================
# Model specifics (XL330-W288-T)
# =========================
TICKS_PER_DEG = 4096.0 / 360.0  # ≈ 11.3778 ticks/deg

# Operating Modes (X-series)
MODE_EXTENDED_POSITION = 4
MODE_PWM = 16

# =========================
# Motor IDs
# =========================
MOTOR100 = 100
MOTOR101 = 101
MOTOR102 = 102
MOTORS = (MOTOR100, MOTOR101, MOTOR102)

# =========================
# Home angles (deg) & direction
# User-provided: negative from home closes for all three.
# Keep these in degrees; we convert to ticks below.
# =========================
HOME_DEG = {
    MOTOR100: 40.0,  # negative is closing
    MOTOR101: 268.0,   # negative is closing
    MOTOR102: 30.0,  # negative is closing
}

def deg_to_ticks(deg: float) -> int:
    return int(round(deg * TICKS_PER_DEG))

HOME_TICKS = {m: deg_to_ticks(HOME_DEG[m]) for m in MOTORS}

# ----------------------
# Soft limits (signed 32-bit range for Extended Position)
# ----------------------
SOFT_MIN = -2_147_483_648
SOFT_MAX =  2_147_483_647

def clamp(v, vmin=SOFT_MIN, vmax=SOFT_MAX):
    return max(vmin, min(vmax, int(v)))

# ----------------------
# Motion profiles (ticks/s and ticks/s^2) for Extended Position
# ----------------------
PROFILE_VEL = 200
PROFILE_ACC = 50

# ----------------------
# PWM drive magnitude (counts). Typical limit ~885 by default.
# Adjust to taste or read PWM_LIMIT if you have a helper.
# ----------------------
PWM_DRIVE = 500

# Per-motor PWM sign so that "close" drives negative-from-home.
# If a finger moves the wrong way, flip its sign here.
PWM_SIGN = {
    MOTOR100: -1,  # negative closes
    MOTOR101: -1,  # negative closes
    MOTOR102: -1,  # negative closes
}

def set_motion_profiles(controller, vel=PROFILE_VEL, acc=PROFILE_ACC):
    """Set velocity/accel profiles (ticks/s and ticks/s^2) for all motors."""
    for m in MOTORS:
        controller.write(m, PROFILE_VELOCITY, int(vel))
        controller.write(m, PROFILE_ACCELERATION, int(acc))
    time.sleep(0.05)

def set_operating_mode(controller, mode, torque_cycle=True):
    """
    Safely switch operating mode:
      - torque off (if torque_cycle)
      - write operating mode
      - torque on (if torque_cycle)
    """
    if torque_cycle:
        for m in MOTORS:
            controller.write(m, TORQUE_ENABLE, 0)
        time.sleep(0.05)

    for m in MOTORS:
        controller.write(m, OPERATING_MODE, int(mode))
    time.sleep(0.05)

    if torque_cycle:
        for m in MOTORS:
            controller.write(m, TORQUE_ENABLE, 1)
        time.sleep(0.05)

def dynamixel_connect():
    controller = DynamixelController(PORT, BAUD, PROTO)

    # GroupSyncWrite for GOAL_POSITION (used in Extended Position mode)
    gsw_pos = GroupSyncWrite(
        controller.port_handler,
        controller.packet_handler,
        GOAL_POSITION[0],
        GOAL_POSITION[1],
    )

    # Reboot motors to a clean state
    for m in MOTORS:
        dxl_comm_result, dxl_error = controller.packet_handler.reboot(
            controller.port_handler, m
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[WARN] Reboot {m} failed: "
                  f"{controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"[WARN] Reboot {m} error: "
                  f"{controller.packet_handler.getRxPacketError(dxl_error)}")
        else:
            print(f"[OK] Motor {m} rebooted")
    time.sleep(1.5)

    # Start in Extended Position mode with torque ON
    set_operating_mode(controller, MODE_EXTENDED_POSITION, torque_cycle=True)
    set_motion_profiles(controller, vel=PROFILE_VEL, acc=PROFILE_ACC)

    return controller, gsw_pos

def go_home(controller, gsw_pos):
    """Drive all motors to HOME (fully open) in Extended Position mode."""
    ok_all = True
    for m in MOTORS:
        goal = clamp(HOME_TICKS[m])
        ok = gsw_pos.addParam(m, int(goal).to_bytes(4, "little", signed=True))
        if not ok:
            ok_all = False
            print(f"[ERROR] addParam failed (home) for ID {m}")
            break

    if not ok_all:
        gsw_pos.clearParam()
        return False

    dxl_comm_result = gsw_pos.txPacket()
    gsw_pos.clearParam()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ERROR] GroupSyncWrite txPacket (home): "
              f"{controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        return False

    return True

def pwm_drive(controller, pwm_map):
    """
    Command PWM directly: dict {motor_id: pwm_value}.
    Typical safe range is -PWM_LIMIT..+PWM_LIMIT.
    """
    for m, val in pwm_map.items():
        controller.write(m, GOAL_PWM, int(val))
    time.sleep(0.02)

def pwm_close(controller, magnitude=PWM_DRIVE):
    """Apply PWM to close (negative from home)."""
    pwm_map = {m: PWM_SIGN[m] * abs(int(magnitude)) for m in MOTORS}
    pwm_drive(controller, pwm_map)

def pwm_open(controller, magnitude=PWM_DRIVE):
    """Apply PWM to open (positive toward/above home)."""
    pwm_map = {m: -PWM_SIGN[m] * abs(int(magnitude)) for m in MOTORS}
    pwm_drive(controller, pwm_map)

def pwm_stop(controller):
    """Stop PWM (0) on all motors; remain in PWM mode."""
    pwm_map = {m: 0 for m in MOTORS}
    pwm_drive(controller, pwm_map)

def torque_off(controller):
    for m in MOTORS:
        controller.write(m, TORQUE_ENABLE, 0)
    time.sleep(0.05)

def position_nudge(controller, gsw_pos, delta_deg):
    """
    Nudge all motors in Extended Position mode by a given delta (deg).
    Negative delta closes; positive opens (per your convention).
    """
    delta_ticks = deg_to_ticks(delta_deg)
    # Read current Present Position (optional). If you prefer open-loop from HOME, comment out reads.
    # Here we just command relative to HOME for simplicity:
    target = {m: clamp(HOME_TICKS[m] + delta_ticks) for m in MOTORS}

    ok_all = True
    for m in MOTORS:
        ok = gsw_pos.addParam(m, int(target[m]).to_bytes(4, "little", signed=True))
        if not ok:
            ok_all = False
            print(f"[ERROR] addParam failed (nudge) for ID {m}")
            break

    if not ok_all:
        gsw_pos.clearParam()
        return False

    dxl_comm_result = gsw_pos.txPacket()
    gsw_pos.clearParam()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ERROR] GroupSyncWrite txPacket (nudge): "
              f"{controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        return False

    return True

def main():
    controller, gsw_pos = dynamixel_connect()
    print("[INFO] Motors initialized in Extended Position mode. Homing...")
    go_home(controller, gsw_pos)
    time.sleep(1.0)
    print("[OK] At home (open).")

    print("\nCommands:")
    print("  g = switch to PWM mode and CLOSE (apply PWM)")
    print("  v = switch to PWM mode and OPEN  (apply PWM)")
    print("  z = set PWM = 0 (stop) (stays in PWM mode)")
    print("  p = position nudge in degrees (Extended Position mode; negative closes)")
    print("  o = switch to Extended Position mode and go HOME (open)")
    print("  s = torque OFF")
    print("  q = quit\n")

    try:
        while True:
            cmd = input("Enter command [g/v/z/p/o/s/q]: ").strip().lower()

            if cmd == "g":
                print("[INFO] Switching to PWM mode… (CLOSE)")
                set_operating_mode(controller, MODE_PWM, torque_cycle=True)
                pwm_close(controller, PWM_DRIVE)
                print(f"[OK] PWM close at ~{PWM_DRIVE} counts.")

            elif cmd == "v":
                print("[INFO] Switching to PWM mode… (OPEN)")
                set_operating_mode(controller, MODE_PWM, torque_cycle=True)
                pwm_open(controller, PWM_DRIVE)
                print(f"[OK] PWM open at ~{PWM_DRIVE} counts.")

            elif cmd == "z":
                print("[INFO] PWM stop (0).")
                pwm_stop(controller)

            elif cmd == "p":
                try:
                    delta_str = input("  Δdeg (negative closes, e.g., -10): ").strip()
                    delta_deg = float(delta_str)
                except ValueError:
                    print("[WARN] Not a number.")
                    continue
                print("[INFO] Switching to Extended Position mode and nudging…")
                set_operating_mode(controller, MODE_EXTENDED_POSITION, torque_cycle=True)
                set_motion_profiles(controller, vel=PROFILE_VEL, acc=PROFILE_ACC)
                if position_nudge(controller, gsw_pos, delta_deg):
                    print(f"[OK] Nudged {delta_deg:+.2f} deg from HOME.")

            elif cmd == "o":
                print("[INFO] Switching to Extended Position mode and homing…")
                set_operating_mode(controller, MODE_EXTENDED_POSITION, torque_cycle=True)
                set_motion_profiles(controller, vel=PROFILE_VEL, acc=PROFILE_ACC)
                if go_home(controller, gsw_pos):
                    print("[OK] Back at HOME (open).")

            elif cmd == "s":
                print("[INFO] Torque OFF for all.")
                torque_off(controller)
                print("[OK] Motors disabled (torque off).")

            elif cmd == "q":
                print("[INFO] Quitting…")
                break

            else:
                print("[WARN] Unknown command. Use g/v/z/p/o/s/q.")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt — stopping.")
    finally:
        try:
            pwm_stop(controller)  # safe if in PWM mode; harmless otherwise
            torque_off(controller)
        except Exception as e:
            print(f"[WARN] Failed to disable torque cleanly: {e}")
        controller.close_port()
        print("[DONE] Motors stopped. Torque off. Port closed.")

if __name__ == "__main__":
    main()
