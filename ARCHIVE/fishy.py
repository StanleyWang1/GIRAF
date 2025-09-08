import time
import math
import threading
from dynamixel_sdk import *  # Uses Dynamixel SDK library
from pynput.keyboard import Listener, Key, KeyCode

# ----------- Parameters -----------
PORT_NAME = 'COM8'
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

DXL_ID = 24

ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64

TORQUE_ENABLE = 1
OPERATING_MODE_POSITION = 3  # Position Control Mode
DXL_MIN_POS = 0
DXL_MAX_POS = 4095
DXL_CENTER = 2048 - 130

# Swim params
f_swim = 3.0      # Hz (initial)
F_MIN = 0.1       # Hz
F_MAX = 10.0      # Hz
F_STEP = 0.25     # Hz per arrow press
theta_mid = DXL_CENTER  # Midpoint in ticks
A_swim = 1000     # Amplitude in ticks
dt = 0.01         # control loop period (s)

# Shared state
state_lock = threading.Lock()
running = True
swimming = False
freq = f_swim

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def on_press(key):
    global running, swimming, freq
    try:
        if key == Key.up:
            with state_lock:
                freq = clamp(freq + F_STEP, F_MIN, F_MAX)
            print(f"[KEY] Frequency increased to {freq:.2f} Hz")
        elif key == Key.down:
            with state_lock:
                freq = clamp(freq - F_STEP, F_MIN, F_MAX)
            print(f"[KEY] Frequency decreased to {freq:.2f} Hz")
        elif key == Key.esc:
            # Optional: ESC also quits
            running = False
        elif key == KeyCode(char='s'):
            with state_lock:
                swimming = not swimming
            print(f"[KEY] Swimming: {'ON' if swimming else 'OFF'}")
        elif key == KeyCode(char='q'):
            running = False
            print("[KEY] Quit requested.")
    except Exception as e:
        print(f"[KEY] Error handling key: {e}")

# ----------- Setup Communication -----------
portHandler = PortHandler(PORT_NAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    raise IOError("Failed to open port")

if not portHandler.setBaudRate(BAUDRATE):
    raise IOError("Failed to set baudrate")

# ----------- Enable Motor -----------
# Disable torque to change mode
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 0)
# Position mode
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION)
# Enable torque
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

# Start keyboard listener
listener = Listener(on_press=on_press)
listener.start()

print("Controls: 's' to start/stop swimming, Up/Down to change frequency, 'q' to quit.")

# ----------- Control Loop -----------
t0 = time.time()
try:
    while running:
        loop_start = time.time()
        with state_lock:
            f = freq
            swim = swimming

        if swim:
            t = loop_start - t0
            theta = int(theta_mid + A_swim * math.sin(2 * math.pi * f * t))
        else:
            theta = theta_mid  # hold center when not swimming

        theta = clamp(theta, DXL_MIN_POS, DXL_MAX_POS)
        packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, theta)

        # simple rate control
        elapsed = time.time() - loop_start
        sleep_time = max(0.0, dt - elapsed)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n[CTRL-C] Exiting...")

finally:
    # ----------- Cleanup -----------
    try:
        packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 0)
    except Exception:
        pass
    try:
        portHandler.closePort()
    except Exception:
        pass
    try:
        listener.stop()
    except Exception:
        passss
    print("Clean shutdown complete.")
