import time
import math
from dynamixel_sdk import * # Uses Dynamixel SDK library

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
DXL_CENTER = 2048 - 120

f_swim = 3.0  # Hz
theta_mid = DXL_CENTER  # Midpoint in ticks
A_swim = 300  # Amplitude in ticks
duration = 25.0  # seconds

# ----------- Setup Communication -----------
portHandler = PortHandler(PORT_NAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    raise IOError("Failed to open port")

if not portHandler.setBaudRate(BAUDRATE):
    raise IOError("Failed to set baudrate")

# ----------- Enable Motor -----------
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 0)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

# ----------- Sinusoidal Actuation -----------
t0 = time.time()
while True:
    t = time.time() - t0
    if t > duration:
        break
    theta = int(theta_mid + A_swim * math.sin(2 * math.pi * f_swim * t))
    theta = max(DXL_MIN_POS, min(DXL_MAX_POS, theta))
    packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, theta)
    time.sleep(0.01)

# ----------- Cleanup -----------
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 0)
portHandler.closePort()
