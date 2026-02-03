"""Test Xbox controller rumble using ctypes and XInput directly."""
import ctypes
import time
from ctypes import wintypes

# XInput constants
XINPUT_GAMEPAD_LEFT_MOTOR = 0
XINPUT_GAMEPAD_RIGHT_MOTOR = 1

# Load XInput DLL
try:
    xinput = ctypes.windll.xinput1_4
except:
    try:
        xinput = ctypes.windll.xinput1_3
    except:
        try:
            xinput = ctypes.windll.xinput9_1_0
        except:
            print("Could not load any XInput DLL")
            exit(1)

# Define structures
class XINPUT_VIBRATION(ctypes.Structure):
    _fields_ = [
        ("wLeftMotorSpeed", wintypes.WORD),
        ("wRightMotorSpeed", wintypes.WORD),
    ]

def set_vibration(controller_id, left_motor, right_motor):
    """Set vibration for Xbox controller.
    
    Args:
        controller_id: 0-3 for controller index
        left_motor: 0-65535 for low frequency motor
        right_motor: 0-65535 for high frequency motor
    """
    vibration = XINPUT_VIBRATION(left_motor, right_motor)
    xinput.XInputSetState(controller_id, ctypes.byref(vibration))

def test_rumble():
    """Test rumble on first controller."""
    print("Testing rumble on controller 0...")
    
    # Low frequency rumble (left motor)
    print("Low frequency rumble...")
    set_vibration(0, 32767, 0)
    time.sleep(0.5)
    set_vibration(0, 0, 0)
    time.sleep(0.3)
    
    # High frequency rumble (right motor)
    print("High frequency rumble...")
    set_vibration(0, 0, 32767)
    time.sleep(0.5)
    set_vibration(0, 0, 0)
    time.sleep(0.3)
    
    # Both motors
    print("Both motors...")
    set_vibration(0, 32767, 32767)
    time.sleep(1.0)
    set_vibration(0, 0, 0)
    
    print("Done!")

if __name__ == "__main__":
    test_rumble()
