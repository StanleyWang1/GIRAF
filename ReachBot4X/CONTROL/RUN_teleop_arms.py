import threading
import time

from joystick_driver import joystick_connect, joystick_read, joystick_disconnect
from dynamixel_driver import (
    dynamixel_connect, dynamixel_drive, dynamixel_disconnect,
)
from control_table import *

## ----------------------------------------------------------------------------------------------------
# Joystick Controller Teleoperation
## ----------------------------------------------------------------------------------------------------

# Control loop params
MOTOR_CTRL_HZ = 1000.0      # motor update rate
JOYSTICK_CTRL_HZ = 200.0    # joystick polling rate

# Global Variables
joystick_data = {"LX":0, "LY":0, "LT":0, "RT":0, "XB":0, "LB":0, "RB":0, "MENULEFT":0, "MENURIGHT":0, "YB":0, "AB":0, "BB":0}
joystick_lock = threading.Lock()

running = True
running_lock = threading.Lock()

# Motor positions for all 12 motors (4 arms × 3 joints)
MOTOR_HOMES = [
    MOTOR21_HOME, MOTOR22_HOME, MOTOR23_HOME,
    MOTOR31_HOME, MOTOR32_HOME, MOTOR33_HOME,
    MOTOR41_HOME, MOTOR42_HOME, MOTOR43_HOME,
    MOTOR51_HOME, MOTOR52_HOME, MOTOR53_HOME,
]
joint_pos = MOTOR_HOMES.copy()
joint_pos_lock = threading.Lock()

# Arm selection (0=ARM1/ID2, 1=ARM2/ID3, 2=ARM3/ID4, 3=ARM4/ID5)
selected_arm = 0
selected_arm_lock = threading.Lock()

# Pitch torque enable state
pitch_torque_enabled = True
pitch_torque_lock = threading.Lock() 

## ----------------------------------------------------------------------------------------------------
# Joystick Monitoring Thread
## ----------------------------------------------------------------------------------------------------
def joystick_monitor():
    global joystick_data, running, selected_arm, pitch_torque_enabled

    js = joystick_connect()
    print("\033[93mTELEOP: Joystick Connected!\033[0m")

    # Track button states for edge detection
    menuleft_prev = 0
    menuright_prev = 0
    yb_prev = 0

    while running:
        with joystick_lock:
            joystick_data = joystick_read(js)
            
            # Arm selection with MENULEFT/MENURIGHT (edge detection)
            menuleft_curr = joystick_data["MENULEFT"]
            menuright_curr = joystick_data["MENURIGHT"]
            yb_curr = joystick_data["YB"]
            
            if menuleft_curr and not menuleft_prev:
                with selected_arm_lock:
                    selected_arm = (selected_arm - 1) % 4
                    print(f"\033[96m→ ARM {selected_arm + 2} selected\033[0m")
            
            if menuright_curr and not menuright_prev:
                with selected_arm_lock:
                    selected_arm = (selected_arm + 1) % 4
                    print(f"\033[96m→ ARM {selected_arm + 2} selected\033[0m")
            
            # Y button toggles pitch torque
            if yb_curr and not yb_prev:
                with pitch_torque_lock:
                    pitch_torque_enabled = not pitch_torque_enabled
                    state_str = "ENABLED" if pitch_torque_enabled else "DISABLED"
                    print(f"\033[95m⚡ Pitch Torque {state_str}\033[0m")
            
            menuleft_prev = menuleft_curr
            menuright_prev = menuright_curr
            yb_prev = yb_curr
            
        time.sleep(1 / JOYSTICK_CTRL_HZ)

    joystick_disconnect(js)
    print("\033[93mTELEOP: Joystick Disconnected!\033[0m")

## ----------------------------------------------------------------------------------------------------
# Motor Control Thread
## ----------------------------------------------------------------------------------------------------
def motor_control():
    global joystick_data, running, joint_pos, selected_arm, pitch_torque_enabled

    dmx_controller, dmx_GSW = dynamixel_connect()
    print("\033[93mTELEOP: Motors Connected!\033[0m")
    time.sleep(0.5)
    
    # Drive to home positions
    with joint_pos_lock:
        dynamixel_drive(dmx_controller, dmx_GSW, joint_pos)
    time.sleep(0.5)

    print("\033[93mTELEOP: Press LB+RB to enable control, XB to quit\033[0m")
    print("\033[93mTELEOP: Use MENULEFT/MENURIGHT to switch arms\033[0m")
    print("\033[93mTELEOP: Y=toggle pitch torque, A=extend all booms, B=retract all booms\033[0m")
    
    # Track previous pitch torque state
    prev_pitch_torque = True
    
    # Pitch motor IDs (index offset 1: motors 22, 32, 42, 52)
    PITCH_MOTORS = [22, 32, 42, 52]

    try:
        while running:
            # unpack joystick data from dict
            with joystick_lock: 
                LX = joystick_data["LX"]
                LY = joystick_data["LY"]
                LT = joystick_data["LT"]
                RT = joystick_data["RT"]
                XB = joystick_data["XB"]
                LB = joystick_data["LB"]
                RB = joystick_data["RB"]
                AB = joystick_data["AB"]
                BB = joystick_data["BB"]
            
            if XB: # stop button engaged - abort process!
                with running_lock:
                    running = False
                break
            
            # Handle pitch torque enable/disable
            with pitch_torque_lock:
                current_pitch_torque = pitch_torque_enabled
            
            if current_pitch_torque != prev_pitch_torque:
                if current_pitch_torque:
                    # Re-enabling torque: read current positions first to avoid jumps
                    with joint_pos_lock:
                        joint_pos[1] = dmx_controller.read(22, PRESENT_POSITION)   # ARM1 pitch
                        joint_pos[4] = dmx_controller.read(32, PRESENT_POSITION)   # ARM2 pitch
                        joint_pos[7] = dmx_controller.read(42, PRESENT_POSITION)   # ARM3 pitch
                        joint_pos[10] = dmx_controller.read(52, PRESENT_POSITION)  # ARM4 pitch
                    # Now enable torque
                    for motor_id in PITCH_MOTORS:
                        dmx_controller.write(motor_id, TORQUE_ENABLE, 1)
                    print("\033[95m✓ Pitch positions synchronized before enabling torque\033[0m")
                else:
                    # Disabling torque
                    for motor_id in PITCH_MOTORS:
                        dmx_controller.write(motor_id, TORQUE_ENABLE, 0)
                prev_pitch_torque = current_pitch_torque

            # Get current arm index
            with selected_arm_lock:
                arm_idx = selected_arm
            
            # Calculate base index for selected arm (each arm has 3 joints)
            # arm_idx=0 → indices 0,1,2 (ARM1/ID21,22,23)
            # arm_idx=1 → indices 3,4,5 (ARM2/ID31,32,33)
            # arm_idx=2 → indices 6,7,8 (ARM3/ID41,42,43)
            # arm_idx=3 → indices 9,10,11 (ARM4/ID51,52,53)
            base_idx = arm_idx * 3
            roll_idx = base_idx + 0  # X1 motor (roll)
            pitch_idx = base_idx + 1  # X2 motor (pitch)
            boom_idx = base_idx + 2   # X3 motor (boom)

            # Drive ratios (gains matching RUN_standup.py)
            TICK_STEP = 5
            roll_gain = TICK_STEP
            pitch_gain = TICK_STEP
            boom_gain = TICK_STEP * 5

            # safety interlock
            if LB and RB:
                with joint_pos_lock:
                    dynamic_gain_scale = 1 - 0.8 / (20000 - MOTOR_HOMES[boom_idx]) * (joint_pos[boom_idx] - MOTOR_HOMES[boom_idx]) # linear scaling of gain based on boom length
                    dynamic_gain_scale = max(0.2, dynamic_gain_scale) # clamp gain scale to >= 0.1

                    # Roll control (LX) - matches RUN_standup sign convention
                    joint_pos[roll_idx] += round(roll_gain * LX * dynamic_gain_scale)
                    
                    # Pitch control (LY) - matches RUN_standup sign convention
                    joint_pos[pitch_idx] += round(pitch_gain * LY * dynamic_gain_scale)
                    
                    # A button - extend all booms (all 4 arms)
                    if AB:
                        joint_pos[2] += round(boom_gain * 0.4)   # ARM1 boom
                        joint_pos[5] += round(boom_gain * 0.4)  # ARM2 boom
                        joint_pos[8] += round(boom_gain * 0.4)  # ARM3 boom
                        joint_pos[11] += round(boom_gain * 0.4) # ARM4 boom
                    
                    # B button - retract all booms (all 4 arms)
                    if BB:
                        joint_pos[2] -= round(boom_gain * 0.4)  # ARM1 boom
                        joint_pos[5] -= round(boom_gain * 0.4)  # ARM2 boom
                        joint_pos[8] -= round(boom_gain * 0.4)  # ARM3 boom
                        joint_pos[11] -= round(boom_gain * 0.4) # ARM4 boom
                        # Clamp all booms to home positions
                        joint_pos[2] = max(joint_pos[2], MOTOR23_HOME)
                        joint_pos[5] = max(joint_pos[5], MOTOR33_HOME)
                        joint_pos[8] = max(joint_pos[8], MOTOR43_HOME)
                        joint_pos[11] = max(joint_pos[11], MOTOR53_HOME)
                    
                    # Boom control (triggers)
                    if RT and not LT: # extend boom
                        joint_pos[boom_idx] += round(boom_gain * RT)
                    elif LT and not RT: # retract boom
                        joint_pos[boom_idx] -= round(boom_gain * LT)
                    
                    # Clamp boom to lower bound (home position)
                    if arm_idx == 0:  # ARM1
                        joint_pos[boom_idx] = max(joint_pos[boom_idx], MOTOR23_HOME)
                    elif arm_idx == 1:  # ARM2
                        joint_pos[boom_idx] = max(joint_pos[boom_idx], MOTOR33_HOME)
                    elif arm_idx == 2:  # ARM3
                        joint_pos[boom_idx] = max(joint_pos[boom_idx], MOTOR43_HOME)
                    elif arm_idx == 3:  # ARM4
                        joint_pos[boom_idx] = max(joint_pos[boom_idx], MOTOR53_HOME)
            
            # Drive motors
            with joint_pos_lock:
                current_pos = joint_pos.copy()
            dynamixel_drive(dmx_controller, dmx_GSW, current_pos)
            
            time.sleep(1 / MOTOR_CTRL_HZ)
            
    finally:
        dynamixel_disconnect(dmx_controller)
        print("\033[93mTELEOP: Motors Disconnected!\033[0m")
        

## ----------------------------------------------------------------------------------------------------
# Main
## ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\033[96mRUN: Teleop Arms — Ctrl+C to stop\033[0m")
    joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
    motor_thread = threading.Thread(target=motor_control, daemon=True)
    
    joystick_thread.start()
    motor_thread.start()
    
    try:
        while True:
            with running_lock:
                if not running:
                    break
            time.sleep(0.5)
    except KeyboardInterrupt:
        with running_lock:
            running = False
    finally:
        joystick_thread.join(timeout=2.0)
        motor_thread.join(timeout=2.0)
        print("\033[92mDONE\033[0m")

