import time
import threading
import msvcrt

from dynamixel_driver import (
    dynamixel_connect, dynamixel_drive, dynamixel_disconnect,
)
from control_table import *

# Control loop params
MOTOR_CTRL_HZ = 500.0      # motor update rate
KEYBOARD_CTRL_HZ = 100.0  # keyboard polling rate
PRINT_HZ = 2.0            # status print rate

# Globals
running = True
running_lock = threading.Lock()

joint_pos = [
    MOTOR21_HOME, MOTOR22_HOME, MOTOR23_HOME,
    MOTOR31_HOME, MOTOR32_HOME, MOTOR33_HOME,
    MOTOR51_HOME, MOTOR52_HOME, MOTOR53_HOME,
]
joint_pos_lock = threading.Lock()

shared = {"Fz": None}
shared_lock = threading.Lock()

start_event = threading.Event()   # <-- new event for starting motor loop

# ------------------------- Motor thread -------------------------
def motor_thread():
    global running, joint_pos
    
    # Connect to motors and home
    dmx_controller, dmx_GSW = dynamixel_connect()
    print("\033[93mCONTROLLER: Motors Connected!\033[0m")
    time.sleep(0.5)
    with joint_pos_lock:
        dynamixel_drive(dmx_controller, dmx_GSW, joint_pos)
    time.sleep(0.5)

    # Wait for user to start control loop (replace input())
    print("\033[93mCONTROLLER: Press ENTER to start controller!\033[0m")
    start_event.wait()   # <-- wait until keyboard thread signals start

    # Performance benchmarking (loop Hz)
    loop_count = 0
    last_time = time.perf_counter()

    print_interval = 1.0 / PRINT_HZ if PRINT_HZ > 0 else 1.0

    try:
        while running:
            with joint_pos_lock:
                current_pos = joint_pos.copy()
                
            dynamixel_drive(dmx_controller, dmx_GSW, current_pos)

            time.sleep(1/MOTOR_CTRL_HZ)

            # perf counting
            loop_count += 1
            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed >= print_interval:
                hz = loop_count / elapsed if elapsed > 0 else float('inf')
                print(f"\033[94m[MOTOR] loop Hz: {hz:.1f}\033[0m")
                loop_count = 0
                last_time = now

    finally:
        dynamixel_disconnect(dmx_controller)
        print("\033[93mCONTROLLER: Motors Disconnected!\033[0m")

def keyboard_thread():
    global running, joint_pos   
    
    TICK_STEP = 10
    
    print("\033[93mKEYBOARD: ENTER to start, W/S to move boom, Q to quit\033[0m")
    
    try:
        while running:
            # Check if a key is pressed (non-blocking on Windows)
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                
                delta1 = 0
                delta2 = 0
                if key == '\r':  # ENTER key
                    start_event.set()   # signal motor_thread to start
                elif key == 'w':
                    delta1 = TICK_STEP
                elif key == 's':
                    delta1 = -TICK_STEP
                elif key == 'e':
                    delta2 = TICK_STEP*5
                elif key == 'r':
                    delta2 = -TICK_STEP*5
                elif key == 'q':
                    with running_lock:
                        running = False
                        
                if delta1 != 0:
                    with joint_pos_lock:
                        joint_pos[1] += delta1
                        joint_pos[4] += delta1
                        joint_pos[7] += delta1
                if delta2 != 0:
                    with joint_pos_lock:
                        joint_pos[2] += delta2
                        joint_pos[5] += delta2
                        joint_pos[8] += delta2
                        
            time.sleep(1 / KEYBOARD_CTRL_HZ)
            
    except Exception as e:
        print(f"\033[91m[KEYBOARD] Error: {e}\033[0m")
    finally:
        print("\033[93mKEYBOARD: stopped\033[0m")

# --------------------------- Main ---------------------------
if __name__ == "__main__":
    print("\033[96mRUN: threaded force→spin — Ctrl+C to stop\033[0m")
    tm = threading.Thread(target=motor_thread,  daemon=True)
    tk = threading.Thread(target=keyboard_thread, daemon=True)
    
    tm.start()
    tk.start()
    
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
        tm.join(timeout=2.0)
        tk.join(timeout=2.0)
        print("\033[92mDONE\033[0m")
