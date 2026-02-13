import time
import threading
import msvcrt

from dynamixel_driver import (
    dynamixel_connect, dynamixel_drive, dynamixel_disconnect,
)
from control_table import *

# Control loop params
MOTOR_CTRL_HZ = 1000.0      # motor update rate
KEYBOARD_CTRL_HZ = 100.0  # keyboard polling rate
PRINT_HZ = 2.0            # status print rate
DIAGNOSTICS_HZ = 10.0     # diagnostics print rate

# Globals
running = True
running_lock = threading.Lock()

joint_pos = [
    MOTOR21_HOME, MOTOR22_HOME, MOTOR23_HOME,
    MOTOR31_HOME, MOTOR32_HOME, MOTOR33_HOME,
    MOTOR41_HOME, MOTOR42_HOME, MOTOR43_HOME,
    MOTOR51_HOME, MOTOR52_HOME, MOTOR53_HOME,
]
joint_pos_lock = threading.Lock()

# Motor loads for diagnostics (motors 22, 32, 42, 52)
motor_loads = [0, 0, 0, 0]  # [ARM1_PITCH, ARM2_PITCH, ARM3_PITCH, ARM4_PITCH]
motor_loads_lock = threading.Lock()

# Motor control Hz tracking
motor_hz = 0.0
motor_hz_lock = threading.Lock()

shared = {"Fz": None}
shared_lock = threading.Lock()

start_event = threading.Event()   # <-- new event for starting motor loop

# ------------------------- Motor thread -------------------------
def motor_thread():
    global running, joint_pos, motor_loads
    
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

    # Track motor control Hz
    loop_count = 0
    last_hz_calc_time = time.perf_counter()

    try:
        while running:
            with joint_pos_lock:
                current_pos = joint_pos.copy()
                
            dynamixel_drive(dmx_controller, dmx_GSW, current_pos)

            # Update Hz calculation
            loop_count += 1
            now = time.perf_counter()
            elapsed = now - last_hz_calc_time
            if elapsed >= 0.5:  # Update Hz twice per second
                hz = loop_count / elapsed
                with motor_hz_lock:
                    global motor_hz
                    motor_hz = hz
                loop_count = 0
                last_hz_calc_time = now

            time.sleep(1/MOTOR_CTRL_HZ)

    finally:
        dynamixel_disconnect(dmx_controller)
        print("\033[93mCONTROLLER: Motors Disconnected!\033[0m")

def keyboard_thread():
    global running, joint_pos   
    
    TICK_STEP = 5
    
    print("\033[93mKEYBOARD: ENTER to start, W/S to move boom, Q to quit\033[0m")
    
    try:
        while running:
            # Check if a key is pressed (non-blocking on Windows)
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                
                delta1 = 0
                delta2 = 0
                delta3 = 0
                if key == '\r':  # ENTER key
                    start_event.set()   # signal motor_thread to start
                elif key == 'w':
                    delta1 = TICK_STEP
                elif key == 's':
                    delta1 = -TICK_STEP
                elif key == 'a':
                    delta2 = TICK_STEP
                elif key == 'd':
                    delta2 = -TICK_STEP
                elif key == 'e':
                    delta3 = TICK_STEP*5
                elif key == 'r':
                    delta3 = -TICK_STEP*5
                elif key == 'q':
                    with running_lock:
                        running = False
                        
                if delta1 != 0: # pitch
                    with joint_pos_lock:
                        joint_pos[1] += delta1
                        joint_pos[4] += delta1
                        joint_pos[7] += delta1
                        joint_pos[10] += delta1
                if delta2 != 0: # roll
                    with joint_pos_lock:
                        joint_pos[0] += delta2
                        joint_pos[3] += delta2
                        joint_pos[6] += delta2
                        joint_pos[9] += delta2
                if delta3 != 0: # boom
                    with joint_pos_lock:
                        joint_pos[2] += delta3
                        joint_pos[5] += delta3
                        joint_pos[8] += delta3
                        joint_pos[11] += delta3
                        # Clamp boom positions to lower bounds
                        joint_pos[2] = max(joint_pos[2], MOTOR23_HOME)
                        joint_pos[5] = max(joint_pos[5], MOTOR33_HOME)
                        joint_pos[8] = max(joint_pos[8], MOTOR43_HOME)
                        joint_pos[11] = max(joint_pos[11], MOTOR53_HOME)
                        
            time.sleep(1 / KEYBOARD_CTRL_HZ)
            
    except Exception as e:
        print(f"\033[91m[KEYBOARD] Error: {e}\033[0m")
    finally:
        print("\033[93mKEYBOARD: stopped\033[0m")

# ------------------------- Diagnostics thread -------------------------
def diagnostics_thread():
    global running, motor_loads
    
    diag_loop_count = 0
    diag_last_time = time.perf_counter()
    
    # Wait for start
    start_event.wait()
    
    # Reserve space for diagnostics
    print("\n" * 2)
    
    try:
        while running:
            with motor_loads_lock:
                loads = motor_loads.copy()
            
            with motor_hz_lock:
                current_motor_hz = motor_hz
            
            # Calculate diagnostics Hz
            diag_loop_count += 1
            now = time.perf_counter()
            
            diag_elapsed = now - diag_last_time
            if diag_elapsed >= 1.0 / DIAGNOSTICS_HZ:
                diag_hz = diag_loop_count / diag_elapsed
                
                # Convert to Amps (value × 2.69 mA / 1000)
                currents_A = [load * 0.00269 for load in loads]
                
                # Move cursor up, clear, and print
                print("\033[2A\r\033[K", end="")
                print(f"\033[96m╔══ DIAGNOSTICS @ {diag_hz:.1f} Hz │ MOTOR CTRL @ {current_motor_hz:.0f} Hz ══╗\033[0m\n\r\033[K", end="")
                print(f"\033[96m║ M22: {currents_A[0]:5.2f}A │ M32: {currents_A[1]:5.2f}A │ M42: {currents_A[2]:5.2f}A │ M52: {currents_A[3]:5.2f}A ║\033[0m", flush=True)
                
                diag_loop_count = 0
                diag_last_time = now
                
            time.sleep(1 / DIAGNOSTICS_HZ)
            
    except Exception as e:
        print(f"\033[91m[DIAGNOSTICS] Error: {e}\033[0m")
    finally:
        print("\033[93mDIAGNOSTICS: stopped\033[0m")

# --------------------------- Main ---------------------------
if __name__ == "__main__":
    print("\033[96mRUN: threaded force→spin — Ctrl+C to stop\033[0m")
    tm = threading.Thread(target=motor_thread,  daemon=True)
    tk = threading.Thread(target=keyboard_thread, daemon=True)
    td = threading.Thread(target=diagnostics_thread, daemon=True)
    
    tm.start()
    tk.start()
    td.start()
    
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
        td.join(timeout=2.0)
        print("\033[92mDONE\033[0m")
