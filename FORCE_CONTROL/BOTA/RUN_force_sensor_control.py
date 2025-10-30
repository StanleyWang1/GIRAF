import time
import sys
import signal
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

from force_sensor_driver import ForceSensorDriver
from dynamixel_driver import (
    dynamixel_connect, dynamixel_drive, dynamixel_disconnect,
)
from control_table import MOTOR11_HOME, MOTOR12_HOME, MOTOR13_HOME, MOTOR12_MIN, MOTOR12_MAX, TORQUE_ENABLE, PRESENT_POSITION

# ===================== User settings =====================
FZ_THRESHOLD_N = -1.0   # spin when Fz < -5 N
STEP_TICKS     = 10    # how many ticks to add per motor update while spinning
SENSOR_TIMEOUT = 0.25   # seconds to wait for a new sensor sample in sensor thread
CTRL_HZ        = 100.0  # motor update rate
PRINT_HZ       = 2.0    # status print rate
TARE_ON_START  = True
PLOT_WINDOW    = 5.0    # seconds of data to show
PLOT_UPDATE_MS = 50     # ms between plot updates
# ========================================================

# Shared state
running = True
running_lock = threading.Lock()

shared = {"Fz": None}
shared_lock = threading.Lock()

boom_pos = MOTOR13_HOME
boom_pos_lock = threading.Lock()

start_event = threading.Event()   # <-- new event for starting motor loop

def handle_sigint(_s, _f):
    global running
    with running_lock:
        running = False
signal.signal(signal.SIGINT, handle_sigint) # Ctrl+C handler

# ------------------------- Sensor thread -------------------------
def sensor_thread():
    global running
    fs = ForceSensorDriver(tare_on_start=TARE_ON_START)
    fs.start()
    print("\033[92mSENSOR: started\033[0m")
    try:
        while True:
            with running_lock:
                if not running:
                    break
            fr = fs.read_blocking(timeout_s=SENSOR_TIMEOUT)
            if fr is None:
                continue
            Fz = float(fr.force[2])
            with shared_lock:
                shared["Fz"] = Fz
    except Exception as e:
        print(f"\033[91m[FATAL][SENSOR] {e}\033[0m", file=sys.stderr)
    finally:
        try:
            fs.stop()
        except Exception:
            pass
        print("\033[93mSENSOR: stopped\033[0m")

# ------------------------- Motor thread -------------------------
def motor_thread():
    global running, boom_pos
    
    PITCH_MOTOR = 12 
    PITCH_TICKS = int((MOTOR12_MIN + MOTOR12_MAX)/2)

    # Controller Parameters
    ts = 1/500.0
    Mv = 1 # virtual mass
    Bv = 500.0 # virtual damping
    Kf = 1000.0 

    # Connect to motors and home
    dmx_controller, dmx_GSW = dynamixel_connect()
    print("\033[93mCONTROLLER: Motors Connected!\033[0m")
    time.sleep(0.5)
    dynamixel_drive(dmx_controller, dmx_GSW, [PITCH_TICKS, boom_pos])
    time.sleep(0.5)

    # Wait for user to start control loop (replace input())
    print("\033[93mCONTROLLER: Press ENTER to start admittance controller!\033[0m")
    start_event.wait()   # <-- wait until keyboard thread signals start

    # Performance benchmarking (loop Hz)
    loop_count = 0
    last_time = time.perf_counter()

    print_interval = 1.0 / PRINT_HZ if PRINT_HZ > 0 else 1.0

    Fz_des = -3.0
    vel = 0.0
    try:
        while running:
            with shared_lock:
                Fz = shared["Fz"]
            F_error = Fz_des - Fz if Fz is not None else 0.0
            F_error = max(-0.5, min(0.5, F_error))
            
            new_vel = (ts/Mv)*(Kf*F_error) + (1 - (ts*Bv)/Mv)*vel
            vel = new_vel
            
            PITCH_TICKS += int(vel)
            PITCH_TICKS = max(MOTOR12_MIN, min(MOTOR12_MAX, PITCH_TICKS))
            
            with boom_pos_lock:
                BOOM_TICKS = boom_pos
                
            dynamixel_drive(dmx_controller, dmx_GSW, [PITCH_TICKS, BOOM_TICKS])
  
            time.sleep(0.001)

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

# ------------------------- Plot thread -------------------------
def plot_thread():
    global running
    
    # Initialize data structures
    t_data = deque(maxlen=int(PLOT_WINDOW * CTRL_HZ))
    fz_data = deque(maxlen=int(PLOT_WINDOW * CTRL_HZ))
    t_start = time.perf_counter()
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], 'b-', label='Fz')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N]')
    ax.set_title('Live Force Reading')
    ax.grid(True)
    ax.set_ylim(-10, 10)  # Adjust force range as needed
    ax.legend()
    
    def update(frame):
        with shared_lock:
            fz = shared["Fz"]
        
        if fz is not None:
            t = time.perf_counter() - t_start
            t_data.append(t)
            fz_data.append(fz)
            
            # Update line data
            line.set_data(t_data, fz_data)
            
            # Adjust x-axis limits to scroll
            if t_data:
                ax.set_xlim(max(0, t - PLOT_WINDOW), t)
        
        return line,
    
    ani = FuncAnimation(fig, update, interval=PLOT_UPDATE_MS, blit=True)
    
    try:
        plt.show()
    except Exception as e:
        print(f"\033[91m[PLOT] Error: {e}\033[0m")
    finally:
        plt.close()
        print("\033[93mPLOT: stopped\033[0m")

# ------------------------- Keyboard thread -------------------------
import sys
import tty
import termios
import select

def keyboard_thread():
    global running, boom_pos
    
    TICK_STEP = 50
    LOOP_HZ = 100
    
    print("\033[93mKEYBOARD: W/S to move boom, Q to quit\033[0m")
    
    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # Set terminal to raw mode
        tty.setraw(sys.stdin.fileno())
        
        while running:
            # Check if input is ready
            if select.select([sys.stdin], [], [], 0)[0] != []:
                key = sys.stdin.read(1).lower()
                
                delta = 0
                if key == '\r' or key == '\n':
                    start_event.set()   # signal motor_thread to start
                elif key == 'w':
                    delta = TICK_STEP
                elif key == 's':
                    delta = -TICK_STEP
                elif key == 'q':
                    with running_lock:
                        running = False
                        
                if delta != 0:
                    with boom_pos_lock:
                        boom_pos += delta
                        
            time.sleep(1/LOOP_HZ)
            
    except Exception as e:
        print(f"\033[91m[KEYBOARD] Error: {e}\033[0m")
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\033[93mKEYBOARD: stopped\033[0m")

# --------------------------- Main ---------------------------
if __name__ == "__main__":
    print("\033[96mRUN: threaded force→spin — Ctrl+C to stop\033[0m")
    ts = threading.Thread(target=sensor_thread, daemon=True)
    tm = threading.Thread(target=motor_thread,  daemon=True)
    tp = threading.Thread(target=plot_thread,   daemon=True)
    tk = threading.Thread(target=keyboard_thread, daemon=True)
    
    ts.start()
    tm.start()
    tp.start()
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
        ts.join(timeout=2.0)
        tm.join(timeout=2.0)
        tp.join(timeout=2.0)
        tk.join(timeout=2.0)
        print("\033[92mDONE\033[0m")
