import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pynput import keyboard
import threading
from collections import deque

# --- System Parameters ---
L = 1.5
M = 0.5
w_n = 12.753
zeta = 0.0194

# Observer matrices (endpoint perpendicular dynamics)
A = np.array([[0, 1], [-w_n**2, -2 * zeta * w_n]])
beta = -L
B = np.array([[0], [beta]])
C = np.array([[-w_n**2, -2 * zeta * w_n]])
D = np.array([[beta]])
L_gain = np.array([[-0.01668588], [0.9887092]])
Kd = 10

# Boom dynamics
k_theta = M * L**2 * w_n**2
c_theta = 2 * zeta * M * L**2 * w_n
J = M * L**2
dt = 0.01

# State
theta = omega = alpha = theta_base = omega_base = 0.0
x_hat = np.zeros((2, 1))  # [displacement; velocity] estimate

# Teleop control
input_velocity = 0.0
lock = threading.Lock()

def on_press(key):
    global input_velocity
    if key == keyboard.Key.up:
        with lock:
            input_velocity = 1.0
    elif key == keyboard.Key.down:
        with lock:
            input_velocity = -1.0

def on_release(key):
    global input_velocity
    if key in (keyboard.Key.up, keyboard.Key.down):
        with lock:
            input_velocity = 0.0

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

# Acceleration buffer (5 sec window)
history_len = int(5 / dt)
accel_times = deque([i * dt - 5.0 for i in range(history_len)], maxlen=history_len)
accel_mags = deque([0.0 for _ in range(history_len)], maxlen=history_len)
time_elapsed = 0.0

# --- Dynamics ---
def update_dynamics():
    global theta, omega, alpha, theta_base, omega_base, time_elapsed, x_hat

    with lock:
        u_task = input_velocity

    # Reference tracking for base (pure reference)
    omega_base = u_task
    theta_base += omega_base * dt

    # Control law for the plant (feedback)
    u = u_task - Kd * x_hat[1, 0]

    tau = -k_theta * (theta - theta_base) - c_theta * (omega - omega_base) + u
    alpha = tau / J
    omega += alpha * dt
    theta += omega * dt

    # IMU-like acceleration (perpendicular)
    a_perp = L * alpha
    y = np.array([[a_perp]])
    u_vec = np.array([[u]])
    dx_hat = A @ x_hat + B @ u_vec + L_gain @ (y - C @ x_hat - D @ u_vec)
    x_hat += dx_hat * dt

    time_elapsed += dt
    accel_times.append(time_elapsed)
    accel_mags.append(a_perp)

    return theta, theta_base



# --- Plotting ---
fig, (ax_boom, ax_accel) = plt.subplots(1, 2, figsize=(10, 4))

line, = ax_boom.plot([], [], 'o-', lw=3)
base_line, = ax_boom.plot([], [], 'k--', lw=1)
ax_boom.set_xlim(-0.5, 2.0)
ax_boom.set_ylim(-0.5, 2.0)
ax_boom.set_aspect('equal')
ax_boom.set_title("Boom Coupled to Spring-Damped Base Angle")

accel_plot, = ax_accel.plot([], [], lw=2)
ax_accel.set_xlim(-5, 0)
ax_accel.set_ylim(-2 * 9.81, 2 * 9.81)
ax_accel.set_title("IMU Acceleration ⊥ to Boom")
ax_accel.set_xlabel("Time (s)")
ax_accel.set_ylabel("a_perp (m/s²)")

def init():
    line.set_data([], [])
    base_line.set_data([], [])
    accel_plot.set_data([], [])
    return line, base_line, accel_plot

def animate(frame):
    theta, theta_base = update_dynamics()
    x = [0, L * np.cos(theta)]
    y = [0, L * np.sin(theta)]
    line.set_data(x, y)

    xb = [0, L * np.cos(theta_base)]
    yb = [0, L * np.sin(theta_base)]
    base_line.set_data(xb, yb)

    t = np.array(accel_times) - accel_times[-1]
    accel_plot.set_data(t, accel_mags)
    return line, base_line, accel_plot

ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=1000, interval=dt * 1000, blit=True
)

plt.tight_layout()
plt.show()
