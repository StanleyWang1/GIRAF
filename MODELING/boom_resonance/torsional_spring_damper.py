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
# zeta = 0.0194
zeta = 0.08

k_theta = M * L**2 * w_n**2
c_theta = 2 * zeta * M * L**2 * w_n
v_input = 0.5
J = M * L**2
dt = 0.01

# --- State ---
theta = omega = alpha = theta_base = omega_base = 0.0
damping_enabled = False
noise_enabled = False

# --- History ---
history_len = int(5 / dt)
accel_times = deque([i * dt - 5.0 for i in range(history_len)], maxlen=history_len)
accel_mags = deque([0.0 for _ in range(history_len)], maxlen=history_len)
damping_states = deque([False for _ in range(history_len)], maxlen=history_len)
time_elapsed = 0.0

# --- Keyboard Control ---
input_velocity = 0.0
lock = threading.Lock()

def on_press(key):
    global input_velocity, damping_enabled, noise_enabled
    try:
        if key == keyboard.Key.up:
            with lock:
                input_velocity = v_input
        elif key == keyboard.Key.down:
            with lock:
                input_velocity = -v_input
        elif key.char == 'd':
            with lock:
                damping_enabled = not damping_enabled
        elif key.char == 'n':
            with lock:
                noise_enabled = not noise_enabled
    except AttributeError:
        pass

def on_release(key):
    global input_velocity
    if key in (keyboard.Key.up, keyboard.Key.down):
        with lock:
            input_velocity = 0.0

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

# --- Dynamics Update ---
def update_dynamics():
    global theta, omega, alpha, theta_base, omega_base, time_elapsed

    with lock:
        damping = damping_enabled
        noise = noise_enabled
        base_input = input_velocity

    if damping:
        omega_base = base_input - 0.05 * alpha
    else:
        omega_base = base_input

    theta_base += omega_base * dt

    tau = -k_theta * (theta - theta_base) - c_theta * (omega - omega_base)
    alpha = tau / J
    omega += alpha * dt
    theta += omega * dt

    a_perp = L * alpha
    if noise:
        a_perp += np.random.normal(0, 1)

    time_elapsed += dt
    accel_times.append(time_elapsed)
    accel_mags.append(a_perp)
    damping_states.append(damping)

    return theta, theta_base

# --- Plotting ---
fig, (ax_boom, ax_accel) = plt.subplots(1, 2, figsize=(10, 4))

line, = ax_boom.plot([], [], 'or-', lw=1.5, markersize=8, label="boom model")
base_line, = ax_boom.plot([], [], 'k--', lw=2, label="pitch motor")

text_damping = ax_boom.text(0.05, 0.95, '', transform=ax_boom.transAxes,
                             fontsize=10, verticalalignment='top')
text_noise = ax_boom.text(0.05, 0.88, '', transform=ax_boom.transAxes,
                          fontsize=10, verticalalignment='top')

ax_boom.set_xlim(-0.5, 2.0)
ax_boom.set_ylim(-0.5, 2.0)
ax_boom.set_aspect('equal')
ax_boom.set_title("Boom Coupled to Spring-Damped Base Angle")
ax_boom.legend(loc='lower right')

accel_plot, = ax_accel.plot([], [], 'gray', lw=1)
scatter_on = ax_accel.scatter([], [], color='green', s=8, label='Damping ON')
scatter_off = ax_accel.scatter([], [], color='red', s=8, label='Damping OFF')

ax_accel.set_xlim(-5, 0)
ax_accel.set_ylim(-3 * 9.81, 3 * 9.81)
ax_accel.set_title("IMU Acceleration ⊥ to Boom")
ax_accel.set_xlabel("Time (s)")
ax_accel.set_ylabel("a_perp (m/s²)")
ax_accel.legend(loc='upper right')

def init():
    line.set_data([], [])
    base_line.set_data([], [])
    accel_plot.set_data([], [])
    scatter_on.set_offsets(np.empty((0, 2)))
    scatter_off.set_offsets(np.empty((0, 2)))
    text_damping.set_text('')
    text_noise.set_text('')
    return line, base_line, accel_plot, scatter_on, scatter_off, text_damping, text_noise

def animate(frame):
    theta, theta_base = update_dynamics()

    x = [0, L * np.cos(theta)]
    y = [0, L * np.sin(theta)]
    line.set_data(x, y)

    with lock:
        if damping_enabled:
            line.set_color("green")
        else:
            line.set_color("red")

    xb = [0, L * np.cos(theta_base)]
    yb = [0, L * np.sin(theta_base)]
    base_line.set_data(xb, yb)

    times = np.array(accel_times) - accel_times[-1]
    accels = np.array(accel_mags)
    states = np.array(damping_states)

    accel_plot.set_data(times, accels)
    scatter_on.set_offsets(np.c_[times[states], accels[states]])
    scatter_off.set_offsets(np.c_[times[~states], accels[~states]])

    with lock:
        if damping_enabled:
            text_damping.set_text("DAMPING ON")
            text_damping.set_color("green")
        else:
            text_damping.set_text("DAMPING OFF")
            text_damping.set_color("red")

        if noise_enabled:
            text_noise.set_text("NOISE ON")
            text_noise.set_color("green")
        else:
            text_noise.set_text("NOISE OFF")
            text_noise.set_color("red")

    return line, base_line, accel_plot, scatter_on, scatter_off, text_damping, text_noise

ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=1000, interval=dt * 1000, blit=True
)

plt.tight_layout()
plt.show()
