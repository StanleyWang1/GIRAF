# rp_force_ellipsoid_gui.py
# Interactive 2D GUI for an RP (revolute-prismatic) planar arm
# - Sliders for theta_1 in [0, pi] and d_3 in [0, 3] meters
# - Sliders for tau_1,max in [0, 10] N·m and F_1,max in [0, 100] N
# - Overlays the FORCE manipulability ellipsoid at the end-effector
#
# The ellipse axes are aligned with:
#   u = [cos(theta_1), sin(theta_1)]        (along boom)
#   u_perp = [-sin(theta_1), cos(theta_1)]  (lateral)
# Semi-axes lengths (in Newtons):
#   a = F_1,max                (along-boom)
#   b = tau_1,max / max(d_3, eps)  (lateral)
#
# For visualization on the geometry plot (meters), we scale forces by SCALE [m/N].

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------- Visualization scaling ----------
SCALE = 0.01  # meters per Newton (purely for drawing the ellipse on the geometry plot)
EPS = 1e-6    # to avoid divide-by-zero when d3 ~ 0

# ---------- Initial parameters ----------
theta1_0 = np.pi/4     # rad
d3_0     = 1.0         # m
tau1max_0 = 20.0        # N·m
F1max_0   = 50.0       # N

# ---------- Figure and axes ----------
plt.close('all')
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(left=0.2, bottom=0.32, right=0.95, top=0.93)

# Arm line and joints
arm_line, = ax.plot([], [], linewidth=3)
base_dot, = ax.plot([], [], 'o', markersize=8)
ee_dot,   = ax.plot([], [], 'o', markersize=6)

# Ellipse line
ellipse_line, = ax.plot([], [], linewidth=2)

# Text annotation for current lateral force limit and scale
info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top')

# Set equal aspect, grid, and initial limits
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
ax.set_title("RP Arm with Force Manipulability Ellipse (at End-Effector)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_xlim(0, 3)      # <-- Fixed x-axis limits
ax.set_ylim(0, 3)      # <-- Fixed y-axis limits

# ---------- Slider axes ----------
ax_theta  = plt.axes([0.20, 0.24, 0.70, 0.03])
ax_d3     = plt.axes([0.20, 0.19, 0.70, 0.03])
ax_tau1   = plt.axes([0.20, 0.14, 0.70, 0.03])
ax_F1     = plt.axes([0.20, 0.09, 0.70, 0.03])

s_theta = Slider(ax_theta, r'$\theta_1$ [rad]', 0.0, np.pi/2, valinit=theta1_0)  # <-- limit is now
s_d3    = Slider(ax_d3,    r'$d_3$ [m]',        0.0, 3.0,   valinit=d3_0)
s_tau1  = Slider(ax_tau1,  r'$\tau_{1,\max}$ [N·m]', 0.0, 20.0, valinit=tau1max_0)
s_F1    = Slider(ax_F1,    r'$F_{1,\max}$ [N]',      0.0, 100.0, valinit=F1max_0)

def compute_arm(theta1, d3):
    """Return base->ee coordinates for the RP arm in 2D."""
    x = d3 * np.cos(theta1)
    y = d3 * np.sin(theta1)
    return np.array([[0.0, x], [0.0, y]])  # two points: base and end-effector

def compute_force_ellipse_points(p, theta1, d3, tau1_max, F1_max, scale=SCALE, n=200):
    """Compute 2D points for the force manipulability ellipse centered at p.

    Ellipse in force space (principal axes):
      along u:     a = F1_max
      along u_perp: b = tau1_max / max(d3, EPS)

    We visualize it on the geometry plot by scaling forces [N] to [m] via 'scale'.
    """
    u = np.array([np.cos(theta1), np.sin(theta1)])       # along the boom
    u_perp = np.array([-np.sin(theta1), np.cos(theta1)]) # lateral

    a = F1_max
    b = tau1_max / max(d3, EPS)

    t = np.linspace(0, 2*np.pi, n)
    # Ellipse param in force space: f = a*cos t * u + b*sin t * u_perp
    # Visualize in meters: p + scale * f
    ellipse = p.reshape(2, 1) + scale * (a * np.cos(t) * u.reshape(2, 1) + b * np.sin(t) * u_perp.reshape(2, 1))
    return ellipse[0, :], ellipse[1, :], a, b

def autoset_limits(P, ex, ey):
    """Auto-set limits to fit both arm and ellipse nicely."""
    all_x = np.concatenate([P[0, :], ex])
    all_y = np.concatenate([P[1, :], ey])
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    dx = max(0.1, (max_x - min_x))
    dy = max(0.1, (max_y - min_y))
    pad = 0.1 * max(dx, dy)
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)

def update(_=None):
    theta1 = s_theta.val
    d3     = s_d3.val
    tau1_m = s_tau1.val
    F1_m   = s_F1.val

    # Arm coordinates
    P = compute_arm(theta1, d3)
    arm_line.set_data(P[0, :], P[1, :])
    base_dot.set_data([0.0], [0.0])
    ee_dot.set_data([P[0, 1]], [P[1, 1]])

    # Ellipse centered at end-effector
    ex, ey, a, b = compute_force_ellipse_points(P[:, 1], theta1, d3, tau1_m, F1_m)

    ellipse_line.set_data(ex, ey)

    # Info text: lateral limit and scale
    info_text.set_text(
        "Along-boom limit: {:.1f} N\nLateral limit: {:.2f} N (={:.1f} N·m / max(d3,ε))\nScale: {:.3f} m per N".format(
            a, b, tau1_m, SCALE
        )
    )

    fig.canvas.draw_idle()

# Initial draw
update()

# Connect sliders
s_theta.on_changed(update)
s_d3.on_changed(update)
s_tau1.on_changed(update)
s_F1.on_changed(update)

plt.show()
