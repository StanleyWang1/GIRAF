# plot_RP_force_ellipsoid_xbase.py
# Interactive 2D GUI for an RP planar arm with movable base and endpoint
# - Sliders for x_base (0 to 3), x_e (0 to 3), y_e (0 to 3)
# - Sliders for tau_1,max in [0, 20] N·m and F_1,max in [0, 100] N
# - Overlays the FORCE manipulability ellipsoid at the end-effector

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------- Visualization scaling ----------
SCALE = 0.01  # meters per Newton (for drawing the ellipse)
EPS = 1e-6    # to avoid divide-by-zero

# ---------- Initial parameters ----------
x_base_0 = 0.5
x_e_0    = 2.0
y_e_0    = 1.0
tau1max_0 = 20.0        # N·m
F1max_0   = 50.0        # N

# ---------- Figure and axes ----------
plt.close('all')
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(left=0.2, bottom=0.37, right=0.95, top=0.93)

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
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

# ---------- Slider axes ----------
ax_xbase = plt.axes([0.20, 0.29, 0.70, 0.03])
ax_xe    = plt.axes([0.20, 0.24, 0.70, 0.03])
ax_ye    = plt.axes([0.20, 0.19, 0.70, 0.03])
ax_tau1  = plt.axes([0.20, 0.14, 0.70, 0.03])
ax_F1    = plt.axes([0.20, 0.09, 0.70, 0.03])

s_xbase = Slider(ax_xbase, r'$x_{base}$ [m]', 0.0, 3.0, valinit=x_base_0)
s_xe    = Slider(ax_xe,    r'$x_e$ [m]',      0.0, 3.0, valinit=x_e_0)
s_ye    = Slider(ax_ye,    r'$y_e$ [m]',      0.0, 3.0, valinit=y_e_0)
s_tau1  = Slider(ax_tau1,  r'$\tau_{1,\max}$ [N·m]', 0.0, 20.0, valinit=tau1max_0)
s_F1    = Slider(ax_F1,    r'$F_{1,\max}$ [N]',      0.0, 100.0, valinit=F1max_0)

def compute_arm(x_base, x_e, y_e):
    """Return base->ee coordinates for the RP arm in 2D."""
    return np.array([[x_base, x_e], [0.0, y_e]])

def compute_force_ellipse_points(p_base, p_ee, tau1_max, F1_max, scale=SCALE, n=200):
    """Compute 2D points for the force manipulability ellipse centered at p_ee.

    Ellipse axes:
      along boom:     a = F1_max
      lateral:        b = tau1_max / max(d3, EPS)
    where d3 is the distance from base to end-effector.
    """
    d_vec = p_ee - p_base
    d3 = np.linalg.norm(d_vec)
    if d3 < EPS:
        u = np.array([1.0, 0.0])
        u_perp = np.array([0.0, 1.0])
    else:
        u = d_vec / d3
        u_perp = np.array([-u[1], u[0]])

    a = F1_max
    b = tau1_max / max(d3, EPS)

    t = np.linspace(0, 2*np.pi, n)
    ellipse = p_ee.reshape(2, 1) + scale * (a * np.cos(t) * u.reshape(2, 1) + b * np.sin(t) * u_perp.reshape(2, 1))
    return ellipse[0, :], ellipse[1, :], a, b, d3

def update(_=None):
    x_base = s_xbase.val
    x_e    = s_xe.val
    y_e    = s_ye.val
    tau1_m = s_tau1.val
    F1_m   = s_F1.val

    # Arm coordinates
    P = compute_arm(x_base, x_e, y_e)
    arm_line.set_data(P[0, :], P[1, :])
    base_dot.set_data([x_base], [0.0])
    ee_dot.set_data([x_e], [y_e])

    # Ellipse centered at end-effector
    ex, ey, a, b, d3 = compute_force_ellipse_points(P[:, 0], P[:, 1], tau1_m, F1_m)

    ellipse_line.set_data(ex, ey)

    # Info text: lateral limit and scale
    info_text.set_text(
        "Along-boom limit: {:.1f} N\nLateral limit: {:.2f} N (={:.1f} N·m / max(d3,ε))\nScale: {:.3f} m per N\nArm length: {:.2f} m".format(
            a, b, tau1_m, SCALE, d3
        )
    )

    fig.canvas.draw_idle()

# Initial draw
update()

# Connect sliders
s_xbase.on_changed(update)
s_xe.on_changed(update)
s_ye.on_changed(update)
s_tau1.on_changed(update)
s_F1.on_changed(update)

plt.show()