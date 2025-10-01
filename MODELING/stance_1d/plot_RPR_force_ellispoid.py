# rpr_force_ellipsoid_gui.py
# Interactive 2D GUI for an RPR (revolute–prismatic–revolute) planar arm
# - Sliders for theta_1 in [0, pi], d_2 in [0, 3] meters, theta_3 in [-pi, pi]
# - Sliders for tau_1,max, F_2,max, tau_3,max
# - Overlays the FORCE manipulability ellipsoid at the end-effector from effort limits:
#     tau = J^T f,    tau^T R tau <= 1  =>  f^T (J R J^T) f <= 1
#   where R = diag(1/tau1_max^2, 1/F2_max^2, 1/tau3_max^2).
#
# Visualization: forces [N] scaled to meters by SCALE for plotting on geometry axes.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------- Constants ----------
L = 0.10         # [m] second link length (constant)
SCALE = 0.01     # [m per N] purely for drawing the ellipse
EPS = 1e-9

# ---------- Initial parameters ----------
theta1_0   = np.deg2rad(45.0)
d2_0       = 1.0
theta3_0   = np.deg2rad(20.0)
tau1max_0  = 20.0     # [N·m] <-- updated default
F2max_0    = 50.0     # [N]   <-- updated default
tau3max_0  = 2.0      # [N·m] <-- updated default

# ---------- Figure ----------
plt.close('all')
fig, ax = plt.subplots(figsize=(7.2, 7.2))
plt.subplots_adjust(left=0.22, bottom=0.36, right=0.95, top=0.93)

# Plot artists
link1_line, = ax.plot([], [], linewidth=3, solid_capstyle='round')
link2_line, = ax.plot([], [], linewidth=3, solid_capstyle='round')
base_dot,   = ax.plot([], [], 'o', markersize=7)
j2_dot,     = ax.plot([], [], 'o', markersize=6)
ee_dot,     = ax.plot([], [], 'o', markersize=6)

ellipse_line, = ax.plot([], [], linewidth=2)

# Info text
info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top')

# Axes cosmetics
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.4)
ax.set_title("RPR Arm with Force Manipulability Ellipse (at End-Effector)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_xlim(0, 2)      # <-- fixed x-axis limits
ax.set_ylim(0, 2)      # <-- fixed y-axis limits

# ---------- Slider axes ----------
ax_theta1 = plt.axes([0.22, 0.28, 0.70, 0.03])
ax_d2     = plt.axes([0.22, 0.23, 0.70, 0.03])
ax_theta3 = plt.axes([0.22, 0.18, 0.70, 0.03])

ax_tau1   = plt.axes([0.22, 0.12, 0.70, 0.03])
ax_F2     = plt.axes([0.22, 0.07, 0.70, 0.03])
ax_tau3   = plt.axes([0.22, 0.02, 0.70, 0.03])

s_theta1 = Slider(ax_theta1, r'$\theta_1$ [rad]', 0.0, np.pi,   valinit=theta1_0)
s_d2     = Slider(ax_d2,     r'$d_2$ [m]',        0.0, 3.0,     valinit=d2_0)
s_theta3 = Slider(ax_theta3, r'$\theta_3$ [rad]', -np.pi, np.pi, valinit=theta3_0)

s_tau1 = Slider(ax_tau1, r'$\tau_{1,\max}$ [N·m]', 0.0, 30.0, valinit=tau1max_0)
s_F2   = Slider(ax_F2,   r'$F_{2,\max}$ [N]',      0.0, 150.0, valinit=F2max_0)
s_tau3 = Slider(ax_tau3, r'$\tau_{3,\max}$ [N·m]', 0.0, 30.0,  valinit=tau3max_0)

# ---------- Kinematics ----------
def fk_points(theta1, d2, theta3, L=L):
    """
    Returns (P0, P1, P2) where:
      P0: base (0,0)
      P1: after prismatic along theta1: [d2*cos(theta1), d2*sin(theta1)]
      P2: end-effector: P1 + L*[cos(theta1+theta3), sin(theta1+theta3)]
    """
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c13, s13 = np.cos(theta1 + theta3), np.sin(theta1 + theta3)
    P0 = np.array([0.0, 0.0])
    P1 = np.array([d2 * c1, d2 * s1])
    P2 = P1 + L * np.array([c13, s13])
    return P0, P1, P2

def J_task(theta1, d2, theta3, L=L):
    """
    Planar (x,y) Jacobian for RPR, columns correspond to [theta1, d2, theta3].
    Matches user's expression:
      [x] = [ d2 cos θ1 + L cos(θ1+θ3) ]
      [y]   [ d2 sin θ1 + L sin(θ1+θ3) ]
    J = [[-d2 sin θ1 - L sin(θ1+θ3),   cos θ1,     -L sin(θ1+θ3)],
         [ d2 cos θ1 + L cos(θ1+θ3),   sin θ1,      L cos(θ1+θ3)]]
    """
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s13, c13 = np.sin(theta1 + theta3), np.cos(theta1 + theta3)
    J = np.array([
        [-d2 * s1 - L * s13,  c1,  -L * s13],
        [ d2 * c1 + L * c13,  s1,   L * c13]
    ])
    return J

# ---------- Ellipse from effort limits ----------
def ellipse_from_effort_limits(J, tau1_max, F2_max, tau3_max, n=200):
    """
    Given J (2x3) and joint effort caps, build A = J R J^T,
    where R = diag(1/tau1_max^2, 1/F2_max^2, 1/tau3_max^2),
    then parameterize the ellipse { f | f^T A f <= 1 }.
    Returns ellipse points in FORCE space (2xn), the semi-axis lengths (a1>=a2),
    and the principal directions (columns of V).
    """
    t1 = max(tau1_max, EPS)
    f2 = max(F2_max,  EPS)
    t3 = max(tau3_max, EPS)
    R = np.diag([1.0/(t1**2), 1.0/(f2**2), 1.0/(t3**2)])
    A = J @ R @ J.T  # 2x2, symmetric PSD

    # Eigen-decompose A
    evals, evecs = np.linalg.eigh(A)
    # Guard small/negative due to numeric issues
    evals = np.clip(evals, EPS, None)

    # Semi-axes in force space are 1/sqrt(lambda)
    axes = 1.0 / np.sqrt(evals)  # [a_small, a_large] if evals sorted ascending
    # Sort descending by axis length for reporting
    order = np.argsort(axes)[::-1]
    axes = axes[order]
    V = evecs[:, order]

    # Parametric ellipse in force space: f = V diag(axes) [cos t; sin t]
    t = np.linspace(0, 2*np.pi, n)
    circle = np.vstack((np.cos(t), np.sin(t)))          # 2xn
    f_ell = V @ (axes.reshape(2, 1) * circle)           # 2xn

    return f_ell, axes, V, A

def autoset_limits(points):
    """Expand axes to fit all provided 2D points columns (2xN)."""
    min_x = np.min(points[0, :])
    max_x = np.max(points[0, :])
    min_y = np.min(points[1, :])
    max_y = np.max(points[1, :])
    dx = max(0.10, max_x - min_x)
    dy = max(0.10, max_y - min_y)
    pad = 0.15 * max(dx, dy)
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)

# ---------- Update ----------
def update(_=None):
    th1  = s_theta1.val
    d2   = s_d2.val
    th3  = s_theta3.val
    t1m  = s_tau1.val
    f2m  = s_F2.val
    t3m  = s_tau3.val

    # FK points
    P0, P1, P2 = fk_points(th1, d2, th3, L=L)

    # Draw links and joints
    link1_line.set_data([P0[0], P1[0]], [P0[1], P1[1]])
    link2_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])
    base_dot.set_data([P0[0]], [P0[1]])
    j2_dot.set_data([P1[0]], [P1[1]])
    ee_dot.set_data([P2[0]], [P2[1]])

    # Jacobian and ellipsoid in FORCE space
    J = J_task(th1, d2, th3, L=L)
    f_ell, axes, V, A = ellipse_from_effort_limits(J, t1m, f2m, t3m, n=300)

    # Plot ellipse centered at EE, scaled to meters for visualization
    ell_geom = P2.reshape(2, 1) + SCALE * f_ell  # 2xN
    ellipse_line.set_data(ell_geom[0, :], ell_geom[1, :])

    # Remove autoset_limits, axes are now fixed

    # Info text: principal force limits (N) and scale
    info_text.set_text(
        "Principal force limits at EE (N):\n"
        f"  a_major = {axes[0]:.2f} N, dir = [{V[0,0]:+.2f}, {V[1,0]:+.2f}]\n"
        f"  a_minor = {axes[1]:.2f} N, dir = [{V[0,1]:+.2f}, {V[1,1]:+.2f}]\n"
        f"Scale = {SCALE:.3f} m/N"
    )

    fig.canvas.draw_idle()

# Initial draw and bindings
update()
s_theta1.on_changed(update)
s_d2.on_changed(update)
s_theta3.on_changed(update)
s_tau1.on_changed(update)
s_F2.on_changed(update)
s_tau3.on_changed(update)

plt.show()
