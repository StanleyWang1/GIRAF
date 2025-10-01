# rpr_opt_pose_gui.py
# GUI to choose task point (x,y) and force (Fx,Fy), then solve/display
# the optimal RPR pose minimizing eta = F^T (J R J^T) F.
#
# Sliders:
#   x,y in [0,2] m, Fx,Fy in [0,5] N
# Actuator limits (fixed):
#   tau1_max = 20 N·m, F2_max = 50 N, tau3_max = 2 N·m
#
# Overlays the end-effector force vector (with visible magnitude)
# and the force ellipsoid for the optimal configuration.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import FancyArrow

# ---------- Constants ----------
L = 0.10            # [m] second link length
EPS = 1e-9
SCALE = 0.01        # [m/N] for drawing the ellipsoid only
FORCE_SCALE = 0.1  # [m/N] for drawing the force vector (bigger so magnitude is visible)

# Joint effort caps (fixed as requested)
TAU1_MAX = 20.0     # [N·m]
F2_MAX   = 50.0     # [N]
TAU3_MAX = 2.0      # [N·m]

# ---------- Jacobian, FK, metric ----------
def J_task(theta1, d2, theta3, L=L):
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s13, c13 = np.sin(theta1 + theta3), np.cos(theta1 + theta3)
    return np.array([
        [-d2*s1 - L*s13,  c1,   -L*s13],
        [ d2*c1 + L*c13,  s1,    L*c13]
    ])

def fk_points(theta1, d2, theta3, L=L):
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c13, s13 = np.cos(theta1 + theta3), np.sin(theta1 + theta3)
    P0 = np.array([0.0, 0.0])
    P1 = np.array([d2*c1, d2*s1])
    P2 = P1 + L*np.array([c13, s13])
    return P0, P1, P2

def eta_value(F, J, tau1_max=TAU1_MAX, F2_max=F2_MAX, tau3_max=TAU3_MAX):
    R = np.diag([1.0/max(tau1_max,EPS)**2,
                 1.0/max(F2_max,EPS)**2,
                 1.0/max(tau3_max,EPS)**2])
    A = J @ R @ J.T
    return float(F.T @ A @ F), A

def ellipse_from_A(A, n=240):
    """Given A (2x2, SPD), return ellipse samples for f^T A f = 1."""
    evals, evecs = np.linalg.eigh(A)
    evals = np.clip(evals, EPS, None)
    axes = 1.0 / np.sqrt(evals)  # semi-axes lengths
    order = np.argsort(axes)[::-1]
    axes = axes[order]
    V = evecs[:, order]
    t = np.linspace(0, 2*np.pi, n)
    circle = np.vstack((np.cos(t), np.sin(t)))
    f_ell = V @ (axes.reshape(2,1) * circle)  # 2xn
    return f_ell, axes, V

# ---------- Redundancy sweep (IK family over theta1) ----------
def solve_opt_pose(p_target, F_task,
                   tau1_max=TAU1_MAX, F2_max=F2_MAX, tau3_max=TAU3_MAX,
                   theta1_grid=np.linspace(0.0, np.pi, 901),
                   d2_bounds=(0.0, 3.0),
                   theta3_bounds=(-np.pi, np.pi)):
    r = np.asarray(p_target, dtype=float).reshape(2)
    F = np.asarray(F_task, dtype=float).reshape(2)
    r2 = float(r @ r)

    best = dict(eta=np.inf, q=None, J=None)

    for th1 in theta1_grid:
        u  = np.array([np.cos(th1), np.sin(th1)])
        ru = float(r @ u)
        disc = ru**2 - (r2 - L**2)
        if disc < 0:
            continue  # no solution for this orientation

        root = np.sqrt(max(disc, 0.0))
        for sign in (+1.0, -1.0):
            d2 = ru + sign*root
            if not (d2_bounds[0] - 1e-9 <= d2 <= d2_bounds[1] + 1e-9):
                continue
            w = r - d2*u
            if np.linalg.norm(w) < 1e-8:
                continue
            u_perp = np.array([-u[1], u[0]])
            th3 = np.arctan2(float(w @ u_perp), float(w @ u))
            # wrap to bounds
            if th3 < theta3_bounds[0]: th3 += 2*np.pi
            if th3 > theta3_bounds[1]: th3 -= 2*np.pi
            if not (theta3_bounds[0] - 1e-9 <= th3 <= theta3_bounds[1] + 1e-9):
                continue

            J = J_task(th1, d2, th3)
            eta, _ = eta_value(F, J, tau1_max, F2_max, tau3_max)
            if eta < best["eta"]:
                best.update(eta=eta, q=np.array([th1, d2, th3]), J=J)

    return best

# ---------- Matplotlib GUI ----------
plt.close('all')
fig, ax = plt.subplots(figsize=(7.6, 7.2))
plt.subplots_adjust(left=0.22, bottom=0.34, right=0.96, top=0.92)

# Artists
link1_line, = ax.plot([], [], linewidth=3, solid_capstyle='round', color='C0')
link2_line, = ax.plot([], [], linewidth=3, solid_capstyle='round', color='C0')
base_dot,   = ax.plot([], [], 'o', markersize=7, color='k')
j2_dot,     = ax.plot([], [], 'o', markersize=6, color='C0')
ee_dot,     = ax.plot([], [], 'o', markersize=6, color='C3')

ellipse_line, = ax.plot([], [], linewidth=2, color='C2', alpha=0.9)
force_arrow = None

# Text info
info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left')
status_text = ax.text(0.98, 0.98, "", transform=ax.transAxes, va='top', ha='right',
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray"))

# Axes cosmetics
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.4)
ax.set_title("RPR Optimal Pose for Task Force (with Force Ellipsoid)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

# Slider axes
ax_x  = plt.axes([0.22, 0.26, 0.70, 0.03])
ax_y  = plt.axes([0.22, 0.21, 0.70, 0.03])
ax_Fx = plt.axes([0.22, 0.16, 0.70, 0.03])
ax_Fy = plt.axes([0.22, 0.11, 0.70, 0.03])

# Sliders: x,y in [0,2] m; Fx,Fy in [0,5] N
s_x  = Slider(ax_x,  r'$x$ [m]',  0.0, 2.0, valinit=1.0)
s_y  = Slider(ax_y,  r'$y$ [m]',  0.0, 2.0, valinit=1.0)
s_Fx = Slider(ax_Fx, r'$F_x$ [N]', 0.0, 5.0, valinit=0.5)
s_Fy = Slider(ax_Fy, r'$F_y$ [N]', 0.0, 5.0, valinit=0.5)

def draw_force_arrow(px, py, Fx, Fy):
    """Draw force arrow with magnitude-proportional head size."""
    global force_arrow
    if force_arrow is not None:
        force_arrow.remove()
    dx, dy = FORCE_SCALE * Fx, FORCE_SCALE * Fy
    mag = max(np.hypot(dx, dy), 1e-9)
    head_len = max(0.06, 0.20 * mag)  # head ~20% of shaft length (min size guard)
    head_w   = 0.5 * head_len
    force_arrow = FancyArrow(px, py, dx, dy,
                             width=0.0,
                             length_includes_head=True,
                             head_width=head_w,
                             head_length=head_len,
                             color='C3', alpha=0.9)
    ax.add_patch(force_arrow)

def update(_=None):
    x = s_x.val
    y = s_y.val
    Fx = s_Fx.val
    Fy = s_Fy.val

    p_star = np.array([x, y])
    F_task = np.array([Fx, Fy])

    sol = solve_opt_pose(p_star, F_task)
    if sol["q"] is None:
        # Clear drawing for unreachable; still show target point & force
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        base_dot.set_data([0.0], [0.0])
        j2_dot.set_data([], [])
        ee_dot.set_data([x], [y])
        ellipse_line.set_data([], [])

        draw_force_arrow(x, y, Fx, Fy)

        info_text.set_text(
            f"Target p* = ({x:.3f}, {y:.3f}) m\n"
            f"Force F = ({Fx:.3f}, {Fy:.3f}) N\n"
            f"Force vector scale = {FORCE_SCALE:.2f} m/N\n"
            f"Status: Unreachable for L={L:.2f} m with d2∈[0,3] m"
        )
        status_text.set_text("No feasible pose")
        status_text.set_bbox(dict(boxstyle="round,pad=0.2", fc="#ffd9d9", ec="gray"))
        fig.canvas.draw_idle()
        return

    th1, d2, th3 = sol["q"]
    J = sol["J"]
    eta, A = eta_value(F_task, J)
    inside = eta < 1.0

    # FK points
    P0, P1, P2 = fk_points(th1, d2, th3)

    # Draw links/joints
    link1_line.set_data([P0[0], P1[0]], [P0[1], P1[1]])
    link2_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])
    base_dot.set_data([0.0], [0.0])
    j2_dot.set_data([P1[0]], [P1[1]])
    ee_dot.set_data([P2[0]], [P2[1]])

    # Force arrow at end-effector
    draw_force_arrow(P2[0], P2[1], Fx, Fy)

    # Force ellipsoid (f^T A f = 1) scaled and centered at EE
    f_ell, axes, V = ellipse_from_A(A)
    ell_geom = P2.reshape(2,1) + SCALE * f_ell
    ellipse_line.set_data(ell_geom[0, :], ell_geom[1, :])

    # Text
    info_text.set_text(
        "Optimal pose:\n"
        f"  θ1 = {th1:.3f} rad, d2 = {d2:.3f} m, θ3 = {th3:.3f} rad\n"
        f"EE = ({P2[0]:.3f}, {P2[1]:.3f}) m ≈ target ({x:.3f}, {y:.3f}) m\n"
        f"η = {eta:.5f}   (τ1={TAU1_MAX:.1f} N·m, F2={F2_MAX:.1f} N, τ3={TAU3_MAX:.1f} N·m)\n"
        f"Ellipse semi-axes (N): a_major={axes[0]:.2f}, a_minor={axes[1]:.2f}\n"
        f"Force vector scale = {FORCE_SCALE:.2f} m/N"
    )
    status_text.set_text("INSIDE (stable)" if inside else "OUTSIDE (violates)")
    status_text.set_bbox(dict(boxstyle="round,pad=0.2",
                              fc=("#e7ffe7" if inside else "#ffd9d9"),
                              ec="gray"))

    fig.canvas.draw_idle()

# Initial draw + callbacks
update()
s_x.on_changed(update)
s_y.on_changed(update)
s_Fx.on_changed(update)
s_Fy.on_changed(update)

plt.show()
