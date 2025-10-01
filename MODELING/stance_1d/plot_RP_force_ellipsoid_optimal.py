# plot_RP_force_ellipsoid_xbase.py
# Adds Fx, Fy sliders, simplified HUD, orange F_task arrow, and stronger force scaling.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches

# ---------- Visualization & numerics ----------
SCALE = 0.015   # meters per Newton (increased from 0.01 -> 0.015)
EPS   = 1e-9
SNAP_TOL = 0.02  # [m] snap when |Δ - Δ_opt| < SNAP_TOL

# ---------- Initial parameters ----------
x_base_0  = 0.5
x_e_0     = 2.0
y_e_0     = 1.0
tau1max_0 = 20.0         # [N·m]
F2max_0   = 50.0         # [N]
Fx_0      = 10.0         # [N]
Fy_0      =  0.0         # [N]

# ---------- Figure and axes ----------
plt.close('all')
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(left=0.22, bottom=0.46, right=0.95, top=0.93)

# Arm line and joints
arm_line, = ax.plot([], [], linewidth=3)
base_dot, = ax.plot([], [], 'o', markersize=8)
ee_dot,   = ax.plot([], [], 'o', markersize=6)

# Ellipse line
ellipse_line, = ax.plot([], [], linewidth=2)

# Guide for optimal base (vertical line at x_opt_base)
opt_line = ax.axvline(0.0, linestyle='--', linewidth=1, alpha=0.6)
opt_line.set_visible(True)

# Orange F_task arrow (created on first update)
task_arrow = None

# Simplified HUD
info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top')

# Axes settings
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
ax.set_title("RP Arm: Force Ellipse & η(Δ) — Snap-to-Optimum")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

# ---------- Slider axes ----------
ax_xbase = plt.axes([0.22, 0.37, 0.70, 0.03])
ax_xe    = plt.axes([0.22, 0.32, 0.70, 0.03])
ax_ye    = plt.axes([0.22, 0.27, 0.70, 0.03])
ax_tau1  = plt.axes([0.22, 0.22, 0.70, 0.03])
ax_F2    = plt.axes([0.22, 0.17, 0.70, 0.03])
ax_Fx    = plt.axes([0.22, 0.12, 0.70, 0.03])
ax_Fy    = plt.axes([0.22, 0.07, 0.70, 0.03])

s_xbase = Slider(ax_xbase, r'$x_{\mathrm{base}}$ [m]',   0.0, 3.0, valinit=x_base_0)
s_xe    = Slider(ax_xe,    r'$x_e$ [m]',                 0.0, 3.0, valinit=x_e_0)
s_ye    = Slider(ax_ye,    r'$y_e$ [m]',                 0.0, 3.0, valinit=y_e_0)
s_tau1  = Slider(ax_tau1,  r'$\tau_{1,\max}$ [N·m]',     0.0, 20.0, valinit=tau1max_0)
s_F2    = Slider(ax_F2,    r'$F_{2,\max}$ [N]',          0.0, 100.0, valinit=F2max_0)
s_Fx    = Slider(ax_Fx,    r'$F_x$ [N]',                -100.0, 100.0, valinit=Fx_0)
s_Fy    = Slider(ax_Fy,    r'$F_y$ [N]',                -100.0, 100.0, valinit=Fy_0)

# ---------- Helpers ----------
def compute_arm(x_base, x_e, y_e):
    """Return base->ee coordinates for the RP arm in 2D."""
    return np.array([[x_base, x_e], [0.0, y_e]])

def compute_force_ellipse_points(p_base, p_ee, tau1_max, F2_max, scale=SCALE, n=200):
    """Force manipulability ellipse centered at p_ee."""
    d_vec = p_ee - p_base
    d3 = np.linalg.norm(d_vec)
    if d3 < EPS:
        u = np.array([1.0, 0.0])
        u_perp = np.array([0.0, 1.0])
    else:
        u = d_vec / d3
        u_perp = np.array([-u[1], u[0]])

    a = F2_max
    b = tau1_max / max(d3, EPS)

    t = np.linspace(0, 2*np.pi, n)
    ellipse = p_ee.reshape(2, 1) + scale * (
        a * np.cos(t) * u.reshape(2, 1) + b * np.sin(t) * u_perp.reshape(2, 1)
    )
    return ellipse[0, :], ellipse[1, :], a, b, d3

def eta_of_delta(Δ, Fx, Fy, y, F2max, tau1max):
    d2 = Δ*Δ + y*y
    term1 = (Fx*Δ + Fy*y)**2 / (max(F2max, EPS)**2 * max(d2, EPS))
    term2 = (-Fx*y + Fy*Δ)**2 / (max(tau1max, EPS)**2)
    return term1 + term2

def optimal_delta(Fx, Fy, y, F2max, tau1max):
    a, b = F2max, tau1max
    cands = []
    if abs(Fy) > 1e-12:
        cands.append(Fx*y / Fy)
    # Quartic: Fy*a^2*(Δ^2+y^2)^2 - b^2*(Fx*y*Δ + Fy*y^2) = 0
    c4 = Fy * a**2
    c3 = 0.0
    c2 = 2 * Fy * a**2 * y**2
    c1 = -b**2 * Fx * y
    c0 = Fy * a**2 * y**4 - b**2 * Fy * y**2
    if abs(c4) > 1e-18:
        roots = np.roots([c4, c3, c2, c1, c0])
        for r in roots:
            if np.isreal(r):
                cands.append(float(np.real(r)))
    else:
        cands.append(0.0)
    # dedupe
    uniq = []
    for v in cands:
        if not any(abs(v - u) < 1e-9 for u in uniq):
            uniq.append(v)
    etas = [eta_of_delta(Δ, Fx, Fy, y, a, b) for Δ in uniq]
    i = int(np.argmin(etas))
    return uniq[i], list(zip(uniq, etas))

# ---------- Main update ----------
_updating = False

def update(_=None):
    global _updating, task_arrow
    if _updating:
        return
    _updating = True
    try:
        # Read sliders
        x_base = s_xbase.val
        x_e    = s_xe.val
        y_e    = s_ye.val
        tau1_m = s_tau1.val
        F2_m   = s_F2.val
        Fx     = s_Fx.val
        Fy     = s_Fy.val

        # Kinematics
        P = compute_arm(x_base, x_e, y_e)
        arm_line.set_data(P[0, :], P[1, :])
        base_dot.set_data([x_base], [0.0])
        ee_dot.set_data([x_e], [y_e])

        # Ellipse at EE
        ex, ey, a, b, d3 = compute_force_ellipse_points(P[:, 0], P[:, 1], tau1_m, F2_m)
        ellipse_line.set_data(ex, ey)

        # Optimal Δ and η
        Δ = x_e - x_base
        Δ_opt, _ = optimal_delta(Fx, Fy, y_e, F2_m, tau1_m)
        x_base_opt = x_e - Δ_opt
        opt_line.set_xdata([x_base_opt, x_base_opt])

        eta_now = eta_of_delta(Δ, Fx, Fy, y_e, F2_m, tau1_m)

        # Light snap-to-optimum
        if abs(Δ - Δ_opt) < SNAP_TOL:
            s_xbase.set_val(x_base_opt)  # retriggers update()

        # Orange F_task arrow at EE (scaled)
        if task_arrow is not None:
            task_arrow.remove()
            task_arrow = None
        dx = SCALE * Fx
        dy = SCALE * Fy
        task_arrow = patches.FancyArrow(
            x_e, y_e, dx, dy, width=0.01, length_includes_head=True,
            head_width=0.06, head_length=0.08, color='orange', alpha=0.9
        )
        ax.add_patch(task_arrow)

        # Simplified HUD
        info_text.set_text(
            r"$\eta$ = {:.5f}   Δ = {:.3f} m   d = {:.3f} m".format(eta_now, Δ, d3)
            + "\n" + r"$x_{{\mathrm{{base,opt}}}}$ = {:.3f} m".format(x_base_opt)
        )

        fig.canvas.draw_idle()
    finally:
        _updating = False

# Initial draw
update()

# Connect sliders
for s in (s_xbase, s_xe, s_ye, s_tau1, s_F2, s_Fx, s_Fy):
    s.on_changed(update)

plt.show()
