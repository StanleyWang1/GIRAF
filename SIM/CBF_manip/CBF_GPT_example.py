import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


TELEOP_KEYS = {'w', 'a', 's', 'd', 'q', 'e', 'space', ' '}


def clear_matplotlib_keybindings(keys):
    normalized_keys = {key.lower() for key in keys}
    for rc_key in plt.rcParams:
        if rc_key.startswith('keymap.'):
            plt.rcParams[rc_key] = [
                key
                for key in plt.rcParams[rc_key]
                if key.lower() not in normalized_keys
            ]


clear_matplotlib_keybindings(TELEOP_KEYS)


# =========================
# User-tunable parameters
# =========================
DT = 1.0 / 100.0
L2 = 0.5
D_MIN = 0.0
D_MAX = 3.0

# Teleop speeds
LIN_SPEED = 1.5       # m/s
ANG_SPEED = 3.0       # rad/s

# RMRC settings
DAMPING = 0.08        # damped pseudoinverse lambda
RMRC_QDOT_LIMITS = np.array([5.0, 3.0, 10.0])  # [rad/s, m/s, rad/s]

# CBF-QP settings
CBF_QDOT_LIMITS = np.array([5.0, 3.0, 10.0])
BARRIER_TYPE = 'sigma_min'  # options: 'sigma_min', 'yoshikawa'
SIGMA_MIN_EPSILON = 0.10    # margin on sigma_min(J)
YOSHIKAWA_EPSILON = 0.20    # margin on sqrt(det(J J^T))
GAMMA = 3.0           # CBF gain
RHO = 5e3             # slack penalty
REG_LAMBDA = 1e-3     # qdot regularization
FD_EPS = 1e-5         # finite-difference step for grad h


def barrier_epsilon(barrier_type: str) -> float:
    if barrier_type == 'sigma_min':
        return SIGMA_MIN_EPSILON
    if barrier_type == 'yoshikawa':
        return YOSHIKAWA_EPSILON
    raise ValueError(f"Unsupported BARRIER_TYPE: {barrier_type!r}")

# Joint angle plotting / limits
THETA1_MIN = -np.inf
THETA1_MAX = np.inf
THETA3_MIN = -np.inf
THETA3_MAX = np.inf

# Initial conditions [theta1, d, theta3]
Q0 = np.array([0.3, 1.2, -0.2], dtype=float)


@dataclass
class RPRArm2D:
    l2: float = L2
    d_min: float = D_MIN
    d_max: float = D_MAX

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        theta1, d, theta3 = q
        phi = theta1 + theta3
        p1 = np.array([d * math.cos(theta1), d * math.sin(theta1)])
        p2 = p1 + np.array([self.l2 * math.cos(phi), self.l2 * math.sin(phi)])
        return p1, p2, phi

    def pose(self, q: np.ndarray) -> np.ndarray:
        _, p2, phi = self.fk(q)
        return np.array([p2[0], p2[1], phi], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        theta1, d, theta3 = q
        phi = theta1 + theta3

        s1 = math.sin(theta1)
        c1 = math.cos(theta1)
        sphi = math.sin(phi)
        cphi = math.cos(phi)

        return np.array(
            [
                [-d * s1 - self.l2 * sphi, c1, -self.l2 * sphi],
                [ d * c1 + self.l2 * cphi, s1,  self.l2 * cphi],
                [1.0,                      0.0, 1.0],
            ],
            dtype=float,
        )

    def sigma_min(self, q: np.ndarray) -> float:
        return float(np.min(np.linalg.svd(self.jacobian(q), compute_uv=False)))

    def yoshikawa(self, q: np.ndarray) -> float:
        J = self.jacobian(q)
        return math.sqrt(max(0.0, float(np.linalg.det(J @ J.T))))

    def barrier_value(self, q: np.ndarray, barrier_type: str = BARRIER_TYPE) -> float:
        if barrier_type == 'sigma_min':
            return self.sigma_min(q)
        if barrier_type == 'yoshikawa':
            return self.yoshikawa(q)
        raise ValueError(f"Unsupported BARRIER_TYPE: {barrier_type!r}")

    def h(self, q: np.ndarray, barrier_type: str = BARRIER_TYPE) -> float:
        return self.barrier_value(q, barrier_type) - barrier_epsilon(barrier_type)

    def grad_h_fd(self, q: np.ndarray, barrier_type: str = BARRIER_TYPE, fd_eps: float = FD_EPS) -> np.ndarray:
        grad = np.zeros(3, dtype=float)
        for i in range(3):
            dq = np.zeros(3, dtype=float)
            dq[i] = fd_eps
            hp = self.h(q + dq, barrier_type=barrier_type)
            hm = self.h(q - dq, barrier_type=barrier_type)
            grad[i] = (hp - hm) / (2.0 * fd_eps)
        return grad

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        q[1] = np.clip(q[1], self.d_min, self.d_max)
        return q


class CBFQPController:
    def __init__(self, dt: float):
        self.dt = dt
        self.qdot = cp.Variable(3)
        self.slack = cp.Variable(nonneg=True)

        self.Jp = cp.Parameter((3, 3))
        self.vdes_p = cp.Parameter(3)
        self.grad_h_p = cp.Parameter(3)
        self.h_p = cp.Parameter(nonneg=False)
        self.qdot_lb_p = cp.Parameter(3)
        self.qdot_ub_p = cp.Parameter(3)

        objective = cp.Minimize(
            0.5 * cp.sum_squares(self.Jp @ self.qdot - self.vdes_p)
            + 0.5 * REG_LAMBDA * cp.sum_squares(self.qdot)
            + 0.5 * RHO * cp.square(self.slack)
        )

        constraints = [
            self.grad_h_p @ self.qdot >= -GAMMA * self.h_p - self.slack,
            self.qdot >= self.qdot_lb_p,
            self.qdot <= self.qdot_ub_p,
        ]

        self.prob = cp.Problem(objective, constraints)
        self.last_qdot = np.zeros(3, dtype=float)
        self.last_slack = 0.0
        self.last_solve_ms = 0.0

    def solve(self, q: np.ndarray, arm: RPRArm2D, v_des: np.ndarray) -> Tuple[np.ndarray, float]:
        J = arm.jacobian(q)
        h_val = arm.h(q, BARRIER_TYPE)
        grad_h = arm.grad_h_fd(q, BARRIER_TYPE)

        pos_lb = np.array([THETA1_MIN, arm.d_min, THETA3_MIN], dtype=float)
        pos_ub = np.array([THETA1_MAX, arm.d_max, THETA3_MAX], dtype=float)

        qdot_lb_pos = (pos_lb - q) / self.dt
        qdot_ub_pos = (pos_ub - q) / self.dt

        qdot_lb = np.maximum(-CBF_QDOT_LIMITS, qdot_lb_pos)
        qdot_ub = np.minimum(CBF_QDOT_LIMITS, qdot_ub_pos)

        self.Jp.value = J
        self.vdes_p.value = v_des
        self.grad_h_p.value = grad_h
        self.h_p.value = h_val
        self.qdot_lb_p.value = qdot_lb
        self.qdot_ub_p.value = qdot_ub

        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, polish=False)
        except Exception:
            self.prob.solve(warm_start=True, verbose=False)
        self.last_solve_ms = (time.perf_counter() - t0) * 1000.0

        if self.qdot.value is None:
            return self.last_qdot.copy(), float(self.last_slack)

        self.last_qdot = np.asarray(self.qdot.value, dtype=float).reshape(3)
        self.last_slack = float(self.slack.value)
        return self.last_qdot.copy(), self.last_slack


def damped_pseudoinverse(J: np.ndarray, damping: float) -> np.ndarray:
    m, n = J.shape
    if m <= n:
        return J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(m))
    return np.linalg.inv(J.T @ J + (damping ** 2) * np.eye(n)) @ J.T


def rmrc_step(q: np.ndarray, arm: RPRArm2D, v_des: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    J = arm.jacobian(q)
    J_dag = damped_pseudoinverse(J, DAMPING)
    qdot = J_dag @ v_des
    qdot = np.clip(qdot, -RMRC_QDOT_LIMITS, RMRC_QDOT_LIMITS)
    q_next = arm.clip_q(q + dt * qdot)
    return q_next, qdot


def cbf_step(q: np.ndarray, arm: RPRArm2D, controller: CBFQPController, v_des: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, float]:
    qdot, slack = controller.solve(q, arm, v_des)
    q_next = arm.clip_q(q + dt * qdot)
    return q_next, qdot, slack


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class KeyTeleop:
    def __init__(self):
        self.pressed: Dict[str, bool] = {}

    def _normalize_key(self, key: str) -> str:
        if key == ' ':
            return 'space'
        return key.lower()

    def on_press(self, event):
        if event.key is not None:
            self.pressed[self._normalize_key(event.key)] = True

    def on_release(self, event):
        if event.key is not None:
            self.pressed[self._normalize_key(event.key)] = False

    def active(self, key: str) -> bool:
        return self.pressed.get(key, False)

    def clear(self, key: str):
        self.pressed[key] = False

    def command(self) -> np.ndarray:
        vx = 0.0
        vy = 0.0
        wz = 0.0

        if self.active('a'):
            vx -= LIN_SPEED
        if self.active('d'):
            vx += LIN_SPEED
        if self.active('s'):
            vy -= LIN_SPEED
        if self.active('w'):
            vy += LIN_SPEED
        if self.active('q'):
            wz += ANG_SPEED
        if self.active('e'):
            wz -= ANG_SPEED

        return np.array([vx, vy, wz], dtype=float)


def make_axes_pretty(ax, title: str):
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-3.7, 3.7)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')


class Simulator:
    def __init__(self):
        self.arm = RPRArm2D()
        self.cbf_controller = CBFQPController(DT)
        self.teleop = KeyTeleop()

        self.q_rmrc = Q0.copy()
        self.q_cbf = Q0.copy()
        self.last_qdot_rmrc = np.zeros(3)
        self.last_qdot_cbf = np.zeros(3)
        self.last_slack = 0.0

        self.xs_l, self.ys_l = [], []
        self.xs_r, self.ys_r = [], []

        self.fig, (self.ax_l, self.ax_r) = plt.subplots(1, 2, figsize=(13, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.teleop.on_press)
        self.fig.canvas.mpl_connect('key_release_event', self.teleop.on_release)

        make_axes_pretty(self.ax_l, 'RMRC / Pseudoinverse')
        make_axes_pretty(self.ax_r, 'CBF-QP')

        self.line_l_1, = self.ax_l.plot([], [], '-o', lw=3, ms=7)
        self.line_l_2, = self.ax_l.plot([], [], '-o', lw=3, ms=7)
        self.line_r_1, = self.ax_r.plot([], [], '-o', lw=3, ms=7)
        self.line_r_2, = self.ax_r.plot([], [], '-o', lw=3, ms=7)

        self.ee_traj_l, = self.ax_l.plot([], [], '--', lw=1.2, alpha=0.8)
        self.ee_traj_r, = self.ax_r.plot([], [], '--', lw=1.2, alpha=0.8)

        self.txt_l = self.ax_l.text(0.02, 0.98, '', transform=self.ax_l.transAxes, va='top', family='monospace')
        self.txt_r = self.ax_r.text(0.02, 0.98, '', transform=self.ax_r.transAxes, va='top', family='monospace')

        help_text = (
            'Click the figure once to give it keyboard focus.  '
            'WASD = Cartesian translation, Q/E = rotate CCW/CW about Z, space = reset'
        )
        self.fig.text(0.5, 0.02, help_text, ha='center', family='monospace')

        self.anim = FuncAnimation(self.fig, self.update, interval=int(DT * 1000), blit=False, cache_frame_data=False)

    def reset(self):
        self.q_rmrc = Q0.copy()
        self.q_cbf = Q0.copy()
        self.last_qdot_rmrc[:] = 0.0
        self.last_qdot_cbf[:] = 0.0
        self.last_slack = 0.0
        self.xs_l.clear(); self.ys_l.clear(); self.xs_r.clear(); self.ys_r.clear()

    def update(self, _frame_idx):
        if self.teleop.active('space'):
            self.reset()
            self.teleop.clear('space')

        v_des = self.teleop.command()

        self.q_rmrc, self.last_qdot_rmrc = rmrc_step(self.q_rmrc, self.arm, v_des, DT)
        self.q_cbf, self.last_qdot_cbf, self.last_slack = cbf_step(self.q_cbf, self.arm, self.cbf_controller, v_des, DT)

        p1_l, p2_l, phi_l = self.arm.fk(self.q_rmrc)
        p1_r, p2_r, phi_r = self.arm.fk(self.q_cbf)

        self.xs_l.append(p2_l[0]); self.ys_l.append(p2_l[1])
        self.xs_r.append(p2_r[0]); self.ys_r.append(p2_r[1])
        self.xs_l[:] = self.xs_l[-400:]; self.ys_l[:] = self.ys_l[-400:]
        self.xs_r[:] = self.xs_r[-400:]; self.ys_r[:] = self.ys_r[-400:]

        self.line_l_1.set_data([0.0, p1_l[0]], [0.0, p1_l[1]])
        self.line_l_2.set_data([p1_l[0], p2_l[0]], [p1_l[1], p2_l[1]])
        self.line_r_1.set_data([0.0, p1_r[0]], [0.0, p1_r[1]])
        self.line_r_2.set_data([p1_r[0], p2_r[0]], [p1_r[1], p2_r[1]])

        self.ee_traj_l.set_data(self.xs_l, self.ys_l)
        self.ee_traj_r.set_data(self.xs_r, self.ys_r)

        sigma_rmrc = self.arm.sigma_min(self.q_rmrc)
        sigma_cbf = self.arm.sigma_min(self.q_cbf)
        yoshikawa_rmrc = self.arm.yoshikawa(self.q_rmrc)
        yoshikawa_cbf = self.arm.yoshikawa(self.q_cbf)
        barrier_cbf = self.arm.barrier_value(self.q_cbf, BARRIER_TYPE)
        h_cbf = self.arm.h(self.q_cbf, BARRIER_TYPE)
        epsilon = barrier_epsilon(BARRIER_TYPE)

        self.txt_l.set_text(
            f"q = [{self.q_rmrc[0]: .2f}, {self.q_rmrc[1]: .2f}, {self.q_rmrc[2]: .2f}]\n"
            f"phi = {wrap_angle(phi_l): .2f} rad\n"
            f"sigma_min(J) = {sigma_rmrc: .3f} | yoshikawa = {yoshikawa_rmrc: .3f}\n"
            f"qdot = [{self.last_qdot_rmrc[0]: .2f}, {self.last_qdot_rmrc[1]: .2f}, {self.last_qdot_rmrc[2]: .2f}]\n"
            f"v_des = [{v_des[0]: .2f}, {v_des[1]: .2f}, {v_des[2]: .2f}]"
        )
        self.txt_r.set_text(
            f"q = [{self.q_cbf[0]: .2f}, {self.q_cbf[1]: .2f}, {self.q_cbf[2]: .2f}]\n"
            f"phi = {wrap_angle(phi_r): .2f} rad\n"
            f"sigma_min(J) = {sigma_cbf: .3f} | yoshikawa = {yoshikawa_cbf: .3f}\n"
            f"h = {BARRIER_TYPE} - eps = {barrier_cbf: .3f} - {epsilon: .3f} = {h_cbf: .3f}\n"
            f"qdot = [{self.last_qdot_cbf[0]: .2f}, {self.last_qdot_cbf[1]: .2f}, {self.last_qdot_cbf[2]: .2f}]\n"
            f"slack = {self.last_slack: .5f} | solve = {self.cbf_controller.last_solve_ms: .2f} ms"
        )

        return (
            self.line_l_1, self.line_l_2, self.line_r_1, self.line_r_2,
            self.ee_traj_l, self.ee_traj_r, self.txt_l, self.txt_r,
        )

    def show(self):
        plt.show()


def main():
    sim = Simulator()
    sim.show()


if __name__ == '__main__':
    main()
