import numpy as np
import casadi as ca

from params import m, n, N, dt, R, v_t, R_t, x0

# 2_decision_variables.py
def make_decision_variables(N, n, m):
    # States X[0..N], Controls U[0..N-1]
    X = ca.SX.sym('X', n, N+1)     # each column is x_t
    U = ca.SX.sym('U', m, N)       # each column is u_t
    return X, U

# 3_dynamics_constraints.py
def build_dynamics_constraints(X, U, v_seq, dt):
    g_list = []
    # Forward Euler: x_{t+1} = x_t + [v_t; u_t] dt
    for t in range(U.shape[1]):  # t = 0..N-1
        v_t = ca.vertcat(*v_seq[t].tolist())       # (2,)
        aug = ca.vertcat(v_t, U[:, t])             # (7,) = [v_t; u_t]
        g_list.append(X[:, t+1] - (X[:, t] + dt*aug))
    g = ca.vertcat(*g_list)  # stacked equality residuals
    return g

# 4_terminal_constraint.py
# 4_terminal_constraint.py
# 4_terminal_constraint.py
def terminal_touch_tangent_with_side(xN, R_t, side: int = +1, cross_margin: float = 1e-8):
    """
    Enforce at t = N:
      (1) ||tip - target|| = R_t                (touch)
      (2) (tip - target) · (tip - base) = 0     (tangent)
      (3) side * cross2D(r, b) >= cross_margin  (choose CW/CCW side)
          side = +1 → boom is CCW from radius (external +90°)
          side = -1 → boom is CW  from radius (external -90°)
    """
    x_t, y_t = xN[0], xN[1]
    x_v, y_v = xN[2], xN[3]
    th_v, th1, d2 = xN[4], xN[5], xN[6]
    phi = th_v + th1

    tip_x = x_v + d2 * ca.cos(phi)
    tip_y = y_v + d2 * ca.sin(phi)

    r = ca.vertcat(tip_x - x_t, tip_y - y_t)      # target -> tip
    b = ca.vertcat(tip_x - x_v, tip_y - y_v)      # base -> tip

    # (1) touch
    c_touch = ca.sqrt(ca.sumsqr(r)) - R_t
    # (2) tangent
    c_perp  = ca.dot(r, b)
    # (3) choose side via signed cross product
    cross2  = r[0]*b[1] - r[1]*b[0]               # scalar
    c_side  = side * cross2                       # >= margin

    # Return stacked constraints; caller must set bounds (==0, ==0, >=margin)
    return ca.vertcat(c_touch, c_perp, c_side)



# 5_objective.py
def control_effort_objective(U, R, dt):
    Rcs = ca.DM(R)  # constant
    J = 0
    for t in range(U.shape[1]):
        J += ca.mtimes([U[:, t].T, Rcs, U[:, t]]) * dt
    return ca.simplify(J)

# 6_pack_nlp.py
def pack_nlp(X, U, J, g_dyn, g_term, x0, u_bounds=None, x_bounds=None, cross_margin=1e-8):
    # Vectorize decision variables
    z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    g = ca.vertcat(g_dyn, g_term)  # all equalities stacked

    # ---- Bounds on constraints (equalities -> 0) ----
    ng = int(g.size1())
    lbg = np.zeros(ng)
    ubg = np.zeros(ng)

    # --- make the *side* constraint an inequality: >= cross_margin
    ng_dyn = int(g_dyn.size1())
    idx_touch   = ng_dyn + 0   # == 0
    idx_perp    = ng_dyn + 1   # == 0
    idx_side    = ng_dyn + 2   # >= cross_margin

    # defaults already 0==g for all; now relax the side row:
    lbg[idx_side] = cross_margin
    ubg[idx_side] = np.inf

    # ---- Bounds on variables ----
    nX = X.size1()*X.size2()
    nU = U.size1()*U.size2()

    # Default: free variables
    lbz = -np.inf*np.ones(nX + nU)
    ubz =  np.inf*np.ones(nX + nU)

    # Fix initial state X[:,0] = x0
    for i in range(X.size1()):
        idx = i + 0*X.size1()  # position in column-major reshape
        lbz[idx] = x0[i]
        ubz[idx] = x0[i]

    # Optional control bounds
    if u_bounds is not None:
        umin, umax = u_bounds  # each shape (m,)
        for t in range(U.size2()):
            for i in range(U.size1()):
                base = nX + i + t*U.size1()
                lbz[base] = umin[i]
                ubz[base] = umax[i]

    # Optional state bounds (apply to all time steps)
    if x_bounds is not None:
        xmin, xmax = x_bounds  # each shape (n,)
        for t in range(X.size2()):
            for i in range(X.size1()):
                idx = i + t*X.size1()
                lbz[idx] = max(lbz[idx], xmin[i])
                ubz[idx] = min(ubz[idx], xmax[i])

    # Initial guess: zeros for U, forward simulate X
    X0 = np.tile(x0.reshape(-1,1), (1, X.size2()))
    U0 = np.zeros((U.size1(), U.size2()))
    # crude rollout with U=0
    for t in range(U.size2()):
        if t == 0: 
            continue
        vt_prev = v_t[t-1]
        X0[:, t] = X0[:, t-1] + dt*np.hstack([vt_prev, np.zeros(U.size1())])
    z0 = np.concatenate([X0.reshape(-1, order='F'),
                         U0.reshape(-1, order='F')])

    nlp = {'x': z, 'f': J, 'g': g}
    bounds = dict(lbx=lbz, ubx=ubz, lbg=lbg, ubg=ubg, x0=z0)
    return nlp, bounds

# 7_solve_ipopt.py
def solve_with_ipopt(nlp, bounds, ipopt_print=False):
    opts = {
        'ipopt': {
            'tol': 1e-6,
            'constr_viol_tol': 1e-6,
            'acceptable_tol': 1e-4,
            'max_iter': 2000,
            'linear_solver': 'mumps',  # good default
        },
        'print_time': False,
        'verbose': ipopt_print
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    sol = solver(x0=bounds['x0'],
                 lbx=bounds['lbx'], ubx=bounds['ubx'],
                 lbg=bounds['lbg'], ubg=bounds['ubg'])
    return sol
