import numpy as np

from params import N, n, m, dt, R, v_t, R_t, x0
import optimization as opt
from utils_io import save_solution

# 8_driver.py
X, U = opt.make_decision_variables(N, n, m)

g_dyn = opt.build_dynamics_constraints(X, U, v_t, dt)
g_term = opt.terminal_touch_tangent_with_side(X[:, -1], R_t, side=+1, cross_margin=1e-8)

J = opt.control_effort_objective(U, R, dt)

# State Bounds
xmin = -np.inf * np.ones(n)
xmax =  np.inf * np.ones(n)
xmin[6] = 0.5      # d2 >= 0.5
xmax[6] = 2.0      # d2 <= 2.0

nlp, bounds = opt.pack_nlp(X, U, J, g_dyn, g_term, x0,
                       u_bounds=(np.array([-0.6, -0.6, -1.0, -1.0, -0.5]),
                                 np.array([+0.6, +0.6, +1.0, +1.0, +0.5])),
                          x_bounds=(xmin, xmax))

sol = opt.solve_with_ipopt(nlp, bounds)

z_opt = np.array(sol['x']).squeeze()
# Unpack back to arrays
nX = n*(N+1)
X_opt = z_opt[:nX].reshape((n, N+1), order='F')
U_opt = z_opt[nX:].reshape((m, N), order='F')

meta = {
    "N": int(N),
    "n": int(n),
    "m": int(m),
    "dt": float(dt),
    "R_t": float(R_t),
}
save_solution("./MODELING/lasso/solutions/soln_down", X_opt, U_opt, meta)