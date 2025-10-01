import numpy as np
from scipy.optimize import minimize
import time

def solve_static_feasibility_2d_scipy(load, giraf_keypoints, m=50, mu=0.7):
    g = 9.81
    px, py = load
    ax, ay = giraf_keypoints['A']
    bx, by = giraf_keypoints['B']
    cx, cy = giraf_keypoints['C']

    # Decision variable order: [f1x, f1y, f2x, f2y]
    def objective(f):
        return 0.0  # Feasibility only

    # Equality constraints (force and torque balance)
    def eq1(f): return f[0] + f[2] + px
    def eq2(f): return f[1] + f[3] - m*g + py
    def eq3(f): return ax*f[1] - ay*f[0] + bx*f[3] - by*f[2] + cx*py - cy*px

    eq_constraints = [{'type': 'eq', 'fun': eq1},
                      {'type': 'eq', 'fun': eq2},
                      {'type': 'eq', 'fun': eq3}]

    # Inequality constraints (nonnegativity and friction cones)
    ineq_constraints = [
        {'type': 'ineq', 'fun': lambda f: f[1]},                      # f1y >= 0
        {'type': 'ineq', 'fun': lambda f: f[3]},                      # f2y >= 0
        {'type': 'ineq', 'fun': lambda f: mu * f[1] - abs(f[0])},     # |f1x| ≤ μ f1y
        {'type': 'ineq', 'fun': lambda f: mu * f[3] - abs(f[2])}      # |f2x| ≤ μ f2y
    ]

    # Combine constraints
    constraints = eq_constraints + ineq_constraints

    # Initial guess
    f0 = np.zeros(4)

    res = minimize(objective, f0, constraints=constraints, method='SLSQP', options={'disp': False})

    if res.success:
        return tuple(res.x)
    else:
        return None, None, None, None
    
def main():
    load = [15, 0] # [px, py]
    giraf_keypoints = {'A' : [-0.25, -0.25], 'B': [0.25, -0.25], 'C': [1, 1]}

    start_time = time.time()
    result = solve_static_feasibility_2d_scipy(load, giraf_keypoints)
    elapsed = time.time() - start_time

    print("Result:", result)
    print(f"SciPy solve time: {elapsed * 1000:.3f} ms")

if __name__ == "__main__":
    main()
