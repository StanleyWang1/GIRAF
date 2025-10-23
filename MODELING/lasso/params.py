import numpy as np

m = 5              # control dim
n = 7              # state  dim
N = 50             # time steps
dt = 0.1           # [s]

# Control weight R (positive definite or semidefinite)
R = np.diag([10.0, 10.0, 0.1, 0.1, 0.1])
# R = np.diag([0.1, 0.1, 1.0, 0.1, 1.0])

# Known target planar velocity sequence v_t \in R^2 for t=0..N-1
v_t = np.tile(np.array([0, -0.25]), (N, 1))  # shape (N,2)

# Target radius
R_t = 0.25  # [m], example

# Initial state x0 = [x_t, y_t, x_v, y_v, th_v, th1, d2]
x0 = np.array([1.8, 1.8, 0.0, 0.0, 0.0, 0.0, 0.5])
