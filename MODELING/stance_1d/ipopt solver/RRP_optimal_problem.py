from ipopt_problem import Program
import jax.numpy as jnp
import numpy as np

class RRP_optimal_pose(Program):
    def __init__(self, task):
        super().__init__(
            name="RRP Optimal Pose Problem",
            num_variables=3,
            num_equality_constraints=1,
            num_inequality_constraints=0
        )
        self.x_task = task.x
        self.y_task = task.y
        self.Fx = task.Fx
        self.Fy = task.Fy

    def initial_guess(self):
        return np.array([0.0, 0.0, 0.0])

    def objective(self, x):
        L = 0.1
        Λ = jnp.diag(jnp.array([20.0 ** 2, 50.0 ** 2, 2.0 ** 2])) # actuator limits squared
        J = jnp.array([
            [-x[1]*jnp.sin(x[0]) - L*jnp.sin(x[0] + x[2]), jnp.cos(x[0]), -L*jnp.sin(x[0] + x[2])],
            [ x[1]*jnp.cos(x[0]) + L*jnp.cos(x[0] + x[2]), jnp.sin(x[0]),  L*jnp.cos(x[0] + x[2])]
        ])
        # Symmetric ellipsoid matrix
        A = J @ jnp.linalg.inv(Λ) @ J.T
        
        # Decompose elements of A (2 x 2)
        a = A[0,0]
        b = A[0,1] 
        c = A[1,0]
        d = A[1,1]
        theta_ellipse = 1/2 * jnp.arctan2(2*b, a - c)

        return x[0]**2 + x[1]**2

    def equality_constraints(self, x):
        # Example: circle constraint x1 + x2 - 1 = 0
        return jnp.array([x[0] + x[1] - 1])

    def inequality_constraints(self, x):
        # No inequalities
        return jnp.zeros((0,)), jnp.zeros((0,)), jnp.zeros((0,))