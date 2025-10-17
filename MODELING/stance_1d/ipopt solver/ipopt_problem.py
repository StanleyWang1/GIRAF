"""Problem formulation for IPOPT/IPYOPT

Originally from Thomas Lew (thomas.lew@stanford.edu)

Usage: Construct a class inheriting from Program with at least the following abstract methods implemented:
- initial_guess
- objective
- equality_constraints
- inequality_constraints

Then, pass it into the Solver (implemented in another file)
"""

# TODO update documentation

import jax.numpy as jnp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)


class Program:
    def __init__(
        self,
        name: str,
        num_variables: int,
        num_equality_constraints: int,
        num_inequality_constraints: int,
        verbose: bool = False,
    ):
        if verbose:
            print("Initializing Program with")
            print("> name          =", name)
            print("> num_variables =", num_variables)
        self._name = name
        self._num_variables = num_variables
        self._num_equality_constraints = num_equality_constraints
        self._num_inequality_constraints = num_inequality_constraints

    @property
    def num_variables(self) -> int:
        return self._num_variables

    @property
    def num_constraints(self) -> int:
        num = self._num_equality_constraints + self._num_inequality_constraints
        return num

    @property
    def num_equality_constraints(self) -> int:
        return self._num_equality_constraints

    @property
    def num_inequality_constraints(self) -> int:
        return self._num_inequality_constraints

    @property
    def name(self) -> str:
        return self._name

    def initial_guess(self) -> np.array:
        raise NotImplementedError

    def solution(self) -> np.array:
        # unknown since stochastically perturbed program
        raise NotImplementedError

    def objective(self, x: jnp.array) -> float:
        """Returns objective f(x) to minimize."""
        raise NotImplementedError

    def equality_constraints(self, x: jnp.array) -> jnp.array:
        """Returns equality constraints.

        Returns h(x) corresponding to the
        equality constraints h(x) = 0.

        Args:
            x: optimization variable,
                (_num_variables, ) array

        Returns:
            h_value: value of h(x),
                (_num_equality_constraints, ) array
        """
        raise NotImplementedError

    def inequality_constraints(
        self, x: jnp.array
    ) -> tuple[jnp.array, jnp.array, jnp.array]:
        """Returns inequality constraints.

        Returns g(x), g_l, g_u corresponding to the
        inequality constraints g_l <= g(x) <= g_u.

        Args:
            x: optimization variable,
                (_num_variables, ) array

        Returns:
            g_value: value of h(x),
                (_num_inequality_constraints, ) array
            g_l: value of g_l,
                (_num_inequality_constraints, ) array
            g_u: value of g_u,
                (_num_inequality_constraints, ) array
        """
        raise NotImplementedError

    def test(self):
        x0 = self.initial_guess()
        f0 = self.objective(x0)
        hs = self.equality_constraints(x0)
        gs, _, _ = self.inequality_constraints(x0)
        assert len(x0) == self.num_variables
        assert len(hs) == self.num_equality_constraints
        assert len(gs) == self.num_inequality_constraints
