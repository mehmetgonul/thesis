from functools import cache

import numpy as np
import torch
from scipy.special import eval_hermitenorm, expit, factorial, roots_hermitenorm

from src.activation_functions.activation_function import ActivationFunction


@cache
def _hermite_quadrature(N, no_samples):
    z, weights = roots_hermitenorm(max(no_samples, N + 1))
    polynomials = eval_hermitenorm(np.arange(N + 1)[:, None], z)
    return z, polynomials * weights / np.sqrt(2 * np.pi)


class Silu(ActivationFunction):
    def __init__(self, stretch: float = 1.0):
        super().__init__()
        self.stretch = stretch

    def torch_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.stretch * x)

    def __call__(self, x):
        return self.torch_fn(x)

    def numpy_fn(self, x):
        z = self.stretch * x
        return z * expit(z)

    def derivative_numpy_fn(self, x):
        z = self.stretch * x
        sigmoid = expit(z)
        return self.stretch * sigmoid * (1.0 + z * (1.0 - sigmoid))

    def calc_coefficients(self, N=10, normalized=False, no_samples=128):
        """Return E[SiLU(stretch * Z) He_n(Z)] for a standard normal Z."""
        z, weighted_polynomials = _hermite_quadrature(N, no_samples)
        coeffs = weighted_polynomials @ self.numpy_fn(z)

        # SiLU(z) - SiLU(-z) = z, so only the first odd coefficient is nonzero.
        coeffs[1::2] = 0.0
        if N >= 1:
            coeffs[1] = self.stretch / 2.0
        if normalized:
            coeffs /= np.sqrt(factorial(np.arange(N + 1)))
        return coeffs

    def scale(self, s: float):
        return Silu(self.stretch * s)

    def __str__(self):
        return "SiLU"
