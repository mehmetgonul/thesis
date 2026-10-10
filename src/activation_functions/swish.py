import numpy as np
import torch
from scipy.special import expit, factorial

from src.activation_functions.activation_function import ActivationFunction


class Swish(ActivationFunction):
    def torch_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.stretch * x)

    def numpy_fn(self, x):
        z = self.stretch * x
        return z * expit(z)

    def derivative_numpy_fn(self, x):
        z = self.stretch * x
        sigmoid = expit(z)
        return self.stretch * sigmoid * (1.0 + z * (1.0 - sigmoid))

    def calc_coefficients(self, N=10, normalized=False, no_samples=128):
        coeffs = super().calc_coefficients(N, normalized=False, no_samples=no_samples)

        # Swish(z) - Swish(-z) = z; only the first odd coefficient is nonzero.
        coeffs[1::2] = 0.0
        if N >= 1:
            coeffs[1] = self.stretch / 2.0
        if normalized:
            coeffs /= np.sqrt(factorial(np.arange(N + 1)))
        return coeffs

