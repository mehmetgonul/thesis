import numpy as np
import torch
from scipy.special import erf, factorial, factorial2

from src.activation_functions.activation_function import ActivationFunction


class GeLU(ActivationFunction):
    def __init__(self, s: float = 1.0):
        super().__init__()
        self.s = s

    def torch_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.s * x)

    def __call__(self, x):
        return self.torch_fn(x)

    def numpy_fn(self, x):
        z = self.s * x
        return z * (1.0 + erf(z / np.sqrt(2.0))) / 2.0

    def derivative_numpy_fn(self, x):
        z = self.s * x
        cdf = (1.0 + erf(z / np.sqrt(2.0))) / 2.0
        return self.s * (cdf + z * np.exp(-z**2 / 2.0) / np.sqrt(2.0 * np.pi))

    def calc_coefficients(self, N=10, normalized=True, no_samples=25):
        """Return exact Hermite coefficients of GELU(s * Z), optionally normalized."""
        s = self.s
        for n in range(len(self.coeffs), N + 1):
            if n == 0:
                cn = s**2 / np.sqrt(2.0 * np.pi * (1.0 + s**2))
            elif n == 1:
                cn = s / 2.0
            elif n % 2:
                cn = 0.0
            else:
                double_factorial = factorial2(n - 3) if n > 2 else 1.0
                cn = (
                    (-1) ** (n // 2 + 1) * double_factorial * s**n * (n + s**2)
                    / (np.sqrt(2.0 * np.pi) * (1.0 + s**2) ** ((n + 1) / 2.0))
                )
            self.coeffs.append(cn)

        coeffs = np.array(self.coeffs[:N + 1])
        if normalized:
            coeffs /= np.sqrt(factorial(np.arange(N + 1)))
        return coeffs

    def scale(self, s: float):
        return GeLU(self.s * s)

    def __str__(self):
        return "GeLU"
