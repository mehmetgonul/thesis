import math

import numpy as np
import torch

from src.activation_functions.activation_function import ActivationFunction


class HermitePolynomial(ActivationFunction):
    def __init__(self, degree: int, input_scale: float = 1.0):
        super().__init__()
        if degree not in {1, 2, 3, 4, 5, 6}:
            raise ValueError(f"Only h1, h2, h3, h4, h5, h6 are supported, got h{degree}")
        self.degree = degree
        self.input_scale = float(input_scale)

    def torch_fn(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_scale * x
        if self.degree == 1:
            return z
        if self.degree == 2:
            return z.pow(2) - 1.0
        if self.degree == 3:
            return z.pow(3) - 3.0 * z
        if self.degree == 5:
            return z.pow(5) - 10.0 * z.pow(3) + 15.0 * z
        if self.degree == 6:
            return z.pow(6) - 15.0 * z.pow(4) + 45.0 * z.pow(2) - 15.0
        return z.pow(4) - 6.0 * z.pow(2) + 3.0

    def numpy_fn(self, x):
        z = self.input_scale * x
        if self.degree == 1:
            return z
        if self.degree == 2:
            return z**2 - 1.0
        if self.degree == 3:
            return z**3 - 3.0 * z
        if self.degree == 5:
            return z**5 - 10.0 * z**3 + 15.0 * z
        if self.degree == 6:
            return z**6 - 15.0 * z**4 + 45.0 * z**2 - 15.0
        return z**4 - 6.0 * z**2 + 3.0 # He_4

    def derivative_numpy_fn(self, x):
        z = self.input_scale * x
        if self.degree == 1:
            return np.ones_like(x, dtype=float) * self.input_scale
        if self.degree == 2:
            return 2.0 * self.input_scale * z
        if self.degree == 3:
            return 3.0 * self.input_scale * (z**2 - 1.0)
        if self.degree == 5:
            return self.input_scale * (5.0 * z**4 - 30.0 * z**2 + 15.0)
        if self.degree == 6:
            return self.input_scale * (6.0 * z**5 - 60.0 * z**3 + 90.0 * z)
        return self.input_scale * (4.0 * z**3 - 12.0 * z) # He_4

    def calc_coefficients(self, N=10, normalized=False, no_samples=25):
        coeffs = np.zeros(N + 1, dtype=float)
        s = self.input_scale

        if self.degree == 1 and N >= 1:
            coeffs[1] = s
        elif self.degree == 2:
            coeffs[0] = s**2 - 1.0
            if N >= 2:
                coeffs[2] = 2.0 * s**2
        elif self.degree == 3:
            if N >= 1:
                coeffs[1] = 3.0 * s * (s**2 - 1.0)
            if N >= 3:
                coeffs[3] = 6.0 * s**3
        elif self.degree == 4:
            coeffs[0] = 3.0 * (s**2 - 1.0) ** 2
            if N >= 2:
                coeffs[2] = 12.0 * s**2 * (s**2 - 1.0)
            if N >= 4:
                coeffs[4] = 24.0 * s**4
        elif self.degree == 5:
            if N >= 1:
                coeffs[1] = 15.0 * s * (s**2 - 1.0) ** 2
            if N >= 3:
                coeffs[3] = 60.0 * s**3 * (s**2 - 1.0)
            if N >= 5:
                coeffs[5] = 120.0 * s**5
        elif self.degree == 6:
            coeffs[0] = 15.0 * (s**2 - 1.0) ** 3
            if N >= 2:
                coeffs[2] = 90.0 * s**2 * (s**2 - 1.0) ** 2
            if N >= 4:
                coeffs[4] = 360.0 * s**4 * (s**2 - 1.0)
            if N >= 6:
                coeffs[6] = 720.0 * s**6

        if normalized:
            coeffs /= np.sqrt([math.factorial(n) for n in range(N + 1)])
        return coeffs

    def scale(self, s: float):
        return HermitePolynomial(self.degree, self.input_scale * s)

    def __call__(self, x):
        return self.torch_fn(x)

    def __str__(self):
        return f"h{self.degree}"
