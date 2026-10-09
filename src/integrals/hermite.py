"""Gaussian moments from a truncated probabilists' Hermite expansion."""

import numpy as np
from scipy.special import factorial

from src.activation_functions.activation_function import ActivationFunction
from .base import Integral


class HermiteIntegral(Integral):
    def __init__(self, g: ActivationFunction, N: int = 30):
        if not isinstance(N, int) or isinstance(N, bool) or N < 1:
            raise ValueError("Hermite order N must be a positive integer")
        self.g = g
        self.N = N

    def _coefficients(self, std, N):
        return np.asarray(
            self.g.scale(std).calc_coefficients(N, normalized=False), dtype=float
        )

    def I2(self, C):
        std1, std2 = np.sqrt([C[0, 0], C[1, 1]])
        if std1 == 0.0 or std2 == 0.0:
            return float(self._coefficients(std1, 0)[0] * self._coefficients(std2, 0)[0])
        rho = C[0, 1] / (std1 * std2)
        n = np.arange(self.N + 1)
        coefficients1 = self._coefficients(std1, self.N)
        coefficients2 = self._coefficients(std2, self.N)
        return float(np.sum(coefficients1 * coefficients2 * rho**n / factorial(n)))

    def J2(self, C):
        std2 = np.sqrt(C[1, 1])
        if std2 == 0.0:
            return 0.0
        return float(C[0, 1] * self._coefficients(std2, 1)[1] / std2)

    def I21(self, C):
        std1 = np.sqrt(C[0, 0])
        if std1 == 0.0:
            return 0.0
        coefficients = self._coefficients(std1, 3)
        return float(C[0, 1] * (coefficients[1] + coefficients[3]) / std1)

    def I22(self, C):
        std1 = np.sqrt(C[0, 0])
        if std1 == 0.0:
            return float(self.g.derivative_numpy_fn(0.0) * C[1, 1])
        coefficients = self._coefficients(std1, 3)
        return float(
            (C[1, 1] * coefficients[1] + C[0, 1]**2 / C[0, 0] * coefficients[3]) / std1
        )

    def I3(self, C):
        std1, std3 = np.sqrt([C[0, 0], C[2, 2]])
        if std1 == 0.0:
            return float(self.g.derivative_numpy_fn(0.0) * self.J2(C[1:, 1:]))
        if std3 == 0.0:
            return float(
                self.g.numpy_fn(0.0) * C[0, 1] / C[0, 0]
                * self._coefficients(std1, 2)[2]
            )
        rho = C[0, 2] / (std1 * std3)
        n = np.arange(self.N + 1)
        weights = rho**n / factorial(n)
        coefficients1 = self._coefficients(std1, self.N)
        coefficients3 = self._coefficients(std3, self.N)
        sum1 = np.sum(weights[:self.N - 1] * coefficients1[2:] * coefficients3[:-2])
        sum2 = np.sum(weights[:self.N] * coefficients1[1:] * coefficients3[1:])
        return float(C[0, 1] / C[0, 0] * sum1 + C[1, 2] / (std1 * std3) * sum2)
