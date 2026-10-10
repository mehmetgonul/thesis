"""Exact centered Gaussian moments for g(x) = sin(x)."""

import numpy as np

from .base import Integral


class SineIntegral(Integral):
    def I2(self, C):
        a, b, p = C[0, 0], C[1, 1], C[0, 1]
        return float((np.exp(-(a + b - 2*p)/2) - np.exp(-(a + b + 2*p)/2)) / 2)

    def J2(self, C):
        return float(C[0, 1] * np.exp(-C[1, 1]/2))

    def I21(self, C):
        return float(C[0, 1] * (1 - C[0, 0]) * np.exp(-C[0, 0]/2))

    def I22(self, C):
        return float((C[1, 1] - C[0, 1]**2) * np.exp(-C[0, 0]/2))

    def I3(self, C):
        a, c, p, q, r = C[0, 0], C[2, 2], C[0, 1], C[0, 2], C[1, 2]
        return float(((r - p)*np.exp(-(a + c - 2*q)/2)
                      + (r + p)*np.exp(-(a + c + 2*q)/2)) / 2)
