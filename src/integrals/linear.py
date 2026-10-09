"""Exact centered Gaussian moments for g(x) = x."""

import numpy as np

from .base import Integral


class LinearIntegral(Integral):
    @staticmethod
    def I2(C: np.ndarray) -> float:
        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for I2.")

        c12 = C[0, 1]
        return c12


    @staticmethod
    def J2(C: np.ndarray) -> float:
        return C[0, 1]

    @staticmethod
    def I21(C: np.ndarray) -> float:
        return C[0, 1]

    @staticmethod
    def I22(C: np.ndarray) -> float:

        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for I22.")

        c22 = C[1, 1]
        return c22

    @staticmethod
    def I3(C: np.ndarray) -> float:

        if C.shape[0] < 3 or C.shape[1] < 3:
            raise ValueError("Covariance matrix C must be at least 3x3 for I3.")

        c23 = C[1, 2]
        return c23
