"""Exact centered Gaussian moments for erf(x / sqrt(2))."""

import math
import numpy as np

from .base import Integral


class ErfIntegral(Integral):
    @staticmethod
    def I2(C: np.ndarray) -> float:
        # sanity check
        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for I2.")
        res = C[0,1] / (math.sqrt(max(1 + C[0,0], 0.0)) * math.sqrt(max(1 + C[1,1], 0.0)))
        res = float(np.clip(res, -1.0 + 1e-9, 1.0 - 1e-9))
        return (2.0 * np.arcsin(res)) / np.pi

    @staticmethod
    def J2(C: np.ndarray) -> float:

        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for J2.")

        c12 = C[0, 1]
        c22 = C[1, 1]

        denom = 1.0 + c22
        # print(f"c22 (J2): {c22}")
        safe_denom = max(denom, 1e-9)
        val = math.sqrt(2.0 / (math.pi * safe_denom)) * (c12)

        return val

    @staticmethod
    def I3(C: np.ndarray) -> float:

        if C.shape[0] < 3 or C.shape[1] < 3:
            raise ValueError("Covariance matrix C must be at least 3x3 for I3.")

        c11 = C[0, 0]
        c33 = C[2, 2]
        c12 = C[0, 1]
        c13 = C[0, 2]
        c23 = C[1, 2]

        denom = (1.0 + c11)*(1.0 + c33) - c13**2
        denom = max(denom, 1e-9)
        denom_sqrt = math.sqrt(denom)

        numerator = c23*(1.0 + c11) - c12*c13

        # Refinetti & Goldt (2022), Appendix D.3: no arcsin in this moment.
        return (2.0 / math.pi) * numerator / ((1.0 + c11) * denom_sqrt)

    @staticmethod
    def I21(C: np.ndarray) -> float:

        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for I21.")

        c11 = C[0, 0]
        c12 = C[0, 1]

        denom = (1.0 + c11)**1.5

        factor = math.sqrt(2.0 / math.pi)

        val = factor * (c12 / denom)
        return val

    @staticmethod
    def I22(C: np.ndarray) -> float:

        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for I22.")

        c11 = C[0, 0]
        c12 = C[0, 1]
        c22 = C[1, 1]

        numerator = c22*(1.0 + c11) - c12**2

        denom = (1.0 + c11)**1.5

        factor = math.sqrt(2.0 / math.pi)

        val = factor * (numerator / denom)
        return val
