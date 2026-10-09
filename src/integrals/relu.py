"""Exact centered Gaussian moments for ReLU."""

import numpy as np

from .base import Integral


class ReLUIntegral(Integral):
    @staticmethod
    def I2(C: np.ndarray) -> float:

        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for I2.")

        C11 = C[0,0]
        C12 = C[0,1]
        C22 = C[1,1]

        if C11 == 0.0 or C22 == 0.0:
            return 0.0
        rho = np.clip(C12 / np.sqrt(C11 * C22), -1.0, 1.0)
        determinant = max(C11 * C22 - C12 * C12, 0.0)
        return C12/4 + C12*np.arcsin(rho)/(2*np.pi) + np.sqrt(determinant)/(2*np.pi)

    @staticmethod
    def J2(C: np.ndarray) -> float:

        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for J2.")

        c12 = C[0, 1]

        val = c12 / 2.0
        return val


    @staticmethod
    def I21(C: np.ndarray) -> float:

        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for J2.")

        c12 = C[0, 1]

        val = c12 / 2.0
        return val

    @staticmethod
    def I22(C: np.ndarray) -> float:

        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for J2.")

        C22 = C[1, 1]

        if C[0, 0] == 0.0:
            return 0.0
        return C22 / 2.0

    @staticmethod
    def I3(C: np.ndarray) -> float:
        if C.shape[0] < 3 or C.shape[1] < 3:
            raise ValueError("Covariance matrix C must be at least 3x3 for I3.")

        C11 = C[0,0]
        C12 = C[0,1]
        C13 = C[0,2]
        # C22 = C[1,1] #not used
        C23 = C[1,2]
        C33 = C[2,2]

        if C11 == 0.0 or C33 == 0.0:
            return 0.0
        rho = np.clip(C13 / np.sqrt(C11 * C33), -1.0, 1.0)
        determinant = max(C11 * C33 - C13 * C13, 0.0)
        return C12*np.sqrt(determinant)/(2*np.pi*C11) + C23*np.arcsin(rho)/(2*np.pi) + C23/4
