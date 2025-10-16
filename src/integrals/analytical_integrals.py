# for Sigmoid, ReLU and Linear see: The dynamics of representation learning in shallow, non-linear autoencoders

import math
import numpy as np
from src.integrals.integrals import Integral

def analytical_integral_factory(activation_function: str):
    if activation_function.lower() == "sigmoid":
        return SigmoidAnalyticIntegral
    elif activation_function.lower() == "relu":
        return ReLUAnalyticIntegral
    elif activation_function.lower() == "linear":
        return LinearAnalyticIntegral
    else:
        raise ValueError(f"Unknown activation function: {activation_function}")

class SigmoidAnalyticIntegral(Integral):
    @staticmethod
    def I2(C: np.ndarray) -> float:
        # sanity check
        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for I2.")
        res = C[0,1] / (math.sqrt(1 + C[0,0]) * math.sqrt(1 + C[1,1]))
        return 2.0 / math.pi * res
    
    @staticmethod
    def J2(C: np.ndarray) -> float:
        
        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for J2.")
        
        c12 = C[0, 1]
        c22 = C[1, 1]

        denom = 1.0 + c22

        val = math.sqrt(2.0 / math.pi) * (c12 / math.sqrt(denom))
        
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
        if denom < 0:
            denom_sqrt = 1e-8
        else:
            denom_sqrt = math.sqrt(denom)

        numerator = c23*(1.0 + c11) - c12*c13

        factor = 2.0 / math.pi
        val = factor * (1.0 / denom_sqrt) * (numerator / (1.0 + c11))
        return val

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


class LinearAnalyticIntegral:
    @staticmethod
    def I2(C: np.ndarray) -> float:
        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for I2.")
        
        c12 = C[0, 1]
        return c12

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


class ReLUAnalyticIntegral:
    @staticmethod
    def I2(C: np.ndarray) -> float:
        
        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for I2.")

        c11 = C[0, 0]
        c12 = C[0, 1]
        c22 = C[1, 1]

        # Step: check the discriminant
        disc = c11 * c22 - c12**2
        disc = max(disc, 1e-8)
        
        sqrt_disc = math.sqrt(disc)
        term1 = 2.0 * sqrt_disc
        term2 = c12 * math.pi
        term3 = 2.0 * c12
        term4 = (c12 / sqrt_disc) * math.atan(c12 / sqrt_disc)

        sum_terms = term1 + term2 + term3 + term4

        val = (1.0 / (8.0 * math.pi)) * sum_terms
        return val

    @staticmethod
    def J2(C: np.ndarray) -> float:
        
        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for J2.")

        c12 = C[0, 1]
        
        # Step: straightforward fraction
        val = c12 / 2.0
        return val