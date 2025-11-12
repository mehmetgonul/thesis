# for Sigmoid, ReLU and Linear see: The dynamics of representation learning in shallow, non-linear autoencoders

import math
import numpy as np
from src.integrals.integrals import Integral
import functools

def analytical_integral_factory(activation_function: str):
    if activation_function.lower() == "erf":
        return SigmoidAnalyticIntegral
    elif activation_function.lower() == "relu":
        return ReLUAnalyticIntegral
    elif activation_function.lower() == "linear":
        return LinearAnalyticIntegral
    elif activation_function.lower() == "gelu":
        return GeLUAnalyticalIntegral.GeLUAnalyticalIntegral
    elif activation_function.lower() == "exponential" or activation_function.lower() == "exp":
        return ExponantialAnalyticalIntegral.ExponantialAnalyticalIntegral
    else:
        raise ValueError(f"Unknown activation function: {activation_function}")

class SigmoidAnalyticIntegral(Integral):
    @staticmethod
    def I2(C: np.ndarray) -> float:
        # sanity check
        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for I2.")
        res = C[0,1] / (math.sqrt(1 + C[0,0]) * math.sqrt(1 + C[1,1]))
        return (2.0 * np.arcsin(res)) / np.pi
    
    @staticmethod
    def J2(C: np.ndarray) -> float:
        
        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for J2.")
        
        c12 = C[0, 1]
        c22 = C[1, 1]

        denom = 1.0 + c22
        # print(f"c22 (J2): {c22}")
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

        C11 = C[0,0]
        C12 = C[0,1]
        C22 = C[1,1]

        return C12/4 + C12*np.arcsin(C12/np.sqrt(C11*C22))/(2*np.pi) + np.sqrt(C11*C22 - C12*C12)/(2*np.pi)

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
        
        return C22 / 2.0
    
    @staticmethod
    def I3(C: np.ndarray) -> float: 
        if C.shape[0] < 2 or C.shape[1] < 2:
            raise ValueError("Covariance matrix C must be at least 2x2 for J2.")
        
            
        C11 = C[0,0]
        C12 = C[0,1]
        C13 = C[0,2]
        C23 = C[1,2]
        C33 = C[2,2]

        return C12*np.sqrt(abs(C11*C33 - C13*C13))/(2*np.pi*C11) + C23*np.arcsin(C13/np.sqrt(C11*C33))/(2*np.pi) + C23/4
    
    
class GeLUAnalyticalIntegral:
    class GeLUAnalyticalIntegral:
        @staticmethod
        def I2(C: np.ndarray) -> float:
            if C.shape[0] < 2 or C.shape[1] < 2:
                raise ValueError("Covariance matrix C must be at least 2x2 for I2.")
            C11 = C[0,0]
            C12 = C[0,1]
            C22 = C[1,1]

            gamma2 = 1 #gamma^2

            det = C11*C22 - C12*C12

            Lambda = (1 + gamma2*C11)*(1 + gamma2*C22)

            return C12/4 + C12*np.arcsin(gamma2*C12/np.sqrt(Lambda))/(2*np.pi) + ((gamma2**3)*C11*C22*det + gamma2*(gamma2*C11*C22*(C11 + C22) + C11*C22 + C12*C12))/(2*np.pi*Lambda*np.sqrt(gamma2*gamma2*det + gamma2*(C11 + C22) + 1))


        @staticmethod
        def J2(C: np.ndarray) -> float:
            if C.shape[0] < 2 or C.shape[1] < 2:
                raise ValueError("Covariance matrix C must be at least 2x2 for J2.")

            
            return C[0, 1] / 2.

        @staticmethod
        def I3(C: np.ndarray) -> float:
            if C.shape[0] < 3 or C.shape[1] < 3:
                raise ValueError("Covariance matrix C must be at least 3x3 for I3.")
            C11 = C[0,0]
            C12 = C[0,1]
            C13 = C[0,2]
            C23 = C[1,2]
            C33 = C[2,2]

            Lambda = (1+C11)*(1+C33) - C13*C13

            D11 = -C13*(1 + C33 + 3*Lambda + C11*(1 + C33 + Lambda))/(4*np.pi*(1+C11)*(1+C11)*np.sqrt(Lambda**3))
            D13 = (1 + C33 + Lambda)/(2*np.pi*np.sqrt(Lambda**3))
            D33 = -C13*(1 + C33 + Lambda)/(4*np.pi*(1+C33)*np.sqrt(Lambda**3))

            return 2*C12*C13*D11 + (C12*C33 + C13*C23)*D13 + 2*C23*C33*D33 + C13*C23/(2*np.pi*(1+C11)*np.sqrt(Lambda)) + C23*np.arctan(C13/np.sqrt(Lambda))/(2*np.pi) + C23/4

        @staticmethod
        def I21(C: np.ndarray) -> float:
            if C.shape[0] < 2 or C.shape[1] < 2:
                raise ValueError("Covariance matrix C must be at least 2x2 for I21.")
            
            return C[0, 1] / 2.

        @staticmethod
        def I22(C: np.ndarray) -> float:
            if C.shape[0] < 2 or C.shape[1] < 2:
                raise ValueError("Covariance matrix C must be at least 2x2 for I22.")
            
            
            return C[1, 1] / 2.
        
        
class ExponantialAnalyticalIntegral:
    class ExponantialAnalyticalIntegral:
        @staticmethod
        def I2(C: np.ndarray) -> float:
            if C.shape[0] < 2 or C.shape[1] < 2:
                raise ValueError("Covariance matrix C must be at least 2x2 for I2.")
            
            C11 = C[0,0]
            C12 = C[0,1]
            C22 = C[1,1]
            
            a = 1
            phi = 0

            return np.exp(a*a*(C11+2*C12+C22)/2 - 2*a*phi)

        @staticmethod
        def J2(C: np.ndarray) -> float:
            if C.shape[0] < 2 or C.shape[1] < 2:
                raise ValueError("Covariance matrix C must be at least 2x2 for J2.")

            a = 1
            phi = 0

            C12= C[0,1]
            C22 = C[1,1]
            
            return a*C12*np.exp(0.5*a*a*C22 - a*phi)

        @staticmethod
        def I3(C: np.ndarray) -> float:
            if C.shape[0] < 3 or C.shape[1] < 3:
                raise ValueError("Covariance matrix C must be at least 3x3 for I3.")
            a = 1
            phi = 0

            C11 = C[0,0]
            C12 = C[0,1]
            C13 = C[0,2]
            C23 = C[1,2]
            C33 = C[2,2]

            return a*(C12+C23)*np.exp(a*a*(C11+2*C13+C33)/2 - 2*a*phi)

        @staticmethod
        def I21(C: np.ndarray) -> float:
            if C.shape[0] < 2 or C.shape[1] < 2:
                raise ValueError("Covariance matrix C must be at least 2x2 for I21.")

            
            a = 1
            phi = 0

            C11= C[0,0]
            C12= C[0,1]
            C22 = C[1,1]
            
            return a*(C12 + a*a*C11*C12)*np.exp(0.5*a*a*C11 - a*phi)
            

        @staticmethod
        def I22(C: np.ndarray) -> float:
            if C.shape[0] < 2 or C.shape[1] < 2:
                raise ValueError("Covariance matrix C must be at least 2x2 for I22.")
            a = 1
            phi = 0

            C11= C[0,0]
            C12= C[0,1]
            C22 = C[1,1]
            
            return a*(C22 + a*a*C12**2)*np.exp(0.5*a*a*C11 - a*phi)