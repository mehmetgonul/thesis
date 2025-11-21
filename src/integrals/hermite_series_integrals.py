import numpy as np
from src.activation_functions import ActivationFunction, activation_function_factory
from src.integrals.integrals import Integral
from scipy.special import hermitenorm, factorial, factorial2, erf

from numpy.polynomial.hermite_e import HermiteE, hermegauss
import functools

class HermiteWrapper():
    def __init__(self, g: str):
        
        self.g: ActivationFunction = activation_function_factory(g)
        
    def I2(self, C, N=10):
        return HermiteSeries.I2(C, self.g, self.g)
    
    def J2(self, C, N=10):
        return HermiteSeries.J2(C, self.g, N)

    def I21(self, C, N=10):
        return HermiteSeries.I21(C, self.g, N)

    def I22(self, C, N=10):
        return HermiteSeries.I22(C, self.g, N)

    def I3(self, C, N=10):
        return HermiteSeries.I3(C, self.g, self.g, N)


class HermiteSeries(Integral):
    @staticmethod
    def I2(C, g1: ActivationFunction, g2: ActivationFunction, N=30) -> float:
        """Computing the I2 integral using hermite series"""
        
        std1, std2 = np.sqrt(np.diagonal(C))
        
        g1 = g1.scale(std1)
        g2 = g2.scale(std2)
        
        rho = C[0,1]/(std1*std2)

        rho_powers = np.array([rho**n for n in range(N+1)])
        
        g1_coeffs = np.asarray(g1.calc_coefficients(N))
        g2_coeffs = np.asarray(g2.calc_coefficients(N))
        
        for i in range(N):
            g1_coeffs[i] /= np.sqrt(factorial(i))
            g2_coeffs[i] /= np.sqrt(factorial(i))
        
        return float(np.sum(g1_coeffs * g2_coeffs * rho_powers))

    @staticmethod
    def J2(C, g: ActivationFunction, N=30) -> float:
        
        std1, std2 = np.diagonal(C)
        std1 = np.sqrt(std1)
        std2 = np.sqrt(std2)
        
        g = g.scale(std2)
        
        rho = C[0, 1] / (std1 * std2)
        
        g2_coeff = g.calc_coefficients(N)
        
        return std1 * rho * g2_coeff[1]

    @staticmethod
    def I21(C, g: ActivationFunction, N=10) -> float:
        """ rho(<H1,g> + <H3, g>) """
        std1, std2 = np.sqrt(np.diagonal(C))
        
        g = g.scale(std1)
        
        rho = C[0,1] / (std1 * std2)
        
        
        g_coeffs = g.calc_coefficients(3)
        
        h1 = g_coeffs[1]
        h3 = g_coeffs[3]
        
        return float(std2 * rho * (h1 + h3))

    @staticmethod
    def I22(C, g: ActivationFunction, N=30) -> float:
        """  """
        std1, std2 = np.sqrt(np.diagonal(C))
        
        g = g.scale(std1)
        
        rho = C[0,1] / (std1 * std2)
        
        g_coeffs = g.calc_coefficients(3)
        
        h1 = g_coeffs[1]
        h3 = g_coeffs[3]
        
        return float((std2**2 / std1) * (h1 + (rho**2) * h3))

    @staticmethod
    def I3(C, g1: ActivationFunction, g3: ActivationFunction, N=30) -> float:
        
        std = np.sqrt(np.diagonal(C))

        g1 = g1.scale(std[0])
        g3 = g3.scale(std[2])
        
        T_inv = np.diag(1/std)
        
        C_norm = (T_inv) @ C @ (T_inv)

        rho12 = C_norm[0,1]
        rho13 = C_norm[0,2]
        rho23 = C_norm[1,2]

        rho13_powers_factor = np.array([rho13**n/factorial(n) for n in range(N+1)])

        g1_coeffs = g1.calc_coefficients(N)
        g3_coeffs = g3.calc_coefficients(N)

        sum1 = np.sum(rho13_powers_factor[:N-1]*g1_coeffs[2:]*g3_coeffs[0:N-1])
        sum2 = np.sum(rho13_powers_factor[:N]*g1_coeffs[1:]*g3_coeffs[1:])
        
        
        return (rho12*sum1 + rho23*sum2)*std[1]/std[0]