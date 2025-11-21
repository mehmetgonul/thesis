import numpy as np
from src.activation_functions.erf import Erf
from src.activation_functions.relu import ReLU
from src.integrals.analytical_integrals import SigmoidAnalyticIntegral, ReLUAnalyticIntegral
from src.integrals.hermite_series_integrals import HermiteWrapper


def random_covariance_matrix22():
    A = np.random.randn(2, 2)
    cov = np.dot(A, A.T)
    return cov

def random_covariance_matrix33():
    A = np.random.randn(3, 3)
    cov = np.dot(A, A.T)
    return cov
from src.utils import set_seed


def test():
    g = ReLU()
    for i in range(50):
        set_seed(42)
        C22 = random_covariance_matrix22()
        C33 = random_covariance_matrix33()
        I21 = ReLUAnalyticIntegral.I21
        I22 = ReLUAnalyticIntegral.I22
        I2 = ReLUAnalyticIntegral.I2
        I3 = ReLUAnalyticIntegral.I3

        h = HermiteWrapper("relu")

        diff_I22 = np.abs(h.I22(C22) - I22(C22))
        diff_I21 = np.abs(h.I21(C22) - I21(C22))
        diff_I2 = np.abs(h.I2(C22) - I2(C22))
        
        diff_I3 = np.abs(h.I3(C33) - I3(C33))
        
        if diff_I22 > 1e-3:
            print(f"Iteration {i}: I22 = {diff_I22}")
        if diff_I21 > 1e-9:
            print(f"Iteration {i}: I21 = {diff_I21}")
        if diff_I2 > 1e-9:
            print(f"Iteration {i}: I2 = {diff_I2}")
        if diff_I3 > 1e-3:
            print(f"Iteration {i}: I3 = {diff_I3}")

if __name__ == "__main__":
    cov_matrix = test()
    print(cov_matrix)
    print("completed")