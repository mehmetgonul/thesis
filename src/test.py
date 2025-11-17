import numpy as np
from src.activation_functions.erf import Erf
from src.integrals.analytical_integrals import SigmoidAnalyticIntegral
from src.integrals.hermite_series_integrals import HermiteWrapper


def random_covariance_matrix():
    A = np.random.randn(2, 2)
    cov = np.dot(A, A.T)
    return cov



def test():
    g = Erf()
    for i in range(50):
        C = random_covariance_matrix()
        I21 = SigmoidAnalyticIntegral.I21
        I22 = SigmoidAnalyticIntegral.I22
        I2 = SigmoidAnalyticIntegral.I2
        I3 = SigmoidAnalyticIntegral.I3

        h = HermiteWrapper(g)

        diff_I22 = np.abs(h.I22(C) - I22(C))
        diff_I21 = np.abs(h.I21(C) - I21(C))
        diff_I2 = np.abs(h.I2(C) - I2(C))
        diff_I3 = np.abs(h.I3(C) - I3(C))
        
        if diff_I22 > 1e-9 or diff_I21 > 1e-9 or diff_I2 > 1e-9 or diff_I3 > 1e-9:
            print(f"Iteration {i}: I22 = {diff_I22}, I21 = {diff_I21}, I2 = {diff_I2}, I3 = {diff_I3}")

if __name__ == "__main__":
    cov_matrix = test()
    print(cov_matrix)
    print("completed")