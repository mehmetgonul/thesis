from src.activation_functions.silu import Silu
from src.integrals.hermite import HermiteIntegral


class SiluIntegral(HermiteIntegral):
    """Evaluate I2, I3, I21, and I22 through Hermite coefficients; J2 is exact."""

    def __init__(self, N: int = 30):
        super().__init__(Silu(), N=N)

    def J2(self, C):
        return float(C[0, 1] / 2.0)

