from .activation_function import ActivationFunction, _polynomial_coefficients


class H3H2(ActivationFunction):
    def torch_fn(self, x):
        z = self.stretch * x
        return z**3 - 3.0 * z + z**2 - 1.0

    numpy_fn = torch_fn

    def derivative_numpy_fn(self, x):
        z = self.stretch * x
        return self.stretch * (3.0 * z**2 - 3.0 + 2.0 * z)

    def calc_coefficients(self, N=10, normalized=False, no_samples=128):
        s = self.stretch
        return _polynomial_coefficients([s*s - 1, 3*s*(s*s - 1), 2*s*s, 6*s**3], N, normalized)
