from .activation_function import ActivationFunction, _polynomial_coefficients


class H4(ActivationFunction):
    def torch_fn(self, x):
        z = self.stretch * x
        return z**4 - 6.0 * z**2 + 3.0

    numpy_fn = torch_fn

    def derivative_numpy_fn(self, x):
        z = self.stretch * x
        return self.stretch * (4.0 * z**3 - 12.0 * z)

    def calc_coefficients(self, N=10, normalized=False, no_samples=128):
        s = self.stretch
        return _polynomial_coefficients(
            [3*(s*s - 1)**2, 0, 12*s*s*(s*s - 1), 0, 24*s**4], N, normalized
        )
