from .activation_function import ActivationFunction, _polynomial_coefficients


class H2(ActivationFunction):
    def torch_fn(self, x):
        return (self.stretch * x)**2 - 1.0

    numpy_fn = torch_fn

    def derivative_numpy_fn(self, x):
        return 2.0 * self.stretch**2 * x

    def calc_coefficients(self, N=10, normalized=False, no_samples=128):
        s = self.stretch
        return _polynomial_coefficients([s*s - 1, 0, 2*s*s], N, normalized)
