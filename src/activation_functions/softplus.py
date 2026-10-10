import numpy as np
import torch
from scipy.special import expit

from .activation_function import ActivationFunction


class Softplus(ActivationFunction):
    def torch_fn(self, x):
        return torch.nn.functional.softplus(self.stretch * x)

    def numpy_fn(self, x):
        return np.logaddexp(0.0, self.stretch * x)

    def derivative_numpy_fn(self, x):
        return self.stretch * expit(self.stretch * x)
