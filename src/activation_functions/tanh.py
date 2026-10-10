import numpy as np
import torch

from .activation_function import ActivationFunction


class Tanh(ActivationFunction):
    def torch_fn(self, x):
        return torch.tanh(self.stretch * x)

    def numpy_fn(self, x):
        return np.tanh(self.stretch * x)

    def derivative_numpy_fn(self, x):
        return self.stretch * (1.0 - self.numpy_fn(x)**2)
