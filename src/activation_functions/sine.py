import numpy as np
import torch

from .activation_function import ActivationFunction


class Sine(ActivationFunction):
    def torch_fn(self, x):
        return torch.sin(self.stretch * x)

    def numpy_fn(self, x):
        return np.sin(self.stretch * x)

    def derivative_numpy_fn(self, x):
        return self.stretch * np.cos(self.stretch * x)
