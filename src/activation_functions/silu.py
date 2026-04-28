from src.activation_functions.activation_function import ActivationFunction
import torch
import numpy as np
from scipy.special import hermitenorm, factorial, factorial2, erf


def factorial2_custom(n: int):
    return factorial2(n) if n > 1 else 1


class Silu(ActivationFunction):
    def __init__(self, stretch = 1.0):
        super().__init__()
        
        self.stretch = stretch
        
        self.coeffs = []
        
    @staticmethod
    def torch_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)
    
    def __call__(self, x):
        return self.torch_fn(x)
