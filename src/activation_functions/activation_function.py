import numpy as np
import math
from numpy import sqrt
import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from scipy.special import hermitenorm, factorial, factorial2, erf
from numpy.polynomial.hermite_e import HermiteE, hermegauss
import functools

@functools.cache
def factorial2_custom(n: int):
    return factorial2(n) if n > 1 else 1

def inner_product(f1, f2, no_samples=100) -> float:
    
    # imports of the sample points and weights for the Gauss-Hermite quadrature with N terms
    z, w = hermegauss(no_samples)

    return np.sum(w*f1(z)*f2(z))/np.sqrt(2*np.pi)

    
class ActivationFunction(ABC):
    def __init__(self):
        super().__init__()
        
        self.coeffs = []
        self.coeffs_normalized = []
    
    @abstractmethod
    def torch_fn(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    @abstractmethod
    def numpy_fn(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        
    @abstractmethod
    def derivative_numpy_fn(self, x):
        raise NotImplementedError()
    
    def calc_coefficients(self, N=10, normalized=True, no_samples=25):
        raise NotImplementedError()
        
    
    def scale(self, s: float):
        raise NotImplementedError()
        
    def hermite_expansion(self, N: int = 10):
        
        coefficients = self.calc_coefficients(N, normalized=True)
        
        return HermiteE(coefficients[:N])