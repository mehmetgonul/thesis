import numpy as np
import math
from numpy import sqrt
import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from scipy.special import hermitenorm, factorial, factorial2, erf, eval_hermitenorm, roots_hermitenorm
from numpy.polynomial.hermite_e import HermiteE, hermegauss
import functools

@functools.cache
def _hermite_quadrature(N, no_samples):
    z, weights = roots_hermitenorm(max(no_samples, N + 1))
    polynomials = eval_hermitenorm(np.arange(N + 1)[:, None], z)
    return z, polynomials * weights / np.sqrt(2 * np.pi)

def _polynomial_coefficients(values, N, normalized):
    coeffs = np.zeros(N + 1)
    coeffs[:min(N + 1, len(values))] = values[:N + 1]
    return coeffs / np.sqrt(factorial(np.arange(N + 1))) if normalized else coeffs

@functools.cache
def factorial2_custom(n: int):
    return factorial2(n) if n > 1 else 1

def inner_product(f1, f2, no_samples=100) -> float:
    
    # imports of the sample points and weights for the Gauss-Hermite quadrature with N terms
    z, w = hermegauss(no_samples)

    return np.sum(w*f1(z)*f2(z))/np.sqrt(2*np.pi)

    
class ActivationFunction(ABC):
    def __init__(self, stretch: float = 1.0):
        super().__init__()
        self.stretch = stretch
        
        self.coeffs = []
        self.coeffs_normalized = []

    def __call__(self, x):
        return self.torch_fn(x)

    def __str__(self):
        return type(self).__name__
    
    @abstractmethod
    def torch_fn(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    @abstractmethod
    def numpy_fn(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        
    @abstractmethod
    def derivative_numpy_fn(self, x):
        raise NotImplementedError()
    
    def calc_coefficients(self, N=10, normalized=True, no_samples=128):
        """Gaussian Hermite coefficients, calculated by quadrature."""
        z, weighted_polynomials = _hermite_quadrature(N, no_samples)
        coeffs = weighted_polynomials @ self.numpy_fn(z)
        return coeffs / np.sqrt(factorial(np.arange(N + 1))) if normalized else coeffs
        
    
    def scale(self, s: float):
        return type(self)(self.stretch * s)
