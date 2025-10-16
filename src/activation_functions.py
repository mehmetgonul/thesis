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

def inner_product(f1, f2, no_samples=100) -> float:
    
    # imports of the sample points and weights for the Gauss-Hermite quadrature with N terms
    z, w = hermegauss(no_samples)

    return np.sum(w*f1(z)*f2(z))/np.sqrt(2*np.pi)

def activation_function_factory(activation_function: str, **kwargs):
    
    activation_function = activation_function.lower()
    
    if activation_function == "erf":
        return Erf(**kwargs)
    elif activation_function == "relu":
        return ReLU(**kwargs)
    elif activation_function == "linear":
        return Linear()
    else:
        raise ValueError(f"Unknown activation function: {activation_function}")

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
    
# def erfscaled_torch(x):
#     return torch.erf(x / math.sqrt(2))    

def erfscaled_numpy(x):
    return erf(x / math.sqrt(2))    


class Erf(ActivationFunction):
    def __init__(self, stretch: float = 1.0):
        super().__init__()
        
        self.stretch = stretch
        
        self.coeffs = []
        self.coeffs_normalized = []
        
    def __str__(self):
        return "erf" 
    
    @staticmethod
    def torch_fn(x):
        return torch.erf((x) / math.sqrt(2))
    
    def numpy_fn(self, x):
        return erfscaled_numpy(x * self.stretch)
    
    def derivative_numpy_fn(self, x):
        return self.stretch*np.exp(-0.5*(x*self.stretch)**2)*np.sqrt(2/np.pi)
    
    def calc_coefficients(self, N=10, normalized=True, no_samples=25):
        
        a = self.stretch
        
        for n in range(len(self.coeffs), N+1):
            if n % 2 == 0:
                c_n = 0
                
            else:
                c_n = np.sqrt(2/np.pi) * (-1)**((n-1)/2) * factorial2(n-2) * (a/np.sqrt(1+a*a))**n
                
            self.coeffs.append(c_n)
            self.coeffs_normalized.append(c_n / (np.sqrt(factorial(n))))

        return self.coeffs if not normalized else self.coeffs_normalized 
    
    
    def scale(self, s: float):
        return Erf(stretch=s)

class ReLU(ActivationFunction):
    def __init__(self, stretch = 1.0):
        super().__init__()
        
        self.stretch = stretch
        
        self.coeffs = []
        self.coeffs_normalized = []
        
    @staticmethod
    def torch_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x)
    
    def numpy_fn(self, x):
        return np.maximum(0.0, self.stretch * x)
    
    def derivative_numpy_fn(self, x):
        return (x > 0).astype(float) * self.stretch
        
    def calc_coefficients(self, N=10, normalized=True, no_samples=25):
        
        a = self.stretch
        
        for n in range(len(self.coeffs), N + 1):
            
            if n == 0:
                c_n = 1 / np.sqrt(2 * np.pi)
                
            elif n == 1:
                c_n = 0.5
                
            elif n % 2 == 0:
                c_n = (-1)**(0.5*(n-2)) * factorial2(n-3) / np.sqrt(2*np.pi)
                
            else:
                c_n = 0 # no coefficients for odd n
                
            self.coeffs.append(c_n)
            self.coeffs_normalized.append(a * c_n/ (np.sqrt(factorial(n))))
            
        return self.coeffs if not normalized else self.coeffs_normalized
        
    def __str__(self):
        return "ReLU"
    
    def scale(self, s: float):
        return ReLU(stretch=s)


class Linear(ActivationFunction):
    def __init__(self):
        super().__init__()
        
        self.g = lambda x: x
        self.dg = lambda x: x * x**-1 # 1
        
        self.coeffs = []
        self.coeffs_normalized = []
    

    def __str__(self):
        return "linear"
    
    @staticmethod
    def torch_fn(x: torch.Tensor) -> torch.Tensor:
        return x
    
    def numpy_fn(self, x):
        return x
    
    def derivative_numpy_fn(self, x):
        return 1
    
    def calc_coefficients(self, N=10, normalized=True, no_samples=25):
        
        for n in range(len(self.coeffs), N + 1):
            
            if n == 1:
                c_n = 1.0
                
            else:
                c_n = 0.0
                
            self.coeffs.append(c_n)
            self.coeffs_normalized.append(c_n/ (np.sqrt(factorial(n))))
            
        return self.coeffs if not normalized else self.coeffs_normalized
    
    def scale(self, s: float):
        return Linear()