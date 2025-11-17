
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

from src.activation_functions.activation_function import ActivationFunction

def erfscaled_numpy(x):
    return erf(x / math.sqrt(2))    

@functools.cache
def factorial2_custom(n: int):
    return factorial2(n) if n > 1 else 1

CACHE = {}

class Erf(ActivationFunction):
    def __init__(self, stretch: float = 1.0):
        super().__init__()
        
        self.stretch = stretch
        
        for i in range(50):
            if i not in CACHE:
                CACHE[i] = np.sqrt(2/np.pi) * ((-1)**((i-1)//2)) * factorial2_custom(i-2)

        
    def __str__(self):
        return "erf" 
    
    @staticmethod
    def torch_fn(x):
        return torch.erf((x) / math.sqrt(2))
    
    def numpy_fn(self, x):
        return erfscaled_numpy(x * self.stretch)
    
    def __call__(self, x):
        return self.torch_fn(x)
    
    def derivative_numpy_fn(self, x):
        return self.stretch*np.exp(-0.5*(x*self.stretch)**2)*np.sqrt(2/np.pi)

    @functools.cache
    def calc_coefficients(self, N=10):
      
        a = self.stretch
        
        coeffs = []

        for n in range(0, N + 1):
            if n % 2 == 0:
                c_n = 0.0                           
            else:
                c_n = CACHE[n] * (a/np.sqrt(1.0 + a*a))**n
                
            coeffs.append(c_n)

        return np.array(coeffs)
    
    
    def scale(self, s: float):
        return Erf(stretch=s)
    
    