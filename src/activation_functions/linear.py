import torch
import numpy as np
from math import factorial
from src.activation_functions.activation_function import ActivationFunction


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
    
    def __call__(self, x):
        return self.torch_fn(x)
    
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