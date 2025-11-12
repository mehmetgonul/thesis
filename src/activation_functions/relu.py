# class ReLU(ActivationFunction):
#     def __init__(self, stretch = 1.0):
#         super().__init__()
        
#         self.stretch = stretch
        
#         self.coeffs = []
#         self.coeffs_normalized = []
        
#     @staticmethod
#     def torch_fn(x: torch.Tensor) -> torch.Tensor:
#         return torch.nn.functional.relu(x)
    
#     def numpy_fn(self, x):
#         return np.maximum(0.0, self.stretch * x)
    
#     def derivative_numpy_fn(self, x):
#         return (x > 0).astype(float) * self.stretch
        
#     def calc_coefficients(self, N=10, normalized=True, no_samples=25):
        
#         a = self.stretch
        
#         for n in range(len(self.coeffs), N + 1):
            
#             if n == 0:
#                 c_n = 1 / np.sqrt(2 * np.pi)
                
#             elif n == 1:
#                 c_n = 0.5
                
#             elif n % 2 == 0:
#                 c_n = (-1)**(0.5*(n-2)) * factorial2_custom(n-3) / np.sqrt(2*np.pi)
                
#             else:
#                 c_n = 0 # no coefficients for odd n
            
#             c_n *= a
            
#             self.coeffs.append(c_n)
#             self.coeffs_normalized.append(c_n/ (np.sqrt(factorial(n))))
            
#         return self.coeffs if not normalized else self.coeffs_normalized
        
#     def __str__(self):
#         return "ReLU"
    
#     def scale(self, s: float):
#         return ReLU(stretch=s)
