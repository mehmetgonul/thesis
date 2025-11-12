
# class GeLU(ActivationFunction):
#     def __init__(self, s = 1.0):
        
#         g = lambda x: x * (1 + erf(x/np.sqrt(2))) / 2
#         dg = lambda x: x * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi) + (1 + erf(x / np.sqrt(2))) / 2
        
#         self.s = s
        
#         self.coeffs = []
#         self.coeffs_normalized = []
        
#     def __str__(self):
#         return "GeLU"
    
#     @staticmethod
#     def torch_fn(x: torch.Tensor) -> torch.Tensor:
#         return torch.nn.functional.gelu(x)
    
#     def numpy_fn(self, x):
#         return x * (1 + erf(x / np.sqrt(2))) / 2
    
#     def derivative_numpy_fn(self, x):
#         return super().derivative_numpy_fn(x)
    
#     def calc_coefficients(self, N=10, normalized=True, no_samples=25):
        
#         a = self.s
        
#         for n in range(len(self.coeffs), N+1):
            
#             if n == 0: cn = a/np.sqrt(2*np.pi*(1+a*a))

#             elif n == 1: cn = 1/2
            
#             elif n%2 != 0: cn = 0
            
#             else:
#                 cn = (1/np.sqrt(2*np.pi)) * (-1)**(n/2 + 1) * factorial2_custom(n-3) * a**(n-1) * (n+a**2) / (1 + a*a)**((n+1)/2)
            
#             self.coeffs.append(self.s*cn)
#             self.coeffs_normalized.append(self.s*cn/np.sqrt(factorial(n)))
            
            
#         if normalized:
#             return np.array(self.coeffs_normalized[:N+1])
#         else:
#             return np.array(self.coeffs[:N+1])
            

#     def scale(self, s: float):
#         return GeLU(s=s)
    