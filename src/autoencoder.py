# implementation adapted from https://github.com/mariaref/NonLinearShallowAE/blob/main/src/models.py


import math
import torch
from torch import nn
import torch.nn.functional as F
from src.config import RunConfig
from src.activation_functions.common import activation_function_factory
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self, config: RunConfig):
        super().__init__()
        self.input_size = config.D 
        self.hidden_size = config.K
        self.g = activation_function_factory(config.activation_function)
        self.tied = config.tied
        self.bias = config.bias
        
        self.scale_factor = 1 / math.sqrt(self.input_size)

        self.encoder = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        
        nn.init.normal_(self.encoder.weight, mean=0.0, std=config.init_std)
            
        if (not self.tied):
            # there is not bias in the decoder
            self.decoder = nn.Linear(self.hidden_size, self.input_size, bias=False)
            
            nn.init.normal_(self.decoder.weight, mean=0.0, std=config.init_std)
        
    def encode(self, x):
        return self.g(self.encoder(x) * self.scale_factor)
    
    def forward(self, x):
        z = self.encode(x)
        
        if self.tied:
            out = F.linear(z, self.encoder.weight.t())
        else:
            out = self.decoder(z)
        
        
        return out
