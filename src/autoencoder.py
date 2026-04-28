# implementation adapted from https://github.com/mariaref/NonLinearShallowAE/blob/main/src/models.py

import torch
import math
from torch import nn
import torch.nn.functional as F
from src.config import RunConfig
from src.activation_functions.common import activation_function_factory
import numpy as np

# TODO: ask about aligned initialization

class AutoEncoder(nn.Module):
    def __init__(self, config: RunConfig):
        super().__init__()
        self.input_size = config.D 
        self.hidden_size = config.K
        self.g = activation_function_factory(config.activation_function)
        self.tied = config.tied
        self.scale = config.scale_online
        self.bias = config.bias
        
        self.scale_factor = 1 / math.sqrt(self.input_size)

        print(f"tied: {self.tied}, scale: {self.scale}, bias: {self.bias}")

        self.encoder = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        # TODO: try xavier
        
        if config.init_uniform:
            nn.init.uniform_(self.encoder.weight, a=-self.scale_factor, b=self.scale_factor)
        else:         
            nn.init.normal_(self.encoder.weight, mean=0.0, std=config.init_std)
            
        if (not self.tied):
            # there is not bias in the decoder
            self.decoder = nn.Linear(self.hidden_size, self.input_size, bias=False)
            
            if config.init_uniform:
                nn.init.uniform_(self.encoder.weight, a=-self.scale_factor, b=self.scale_factor)
            else:
                nn.init.normal_(self.decoder.weight, mean=0.0, std=config.init_std)
        
    def forward(self, x):
        
        # encoder
        
        if self.scale:
            # normalize by input dimension
            x = self.g(self.encoder(x) * self.scale_factor)
        else:
            x = self.g(self.encoder(x))
            
            
        # decoder
        
        if self.tied:
            x = F.linear(x, self.encoder.weight.t())
        else:
            x = self.decoder(x)
            
        return x