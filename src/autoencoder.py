# implementation adapted from https://github.com/mariaref/NonLinearShallowAE/blob/main/src/models.py

import torch
from torch import nn
import torch.nn.functional as F
from src.config import RunConfig
from src.activation_functions.factory import activation_function_factory
import numpy as np

# TODO: ask about aligned initialization

class AutoEncoder(nn.Module):
    def __init__(self, config: RunConfig):
        super().__init__()
        self.input_size = config.D 
        self.hidden_size = config.K
        self.g = activation_function_factory(config.activation_function)
        self.tied = config.tied
        self.scale = config.scale_online_training
        self.bias = config.bias

        self.encoder = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.01)
            
        if (not self.tied):
            self.decoder = nn.Linear(self.hidden_size, self.input_size, bias=self.bias)
            nn.init.normal_(self.decoder.weight, mean=0.0, std=0.01)
        
    def forward(self, x):
        
        # encoder
        
        if self.scale:
            # normalize by input dimension
            x = self.g(self.encoder(x) / np.sqrt(self.input_size))
        else:
            x = self.g(self.encoder(x))
            
            
        # decoder
        
        if self.tied:
            x = F.linear(x, self.encoder.weight.t())
        else:
            x = self.decoder(x)
            
        return x