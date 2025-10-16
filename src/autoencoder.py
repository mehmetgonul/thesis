# implementation adapted from https://github.com/mariaref/NonLinearShallowAE/blob/main/src/models.py

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self, input_size,  hidden_size, layers = 1, g = F.relu, tied = False, scale = True, bias = False, std = 0.1):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.g = g 
        self.tied = tied
        self.scale = scale
        self.layers = layers

        self.encoder = nn.Linear(input_size, hidden_size, bias=bias)
        nn.init.normal_(self.encoder.weight, mean=0.0, std=std)
            
        if (not self.tied):
            self.decoder = nn.Linear(hidden_size, input_size, bias=bias)
            nn.init.normal_(self.decoder.weight, mean=0.0, std=std)
        
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