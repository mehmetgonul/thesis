from src.autoencoder import AutoEncoder
from src.config import RunConfig, DataManagerConfig
from src.utils import set_seed
from src.data_manager import DataManager

import numpy as np
import math

import torch
from torch import optim
from torch import nn


SEEDS = [500]

class OnlineLearning:

    def __init__(self, config: RunConfig, device):
        self.config = config
        self.dm_config = config.dm_config
        self.device = device
        self.lr = config.learning_rate
        
        self.model = None
        self.optim = None
        self.dm: DataManager = None # to be set in the run loop for different seeds
        self.data_stats = None
        self.trainloader = None
        self.testloader = None

    def _init_model(self):
        self.model = AutoEncoder(config=self.config)
        self.model.to(self.device)
        # self.optim = optim.SGD(self.model.parameters(), lr=self.lr)
        self.optim = optim.SGD(
            [
                {"params": self.model.encoder.parameters(),
                 "lr": self.lr / self.config.D},  # η_W
                {"params": self.model.decoder.parameters(),
                 "lr": self.lr},                  # η_V
            ]
        )
        
    def _set_data(self, seed):
        self.dm = DataManager(seed, self.config)
        self.trainloader, _ = self.dm.get_trainloader()
        self.testloader, self.data_stats = self.dm.get_testloader()
        # import code; code.interact(local=locals())
    
    def _train_step(self, input):
        self.model.train()
        
        reconstruction = self.model(input)
        
        loss = nn.functional.mse_loss(input, reconstruction) * 0.5
        
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss
        
    
    def _test(self):
        self.model.eval()
        
        with torch.no_grad():
            avg_test_loss = 0.
            
            for test_input, _ in self.testloader:
                test_input = test_input.to(self.device)
                test_reconstruction = self.model(test_input)
                loss = nn.functional.mse_loss(test_input, test_reconstruction) * 0.5
                
                avg_test_loss += loss
                
            avg_test_loss /= len(self.testloader)
            
        return avg_test_loss.item()
                
    
    def _train(self):
        train_iter = iter(self.trainloader)
        
        loss_per_step = []
        ge_per_step =[]
        step = 0.
        steps = []
        
        while step < self.config.alpha:
            try:
                input, _ = next(train_iter)    
            except StopIteration:
                train_iter = iter(self.trainloader)
                input, _ = next(train_iter)

            input = input.to(self.device)

            loss = self._train_step(input)
            loss_per_step.append(loss.item())
            
            # generalizaiton error
            ge = self._test()
            ge_per_step.append(ge)
            
            
            step += 1/self.config.D
            steps.append(step)
            
            print(f"Step {step}: Training Loss: {loss}, Generalizaiton Error: {ge}")
            
        return steps, ge_per_step       
    
    def start_training(self, run_count = 1):
        
        
        for seed in SEEDS:
            
            set_seed(seed)

            self._set_data(seed)
            self._init_model()
            
            return self._train()

            