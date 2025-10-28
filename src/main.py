from src.integrals.analytical_integrals import analytical_integral_factory
from src.integrals.hermite_series_integrals import HermiteSeries, HermiteWrapper
from src.activation_functions import activation_function_factory
from src.integrals.hermite_series_integrals import HermiteSeries
from src.data_managers import DataManager
from src.autoencoder import AutoEncoder
from src.order_parameters import simulate
from src.plot import plot_graph
from src.config import ExperimentConfig, RunConfig, DataManagerConfig, parse_config

import torch.nn.functional as F
import torch, math
import numpy as np
import os
import sys
import random


config: ExperimentConfig = parse_config(sys.argv[1])

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


for run in config.runs:
    dm_config = run.data_manager
    print(run)

    dataset = dm_config.dataset
    D = run.D
    
    if dataset == "cifar-10":
        D = 1024
    elif dataset == "fashion_mnist":
        D = 784
    
    
    K = run.K
    lr = run.learning_rate
    online_training = True if run.training_method == "sgd" else False
    activation_function = run.activation_function
    noise = dm_config.noise
    gaussian = dm_config.gaussian

    if dataset == "random" and gaussian:
        raise ValueError("Gaussian noise cannot be applied to a randomly generated data.")

    if dataset == "random":
        low_rank = K
    else:
        low_rank = D

    act_func = activation_function_factory(run.activation_function)


    # load the model
    model = AutoEncoder(
        input_size=D,
        hidden_size=K,
        layers=1,
        g=act_func.torch_fn,
        tied=run.tied,
        scale=False,
        bias=run.bias
    )
    
    if run.training_method == "analytical_integrals":
        integral = analytical_integral_factory(run.activation_function)
    
    elif run.training_method == "hermite_integrals":
        integral = HermiteWrapper(act_func)
    elif run.training_method == "sgd":
        print("The training with stochasitic gradient descent will be done. No integral is needed.")
        
    else:
        raise ValueError()



    if 'integral' in locals():
        simulate(D, K, act_func, low_rank, dataset, noise, gaussian, lr, integral, config=config)

    else:
        # online learning, no epochs
        data_manager = DataManager(dataset=dataset, rank=1, D=D, noise=noise, gaussian=gaussian)

        trainloader, _ = data_manager.get_trainloader()
        testloader, _ = data_manager.get_testloader()

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        online_learning_results = {
            "train_loss": [],
            "test_loss": [],
            "steps": []
        }

        step = 0
        print("online")
        
        
        while step < 120:    
            try:
                inputs, _ = next(iter(trainloader))
            except StopIteration:
                trainloader, _ = data_manager.get_trainloader()
                inputs, _ = next(iter(trainloader))
            inputs = inputs.to(device)
            
            model.train()
            
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            online_learning_results["train_loss"].append(loss.item())
            online_learning_results["steps"].append(step)

            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for test_inputs, _ in testloader:
                    test_outputs = model(test_inputs)
                    t_loss = criterion(test_outputs, test_inputs)
                    test_loss += t_loss.item()
            
            avg_test_loss = test_loss / len(testloader)
            online_learning_results["test_loss"].append(avg_test_loss)
            print(f'Step {step}, Train Loss: {loss.item():.4f}, Test Loss: {avg_test_loss:.4f}')

            step += 1

        # final eval
        model.eval()
        final_test_loss = 0.0
        with torch.no_grad():
            for test_inputs, _ in testloader:
                test_outputs = model(test_inputs)
                t_loss = criterion(test_outputs, test_inputs)
                final_test_loss += t_loss.item()

        avg_final_test_loss = final_test_loss / len(testloader)
        print(f'Final Test Loss: {avg_final_test_loss:.4f}')

        # plot online learning results
        plot_graph(
            values=online_learning_results["train_loss"],
            x_title="Step",
            y_title="Train Loss",
            plot_title="Online Learning Train Loss through steps"
        )

        plot_graph(
            values=online_learning_results["test_loss"],
            x_title="Step",
            y_title="Test Loss",
            plot_title="Online Learning Test Loss through steps"
        )
        