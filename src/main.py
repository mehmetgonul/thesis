from src.integrals.analytical_integrals import analytical_integral_factory
from src.integrals.hermite_series_integrals import HermiteSeries, HermiteWrapper
from src.activation_functions.erf import *
from src.integrals.hermite_series_integrals import HermiteSeries
from src.data_managers import DataManager
from src.autoencoder import AutoEncoder
from src.order_parameters import simulate
import matplotlib.pyplot as plt
from src.config import ExperimentConfig, RunConfig, DataManagerConfig, parse_config

import torch.nn.functional as F
import torch, math
import numpy as np
import os
import sys
import random

def activation_function_factory(activation_function: str, **kwargs):
    
    activation_function = activation_function.lower()
    
    if activation_function == "erf":
        return Erf(**kwargs)
    # elif activation_function == "relu":
    #     return ReLU(**kwargs)
    # elif activation_function == "linear":
    #     return Linear()
    # elif activation_function == "gelu":
    #     return GeLU(**kwargs)
    # elif activation_function == "exponential" or activation_function == "exp": # TODO: repo has mixed usage, unify it sometime
    #     return Exponential(**kwargs)
    else:
        raise ValueError(f"Unknown activation function: {activation_function}")
    


config: ExperimentConfig = parse_config(sys.argv[1])

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

results = []
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
        scale=True,
        bias=False
    )
    
    if run.training_method == "analytical_integrals":
        integral = analytical_integral_factory(run.activation_function)
    
    elif run.training_method == "hermite_integrals":
        integral = HermiteWrapper(act_func)
    elif run.training_method == "sgd":
        print("The training with stochasitic gradient descent will be done. No integral is needed.")
        
    else:
        raise ValueError()

    
    import time
    s = time.time()

    if 'integral' in locals():
        alpha, generalization_error = simulate(D, K, act_func, low_rank, dataset, noise, gaussian, lr, integral, config=config)
        results.append({"method": run.training_method, "alpha": alpha, "ge": generalization_error, "D": D, "color": run.line_color, "style": run.line_type})
        del integral
    else:
        # online learning, no epochs
        data_manager = DataManager(seed= SEED, alpha=120, dataset=dataset, rank=dm_config.rank, D=D, noise=noise, gaussian=gaussian, online=True)

        trainloader, _ = data_manager.get_trainloader()
        testloader, _ = data_manager.get_testloader()

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=(lr))
        criterion = torch.nn.MSELoss()

        online_learning_results = {
            "train_loss": [],
            "test_loss": [],
            "steps": []
        }

        step = 0
        print("online")
        alpha= []
        train_iter = iter(trainloader)
        while step < 200:    
            alpha.append(step)
            try:
                inputs, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(trainloader)
                inputs, _ = next(train_iter)
            inputs = inputs.to(device)
            
            test_loss = 0.0
            with torch.no_grad():
                model.eval()
                for test_inputs, _ in testloader:
                    test_inputs = test_inputs.to(device)
                    test_outputs = model(test_inputs)
                    t_loss = criterion(test_outputs, test_inputs)
                    test_loss += t_loss.item()
            
            avg_test_loss = test_loss / len(testloader)
            online_learning_results["test_loss"].append(avg_test_loss)
            
            model.train()
            
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            online_learning_results["train_loss"].append(loss.item())
            print(f"loss: {loss}")
            online_learning_results["steps"].append(step)

            
            print(f'Step {step}, Train Loss: {loss.item():.4f}, Test Loss: {avg_test_loss:.4f}')

            step += 1/D

        # final eval
        model.eval()
        final_test_loss = 0.0
        with torch.no_grad():
            for test_inputs, _ in testloader:
                test_inputs = test_inputs.to(device)
                test_outputs = model(test_inputs)
                t_loss = criterion(test_outputs, test_inputs)
                final_test_loss += t_loss.item()

        avg_final_test_loss = final_test_loss / len(testloader)
        print(f'Final Test Loss: {avg_final_test_loss:.4f}')

        results.append({"method": run.training_method, "alpha": alpha, "ge": online_learning_results["test_loss"], "D": D, "color": run.line_color, "style": run.line_type})
    
    print(f"Time taken: {(time.time() - s) / 60} min")

for res in results:
    ge = res["ge"]
    print(res["alpha"][-1])
    step_count = np.arange(0, (res["alpha"][-1]) + 1)
    plt.plot(
        res["alpha"],
        ge,
        color=res["color"],
        linestyle=res["style"],
        label=res["method"],
    )
    
plt.xlabel(config.x_label)
plt.ylabel(config.y_label)
plt.title(config.plot_name)
plt.grid(True)
plt.legend(title="Runs")

safe_title = config.plot_name.replace(" ", "_").lower()
safe_x = config.x_label.replace(" ", "_").lower()
safe_y = config.y_label.replace(" ", "_").lower()

plt.savefig(f"{safe_title}_{safe_x}_{safe_y}.png")
plt.close()