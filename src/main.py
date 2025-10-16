from src.arg_parser import get_arguments
from src.integrals.analytical_integrals import SigmoidAnalyticIntegral, ReLUAnalyticIntegral, LinearAnalyticIntegral
from src.integrals.hermite_series_integrals import HermiteSeries, HermiteWrapper
from src.activation_functions import Erf, ReLU, Linear
from src.integrals.hermite_series_integrals import HermiteSeries
from src.data_managers import DataManager
from src.autoencoder import AutoEncoder
from src.order_parameters import *
from src.plot import plot_graph

import torch.nn.functional as F
import torch, math
import numpy as np

# args = get_arguments()
dataset = "cifar-10"
D = 1024
K = 4
lr = 1
online_training = False
activation_function = "erf"
noise = True
gaussian = False

if dataset == "random" and gaussian:
    raise ValueError("Gaussian noise cannot be applied to a randomly generated data.")

if dataset == "random":
    low_rank = K
else:
    low_rank = D


if (activation_function == "erf"):
    act_func = Erf()
elif (activation_function == "relu"):
    act_func = ReLU()
elif (activation_function == "linear"):
    act_func = Linear()

# training with analytical solution 
if activation_function == "erf":
    integral = SigmoidAnalyticIntegral()
elif activation_function == "relu":
    integral = ReLUAnalyticIntegral()
else:
    integral = LinearAnalyticIntegral()







print("analytical")
simulate(D, K, act_func, low_rank, dataset, noise, gaussian, lr, integral)


# train with the hermite approximation of the activation function and the integrals

print("hermtie")
integral = HermiteWrapper(act_func)
# simulate(D, K, act_func, low_rank, dataset, noise, gaussian, lr, integral)

# online learning, no epochs

# trainloader, _ = data_manager.get_trainloader()
# testloader, _ = data_manager.get_testloader()

# device = "mps" if torch.backends.mps.is_available() else "cpu"
# model.to(device)

# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# criterion = torch.nn.MSELoss()

# online_learning_results = {
#     "train_loss": [],
#     "test_loss": [],
#     "steps": []
# }

# step = 0
# print("online")
# for inputs, _ in trainloader:
#     if step > 000:
#         break
#     model.train()
    
#     outputs = model(inputs)
#     loss = criterion(outputs, inputs)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     online_learning_results["train_loss"].append(loss.item())
#     online_learning_results["steps"].append(step)

#     model.eval()
#     test_loss = 0.0
#     with torch.no_grad():
#         for test_inputs, _ in testloader:
#             test_outputs = model(test_inputs)
#             t_loss = criterion(test_outputs, test_inputs)
#             test_loss += t_loss.item()
    
#     avg_test_loss = test_loss / len(testloader)
#     online_learning_results["test_loss"].append(avg_test_loss)
#     print(f'Step {step}, Train Loss: {loss.item():.4f}, Test Loss: {avg_test_loss:.4f}')

#     step += 1

# # final eval
# model.eval()
# final_test_loss = 0.0
# with torch.no_grad():
#     for test_inputs, _ in testloader:
#         test_outputs = model(test_inputs)
#         t_loss = criterion(test_outputs, test_inputs)
#         final_test_loss += t_loss.item()

# avg_final_test_loss = final_test_loss / len(testloader)
# print(f'Final Test Loss: {avg_final_test_loss:.4f}')

# # plot online learning results
# plot_graph(
#     values=online_learning_results["train_loss"],
#     x_title="Step",
#     y_title="Train Loss",
#     plot_title="Online Learning Train Loss through steps"
# )

# plot_graph(
#     values=online_learning_results["test_loss"],
#     x_title="Step",
#     y_title="Test Loss",
#     plot_title="Online Learning Test Loss through steps"
# )