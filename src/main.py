from src.integrals.analytical_integrals import analytical_integral_factory
from src.integrals.hermite_series_integrals import HermiteSeries, HermiteWrapper
from src.activation_functions.erf import *
from src.integrals.hermite_series_integrals import HermiteSeries
from src.autoencoder import AutoEncoder
from src.order_parameters import solve_autoencoder
import matplotlib.pyplot as plt
from src.config import ExperimentConfig, RunConfig, DataManagerConfig, parse_config
from src.utils import set_seed
from src.online_learning import OnlineLearning

import torch.nn.functional as F
import torch, math
import numpy as np
import os
import sys
import random    


config: ExperimentConfig = parse_config(sys.argv[1])

SEED = 1

set_seed(SEED)


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps" # for mac m series
else:
    DEVICE = "cpu"

results = []
for run_config in config.runs:
    dm_config = run_config.dm_config
    print(run_config)

    dataset = dm_config.dataset
    

    K = run_config.K
    lr = run_config.learning_rate
    online_training = True if run_config.training_method == "sgd" else False
    activation_function = run_config.activation_function
    noise = dm_config.noise
    gaussian = dm_config.gaussian
    
    if dataset == "cifar-10":
        run_config.D = 1024
        low_rank = run_config.D
    elif dataset == "fashion_mnist":
        run_config.D = 784
        low_rank = run_config.D
    else:
        run_config.D = 1000
        low_rank = K



    # load the model
    model = AutoEncoder(config=run_config)
    
    if run_config.training_method == "analytical_integral":
        integral = analytical_integral_factory(run_config.activation_function)
    elif run_config.training_method == "hermite_integral":
        integral = HermiteWrapper(run_config.activation_function)
    elif run_config.training_method == "sgd":
        print("The training with stochasitic gradient descent will be done. No integral is needed.")
        
    else:
        raise ValueError()

    
    import time
    s = time.time()

    if 'integral' in locals():
        alpha, generalization_error = solve_autoencoder(SEED, run_config, low_rank, integral)
        D = run_config.D
        results.append({"label": run_config.plot_label, "alpha": alpha, "ge": generalization_error, "D": D, "color": run_config.line_color, "style": run_config.line_type})
        del integral
    else:
        D = run_config.D
        online_learning = OnlineLearning(run_config, device=DEVICE)
        alpha, res = online_learning.start_training()
        results.append({"label": run_config.plot_label, "alpha": alpha, "ge": res, "D": D, "color": run_config.line_color, "style": run_config.line_type})      



for res in results:
    ge = res["ge"]
    alpha = res["alpha"]
    plt.plot(
        alpha,
        ge,
        color=res["color"],
        linestyle=res["style"],
        label=res["label"],
    )
diff_flag = False
# Check if at any points the difference between the curves is bigger than 1e-3
if len(results) > 1:
    min_len = min(len(res["ge"]) for res in results)
    ge_arrays = [np.array(res["ge"][:min_len]) for res in results]
    for i in range(len(ge_arrays)):
        for j in range(i + 1, len(ge_arrays)):
            diff = np.abs(ge_arrays[i] - ge_arrays[j])
            if np.any(diff > 1e-3):
                diff_flag = True
                print(f"Difference between curves {i} and {j} exceeds 1e-3 at some points.")
                
if diff_flag:
    print("something is not correct")
else:
    print("all good")

plt.xlabel(config.x_label)
plt.ylabel(config.y_label)
plt.title(config.plot_name)
plt.grid(True)
plt.legend(title="Runs")

safe_title = config.plot_name.replace(" ", "_").lower()
safe_x = config.x_label.replace(" ", "_").lower()
safe_y = config.y_label.replace(" ", "_").lower()

plt.savefig(f"{safe_title}_{safe_x}_{safe_y}_{time.time()}.png")
plt.close()