from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from src.autoencoder import AutoEncoder
from src.config import parse_config
from src.data_manager import DataManager
from src.hash import make_hash
from src.integrals import analytical_integral_factory

from itertools import product

from src.integrals.integrals import Integral
from src.utils import set_seed



SEEDS = [40, 41, 42]

def compute_order_parameters_from_weights(model: AutoEncoder, omega):
    W = model.encoder.weight.data.detach().cpu().numpy()
    if model.tied: 
        V = model.encoder.weight.detach().cpu().numpy()
    else:
        V = model.decoder.weight.detach().cpu().numpy().T

    
    D, _ = omega.shape

    T1 = V @ omega @ V.T/(D**2)
    Q1 = W @ omega @ W.T/D
    
    R1 = V @ omega @ W.T/(D**1.5)
    # R1 = W @ omega @ V.T/(D**1.5)
    
    
    T0 = V@V.T/D
    Q0 = W@W.T/D
    R0 = V@W.T/D
    
    return T1, Q1, R1, T0, Q0, R0

def test_error_with_order_parameters(T0, T1, Q1, R1, omega, integral: Integral):
    ## taken from NonLinearShallowAE
    
    D, _ = omega.shape
    K, _ = Q1.shape
    
    C2Vec = np.zeros((K, 2, 2))
    C2Mat = np.zeros((K, K, 2, 2))
    
    for k in range(K):
        C2Vec[k, 0, 0] = T1[k, k]
        C2Vec[k, 0, 1] = R1[k, k]
        # C2Vec[k, 1, 0] = R1[k, k]
        C2Vec[k, 1, 1] = Q1[k, k]
        
        for l in range(K):
            C2Mat[k, l, 0, 0] = Q1[k, k]
            C2Mat[k, l, 0, 1] = Q1[k, l]
            # C2Mat[k, l, 1, 0] = Q1[k, l]
            C2Mat[k, l, 1, 1] = Q1[l, l]


    I2m = np.asarray([integral.I2(C2Mat[k,l])* T0[k,l] for k, l in product(range(K),range(K))])
    J2m = sum([integral.J2(C2Vec[k]) for k in range(K)])
    
    error  = 0.5*np.trace(omega)/D
    
    error += 0.5*np.sum(I2m)
    error -= np.sum(J2m)
    return error , (0.5*np.trace(omega)/D, np.sum(J2m), 0.5*np.sum(I2m) )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    artifacts_dir = Path("artifacts")

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    run_config = parse_config(args.config)

    run_hash = make_hash(run_config=run_config)
    integral = analytical_integral_factory(run_config.activation_function)

    for seed in SEEDS:
        out_dir = artifacts_dir / run_hash / "autoencoder" / f"seed_{seed}"
        model_path = out_dir / "model.pt"
        loss_path = out_dir / "loss.jsonl"

        if not args.overwrite and model_path.exists() and loss_path.exists():
            print(f"skip: {out_dir}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        if loss_path.exists():
            loss_path.unlink()
        if model_path.exists():
            model_path.unlink()

        set_seed(seed)

        dm = DataManager(seed, run_config)

        trainloader, (omega, _, _, _) = dm.get_trainloader()

        model = AutoEncoder(config=run_config).to(device)

        params = [{"params": model.encoder.weight, "lr": run_config.learning_rate / run_config.D}]
        if run_config.bias and model.encoder.bias is not None:
            params.append({"params": model.encoder.bias, "lr": run_config.learning_rate})
        if not run_config.tied:
            params.append({"params": model.decoder.parameters(), "lr": run_config.learning_rate})

        optimizer = optim.SGD(params)
        train_iter = iter(trainloader)
        alpha = 0.0
        step = 0

        with loss_path.open("w", encoding="utf-8") as f:
            while alpha < run_config.alpha:
                try:
                    x, _ = next(train_iter)
                except StopIteration:
                    train_iter = iter(trainloader)
                    x, _ = next(train_iter)

                x = x.to(device)

                model.train()
                reconstruction = model(x)
                train_loss = 0.5 * nn.functional.mse_loss(reconstruction, x)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    # also order parameters are used for a more robust result
                    T1, Q1, R1, T0, Q0, R0 = compute_order_parameters_from_weights(
                        model=model,
                        omega=omega,
                    )
                    ge, _ = test_error_with_order_parameters(
                        T0,
                        T1,
                        Q1,
                        R1,
                        omega,
                        integral,
                    )

                alpha += 1.0 / run_config.D
                row = {
                    "step": step,
                    "alpha": alpha,
                    "loss": float(train_loss.detach().cpu()),
                    "generalization_error": float(ge),
                }
                f.write(json.dumps(row) + "\n")

                print(
                    f"seed={seed} "
                    f"step={step} alpha={alpha:.6f} "
                    f"loss={row['loss']:.6g} ge={row['generalization_error']:.6g}"
                )
                step += 1

        torch.save(model.state_dict(), model_path)
        print(f"saved: {model_path}")


if __name__ == "__main__":
    main()
