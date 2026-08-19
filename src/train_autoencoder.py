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
from src.integrals.hermite_series_integrals import HermiteWrapper
from src.ode_solver import (
    MAX_RK45_ALPHA_STEP,
    MIN_ODE_ALPHA,
    ODESample,
    solve_ode_from_model,
)

from itertools import product

from src.integrals.integrals import Integral
from src.utils import set_seed



SEEDS = [40, 41, 42]
HERMITE_ACTIVATIONS = {"h1", "h2", "h3", "h4", "h5", "h6"}
ODE_RESULT_ALPHA_STEP = 1.0


def integral_for_activation(activation_function: str, N: int | None = None):
    activation_function = activation_function.lower()
    if activation_function in HERMITE_ACTIVATIONS:
        degree = int(activation_function[1:])
        return HermiteWrapper(activation_function, N=max(N or degree, degree))
    return analytical_integral_factory(activation_function)

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
    J2m = sum([integral.J2(C2Vec[k]) for k in range(K)]) # 0 for even hermite series activation functions
    
    error  = 0.5*np.trace(omega)/D
    
    error += 0.5*np.sum(I2m)
    error -= np.sum(J2m)
    return error , (0.5*np.trace(omega)/D, np.sum(J2m), 0.5*np.sum(I2m) )



def reconstruction_loss(model: AutoEncoder, dataloader, device: torch.device) -> float:
    # empirical reconstruction loss for experiments with bias and even activation function
    
    
    total_loss = 0.0
    total_elements = 0

    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            reconstruction = model(x)
            loss = 0.5 * nn.functional.mse_loss(
                reconstruction,
                x,
                reduction="sum",
            )
            total_loss += float(loss.detach().cpu())
            total_elements += x.numel()

    return total_loss / total_elements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_false")
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

    if run_config.ode_sol and (run_config.tied or run_config.bias):
        reasons = []
        if run_config.tied:
            reasons.append("tied: true")
        if run_config.bias:
            reasons.append("bias: true")
        print(
            "ode_sol: true is unsupported with "
            + ", ".join(reasons)
            + "; overriding ode_sol to false and running online SGD"
        )
        run_config.ode_sol = False

    if run_config.ode_sol and run_config.test_loss:
        print(
            "warning: test_loss is ignored in ODE mode because no final "
            "PyTorch model is evolved"
        )

    if run_config.ode_sol and 0.0 <= float(run_config.alpha) < MIN_ODE_ALPHA:
        print(
            f"ODE alpha={run_config.alpha} is below the minimum horizon; "
            f"solving through alpha={MIN_ODE_ALPHA:g}"
        )

    run_hash = make_hash(run_config=run_config)
    integral = integral_for_activation(run_config.activation_function, run_config.N)

    for seed in SEEDS:
        out_dir = artifacts_dir / run_hash / "autoencoder" / f"seed_{seed}"
        model_path = out_dir / "model.pt"
        loss_path = out_dir / "loss.jsonl"
        ode_loss_path = out_dir / "ode_loss.jsonl"
        ode_state_path = out_dir / "ode_state.npz"
        ode_meta_path = out_dir / "ode_meta.json"
        initial_model_path = out_dir / "initial_model.pt"

        out_dir_exists = out_dir.exists()
        if run_config.ode_sol:
            checkpoint_exists = (
                ode_state_path.exists()
                and ode_meta_path.exists()
                and initial_model_path.exists()
            )
            checkpoint_description = f"ode_state_path={ode_state_path.exists()}"
        else:
            checkpoint_exists = model_path.exists()
            checkpoint_description = f"model_path={checkpoint_exists}"
        active_loss_path = ode_loss_path if run_config.ode_sol else loss_path
        loss_exists = active_loss_path.exists()
        print(
            f"exists: out_dir={out_dir_exists} "
            f"{checkpoint_description} loss_path={loss_exists}"
        )

        # if model isn't done (experiment isn't done), do experiment
        # skip otherwise
        if checkpoint_exists and loss_exists:
            print(f"skip: {out_dir}")
            continue

        # if partial run is done, erase past run data
        out_dir.mkdir(parents=True, exist_ok=True)
        partial_outputs = (
            ode_loss_path,
            ode_state_path,
            ode_meta_path,
            initial_model_path,
        ) if run_config.ode_sol else (loss_path, model_path)
        for partial_output in partial_outputs:
            if partial_output.exists():
                partial_output.unlink()

        set_seed(seed)

        # Initialize before touching a DataLoader so SGD and ODE configurations
        # with the same seed start from exactly the same weights.
        model = AutoEncoder(config=run_config)

        dm = DataManager(seed, run_config)

        if run_config.ode_sol:
            omega, _, eigenvalues, eigenvectors = dm.get_ode_statistics()
            torch.save(model.state_dict(), initial_model_path)

            with ode_loss_path.open("w", encoding="utf-8") as loss_file:
                def write_ode_sample(sample: ODESample) -> None:
                    row = {
                        "step": int(round(sample.alpha * run_config.D)),
                        "alpha": sample.alpha,
                        "generalization_error": sample.generalization_error,
                    }
                    loss_file.write(json.dumps(row) + "\n")

                ode_result = solve_ode_from_model(
                    model=model,
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                    integral=integral,
                    learning_rate=run_config.learning_rate,
                    alpha_end=float(run_config.alpha),
                    sample_step=ODE_RESULT_ALPHA_STEP,
                    on_sample=write_ode_sample,
                )

            np.savez_compressed(
                ode_state_path,
                q=ode_result.final_state.q,
                r=ode_result.final_state.r,
                t=ode_result.final_state.t,
                T0=ode_result.final_state.T0,
                eigenvalues=eigenvalues,
                alpha=np.asarray(ode_result.final_alpha),
            )
            ode_metadata = {
                "method": "RK45",
                "requested_alpha_end": float(run_config.alpha),
                "minimum_alpha_end": MIN_ODE_ALPHA,
                "max_rk45_alpha_step": MAX_RK45_ALPHA_STEP,
                "result_alpha_step": ode_result.sample_step,
                "rtol": ode_result.rtol,
                "atol": ode_result.atol,
                "nfev": ode_result.nfev,
                "accepted_steps": ode_result.accepted_steps,
                "emitted_samples": ode_result.emitted_samples,
                "final_alpha": ode_result.final_alpha,
                "covariance_trace": float(np.trace(omega)),
            }
            ode_meta_path.write_text(
                json.dumps(ode_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"saved: {ode_state_path}")
            continue

        # Setting up datasets and covariance matrices for the SGD path.
        trainloader, (omega, _, _, _) = dm.get_trainloader()
        testloader, (omega_test, _, _, _) = dm.get_testloader()
        
        if not run_config.test_loss:
            testloader = None

        model = model.to(device)

    
        # scaling learning rate
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
                
                # looped iteration over training set
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

                completed_steps = step + 1
                test_loss = None
                model.eval()
                
                
                with torch.no_grad():
                    # also order parameters are used for a more robust result
                    T1, Q1, R1, T0, Q0, R0 = compute_order_parameters_from_weights(
                        model=model,
                        omega=omega_test, # test set covariance is used for OP
                    )
                    ge, _ = test_error_with_order_parameters(
                        T0,
                        T1,
                        Q1,
                        R1,
                        omega_test, # test set for correct generalization error
                        integral,
                    )
                    
                # computed every alpha value, not D * alpha times like generalization error
                if testloader is not None and completed_steps % run_config.D == 0:
                    test_loss = reconstruction_loss(model, testloader, device)

                alpha += 1.0 / run_config.D
                row = {
                    "step": step,
                    "alpha": alpha,
                    "loss": float(train_loss.detach().cpu()),
                    "generalization_error": float(ge),
                }
                if test_loss is not None:
                    row["test_loss"] = float(test_loss)
                if model.encoder.bias is not None:
                    row["bias_values"] = model.encoder.bias.detach().cpu().tolist()
                f.write(json.dumps(row) + "\n")

                # print(
                #     f"seed={seed} "
                #     f"step={step} alpha={alpha:.6f} "
                #     f"loss={row['loss']:.6g} ge={row['generalization_error']:.6g}"
                # )
                step += 1
        
        torch.save(model.state_dict(), model_path)
        print(f"saved: {model_path}")


if __name__ == "__main__":
    main()
