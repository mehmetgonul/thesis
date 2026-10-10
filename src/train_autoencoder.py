from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from src.autoencoder import AutoEncoder
from src.config import parse_config, resolve_run_modes
from src.data_manager import DataManager
from src.hash import (
    INTEGRAL_VERSIONS,
    ODE_INTEGRATION_COORDINATES,
    ODE_ORDER_PARAMETER_LOGGING_VERSION,
    is_current_ode_metadata,
    make_hash,
)
from src.integrals import Integral, integral_for_activation
from src.ode_solver import (
    MIN_ODE_ALPHA,
    ORDER_PARAMETER_NAMES,
    ODESample,
    solve_ode_from_model,
)

from itertools import product

from src.utils import set_seed



SEEDS = [43]
ODE_RESULT_ALPHA_STEP = 1.0


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


def normalize_order_parameters(order_parameters, final_order_parameters):
    """Scale matrices by final recorded norms; zero final norms become null."""
    normalized = {}
    for suffix in ("0", "1"):
        q_std = np.sqrt(np.maximum(np.diag(final_order_parameters[f"Q{suffix}"]), 0))
        t_std = np.sqrt(np.maximum(np.diag(final_order_parameters[f"T{suffix}"]), 0))
        for name, left, right in (
            (f"Q{suffix}", q_std, q_std),
            (f"T{suffix}", t_std, t_std),
            (f"R{suffix}", t_std, q_std),
        ):
            denominator = np.outer(left, right)
            valid = denominator > 0
            values = np.divide(
                order_parameters[name], denominator,
                out=np.zeros_like(denominator), where=valid,
            )
            normalized[f"{name}_normalized"] = np.where(valid, values, None).tolist()
    return normalized


def normalize_order_parameter_log(path: Path):
    """Rescale a completed history using its last recorded parameter norms."""
    final_line = None
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if '"Q0"' in line:
                final_line = line
    if final_line is None:
        return
    final_order_parameters = json.loads(final_line)

    temporary_path = path.with_suffix(".normalized.tmp")
    try:
        with path.open("r", encoding="utf-8") as source, temporary_path.open(
            "w", encoding="utf-8"
        ) as target:
            for line in source:
                if '"Q0"' in line:
                    row = json.loads(line)
                    row.update(normalize_order_parameters(row, final_order_parameters))
                    line = json.dumps(row, allow_nan=False) + "\n"
                target.write(line)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
    parser.add_argument("--overwrite", action="store_false")
    args = parser.parse_args()
    artifacts_dir = Path("artifacts")

    device = torch.device("cpu")

    run_config = parse_config(args.config)

    ode_requested = run_config.ode_sol
    ode_supported = not run_config.tied and not run_config.bias
    run_sgd, run_ode = resolve_run_modes(
        run_config,
        ode_supported=ode_supported,
    )

    if ode_requested and not ode_supported:
        reasons = []
        if run_config.tied:
            reasons.append("tied: true")
        if run_config.bias:
            reasons.append("bias: true")
        message = "ode_sol: true is unsupported with " + ", ".join(reasons)
        if not run_sgd:
            raise ValueError(message + "; enable sgd or use an untied, unbiased model")
        print(message + "; skipping ODE and running online SGD")

    if not run_sgd and not run_ode:
        raise ValueError("At least one of sgd or ode_sol must be enabled")

    if run_ode and not run_sgd and run_config.test_loss:
        print(
            "warning: test_loss is ignored in ODE mode because no final "
            "PyTorch model is evolved"
        )

    if run_ode and 0.0 <= float(run_config.alpha) < MIN_ODE_ALPHA:
        print(
            f"ODE alpha={run_config.alpha} is below the minimum horizon; "
            f"solving through alpha={MIN_ODE_ALPHA:g}"
        )

    run_hash = make_hash(run_config=run_config)
    integral = integral_for_activation(
        run_config.activation_function,
        run_config.N,
    )

    for seed in SEEDS:
        out_dir = artifacts_dir / run_hash / "autoencoder" / f"seed_{seed}"
        model_path = out_dir / "model.pt"
        loss_path = out_dir / "loss.jsonl"
        ode_loss_path = out_dir / "ode_loss.jsonl"
        ode_meta_path = out_dir / "ode_meta.json"
        initial_model_path = out_dir / "initial_model.pt"

        out_dir_exists = out_dir.exists()
        ode_complete = (
            ode_loss_path.exists()
            and ode_meta_path.exists()
            and initial_model_path.exists()
            and is_current_ode_metadata(ode_meta_path)
        )
        sgd_complete = model_path.exists() and loss_path.exists()
        statuses = []
        if run_ode:
            statuses.append(f"ode_complete={ode_complete}")
        if run_sgd:
            statuses.append(f"sgd_complete={sgd_complete}")
        print(
            f"exists: out_dir={out_dir_exists} "
            + " ".join(statuses)
        )

        if (not run_ode or ode_complete) and (not run_sgd or sgd_complete):
            print(f"skip: {out_dir}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_config.json").write_text(
            json.dumps(asdict(run_config), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        partial_outputs = []
        if run_ode and not ode_complete:
            partial_outputs.extend(
                [ode_loss_path, ode_meta_path, initial_model_path]
            )
        if run_sgd and not sgd_complete:
            partial_outputs.extend([loss_path, model_path])
        for partial_output in partial_outputs:
            if partial_output.exists():
                partial_output.unlink()

        set_seed(seed)

        # Initialize before touching a DataLoader so SGD and ODE configurations
        # with the same seed start from exactly the same weights.
        model = AutoEncoder(config=run_config)

        dm = DataManager(seed, run_config)

        if run_ode and not ode_complete:
            omega, _, eigenvalues, eigenvectors = dm.get_ode_statistics()
            torch.save(model.state_dict(), initial_model_path)

            with ode_loss_path.open("w", encoding="utf-8") as loss_file:
                def write_ode_sample(sample: ODESample) -> None:
                    row = {
                        "step": int(round(sample.alpha * run_config.D)),
                        "alpha": sample.alpha,
                        "generalization_error": sample.generalization_error,
                    }
                    row.update({
                        name: sample.order_parameters[name].tolist()
                        for name in ORDER_PARAMETER_NAMES
                    })
                    loss_file.write(json.dumps(row, allow_nan=False) + "\n")

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

            normalize_order_parameter_log(ode_loss_path)
            ode_metadata = {
                "method": "RK45",
                "integration_coordinates": ODE_INTEGRATION_COORDINATES,
                "order_parameter_logging_version": ODE_ORDER_PARAMETER_LOGGING_VERSION,
                "order_parameter_names": list(ORDER_PARAMETER_NAMES),
                "D": run_config.D,
                "K": run_config.K,
                "activation_function": run_config.activation_function,
                "integral_version": INTEGRAL_VERSIONS.get(run_config.activation_function.lower(), 1),
                "hermite_order": getattr(integral, "N", None),
                "eigenvalues": eigenvalues.tolist(),
                "requested_alpha_end": float(run_config.alpha),
                "minimum_alpha_end": MIN_ODE_ALPHA,
                "max_rk45_alpha_step": ode_result.max_step,
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
            print(f"saved: {ode_loss_path}")

        if not run_sgd or sgd_complete:
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
                alpha = completed_steps / run_config.D
                ge = None
                order_parameters = None
                test_loss = None

                # Evaluate generalization only at completed whole-alpha intervals.
                if completed_steps % run_config.D == 0:
                    model.eval()
                    with torch.no_grad():
                        T1, Q1, R1, T0, Q0, R0 = compute_order_parameters_from_weights(
                            model=model,
                            omega=omega_test,
                        )
                        order_parameters = {
                            "Q0": Q0, "Q1": Q1, "R0": R0,
                            "R1": R1, "T0": T0, "T1": T1,
                        }
                        ge, _ = test_error_with_order_parameters(
                            T0,
                            T1,
                            Q1,
                            R1,
                            omega_test,
                            integral,
                        )
                    if testloader is not None:
                        test_loss = reconstruction_loss(model, testloader, device)

                row = {
                    "step": step,
                    "alpha": alpha,
                    "loss": float(train_loss.detach().cpu()),
                }
                if ge is not None:
                    row["generalization_error"] = float(ge)
                if order_parameters is not None:
                    row.update({
                        name: matrix.tolist()
                        for name, matrix in order_parameters.items()
                    })
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
        
        normalize_order_parameter_log(loss_path)
        torch.save(model.state_dict(), model_path)
        print(f"saved: {model_path}")


if __name__ == "__main__":
    main()
