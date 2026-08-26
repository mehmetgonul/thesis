# Autoencoder Experiments

This repository contains the complete experimental setup for the thesis "The Role of Activation Function in Shallow Autoencoders."

## Setup

For the smoothest setup, use Python 3.11.10. Other Python versions may also work.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data

The required dataset is downloaded automatically into `data/` the first time an experiment is run, so no manual dataset setup is needed.

Use `dataset: "cifar-10_gaussian"` for online SGD with a fresh Gaussian
sample at every update. The Gaussian covariance, and the covariance evolved by
the ODE, are computed from the 10,000 grayscale CIFAR-10 test images. Use
`dataset: "cifar-10_real"` to train SGD on the actual grayscale CIFAR-10
training images; the loader reshuffles them whenever it starts a new pass.

## Run an experiment

Each Bash file in the repository root represents one set of experiments from the thesis. For example:

```bash
bash run_exp_general.sh
```

Completed runs are detected and skipped automatically. Therefore, if a script is stopped halfway, rerunning it will not repeat the completed runs; only the unfinished runs will be executed.

### Run the autonomous ODE path

Set the following flag in an experiment config:

```yaml
ode_sol: true
sgd: false
```

Then use the normal autoencoder entry point:

```bash
python -m src.train_autoencoder path/to/config.yaml
```

`ode_sol` and `sgd` independently select the two run modes. Use `ode_sol: true` with `sgd: false` for ODE only, or set both to `true` to produce both results from the same seeded initialization. If `sgd` is omitted, existing behavior is preserved: ODE configurations run only ODE, while other configurations run SGD.

ODE mode initializes the configured autoencoder once and evolves full-spectrum rank-one factors independently with adaptive RK45. Density order parameters are reconstructed from those factors at every RK stage. It does not keep the analytical state aligned to a PyTorch model. The current implementation requires `tied: false` and `bias: false`; `test_loss` is ignored during an ODE-only run because no final PyTorch model is evolved.

Every ODE run evolves through at least `alpha: 1`. If a non-negative value below one is configured, RK45 extends the endpoint to one and the trainer prints the override; online SGD continues to use the configured value unchanged. RK45 is limited to at most one alpha per accepted integration step, while remaining free to take smaller steps for accuracy. `ode_loss.jsonl` contains the generalization error at `alpha = 0, 1, 2, ...` through the configured endpoint.

`ode_sol` and `sgd` are excluded from the experiment hash. This lets online and ODE results coexist under the same seed directory: online mode uses `model.pt` and `loss.jsonl`, while ODE mode uses `initial_model.pt`, `ode_loss.jsonl`, `ode_state.npz`, and `ode_meta.json`. ODE metadata records the integration-coordinate version so legacy raw-density results are recomputed. Running the other mode later adds its missing artifacts without overwriting the completed mode.

If `tied: true` or `bias: true` is configured together with `ode_sol: true`, the trainer skips the unsupported ODE mode when SGD is enabled. An explicit ODE-only configuration fails with an explanation. ODE mode does not save a final `model.pt`, because its final order parameters do not uniquely determine model weights.

## Generate plots

Run the corresponding plotting script from the repository root after an experiment finishes. For example:

```bash
python plots/plot_act_func_compare.py
```

The config paths for each experiment are already hardcoded in the corresponding plotting functions, so no config arguments are required.

## Deactivate the environment

```bash
deactivate
```
