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

## Run an experiment

Each Bash file in the repository root represents one set of experiments from the thesis. For example:

```bash
bash run_exp_general.sh
```

Completed runs are detected and skipped automatically. Therefore, if a script is stopped halfway, rerunning it will not repeat the completed runs; only the unfinished runs will be executed.

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
