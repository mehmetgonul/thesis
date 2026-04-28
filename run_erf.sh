#!/usr/bin/env bash
set -euo pipefail

python -m src.train_autoencoder configs/erf_cifar.yaml

python -m src.train_classifier configs/erf_cifar.yaml

