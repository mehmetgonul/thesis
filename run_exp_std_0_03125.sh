#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/erf/erf_cifar_init_std=0.03125_k=3.yaml
  configs/relu/relu_cifar_init_std=0.03125_k=3.yaml
  configs/gelu/gelu_cifar_init_std=0.03125_k=3.yaml
  configs/h1/h1_cifar_init_std=0.03125_k=3.yaml
  configs/h3/h3_cifar_init_std=0.03125_k=3.yaml
)

for config in "${configs[@]}"; do
  python -m src.train_autoencoder "$config"
done
