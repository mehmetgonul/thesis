#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/erf/initialization/erf_cifar_init_std=0.03125_k=3.yaml
  configs/relu/initialization/relu_cifar_init_std=0.03125_k=3.yaml
  configs/gelu/initialization/gelu_cifar_init_std=0.03125_k=3.yaml
  configs/h1/initialization/h1_cifar_init_std=0.03125_k=3.yaml
  configs/h3/initialization/h3_cifar_init_std=0.03125_k=3.yaml
)

for config in "${configs[@]}"; do
  python -m src.train_autoencoder "$config"
done
