#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/relu/relu_cifar_bias.yaml
  configs/relu/no_bias.yaml
  configs/gelu/gelu_cifar_bias.yaml
  configs/gelu/no_bias.yaml

)

for config in "${configs[@]}"; do
  python -m src.train_autoencoder "$config"
done
