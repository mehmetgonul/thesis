#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/relu/bias/relu_cifar_bias.yaml
  configs/relu/bias/no_bias.yaml
  configs/gelu/bias/gelu_cifar_bias.yaml
  configs/gelu/bias/no_bias.yaml

)

for config in "${configs[@]}"; do
  python -m src.train_autoencoder "$config"
done
