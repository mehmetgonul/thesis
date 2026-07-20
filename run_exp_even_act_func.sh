#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/h2/preprocess/k=3.yaml
  configs/h4/preprocess/k=3.yaml
  configs/h6/preprocess/k=3.yaml
  configs/h2/no_preprocess/k=3.yaml
  configs/h4/no_preprocess/k=3.yaml
  configs/h6/no_preprocess/k=3.yaml
)

for config in "${configs[@]}"; do
  echo "Running $config"
  python -m src.train_autoencoder "$config" "$@"
done
