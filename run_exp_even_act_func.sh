#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/h2/preprocess_k=3.yaml
  configs/h4/preprocess_k=3.yaml
  configs/h6/preprocess_k=3.yaml
  configs/h2/no_preprocess_k=3.yaml
  configs/h4/no_preprocess_k=3.yaml
  configs/h6/no_preprocess_k=3.yaml
)

for config in "${configs[@]}"; do
  echo "Running $config"
  python -m src.train_autoencoder "$config" "$@"
done
