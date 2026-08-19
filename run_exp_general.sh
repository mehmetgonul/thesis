#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/erf/internal_unit_k=3.yaml
  configs/relu/internal_unit_k=3.yaml
  configs/gelu/internal_unit_k=3.yaml
  configs/h1/internal_unit_k=3.yaml
  configs/h2/internal_unit_k=3.yaml
  configs/h3/internal_unit_k=3.yaml
  configs/h4/internal_unit_k=3.yaml
  configs/h5/internal_unit_k=3.yaml
  configs/h6/internal_unit_k=3.yaml
)

for config in "${configs[@]}"; do
  python -m src.train_autoencoder "$config"
done
