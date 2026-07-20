#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/erf/internal_unit/k=3.yaml
  configs/erf/internal_unit/k=4.yaml
  configs/erf/internal_unit/k=10.yaml
  configs/relu/internal_unit/k=3.yaml
  configs/relu/internal_unit/k=4.yaml
  configs/relu/internal_unit/k=10.yaml
  configs/gelu/internal_unit/k=3.yaml
  configs/gelu/internal_unit/k=4.yaml
  configs/gelu/internal_unit/k=10.yaml
  configs/h1/internal_unit/k=3.yaml
  configs/h1/internal_unit/k=4.yaml
  configs/h1/internal_unit/k=10.yaml
  configs/h2/internal_unit/k=3.yaml
  configs/h2/internal_unit/k=4.yaml
  configs/h2/internal_unit/k=10.yaml
  configs/h3/internal_unit/k=3.yaml
  configs/h3/internal_unit/k=4.yaml
  configs/h3/internal_unit/k=10.yaml
  configs/h4/internal_unit/k=3.yaml
  configs/h4/internal_unit/k=4.yaml
  configs/h4/internal_unit/k=10.yaml
  configs/h5/internal_unit/k=3.yaml
  configs/h5/internal_unit/k=4.yaml
  configs/h5/internal_unit/k=10.yaml
  configs/h6/internal_unit/k=3.yaml
  configs/h6/internal_unit/k=4.yaml
  configs/h6/internal_unit/k=10.yaml
)

for config in "${configs[@]}"; do
  python -m src.train_autoencoder "$config"
  python -m src.train_classifier "$config"
done
