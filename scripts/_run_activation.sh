#!/usr/bin/env bash
set -euo pipefail
shopt -s failglob

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
activation="${1:?Activation name required}"

for config in "configs/$activation"/*.yaml; do
  echo "[$activation] Running $config"
  python -m src.train_autoencoder "$config"
done
