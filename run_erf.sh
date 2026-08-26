#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/erf/test_erf.yaml}"

mkdir -p "$(dirname -- "$config")"

python -m src.train_autoencoder "$config"
