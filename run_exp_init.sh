#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/erf/erf_cifar_init_std=0.5_k=3.yaml
  configs/erf/erf_cifar_init_uniform_k=3.yaml
  configs/erf/erf_cifar_init_unit_sphere_scale=0.1_k=3.yaml
  configs/erf/erf_cifar_init_unit_sphere_scale=0.5_k=3.yaml
  configs/erf/erf_cifar_init_unit_sphere_scale=1_k=3.yaml
  configs/erf/erf_cifar_init_unit_sphere_scale=2_k=3.yaml
  configs/erf/erf_cifar_init_unit_sphere_scale=5_k=3.yaml
  configs/gelu/gelu_cifar_init_std=0.5_k=3.yaml
  configs/gelu/gelu_cifar_init_uniform_k=3.yaml
  configs/gelu/gelu_cifar_init_unit_sphere_scale=0.1_k=3.yaml
  configs/gelu/gelu_cifar_init_unit_sphere_scale=0.5_k=3.yaml
  configs/gelu/gelu_cifar_init_unit_sphere_scale=1_k=3.yaml
  configs/gelu/gelu_cifar_init_unit_sphere_scale=2_k=3.yaml
  configs/gelu/gelu_cifar_init_unit_sphere_scale=5_k=3.yaml
  configs/relu/relu_cifar_init_std=0.5_k=3.yaml
  configs/relu/relu_cifar_init_uniform_k=3.yaml
  configs/relu/relu_cifar_init_unit_sphere_scale=0.1_k=3.yaml
  configs/relu/relu_cifar_init_unit_sphere_scale=0.5_k=3.yaml
  configs/relu/relu_cifar_init_unit_sphere_scale=1_k=3.yaml
  configs/relu/relu_cifar_init_unit_sphere_scale=2_k=3.yaml
  configs/relu/relu_cifar_init_unit_sphere_scale=5_k=3.yaml
  configs/h1/h1_cifar_init_std=0.5_k=3.yaml
  configs/h1/h1_cifar_init_uniform_k=3.yaml
  configs/h1/h1_cifar_init_unit_sphere_scale=0.1_k=3.yaml
  configs/h1/h1_cifar_init_unit_sphere_scale=0.5_k=3.yaml
  configs/h1/h1_cifar_init_unit_sphere_scale=1_k=3.yaml
  configs/h1/h1_cifar_init_unit_sphere_scale=2_k=3.yaml
  configs/h1/h1_cifar_init_unit_sphere_scale=5_k=3.yaml
  configs/h3/h3_cifar_init_std=0.5_k=3.yaml
  configs/h3/h3_cifar_init_uniform_k=3.yaml
  configs/h3/h3_cifar_init_unit_sphere_scale=0.1_k=3.yaml
  configs/h3/h3_cifar_init_unit_sphere_scale=0.5_k=3.yaml
  configs/h3/h3_cifar_init_unit_sphere_scale=1_k=3.yaml
  configs/h3/h3_cifar_init_unit_sphere_scale=2_k=3.yaml
  configs/h3/h3_cifar_init_unit_sphere_scale=5_k=3.yaml
  
)

for config in "${configs[@]}"; do
  python -m src.train_autoencoder "$config"
done
