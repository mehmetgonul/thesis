#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && $1 != --artifacts ) ]]; then
  echo "Usage: $0 [--artifacts]" >&2
  exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

find . -name .git -prune -o -type d -name __pycache__ -prune -exec rm -rf -- {} +
find . -name .git -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' \) -exec rm -f -- {} +

if [[ ${1:-} == --artifacts ]]; then
  rm -rf -- artifacts
fi
