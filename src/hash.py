from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _to_dict(value):
    return asdict(value) if is_dataclass(value) else value


def make_hash(run_config=None, seed=None):
    payload = {
        k: v
        for k, v in {
            "run_config": _to_dict(run_config) if run_config is not None else None,
            "seed": int(seed) if seed is not None else None,
        }.items()
        if v is not None
    }

    if not payload:
        raise ValueError("empty hash payload")

    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_hash(run_config=None, seed=None, artifacts_dir=Path("artifacts")):
    run_hash = make_hash(run_config=run_config)
    base = artifacts_dir / run_hash / "autoencoder" / f"seed_{seed}"
    checkpoint = base / "model.pt"
    results = base / "loss.jsonl"
    return (str(checkpoint), str(results)) if checkpoint.is_file() and results.is_file() else None
