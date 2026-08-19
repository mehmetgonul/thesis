from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _to_dict(value):
    payload = asdict(value) if is_dataclass(value) else value
    if isinstance(payload, dict):
        payload = dict(payload)
        
        if payload.get("init_unit_sphere") is False:
            payload.pop("init_unit_sphere")
            
        if payload.get("unit_sphere_scale") in (None, 1, 1.0):
            payload.pop("unit_sphere_scale")

        if payload.get("preprocess_data") is True:
            payload.pop("preprocess_data")

        if payload.get("test_loss") is False:
            payload.pop("test_loss")

        # The solver selects which artifact is produced, not the identity of
        # the experiment. Online and ODE results therefore share one hash.
        payload.pop("ode_sol", None)
        
    return payload


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
    if isinstance(run_config, dict):
        ode_sol = bool(run_config.get("ode_sol", False))
        tied = bool(run_config.get("tied", False))
        bias = bool(run_config.get("bias", False))
    else:
        ode_sol = bool(getattr(run_config, "ode_sol", False))
        tied = bool(getattr(run_config, "tied", False))
        bias = bool(getattr(run_config, "bias", False))
    ode_sol = ode_sol and not tied and not bias
    checkpoint = base / ("ode_state.npz" if ode_sol else "model.pt")
    results = base / ("ode_loss.jsonl" if ode_sol else "loss.jsonl")
    required = [checkpoint, results]
    if ode_sol:
        required.extend([base / "ode_meta.json", base / "initial_model.pt"])
    return (
        (str(checkpoint), str(results))
        if all(path.is_file() for path in required)
        else None
    )
