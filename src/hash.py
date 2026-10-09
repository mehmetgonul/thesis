from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from src.config import resolve_run_modes


ODE_INTEGRATION_COORDINATES = "rank_one_factor_v1"
ODE_ORDER_PARAMETER_LOGGING_VERSION = 1
# ERF I3 was corrected; GELU now uses a truncated Hermite expansion.
INTEGRAL_VERSIONS = {"erf": 2, "gelu": 2}


def is_current_ode_metadata(path: Path) -> bool:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(metadata, dict)
        and metadata.get("integration_coordinates")
        == ODE_INTEGRATION_COORDINATES
        and metadata.get("order_parameter_logging_version")
        == ODE_ORDER_PARAMETER_LOGGING_VERSION
    )


def _to_dict(value):
    payload = asdict(value) if is_dataclass(value) else value
    if isinstance(payload, dict):
        payload = dict(payload)
        activation = payload.get("activation_function", "").lower()
        if activation in INTEGRAL_VERSIONS:
            payload["integral_version"] = INTEGRAL_VERSIONS[activation]
        
        if payload.get("init_unit_sphere") is False:
            payload.pop("init_unit_sphere")
            
        if payload.get("unit_sphere_scale") in (None, 1, 1.0):
            payload.pop("unit_sphere_scale")

        if payload.get("preprocess_data") is True:
            payload.pop("preprocess_data")

        if payload.get("test_loss") is False:
            payload.pop("test_loss")

        if payload.get("wishart_samples") is None:
            payload.pop("wishart_samples", None)

        if payload.get("spike_multipliers") is None:
            payload.pop("spike_multipliers", None)

        # The solver selects which artifact is produced, not the identity of
        # the experiment. Online and ODE results therefore share one hash.
        payload.pop("ode_sol", None)
        payload.pop("sgd", None)
        
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
    """Return completed artifact paths; ODE-only returns its trajectory and metadata."""
    run_hash = make_hash(run_config=run_config)
    base = artifacts_dir / run_hash / "autoencoder" / f"seed_{seed}"
    if isinstance(run_config, dict):
        tied = bool(run_config.get("tied", False))
        bias = bool(run_config.get("bias", False))
    else:
        tied = bool(getattr(run_config, "tied", False))
        bias = bool(getattr(run_config, "bias", False))
    run_sgd, run_ode = resolve_run_modes(
        run_config,
        ode_supported=not tied and not bias,
    )
    if not run_sgd and not run_ode:
        return None

    model_path = base / "model.pt"
    loss_path = base / "loss.jsonl"
    ode_loss_path = base / "ode_loss.jsonl"
    required = []
    if run_sgd:
        required.extend([model_path, loss_path])
    metadata = None
    if run_ode:
        metadata = base / "ode_meta.json"
        required.extend(
            [ode_loss_path, metadata, base / "initial_model.pt"]
        )
    if not all(path.is_file() for path in required):
        return None
    if metadata is not None and not is_current_ode_metadata(metadata):
        return None
    if run_sgd:
        return str(model_path), str(loss_path)
    return str(ode_loss_path), str(metadata)
