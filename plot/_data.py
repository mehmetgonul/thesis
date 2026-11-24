import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.config import parse_config
from src.hash import is_current_ode_metadata, make_hash


def evaluation_rows(path):
    """Read a fixed snapshot, ignoring an unfinished trailing JSONL row."""
    path = Path(path)
    previous_alpha = None
    with path.open("rb") as source:
        snapshot_size = source.seek(0, 2)
        source.seek(0)
        consumed = 0
        for number, line in enumerate(source, 1):
            consumed += len(line)
            if consumed > snapshot_size or not line.endswith(b"\n"):
                break
            if b'"generalization_error"' not in line:
                continue
            try:
                row = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{number}: invalid evaluation JSON") from error
            if row.get("generalization_error") is None:
                continue
            alpha, error = float(row["alpha"]), float(row["generalization_error"])
            if not np.isfinite((alpha, error)).all():
                raise ValueError(f"{path}:{number}: nonfinite evaluation")
            if previous_alpha is not None and alpha <= previous_alpha:
                raise ValueError(f"{path}:{number}: alpha must increase")
            previous_alpha = alpha
            yield row


def read_curve(path):
    """Keep only evaluation coordinates, without retaining matrix histories."""
    alphas, errors = [], []
    for row in evaluation_rows(path):
        alphas.append(row["alpha"])
        errors.append(row["generalization_error"])
    return np.asarray(alphas, dtype=float), np.asarray(errors, dtype=float)


def read_last_point(path):
    """Find the last complete evaluation by reading backwards from the log end."""
    path = Path(path)
    with path.open("rb") as source:
        end = source.seek(0, 2)
        window = 65536
        while True:
            start = max(0, end - window)
            source.seek(start)
            if start:
                source.readline()
            lines = source.read(max(0, end - source.tell())).splitlines(keepends=True)
            for line in reversed(lines):
                if not line.endswith(b"\n") or b'"generalization_error"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"{path}: invalid evaluation JSON") from error
                if row.get("generalization_error") is not None:
                    if not np.isfinite((row["alpha"], row["generalization_error"])).all():
                        raise ValueError(f"{path}: nonfinite final evaluation")
                    return [row]
            if start == 0:
                return []
            window *= 2


def load_run(config_path, artifacts_dir, seed):
    """Load completed ODE encoder and decoder histories for a paper figure."""
    config = parse_config(config_path)
    directory = Path(artifacts_dir) / make_hash(run_config=config) / "autoencoder" / f"seed_{seed}"
    log = directory / "ode_loss.jsonl"
    metadata_path = directory / "ode_meta.json"
    last = read_last_point(log) if log.is_file() else []
    metadata = {}
    if is_current_ode_metadata(metadata_path):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    complete = bool(
        last
        and np.isclose(metadata.get("final_alpha", -1), config.alpha, rtol=0, atol=1e-8)
        and np.isclose(last[0]["alpha"], config.alpha, rtol=0, atol=1e-8)
    )
    if not complete:
        final_alpha = last[0]["alpha"] if last else None
        print(f"Pending: {config_path} (last alpha: {final_alpha})")
        return None

    alphas, encoder, decoder = [], [], []
    for row in evaluation_rows(log):
        alphas.append(row["alpha"])
        encoder.append(row["Q0"])
        decoder.append(row["T0"])
    encoder = np.asarray(encoder, dtype=float)
    decoder = np.asarray(decoder, dtype=float)
    expected = (len(alphas), config.K, config.K)
    if encoder.shape != expected or decoder.shape != expected:
        raise ValueError(f"{log}: invalid order-parameter shape")
    if not np.isfinite(encoder).all() or not np.isfinite(decoder).all():
        raise ValueError(f"{log}: nonfinite order parameters")
    return SimpleNamespace(config=config, alpha=np.asarray(alphas, dtype=float),
                           encoder=encoder, decoder=decoder)


def instantaneous_correlations(matrices):
    """Normalize each matrix by its own contemporaneous diagonal norms."""
    matrices = np.asarray(matrices, dtype=float)
    norms = np.sqrt(np.maximum(np.diagonal(matrices, axis1=-2, axis2=-1), 0))
    denominator = norms[..., :, None] * norms[..., None, :]
    return np.divide(matrices, denominator, out=np.full_like(matrices, np.nan),
                     where=denominator > 0)
