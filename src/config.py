from dataclasses import MISSING, dataclass
from typing import Optional
import yaml
from pathlib import Path
from dataclasses import fields, is_dataclass
from typing import get_origin, get_args, Union, Any


@dataclass(frozen=False)
class RunConfig:
    dataset: str # "cifar-10_gaussian" | "cifar-10_real" | "fashion_mnist" | "spiked_wishart"
    activation_function: str # erf, relu, linear, gelu, swish, softplus, tanh, h2, h3, h4, sine, h3+h2
    tied: bool
    bias: bool
    alpha: int
    learning_rate: float
    D: int # input size - determined by the data set choice
    K: int # hidden state dimension
    preprocess_data: bool = True
    init_std: float = 0.01
    N: Optional[int] = None # GELU/Swish/Softplus/tanh Hermite order; defaults to 30
    test_loss: bool = False
    ode_sol: bool = False
    sgd: Optional[bool] = None
    wishart_samples: Optional[int] = None
    spike_multipliers: Optional[list[float]] = None


def resolve_run_modes(
    run_config: RunConfig | dict,
    *,
    ode_supported: bool = True,
) -> tuple[bool, bool]:
    """Return ``(run_sgd, run_ode)`` with legacy defaults preserved."""

    if isinstance(run_config, dict):
        ode_requested = bool(run_config.get("ode_sol", False))
        sgd_setting = run_config.get("sgd")
    else:
        ode_requested = bool(run_config.ode_sol)
        sgd_setting = run_config.sgd

    run_sgd = not ode_requested if sgd_setting is None else bool(sgd_setting)
    if ode_requested and not ode_supported and sgd_setting is None:
        run_sgd = True
    return run_sgd, ode_requested and ode_supported


def _unwrap_optional(annotation):
    """Optional[T] | Union[T, NoneType]  →  T"""
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _build(cls, payload: Any):
    """
    Convert *payload* (dict/list/scalar) into an instance of **cls**
    according to the dataclass field annotations found in *cls*.
    """
    cls = _unwrap_optional(cls)

    # Primitive or already‑converted value
    if not is_dataclass(cls):
        return payload

    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(payload).__name__}")

    kwargs = {}
    for f in fields(cls):
        raw = payload.get(f.name)
        if raw is None:
            if f.default is not MISSING:
                kwargs[f.name] = f.default
                continue
            kwargs[f.name] = None
            continue

        anno   = f.type
        origin = get_origin(anno)
        args   = get_args(anno)

        if origin is list:
            kwargs[f.name] = [_build(args[0], item) for item in raw]
        elif origin is dict:
            k_t, v_t = args
            kwargs[f.name] = {k: _build(v_t, v) for k, v in raw.items()}
        else:
            kwargs[f.name] = _build(anno, raw)

    return cls(**kwargs)


def parse_config(src: Union[str, Path, dict]) -> RunConfig:
    """
    Accepts
      • a `str` / `Path` pointing to a YAML/JSON file, **or**
      • an in memory `dict`.
    Returns a fully populated `RunConfig`.
    """
    if isinstance(src, (str, Path)):
        with open(src, "r", encoding="utf-8") as fh:
            cfg_dict = yaml.safe_load(fh)
    elif isinstance(src, dict):
        cfg_dict = src
    else:
        raise TypeError(f"parse_config() needs str, Path or dict - got {type(src).__name__}")

    if "runs" in cfg_dict:
        raise ValueError("Expected a single RunConfig, not an experiment config with runs.")

    return _build(RunConfig, cfg_dict)
