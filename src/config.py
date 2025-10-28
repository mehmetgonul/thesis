from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path
from dataclasses import fields, is_dataclass, fields
from typing import get_origin, get_args, Union, Type, Any, Dict

@dataclass(frozen=True)
class DataManagerConfig:
    dataset: str # "wishart" | "cifar-10" | "fashion_mnist"
    rank: int
    batch_size: int
    noise: float = 0.
    gaussian: bool = False

@dataclass(frozen=True)
class RunConfig:
    data_manager: DataManagerConfig
    activation_function: str # "erf" | "relu" | "linear" | "gelu" | "exp"
    tied: bool
    bias: bool
    training_method: str # "analytical_integrals" | "hermite_integrals" | "sgd"
    learning_rate: float
    step_count: int
    N: Optional[int]
    
    line_type: str # "dashed" | "solid"
    line_color: str # color of the lines
    
    D: int # I/O dimension (if the dataset is synthetic this will be overriden)
    K: int # hidden state dimension
    

@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    runs: list[RunConfig]
    
    plot_name: str
    x_label: str
    y_label: str
    
    
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


def parse_config(src: Union[str, Path, dict]) -> ExperimentConfig:
    """
    Accepts
      • a `str` / `Path` pointing to a YAML/JSON file, **or**
      • an in memory `dict`.
    Returns a fully populated `ExperimentConfig`.
    """
    if isinstance(src, (str, Path)):
        with open(src, "r", encoding="utf-8") as fh:
            cfg_dict = yaml.safe_load(fh)
    elif isinstance(src, dict):
        cfg_dict = src
    else:
        raise TypeError(f"parse_config() needs str, Path or dict - got {type(src).__name__}")

    return _build(ExperimentConfig, cfg_dict)