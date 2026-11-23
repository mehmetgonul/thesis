"""Select analytical moments or a Hermite approximation by activation."""

from .base import Integral
from .erf import ErfIntegral
from .gelu import GeLUIntegral
from .linear import LinearIntegral
from .relu import ReLUIntegral
from .sine import SineIntegral
from .hermite import HermiteIntegral
from src.activation_functions.common import activation_function_factory


def integral_for_activation(activation_function: str, N: int | None = None) -> Integral:
    name = activation_function.lower()
    if name == "gelu":
        return GeLUIntegral(N=30 if N is None else N)
    if name in {"swish", "softplus", "tanh"}:
        return HermiteIntegral(activation_function_factory(name), N=30 if N is None else N)
    if name in {"h2", "h3", "h4", "h3+h2"}:
        return HermiteIntegral(activation_function_factory(name), N={"h2": 2, "h3": 3, "h4": 4, "h3+h2": 3}[name])
    analytical = {"erf": ErfIntegral, "relu": ReLUIntegral,
                  "linear": LinearIntegral, "sine": SineIntegral}
    if name in analytical:
        return analytical[name]()
    raise ValueError(f"Unknown activation function: {activation_function}")
