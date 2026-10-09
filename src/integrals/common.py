"""Select analytical moments or a Hermite approximation by activation."""

from .base import Integral
from .erf import ErfIntegral
from .gelu import GeLUIntegral
from .linear import LinearIntegral
from .relu import ReLUIntegral
from .silu import SiluIntegral


def integral_for_activation(activation_function: str, N: int | None = None) -> Integral:
    name = activation_function.lower()
    if name == "gelu":
        return GeLUIntegral(N=30 if N is None else N)
    if name == "silu":
        return SiluIntegral(N=30 if N is None else N)
    analytical = {"erf": ErfIntegral, "relu": ReLUIntegral, "linear": LinearIntegral}
    if name in analytical:
        return analytical[name]()
    raise ValueError(f"Unknown activation function: {activation_function}")
