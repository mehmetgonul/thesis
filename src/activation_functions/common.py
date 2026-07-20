from src.activation_functions.erf import Erf
from src.activation_functions.relu import ReLU
from src.activation_functions.gelu import GeLU
from src.activation_functions.linear import Linear
from src.activation_functions.hermite import HermitePolynomial
from src.activation_functions.silu import Silu

def activation_function_factory(activation_function: str, **kwargs):
    
    activation_function = activation_function.lower()
    
    if activation_function == "erf":
        return Erf(**kwargs)
    elif activation_function == "relu":
        return ReLU(**kwargs)
    elif activation_function == "linear":
        return Linear(**kwargs)
    elif activation_function in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return HermitePolynomial(int(activation_function[1:])) # only pass the integer value for easier indexing
    elif activation_function == "gelu":
        return GeLU(**kwargs)
    elif activation_function == "silu":
        return Silu(**kwargs)
    else:
        raise ValueError(f"Unknown activation function: {activation_function}")
