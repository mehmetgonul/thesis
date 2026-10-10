from src.activation_functions.erf import Erf
from src.activation_functions.relu import ReLU
from src.activation_functions.gelu import GeLU
from src.activation_functions.linear import Linear
from src.activation_functions.swish import Swish
from src.activation_functions.softplus import Softplus
from src.activation_functions.tanh import Tanh
from src.activation_functions.h2 import H2
from src.activation_functions.h3 import H3
from src.activation_functions.h3_h2 import H3H2
from src.activation_functions.sine import Sine

def activation_function_factory(activation_function: str, **kwargs):
    functions = {
        "erf": Erf, "relu": ReLU, "linear": Linear, "gelu": GeLU,
        "swish": Swish, "softplus": Softplus, "tanh": Tanh,
        "h2": H2, "h3": H3, "h3+h2": H3H2, "sine": Sine,
    }
    activation_function = activation_function.lower()
    if activation_function not in functions:
        raise ValueError(f"Unknown activation function: {activation_function}")
    return functions[activation_function](**kwargs)
