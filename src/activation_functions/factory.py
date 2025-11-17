from src.activation_functions.erf import Erf

def activation_function_factory(activation_function: str, **kwargs):
    
    activation_function = activation_function.lower()
    
    if activation_function == "erf":
        return Erf(**kwargs)
    # elif activation_function == "relu":
    #     return ReLU(**kwargs)
    # elif activation_function == "linear":
    #     return Linear()
    # elif activation_function == "gelu":
    #     return GeLU(**kwargs)
    # elif activation_function == "exponential" or activation_function == "exp": # TODO: repo has mixed usage, unify it sometime
    #     return Exponential(**kwargs)
    else:
        raise ValueError(f"Unknown activation function: {activation_function}")
