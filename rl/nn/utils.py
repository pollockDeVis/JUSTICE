import torch.nn as nn
from rl.nn.enumerations import Activations


def get_activation_function_from_enum(activation: Activations):
    if activation == Activations.RELU:
        return nn.ReLU()
    elif activation == Activations.PRELU:
        return nn.PReLU()
    else:
        raise NotImplementedError(f"Activation {activation} is not supported.")
